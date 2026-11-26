"""One GPU worker thread (one job at a time) and the retention sweeper."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from ..config import Settings
from ..errors import ApiError
from ..imaging import pipeline
from ..imaging.geometry import Rect
from ..imaging.tile_cache import TileCache
from ..models import Registry
from ..store import Store, dumps

log = logging.getLogger("painite.worker")


def source_path(settings: Settings, owner: str, sha: str) -> Path:
    return settings.sources_dir / owner / sha


class Worker:
    def __init__(self, settings: Settings, store: Store, registry: Registry):
        self.settings = settings
        self.tile_cache = TileCache(settings.tile_cache_mb * 1024 * 1024)
        self.store = store
        self.registry = registry
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # Core ML (Macs) converts each model for the window every tile gets.
        for model in registry.models.values():
            if hasattr(model, "tiling"):
                model.tiling = (settings.tile_size, settings.tile_overlap)

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._run, name="painite-gpu-worker", daemon=True),
            threading.Thread(target=self._warm, name="painite-warmup", daemon=True),
            threading.Thread(target=self._sweep_loop, name="painite-sweeper", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        for t in self._threads:
            t.join(timeout=10)

    def notify(self) -> None:
        self._wake.set()

    def _warm(self) -> None:
        """Load the default models at startup so the first job is not a cold start."""
        for model_id in self.registry.defaults.values():
            model = self.registry.get(model_id)
            load = getattr(model, "prepare_coreml", None) or getattr(model, "load", None)
            if load is None or self._stop.is_set():
                continue
            try:
                load()
            except Exception:  # noqa: BLE001 - the job will report the failure
                log.exception("could not load %s", model_id)

    # Worker -----------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self.store.next_queued()
            if job is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            if not self.store.claim(job["id"]):
                continue
            self.process(job)

    def process(self, job: dict) -> None:
        job_id = job["id"]
        started = time.time()
        timings = {"queue": (started - job["created_at"]) * 1000}
        steps = []
        missing = None
        for st in _job_steps(job):
            m = self.registry.get(st["model"])
            if m is None:
                missing = missing or st["model"]
            steps.append(pipeline.Step(model=m, params=st["params"]))
        last_progress = [0.0]

        def on_progress(p: float) -> None:
            if p - last_progress[0] >= 0.05 or p >= 1.0:
                last_progress[0] = p
                self.store.update_job(job_id, progress=round(p, 3))

        def is_canceled() -> bool:
            row = self.store.get_job(None, job_id)
            return row is None or bool(row["cancel_requested"])

        log.info("job start id=%s kind=%s op=%s model=%s", job_id, job["kind"], job["operation"], job["model"])
        try:
            if missing is not None:
                raise ApiError(409, "model_unavailable", f"model {missing} is not installed")
            region = Rect(**json.loads(job["region"])) if job["region"] else None
            out = pipeline.run(
                source_path=source_path(self.settings, job["owner"], job["source_sha"]),
                steps=steps,
                region=region,
                output=json.loads(job["output"]),
                work_dir=self.settings.work_dir,
                results_dir=self.settings.results_dir,
                job_id=job_id,
                tile=self.settings.tile_size,
                overlap=self.settings.tile_overlap,
                max_output_pixels=self.settings.max_output_pixels,
                on_progress=on_progress,
                is_canceled=is_canceled,
                cache=self.tile_cache,
                cache_key=(job["owner"], job["source_sha"]),
                cache_write=job["kind"] == "preview",
            )
        except pipeline.Canceled:
            self._finish(job_id, "canceled", timings, started, code="canceled", message="canceled by client")
            return
        except ApiError as e:
            self._finish(job_id, "failed", timings, started, code=e.code, message=e.message)
            return
        except Exception:  # noqa: BLE001 - a failed job must not kill the worker
            log.exception("job failed id=%s", job_id)
            self._finish(job_id, "failed", timings, started, code="processing_failed", message="processing failed")
            return

        timings.update(out.timings_ms)
        now = time.time()
        timings["total"] = (now - job["created_at"]) * 1000
        if is_canceled():
            out.path.unlink(missing_ok=True)
            self._finish(job_id, "canceled", timings, started, code="canceled", message="canceled by client")
            return
        self.store.update_job(
            job_id,
            state="completed",
            progress=1.0,
            finished_at=now,
            timings=dumps(_round(timings)),
            result_path=str(out.path),
            result_bytes=out.bytes,
            result_sha=out.sha256,
            result_format=out.format,
            result_width=out.width,
            result_height=out.height,
            info=dumps(out.info) if out.info else None,
            result_expires_at=now + self.settings.result_retention_s,
        )
        self.store.touch_source(job["owner"], job["source_sha"])
        log.info("job done id=%s bytes=%d timings=%s", job_id, out.bytes, _round(timings))

    def _finish(self, job_id: str, state: str, timings: dict, started: float, code: str, message: str) -> None:
        now = time.time()
        timings["total"] = (now - started) * 1000 + timings.get("queue", 0)
        self.store.update_job(
            job_id, state=state, finished_at=now, timings=dumps(_round(timings)), error_code=code,
            error_message=message,
        )
        log.info("job %s id=%s code=%s", state, job_id, code)

    # Sweeper ----------------------------------------------------------------

    def _sweep_loop(self) -> None:
        while not self._stop.wait(self.settings.sweep_interval_s):
            try:
                self.sweep()
            except Exception:  # noqa: BLE001
                log.exception("sweep failed")

    def sweep(self, startup: bool = False) -> None:
        now = time.time()
        s = self.settings
        for row in self.store.expired_results(now):
            Path(row["result_path"]).unlink(missing_ok=True)
            self.store.update_job(row["id"], result_deleted=1)
        for row in self.store.idle_sources(now - s.source_idle_s):
            source_path(s, row["owner"], row["sha256"]).unlink(missing_ok=True)
            self.store.delete_source(row["owner"], row["sha256"])
        self.store.purge_old_jobs(now - s.idempotency_window_s)
        if startup:
            # Crash cleanup: temp files, and files the database no longer knows about.
            for p in s.work_dir.glob("*"):
                p.unlink(missing_ok=True)
            live = self.store.live_result_paths()
            for p in s.results_dir.glob("*"):
                if str(p) not in live:
                    p.unlink(missing_ok=True)
            known = {(r["owner"], r["sha256"]) for r in self.store.all_sources()}
            for p in s.sources_dir.glob("*/*"):
                if (p.parent.name, p.name) not in known:
                    p.unlink(missing_ok=True)


def _job_steps(job: dict) -> list[dict]:
    if job.get("steps"):
        return json.loads(job["steps"])
    return [{"operation": job["operation"], "model": job["model"], "params": json.loads(job["params"])}]


def disk_usage(settings: Settings) -> int:
    total = 0
    for d in (settings.sources_dir, settings.work_dir, settings.results_dir):
        for p in d.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


def _round(t: dict) -> dict:
    return {k: round(v, 1) for k, v in t.items()}
