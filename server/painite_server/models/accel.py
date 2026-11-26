"""Optional accelerated backends for a model's network on NVIDIA GPUs:
TensorRT engines and torch.compile. PyTorch stays the reference and the
fallback; Apple MPS and the CPU always run PyTorch.

Which backend a model asks for is in its manifest's `config.accel`:

    "accel": {"backend": "tensorrt", "precision": "fp16"}   # TensorRT, FP16 (weakly typed)
    "accel": {"backend": "tensorrt", "precision": "fp16", "typing": "strong"}
    "accel": {"backend": "compile"}                          # torch.compile
    "accel": {"backend": "pytorch"}                          # (or no key) plain PyTorch

An installed manifest without the key takes it from the server's catalog entry
for the same model id, so models installed before this feature need no re-pull.
BF16 engines are refused: TensorRT BF16 builds can run away (~46 GB RAM,
20 min) and BF16 is less accurate than FP16 for these models.

PAINITE_ACCEL turns it off (`off`), allows every backend (`auto`, the default)
or only the listed ones (`tensorrt`, `compile`, `tensorrt,compile`).

Life of an accelerated model (the server serves PyTorch throughout):

1. When the model's weights load on a CUDA device, a build is queued on one
   background builder thread (builds never run in parallel).
2. TensorRT: a subprocess (`python -m painite_server.models.accel_build`)
   exports the network at the fixed tile-window shape, builds the engine,
   reloads it from disk and checks it against PyTorch FP32 on fixed golden
   tiles. The engine and a JSON record of the result are cached under
   `<data_dir>/engines/`, keyed by everything that makes an engine invalid
   (`cache_key`). The subprocess runs under a time limit and is killed after it.
   torch.compile: compiled and checked in-process (inductor keeps its own
   cache under the same directory).
3. A build that passes is staged and switched to between jobs (`begin_job`),
   never during one. A build or check that fails is logged and recorded in the
   cache, so it is not retried on every start; the model stays on PyTorch. A
   torch.compile that raises disables torch.compile for the whole process (a
   failed compile can leave dynamo's state broken for later ones).
4. Windows of any other shape (images smaller than a tile window) and any
   error from the accelerated network at run time fall back to PyTorch.

The active backend is part of the tile cache key (imaging/pipeline.py), so a
preview rendered by PyTorch while an engine was building is never stitched
into an export rendered by TensorRT.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("painite.models")

BACKENDS = ("tensorrt", "compile")
PRECISIONS = ("fp16", "fp32")
TYPINGS = ("weak", "strong")
# Bump when the engine format or the build recipe changes: old engines are rebuilt.
FORMAT = 1
DEFAULT_TIMEOUT_S = 900
DEFAULT_MIN_PSNR = 50.0
# The candidate's max abs diff vs FP32 (8-bit) may be at most this, or twice
# the PyTorch server path's own max diff, whichever is larger.
MAX_ABS_FLOOR = 16


def requested() -> frozenset[str]:
    """The backends PAINITE_ACCEL allows."""
    raw = (os.environ.get("PAINITE_ACCEL") or "auto").strip().lower()
    if raw in ("auto", "on", "1", "true", "yes"):
        return frozenset(BACKENDS)
    if raw in ("off", "0", "false", "no", "none", "pytorch"):
        return frozenset()
    want = {p.strip() for p in raw.split(",") if p.strip()}
    bad = want - set(BACKENDS)
    if bad:
        log.warning("PAINITE_ACCEL: ignoring unknown backend(s) %s (known: %s)", sorted(bad), BACKENDS)
    return frozenset(want & set(BACKENDS))


def mode_label() -> str:
    want = requested()
    if not want:
        return "off"
    return "auto" if want == set(BACKENDS) else ",".join(b for b in BACKENDS if b in want)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        log.warning("%s is not a number; using %s", name, default)
        return default


def build_timeout_s() -> float:
    return _env_float("PAINITE_ACCEL_BUILD_TIMEOUT_S", DEFAULT_TIMEOUT_S)


def min_psnr() -> float:
    return _env_float("PAINITE_ACCEL_MIN_PSNR", DEFAULT_MIN_PSNR)


@dataclass(frozen=True)
class Spec:
    backend: str  # tensorrt | compile
    precision: str = "fp16"  # tensorrt only
    typing: str = "weak"  # tensorrt only: weak (TensorRT picks per-layer precision) | strong (.half() network)

    @property
    def label(self) -> str:
        """The backend's name in status output and tile cache keys."""
        if self.backend == "tensorrt":
            return f"tensorrt-{self.precision}" + ("-strong" if self.typing == "strong" else "")
        return self.backend


def parse_spec(raw: Any) -> Spec | None:
    """`config.accel` -> Spec, or None for plain PyTorch. Raises ValueError."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"backend": raw}
    if not isinstance(raw, dict):
        raise ValueError("config.accel must be an object")
    backend = raw.get("backend", "pytorch")
    if backend == "pytorch":
        return None
    if backend not in BACKENDS:
        raise ValueError(f"config.accel.backend must be pytorch or one of {BACKENDS}")
    if backend == "compile":
        return Spec("compile", precision="", typing="")
    precision = raw.get("precision", "fp16")
    if precision == "bf16":
        raise ValueError("config.accel: BF16 TensorRT engines are not supported (builds can run away)")
    if precision not in PRECISIONS:
        raise ValueError(f"config.accel.precision must be one of {PRECISIONS}")
    typing = raw.get("typing", "weak")
    if typing not in TYPINGS:
        raise ValueError(f"config.accel.typing must be one of {TYPINGS}")
    return Spec("tensorrt", precision, typing)


def spec_for(manifest, catalog: Callable[[], dict] | None = None) -> Spec | None:
    """The installed manifest's accel spec, else the catalog entry's for the same id."""
    if "accel" in manifest.config:
        return parse_spec(manifest.config["accel"])
    if catalog is None:
        from .manifest import catalog  # noqa: PLC0415 - avoid import cycle
    try:
        entry = catalog().get(manifest.id)
    except Exception:  # noqa: BLE001 - a broken catalog must not stop the model loading
        log.exception("could not read the model catalog")
        return None
    return parse_spec(entry.config.get("accel")) if entry is not None else None


def cache_key(model_id: str, weights: list[str], config: dict, spec: Spec, shape: tuple, env: dict) -> str:
    """Names an engine (or a compile check) by everything that invalidates it:
    the weights and network code (manifest SHA-256s), the model config, the
    backend and precision, the input shape, and the software and GPU it was
    built with. Engines are not portable across TensorRT versions or GPUs."""
    blob = json.dumps({"format": FORMAT, "model": model_id, "weights": sorted(weights), "config": config,
                       "spec": asdict(spec), "shape": list(shape), "env": env}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def environment(torch, backend: str) -> dict:
    """Software and GPU versions that go into the cache key (call on a CUDA machine)."""
    major, minor = torch.cuda.get_device_capability(0)
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
           "sm": f"{major}.{minor}"}
    if backend == "tensorrt":
        import tensorrt  # noqa: PLC0415
        import torch_tensorrt  # noqa: PLC0415

        env["tensorrt"] = tensorrt.__version__
        env["torch_tensorrt"] = torch_tensorrt.__version__
    return env


# Golden tiles ---------------------------------------------------------------

def golden_tiles(h: int, w: int, n: int = 4) -> list[np.ndarray]:
    """n fixed HxWx3 float32 tiles in [0, 1]: noisy gradient, fractal texture,
    blurred hard edges, and clipped highlights/shadows. Synthetic, so the
    server needs no test images; they are what exposed TensorRT's silent FP16
    errors in the evaluation (garbage on structure, NaN on bright content)."""
    from . import _resize_axis  # noqa: PLC0415

    rng = np.random.default_rng(20261126)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= max(h - 1, 1)
    xx /= max(w - 1, 1)

    def up(a):  # bilinear resize of a small grid to h x w
        return _resize_axis(_resize_axis(a, 0, h), 1, w)

    def fractal():
        out = np.zeros((h, w, 3), np.float32)
        for octave in range(1, 7):
            g = 2 ** octave
            out += up(rng.random((g + 1, g + 1, 3), dtype=np.float32)) / 1.6 ** octave
        out -= out.min()
        return out / max(float(out.max()), 1e-6)

    def box_blur(a, r):
        k = 2 * r + 1
        p = np.pad(a, ((r, r), (r, r), (0, 0)), mode="edge")
        c = p.cumsum(0).cumsum(1)
        c = np.pad(c, ((1, 0), (1, 0), (0, 0)))
        return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)

    tiles = []
    # 1. smooth colour gradient with sensor-like noise
    t = np.stack([0.2 + 0.6 * xx, 0.3 + 0.4 * yy, 0.5 + 0.3 * (xx - yy) / 2], -1)
    tiles.append(t + rng.normal(0, 0.05, t.shape).astype(np.float32))
    # 2. natural-ish multi-scale texture
    tiles.append(fractal())
    # 3. hard-edged shapes and thin lines, slightly blurred, with light noise
    t = np.full((h, w, 3), 0.5, np.float32)
    for _ in range(40):
        y0, x0 = rng.integers(0, h), rng.integers(0, w)
        hh, ww = rng.integers(4, h // 3), rng.integers(4, w // 3)
        t[y0:y0 + hh, x0:x0 + ww] = rng.random(3)
    for _ in range(30):
        if rng.random() < 0.5:
            t[rng.integers(0, h), :] = rng.random(3)
        else:
            t[:, rng.integers(0, w)] = rng.random(3)
    t = box_blur(t, 2)
    tiles.append(t + rng.normal(0, 0.01, t.shape).astype(np.float32))
    # 4. bright and dark extremes: clipped highlights, deep shadows, texture between
    t = fractal() * 1.8 - 0.4
    tiles.append(t + rng.normal(0, 0.02, t.shape).astype(np.float32))
    return [np.clip(x, 0, 1).astype(np.float32) for x in tiles[:n]]


def compare(outs: list, refs: list) -> dict:
    """min PSNR (dB), max and mean abs diff of 8-bit outputs (torch tensors)."""
    psnrs, maxd, meand, finite = [], 0, [], True
    for a, b in zip(outs, refs):
        finite = finite and bool(a.isfinite().all())
        ua = (a.float().nan_to_num(0).clamp(0, 1) * 255).round()
        ub = (b.float().clamp(0, 1) * 255).round()
        d = (ua - ub).abs()
        mse = float((d ** 2).mean())
        psnrs.append(99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse))
        maxd = max(maxd, int(d.max()))
        meand.append(float(d.mean()))
    return {"psnr_min": round(min(psnrs), 2), "max_abs": maxd, "mean_abs": round(float(np.mean(meand)), 4),
            "finite": finite}


def verdict(candidate: dict, server: dict, threshold: float | None = None) -> tuple[bool, str]:
    """Is the candidate close enough to PyTorch FP32? `server` is today's
    PyTorch path measured the same way, for the max-diff allowance."""
    threshold = min_psnr() if threshold is None else threshold
    if not candidate["finite"]:
        return False, "output has NaN or Inf"
    if candidate["psnr_min"] < threshold:
        return False, f"PSNR {candidate['psnr_min']} dB < {threshold} dB"
    limit = max(MAX_ABS_FLOOR, 2 * server["max_abs"])
    if candidate["max_abs"] > limit:
        return False, f"max abs diff {candidate['max_abs']} > {limit}"
    return True, f"PSNR {candidate['psnr_min']} dB, max abs diff {candidate['max_abs']} (PyTorch path: " \
                 f"{server['psnr_min']} dB, {server['max_abs']})"


def check(torch, fn: Callable, ref: Callable, server: Callable, shape: tuple, device: str = "cuda") -> dict:
    """Run golden tiles through the candidate `fn`, the FP32 reference and the
    server's PyTorch path (all fp32 NCHW in -> fp32 out). Returns the record."""
    tiles = golden_tiles(shape[-2], shape[-1])
    xs = [torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1)))[None].to(device) for t in tiles]
    with torch.inference_mode():
        refs = [ref(x).float().clamp(0, 1) for x in xs]
        cand = compare([fn(x) for x in xs], refs)
        srv = compare([server(x) for x in xs], refs)
    ok, why = verdict(cand, srv)
    return {"passed": ok, "reason": why, "candidate": cand, "pytorch": srv, "tiles": len(xs)}


# Records on disk ------------------------------------------------------------

def _model_dir(engines_dir: Path, model_id: str) -> Path:
    return engines_dir / model_id.replace("/", "_").replace("@", "_")


def read_record(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def prune(model_dir: Path, keep: str) -> None:
    """Remove engines and records for other keys (an older TensorRT, driver or weights)."""
    for p in model_dir.glob("*"):
        if p.is_file() and not p.name.startswith(keep):
            p.unlink(missing_ok=True)


# The builder thread --------------------------------------------------------

_queue: queue.Queue = queue.Queue()
_builder: threading.Thread | None = None
_builder_lock = threading.Lock()
# Backends turned off for this process after an in-process failure.
_disabled: set[str] = set()


def _builder_loop() -> None:
    while True:
        acc, model = _queue.get()
        try:
            acc._build(model)
        except Exception as e:  # noqa: BLE001 - never let one model kill the builder
            log.exception("accel: %s build crashed", acc.model_id)
            acc._fail(f"build crashed: {e}", persist=False)
        finally:
            _queue.task_done()


def _enqueue(acc: Accelerator, model) -> None:
    global _builder  # noqa: PLW0603
    with _builder_lock:
        if _builder is None or not _builder.is_alive():
            _builder = threading.Thread(target=_builder_loop, name="painite-accel-builder", daemon=True)
            _builder.start()
    _queue.put((acc, model))


def wait_idle() -> None:
    """Block until every queued build has finished (tests, benchmarks)."""
    _queue.join()


class Accelerator:
    """One model's accelerated network: its build state, and the switch between
    it and PyTorch. The model supplies (see SpandrelModel):

        model.device                 resolved device ("cuda" to accelerate)
        model.accel_module()         the bare network in the server dtype
        model.accel_call(net)        net -> callable, fp32 NCHW in -> fp32 out, the server's dtype/autocast
        model.accel_model_dir        install dir (for the TensorRT build subprocess)
    """

    def __init__(self, model_id: str, spec: Spec, engines_dir: Path, weights: list[str], config: dict,
                 shape: tuple[int, int, int, int]):
        self.model_id = model_id
        self.spec = spec
        self.engines_dir = engines_dir
        self.weights = weights
        self.config = {k: v for k, v in config.items() if k != "accel"}
        self.shape = tuple(shape)
        self.state = "idle"  # idle | queued | building | ready | active | failed | unavailable | off
        self.detail = ""
        self.key: str | None = None
        self._lock = threading.Lock()
        self._staged: Callable | None = None
        self._active: Callable | None = None
        self._active_label = "pytorch"

    # Serving ---------------------------------------------------------------

    @property
    def backend(self) -> str:
        """What runs the network right now (part of the tile cache key)."""
        return self._active_label

    def begin_job(self) -> None:
        """Switch to a staged engine. Called between jobs only, so one job never mixes backends."""
        with self._lock:
            if self._staged is not None:
                self._active, self._staged = self._staged, None
                self._active_label = self.spec.label
                self.state = "active"
                log.info("accel: %s now runs on %s", self.model_id, self.spec.label)

    def call(self, x, fallback: Callable):
        """The accelerated network for the engine's shape, PyTorch otherwise."""
        fn = self._active
        if fn is None or tuple(x.shape) != self.shape:
            return fallback(x)
        try:
            return fn(x)
        except Exception as e:  # noqa: BLE001 - fall back, and stop using this engine
            log.exception("accel: %s failed at run time on %s; back to PyTorch", self.model_id, self.spec.label)
            with self._lock:
                self._active = None
                self._active_label = "pytorch"
            self._fail(f"run-time error: {e}", persist=False)
            return fallback(x)

    def status(self) -> dict:
        return {"backend": self.backend, "target": self.spec.label, "state": self.state, "detail": self.detail}

    # Building --------------------------------------------------------------

    def start(self, model) -> None:
        """Queue the build once the model is loaded (no-op unless it can run)."""
        with self._lock:
            if self.state != "idle":
                return
            if model.device != "cuda":
                self.state, self.detail = "off", f"{model.device} device (accelerated backends need CUDA)"
                return
            if self.spec.backend not in requested():
                self.state, self.detail = "off", f"PAINITE_ACCEL={mode_label()}"
                return
            self.state = "queued"
        log.info("accel: %s queued for %s", self.model_id, self.spec.label)
        _enqueue(self, model)

    def _fail(self, why: str, persist: bool = True, record: dict | None = None) -> None:
        with self._lock:
            self.state, self.detail = "failed", why
            self._staged = None
        log.warning("accel: %s stays on PyTorch: %s", self.model_id, why)
        if persist and self.key:
            write_record(self._paths()[1], {**(record or {}), "status": "failed", "reason": why,
                                            "model": self.model_id, "spec": asdict(self.spec), "at": time.time()})

    def _stage(self, fn: Callable, why: str) -> None:
        with self._lock:
            self._staged = fn
            self.state, self.detail = "ready", why
        log.info("accel: %s %s ready (%s); switching at the next job", self.model_id, self.spec.label, why)

    def _paths(self) -> tuple[Path, Path]:
        d = _model_dir(self.engines_dir, self.model_id)
        return d / f"{self.key}.ep", d / f"{self.key}.json"

    def _build(self, model) -> None:
        torch = _torch()
        if self.spec.backend in _disabled:
            self._fail(f"{self.spec.backend} disabled in this process after an earlier failure", persist=False)
            return
        try:
            env = environment(torch, self.spec.backend)
        except ImportError as e:
            with self._lock:
                self.state, self.detail = "unavailable", f"{self.spec.backend} is not installed ({e})"
            log.warning("accel: %s: %s", self.model_id, self.detail)
            return
        self.key = cache_key(self.model_id, self.weights, self.config, self.spec, self.shape, env)
        engine, rec_path = self._paths()
        rec = read_record(rec_path)
        if rec and rec.get("status") == "failed":
            with self._lock:
                self.state = "failed"
                self.detail = f"{rec.get('reason')} (cached; delete {rec_path} to retry)"
            log.warning("accel: %s stays on PyTorch: %s", self.model_id, self.detail)
            return
        with self._lock:
            self.state, self.detail = "building", ""
        if self.spec.backend == "compile":
            self._build_compile(torch, model, env)
        else:
            self._build_tensorrt(torch, model, env, engine, rec_path, rec)

    def _build_compile(self, torch, model, env: dict) -> None:
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(self.engines_dir / "inductor"))
        net = model.accel_module()
        ref_net = copy.deepcopy(net).float()
        server = model.accel_call(net)
        result: dict[str, Any] = {}
        t0 = time.perf_counter()

        def first_call():  # compile happens on the first call at the engine shape
            try:
                result["fn"] = model.accel_call(torch.compile(copy.deepcopy(net), dynamic=False))
                with torch.inference_mode():
                    result["fn"](torch.rand(self.shape, device="cuda"))
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                result["error"] = e

        th = threading.Thread(target=first_call, name="painite-accel-compile", daemon=True)
        th.start()
        th.join(build_timeout_s())
        if th.is_alive():
            _disabled.add("compile")
            self._fail(f"torch.compile took longer than {build_timeout_s():.0f} s; torch.compile is off "
                       "for this process")
            return
        if "error" in result:
            _disabled.add("compile")
            self._fail(f"torch.compile failed: {result['error']!r}"[:500])
            return
        build_s = round(time.perf_counter() - t0, 1)
        fn = result["fn"]
        checked = check(torch, fn, ref_net, server, self.shape)
        del ref_net
        torch.cuda.empty_cache()
        record = {"model": self.model_id, "spec": asdict(self.spec), "shape": list(self.shape), "env": env,
                  "build_s": build_s, "validation": checked, "at": time.time()}
        if not checked["passed"]:
            self._fail(f"output check failed: {checked['reason']}", record=record)
            return
        write_record(self._paths()[1], {**record, "status": "ok"})
        prune(self._paths()[1].parent, self.key)
        self._stage(fn, f"compiled in {build_s} s; {checked['reason']}")

    def _build_tensorrt(self, torch, model, env: dict, engine: Path, rec_path: Path, rec: dict | None) -> None:
        if not (rec and rec.get("status") == "ok" and engine.is_file()):
            engine.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "painite_server.models.accel_build", "--model-dir",
                   str(model.accel_model_dir), "--shape", ",".join(map(str, self.shape)),
                   "--precision", self.spec.precision, "--typing", self.spec.typing,
                   "--out", str(engine), "--record", str(rec_path), "--min-psnr", str(min_psnr())]
            log.info("accel: building %s engine for %s (limit %.0f s)", self.spec.label, self.model_id,
                     build_timeout_s())
            t0 = time.perf_counter()
            with open(engine.with_suffix(".log"), "w") as logf:
                try:
                    proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=build_timeout_s(),
                                          check=False)
                except subprocess.TimeoutExpired:
                    engine.unlink(missing_ok=True)
                    self._fail(f"engine build took longer than {build_timeout_s():.0f} s")
                    return
            rec = read_record(rec_path)
            if rec is None or proc.returncode != 0:
                engine.unlink(missing_ok=True)
                self._fail(f"engine build failed (exit {proc.returncode}); see {engine.with_suffix('.log')}",
                           record={"env": env})
                return
            if rec.get("status") != "ok":  # the subprocess already recorded why
                engine.unlink(missing_ok=True)
                self._fail(str(rec.get("reason")), persist=False)
                return
            rec["env"] = env
            write_record(rec_path, rec)
            prune(engine.parent, self.key)
            how = f"built in {time.perf_counter() - t0:.0f} s"
        else:
            how = "loaded from cache"
        try:
            import torch_tensorrt  # noqa: PLC0415, F401 - registers the TensorRT runtime ops

            mod = torch_tensorrt.load(str(engine)).module()
            in_dtype = torch.float16 if rec.get("input_dtype") == "float16" else torch.float32

            def fn(x, _m=mod, _dt=in_dtype):
                y = _m(x.to(_dt))
                return (y[0] if isinstance(y, (tuple, list)) else y).float()

            with torch.inference_mode():
                y = fn(torch.rand(self.shape, device="cuda"))
            if not bool(y.isfinite().all()):
                raise RuntimeError("engine gives NaN/Inf after loading")
        except Exception as e:  # noqa: BLE001
            log.exception("accel: could not load %s", engine)
            self._fail(f"could not load the engine: {e!r}"[:500], persist=False)
            return
        v = rec.get("validation", {})
        self._stage(fn, f"{how}; {v.get('reason', 'checked')}")


def _torch():
    import torch  # noqa: PLC0415

    return torch


def attach(model, manifest, data_dir: Path, tile: int, overlap: int) -> None:
    """Give a CUDA-capable model its Accelerator, if its manifest asks for one."""
    try:
        spec = spec_for(manifest)
    except ValueError as e:
        log.warning("model %s: %s; running on PyTorch", manifest.id, e)
        return
    if spec is None or not hasattr(model, "accel_module"):
        return
    window = tile + 2 * max(overlap, model.receptive_padding)
    window += -window % int(getattr(model, "accel_multiple", 1))
    model.accel = Accelerator(manifest.id, spec, data_dir / "engines", [f.sha256 for f in manifest.files],
                              dict(manifest.config), (1, 3, window, window))
    model.accel_model_dir = Path(model.weights_path).parent
