"""Accelerated backends (models/accel.py): selection, cache keys, fallback and
the tile cache. No GPU needed: builds are faked; the real TensorRT and
torch.compile paths are checked on the DGX Spark."""

from __future__ import annotations

import json
import subprocess
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from conftest import decode, make_image, pixels
from painite_server.imaging import pipeline
from painite_server.imaging.geometry import Rect
from painite_server.models import Model, accel, default_registry
from painite_server.models import manifest as mf

SCUNET = "denoise/scunet-real-gan@1.0.0"


@pytest.fixture(autouse=True)
def clean_accel(monkeypatch):
    monkeypatch.delenv("PAINITE_ACCEL", raising=False)
    monkeypatch.setattr(accel, "_disabled", set())


# Selection ------------------------------------------------------------------

@pytest.mark.parametrize(("env", "want", "label"), [
    (None, {"tensorrt", "compile"}, "auto"),
    ("auto", {"tensorrt", "compile"}, "auto"),
    ("off", set(), "off"),
    ("pytorch", set(), "off"),
    ("compile", {"compile"}, "compile"),
    ("tensorrt, compile", {"tensorrt", "compile"}, "auto"),
    ("tensorrt,bogus", {"tensorrt"}, "tensorrt"),
])
def test_env_switch(monkeypatch, env, want, label):
    if env is not None:
        monkeypatch.setenv("PAINITE_ACCEL", env)
    assert accel.requested() == want
    assert accel.mode_label() == label


def test_parse_spec():
    assert accel.parse_spec(None) is None
    assert accel.parse_spec({"backend": "pytorch"}) is None
    assert accel.parse_spec("compile").label == "compile"
    assert accel.parse_spec({"backend": "tensorrt"}) == accel.Spec("tensorrt", "fp16", "weak")
    assert accel.parse_spec({"backend": "tensorrt", "precision": "fp32", "typing": "strong"}).label == \
        "tensorrt-fp32-strong"
    for bad in ({"backend": "tensorrt", "precision": "bf16"}, {"backend": "onnx"},
                {"backend": "tensorrt", "typing": "loose"}, ["tensorrt"]):
        with pytest.raises(ValueError):
            accel.parse_spec(bad)


def test_catalog_defaults_follow_the_evaluation():
    """TensorRT FP16 for the CNN/GAN ops, torch.compile for Stripformer (TensorRT
    FP16 is silently wrong on it), plain PyTorch for the face model."""
    want = {
        "denoise": "tensorrt-fp16", "sharpen": "tensorrt-fp16", "upscale2x": "tensorrt-fp16",
        "unblur_focus": "tensorrt-fp16", "unblur_motion": "compile", "face": None,
    }
    for m in mf.catalog().values():
        spec = accel.parse_spec(m.config.get("accel"))
        assert (spec.label if spec else None) == want[m.operation], m.id


def test_installed_manifest_wins_over_catalog():
    cat = mf.catalog()
    old = mf.Manifest.parse({**cat[SCUNET].to_json(), "config": {"weights": "w.pth"}})  # installed before accel
    assert accel.spec_for(old, mf.catalog).label == "tensorrt-fp16"
    off = mf.Manifest.parse({**old.to_json(), "config": {"weights": "w.pth", "accel": {"backend": "pytorch"}}})
    assert accel.spec_for(off, mf.catalog) is None
    other = mf.Manifest.parse({**old.to_json(), "id": "denoise/custom@1.0.0"})
    assert accel.spec_for(other, mf.catalog) is None


def test_cache_key_changes_with_everything_that_invalidates_an_engine():
    spec = accel.Spec("tensorrt")
    env = {"torch": "2.9", "cuda": "13.0", "gpu": "NVIDIA GB10", "sm": "12.1", "tensorrt": "10.13.3.9"}
    base = dict(model_id=SCUNET, weights=["aa", "bb"], config={"dtype": "fp16"}, spec=spec,
                shape=(1, 3, 640, 640), env=env)
    k = accel.cache_key(**base)
    assert k == accel.cache_key(**{**base, "weights": ["bb", "aa"]})  # order does not matter
    changes = [
        {"weights": ["aa", "cc"]},
        {"config": {"dtype": "fp32"}},
        {"spec": accel.Spec("tensorrt", "fp32")},
        {"spec": accel.Spec("tensorrt", "fp16", "strong")},
        {"shape": (1, 3, 384, 384)},
        {"env": {**env, "tensorrt": "10.14.0"}},
        {"env": {**env, "gpu": "NVIDIA H100", "sm": "9.0"}},
        {"model_id": "denoise/scunet-real-gan@1.0.1"},
    ]
    keys = {accel.cache_key(**{**base, **c}) for c in changes}
    assert k not in keys and len(keys) == len(changes)


# Golden tiles and the check -------------------------------------------------

def test_golden_tiles_are_fixed():
    a = accel.golden_tiles(96, 80)
    b = accel.golden_tiles(96, 80)
    assert len(a) == 4
    for x, y in zip(a, b):
        assert x.shape == (96, 80, 3) and x.dtype == np.float32
        assert np.array_equal(x, y) and 0 <= x.min() and x.max() <= 1
    assert a[3].max() == 1.0 and a[3].min() == 0.0  # clipped extremes


def test_verdict():
    good = {"psnr_min": 55.0, "max_abs": 9, "mean_abs": 0.1, "finite": True}
    srv = {"psnr_min": 58.0, "max_abs": 11, "mean_abs": 0.1, "finite": True}
    assert accel.verdict(good, srv, 50)[0]
    assert accel.verdict({**good, "max_abs": 22}, srv, 50)[0]  # within 2x the PyTorch path
    assert not accel.verdict({**good, "max_abs": 23}, srv, 50)[0]
    assert not accel.verdict({**good, "psnr_min": 24.2}, srv, 50)[0]  # Stripformer TensorRT FP16
    assert not accel.verdict({**good, "finite": False}, srv, 50)[0]  # RetinaFace TensorRT FP16


def test_check_catches_a_wrong_network():
    torch = pytest.importorskip("torch")
    ref = lambda x: x * 0.9  # noqa: E731
    good = accel.check(torch, lambda x: x * 0.9 + 1e-4, ref, ref, (1, 3, 64, 64), device="cpu")
    assert good["passed"], good
    check = accel.check(torch, lambda x: x * 0.8, ref, ref, (1, 3, 64, 64), device="cpu")
    assert not check["passed"] and "PSNR" in check["reason"]
    check = accel.check(torch, lambda x: x * float("nan"), ref, ref, (1, 3, 64, 64), device="cpu")
    assert not check["passed"] and "NaN" in check["reason"]


# The accelerator -------------------------------------------------------------

class FakeModel:
    device = "cuda"
    accel_model_dir = Path("/nonexistent")


def make_acc(tmp_path, backend="tensorrt") -> accel.Accelerator:
    spec = accel.parse_spec({"backend": backend})
    return accel.Accelerator(SCUNET, spec, tmp_path / "engines", ["aa"], {"dtype": "fp16", "accel": {}},
                             (1, 3, 8, 8))


def fake_env(monkeypatch):
    monkeypatch.setattr(accel, "_torch", lambda: types.SimpleNamespace())
    monkeypatch.setattr(accel, "environment", lambda torch, backend: {"gpu": "fake"})


class Tensor:
    def __init__(self, shape):
        self.shape = shape


def test_serves_pytorch_until_an_engine_is_staged_and_a_job_starts(tmp_path):
    acc = make_acc(tmp_path)
    fallback = lambda x: "pytorch"  # noqa: E731
    assert acc.backend == "pytorch" and acc.call(Tensor((1, 3, 8, 8)), fallback) == "pytorch"
    acc._stage(lambda x: "engine", "test")
    assert acc.state == "ready" and acc.backend == "pytorch"  # not mid-job
    assert acc.call(Tensor((1, 3, 8, 8)), fallback) == "pytorch"
    acc.begin_job()
    assert acc.state == "active" and acc.backend == "tensorrt-fp16"
    assert acc.call(Tensor((1, 3, 8, 8)), fallback) == "engine"
    assert acc.call(Tensor((1, 3, 8, 16)), fallback) == "pytorch"  # other window shape


def test_run_time_error_falls_back_and_stays_on_pytorch(tmp_path):
    acc = make_acc(tmp_path)

    def broken(x):
        raise RuntimeError("engine exploded")

    acc._stage(broken, "test")
    acc.begin_job()
    assert acc.call(Tensor((1, 3, 8, 8)), lambda x: "pytorch") == "pytorch"
    assert acc.backend == "pytorch" and acc.state == "failed"
    acc.begin_job()
    assert acc.backend == "pytorch"


def test_start_needs_cuda_and_an_allowed_backend(tmp_path, monkeypatch):
    queued = []
    monkeypatch.setattr(accel, "_enqueue", lambda acc, model: queued.append(acc))
    cpu = FakeModel()
    cpu.device = "mps"
    acc = make_acc(tmp_path)
    acc.start(cpu)
    assert acc.state == "off" and "CUDA" in acc.detail and not queued
    monkeypatch.setenv("PAINITE_ACCEL", "compile")
    acc = make_acc(tmp_path)
    acc.start(FakeModel())
    assert acc.state == "off" and "PAINITE_ACCEL=compile" in acc.detail and not queued
    monkeypatch.setenv("PAINITE_ACCEL", "auto")
    acc = make_acc(tmp_path)
    acc.start(FakeModel())
    acc.start(FakeModel())  # once only
    assert acc.state == "queued" and queued == [acc]


def test_builder_thread_runs_builds_in_order(tmp_path, monkeypatch):
    done = []
    monkeypatch.setattr(accel.Accelerator, "_build", lambda self, model: done.append(self.model_id))
    for i in range(3):
        acc = make_acc(tmp_path)
        acc.model_id = f"m{i}"
        acc.start(FakeModel())
    accel.wait_idle()
    assert done == ["m0", "m1", "m2"]


def test_a_crashing_build_does_not_kill_the_builder(tmp_path, monkeypatch):
    def boom(self, model):
        raise RuntimeError("boom")

    monkeypatch.setattr(accel.Accelerator, "_build", boom)
    a, b = make_acc(tmp_path), make_acc(tmp_path)
    a.start(FakeModel())
    b.start(FakeModel())
    accel.wait_idle()
    assert a.state == b.state == "failed" and a.backend == "pytorch"


def test_failed_build_is_remembered_and_not_retried(tmp_path, monkeypatch):
    fake_env(monkeypatch)
    calls = []

    def run(cmd, **kw):  # the build subprocess: writes a failed record, as accel_build does
        calls.append(cmd)
        rec = Path(cmd[cmd.index("--record") + 1])
        accel.write_record(rec, {"status": "failed", "reason": "PSNR 24.2 dB < 50.0 dB"})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(accel.subprocess, "run", run)
    acc = make_acc(tmp_path)
    acc._build(FakeModel())
    assert acc.state == "failed" and "PSNR" in acc.detail and len(calls) == 1
    acc2 = make_acc(tmp_path)  # the next server start
    acc2._build(FakeModel())
    assert acc2.state == "failed" and "cached" in acc2.detail and len(calls) == 1
    assert acc2.backend == "pytorch"


def test_build_time_limit(tmp_path, monkeypatch):
    fake_env(monkeypatch)
    monkeypatch.setenv("PAINITE_ACCEL_BUILD_TIMEOUT_S", "5")

    def run(cmd, timeout, **kw):
        assert timeout == 5
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(accel.subprocess, "run", run)
    acc = make_acc(tmp_path)
    acc._build(FakeModel())
    assert acc.state == "failed" and "longer than 5 s" in acc.detail
    rec = json.loads(acc._paths()[1].read_text())
    assert rec["status"] == "failed"


def test_build_crash_is_recorded(tmp_path, monkeypatch):
    fake_env(monkeypatch)
    monkeypatch.setattr(accel.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, -11))
    acc = make_acc(tmp_path)
    acc._build(FakeModel())
    assert acc.state == "failed" and "exit -11" in acc.detail
    assert json.loads(acc._paths()[1].read_text())["status"] == "failed"


def test_failed_compile_turns_compile_off_for_the_process(tmp_path, monkeypatch):
    fake_env(monkeypatch)

    def compile_(*a, **k):
        raise RuntimeError("inductor failed")

    torch = types.SimpleNamespace(compile=compile_)
    monkeypatch.setattr(accel, "_torch", lambda: torch)
    model = FakeModel()
    model.accel_module = lambda: types.SimpleNamespace(float=lambda: None)
    model.accel_call = lambda net: net
    acc = make_acc(tmp_path, "compile")
    acc._build(model)
    assert acc.state == "failed" and "compile" in accel._disabled
    other = make_acc(tmp_path, "compile")
    other.model_id = "unblur_focus/x@1"
    other._build(model)
    assert other.state == "failed" and "disabled" in other.detail


def test_unavailable_tensorrt(tmp_path, monkeypatch):
    def env(torch, backend):
        raise ImportError("No module named 'tensorrt'")

    monkeypatch.setattr(accel, "_torch", lambda: types.SimpleNamespace())
    monkeypatch.setattr(accel, "environment", env)
    acc = make_acc(tmp_path)
    acc._build(FakeModel())
    assert acc.state == "unavailable" and acc.backend == "pytorch"


def test_prune_keeps_only_the_current_key(tmp_path):
    for name in ("old.ep", "old.json", "new.ep", "new.json", "new.log"):
        (tmp_path / name).write_text("x")
    accel.prune(tmp_path, "new")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["new.ep", "new.json", "new.log"]


# Registry, tile cache, health -------------------------------------------------

def install(data_dir: Path, model_id: str, extra_config: dict | None = None) -> None:
    m = mf.catalog()[model_id]
    d = m.install_dir(data_dir)
    d.mkdir(parents=True)
    for f in m.files:
        (d / f.name).write_bytes(b"x")
    data = m.to_json()
    for f in data["files"]:
        f["bytes"] = None
    data["config"] = {**data["config"], **(extra_config or {})}
    (d / "manifest.json").write_text(json.dumps(data))


@pytest.mark.parametrize(("model_id", "tile", "overlap", "window", "label"), [
    (SCUNET, 512, 64, 640, "tensorrt-fp16"),
    (SCUNET, 256, 32, 384, "tensorrt-fp16"),
    ("unblur_motion/stripformer-realblurj@1.0.0", 512, 64, 640, "compile"),
    ("unblur_focus/lakdnet-defocus-lfdpdd-s@1.0.0", 500, 64, 632, "tensorrt-fp16"),  # padded to 8
])
def test_registry_attaches_accelerators(tmp_path, model_id, tile, overlap, window, label):
    install(tmp_path, model_id)
    reg = default_registry(tmp_path, tile, overlap)
    m = reg.get(model_id)
    assert m.accel.spec.label == label and m.accel.shape == (1, 3, window, window)
    assert m.backend == "pytorch" and m.accel_status()["target"] == label
    assert m.accel.engines_dir == tmp_path / "engines"
    assert reg.get("denoise/identity@0.1.0").accel_status() is None


def test_manifest_can_turn_acceleration_off(tmp_path):
    install(tmp_path, SCUNET, {"accel": {"backend": "pytorch"}})
    m = default_registry(tmp_path).get(SCUNET)
    assert m.accel is None and m.backend == "pytorch" and m.accel_status() is None


def test_bad_accel_config_still_loads_the_model(tmp_path):
    install(tmp_path, SCUNET, {"accel": {"backend": "tensorrt", "precision": "bf16"}})
    m = default_registry(tmp_path).get(SCUNET)
    assert m is not None and m.accel is None


@dataclass
class Switchable(Model):
    """Counts its calls; `backend` is settable, as an engine switching in."""

    receptive_padding: int = 4
    calls: int = 0
    _backend: str = "pytorch"

    @property
    def backend(self) -> str:
        return self._backend

    def run(self, tile: np.ndarray) -> np.ndarray:
        self.calls += 1
        return 0.5 * tile + 0.5 * tile.mean(axis=(0, 1), keepdims=True)


def test_tile_cache_key_includes_the_backend(tmp_path):
    """Tiles a preview rendered on PyTorch are not reused once TensorRT is
    active, for this step or for a later step fed by it."""
    from painite_server.imaging.tile_cache import TileCache

    src = make_image(tmp_path / "in.tif", 150, 110, fmt="tiff")
    m = Switchable(id="denoise/s@1", operation="denoise", version="1")
    later = Switchable(id="sharpen/s@1", operation="sharpen", version="1")
    cache = TileCache(64 * 1024 * 1024)
    kw = dict(output={"format": "tiff"}, work_dir=tmp_path, results_dir=tmp_path, tile=32, overlap=8,
              max_output_pixels=10**9, on_progress=lambda p: None, is_canceled=lambda: False, cache=cache,
              cache_key=("owner", "sha"), cache_write=True)
    region = Rect(36, 36, 56, 56)
    chain = [pipeline.Step(m, {"strength": 1.0}), pipeline.Step(later, {"strength": 1.0})]
    pipeline.run(src, chain, region, job_id="a", **kw)
    first = (m.calls, later.calls)  # the first step also renders the second's context ring
    assert later.calls == 4 and m.calls > 4
    pipeline.run(src, chain, region, job_id="b", **kw)
    assert (m.calls, later.calls) == first  # cached
    m._backend = "tensorrt-fp16"
    pipeline.run(src, chain, region, job_id="c", **kw)
    assert (m.calls, later.calls) == (2 * first[0], 2 * first[1])  # both steps miss
    out = pipeline.run(src, chain, region, job_id="d", **kw)
    assert (m.calls, later.calls) == (2 * first[0], 2 * first[1])
    assert pixels(decode(out.path.read_bytes())).shape[:2] == (56, 56)


def test_tile_store_skips_tiles_after_a_fallback():
    from painite_server.imaging.tile_cache import TileCache

    m = Switchable(id="denoise/s@1", operation="denoise", version="1", _backend="tensorrt-fp16")
    cache = TileCache(1 << 20)
    store = pipeline._TileStore(cache, ("k", (), m.id, m.backend), True, m)
    store.put((0, 0, 1, 1), np.zeros((1, 1, 3), np.float32))
    m._backend = "pytorch"  # the engine failed mid-job
    store.put((0, 0, 2, 2), np.zeros((1, 1, 3), np.float32))
    assert len(cache) == 1


def test_health_reports_backends(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from painite_server.app import create_app
    from painite_server.config import Settings

    install(tmp_path, SCUNET)
    monkeypatch.setenv("PAINITE_ACCEL", "tensorrt")
    app = create_app(Settings(data_dir=tmp_path), start_worker=False)
    with TestClient(app) as client:
        body = client.get("/v1/health").json()
    assert body["accel"]["mode"] == "tensorrt"
    assert body["accel"]["models"][SCUNET] == {"backend": "pytorch", "target": "tensorrt-fp16", "state": "idle",
                                               "detail": ""}


def test_worker_switches_backends_between_jobs(tmp_path):
    """The worker calls begin_job on every step's model before it runs."""
    from painite_server.jobs.worker import Worker

    calls = []

    class M(Switchable):
        def begin_job(self):
            calls.append(self.id)

    m = M(id="denoise/s@1", operation="denoise", version="1")
    reg = types.SimpleNamespace(get=lambda mid: m if mid == m.id else None)
    store = types.SimpleNamespace(update_job=lambda *a, **k: None, get_job=lambda *a: None)
    w = Worker.__new__(Worker)
    w.registry, w.store = reg, store
    w.settings = types.SimpleNamespace(work_dir=tmp_path, results_dir=tmp_path, tile_size=32, tile_overlap=8,
                                       max_output_pixels=10**9, result_retention_s=1, data_dir=tmp_path)
    w._finish = lambda *a, **k: None
    job = {"id": "j", "created_at": 0.0, "steps": json.dumps([{"operation": "denoise", "model": m.id, "params": {}}]),
           "owner": "o", "source_sha": "s", "kind": "preview", "operation": "denoise", "model": m.id,
           "region": None, "output": "{}"}
    w.tile_cache = None
    w.process(job)
    assert calls == [m.id]
