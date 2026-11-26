"""Core ML on Macs (models/coreml.py): the choice per model, the fallback to
PyTorch, and, on an Apple Silicon Mac with coremltools, a real conversion."""

from __future__ import annotations

import importlib.util
import platform
import sys

import numpy as np
import pytest

from painite_server.models import ADAPTERS, coreml
from painite_server.models import manifest as mf

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "torchvision", "spandrel"))
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch, torchvision and spandrel not installed")
ON_MAC = (sys.platform == "darwin" and platform.machine() == "arm64" and HAVE_TORCH
          and importlib.util.find_spec("coremltools") is not None)
needs_coreml = pytest.mark.skipif(not ON_MAC, reason="needs an Apple Silicon Mac with coremltools")


@pytest.mark.parametrize("value", [None, True, False, {}, {"compute_units": "gpu"},
                                   {"compute_units": "neural_engine", "precision": "fp32"}])
def test_validate_accepts(value):
    coreml.validate_config(value)


@pytest.mark.parametrize("value,msg", [
    ("yes", "true, false or an object"),
    ({"compute_units": "ane"}, "compute_units"),
    ({"precision": "int8"}, "precision"),
])
def test_validate_rejects(value, msg):
    with pytest.raises(ValueError, match=msg):
        coreml.validate_config(value)


def test_choose(monkeypatch):
    monkeypatch.setattr(coreml, "available", lambda: True)
    assert coreml.choose("ESRGAN", {}, "mps") == ("neural_engine", "fp16")
    assert coreml.choose("SCUNet", {}, "mps") == ("gpu", "fp16")
    assert coreml.choose("GPEN", {}, "mps") == ("gpu", "fp32")
    # the manifest can change or turn off the choice
    assert coreml.choose("ESRGAN", {"coreml": {"compute_units": "gpu"}}, "mps") == ("gpu", "fp16")
    assert coreml.choose("ESRGAN", {"coreml": False}, "mps") is None
    # an architecture nobody measured stays on PyTorch unless the manifest names a unit
    assert coreml.choose("CodeFormer", {}, "mps") is None
    assert coreml.choose("CodeFormer", {"coreml": {"compute_units": "gpu"}}, "mps") == ("gpu", "fp16")
    # only replaces MPS
    assert coreml.choose("ESRGAN", {}, "cuda") is None
    assert coreml.choose("ESRGAN", {}, "cpu") is None
    monkeypatch.setattr(coreml, "available", lambda: False)
    assert coreml.choose("ESRGAN", {}, "mps") is None


def test_off_and_bad_mode(monkeypatch):
    monkeypatch.setenv("PAINITE_COREML", "off")
    assert coreml.requested() == "off" and not coreml.available()
    monkeypatch.setenv("PAINITE_COREML", "maybe")
    assert coreml.requested() == "auto"


def test_runner_handles_only_prepared_square_windows(tmp_path):
    r = coreml.Runner("x/y@1", build=lambda: None, cache_dir=tmp_path, units="gpu", precision="fp16")
    r.sizes.add(640)
    assert r.handles((1, 3, 640, 640))
    assert not r.handles((1, 3, 640, 512))  # an image smaller than one window: PyTorch
    assert not r.handles((1, 3, 512, 512))
    r.failed = True
    assert not r.handles((1, 3, 640, 640))
    assert r.run(np.zeros((1, 3, 512, 512), np.float32)) is None


@pytest.fixture
def tiny_esrgan_model(tmp_path):
    import torch
    from spandrel.architectures.ESRGAN import ESRGAN

    torch.manual_seed(0)
    net = ESRGAN(in_nc=3, out_nc=3, num_filters=8, num_blocks=1, scale=1)
    path = tmp_path / "tiny.pth"
    torch.save(net.state_dict(), path)
    m = mf.Manifest(id="sharpen/tiny@0.0.1", adapter="spandrel", scale=1, license={"name": "test"}, files=[],
                    config={"weights": path.name, "dtype": "fp32", "context": 16})
    return ADAPTERS["spandrel"](m, tmp_path)


class FakeRunner:
    """Stands in for Core ML: answers 0.25 everywhere for its one size."""

    failed = False

    def __init__(self, size):
        self.size = size
        self.calls = 0

    def describe(self):
        return "fake"

    def run(self, x):
        if x.shape[2:] != (self.size, self.size):
            return None
        self.calls += 1
        return np.full(x.shape, 0.25, np.float32)


@needs_torch
def test_adapter_uses_core_ml_for_the_standard_window_only(tiny_esrgan_model):
    model = tiny_esrgan_model
    model.device = "cpu"
    model.tiling = (48, 16)
    assert model.coreml_window(48, 16) == 80
    model.load()
    assert model._coreml is None and model.engine() == "PyTorch"  # CPU: never Core ML
    model._coreml = fake = FakeRunner(80)
    rng = np.random.default_rng(0)
    out = model.run(rng.random((80, 80, 3), dtype=np.float32))
    assert fake.calls == 1 and np.allclose(out, 0.25)
    out = model.run(rng.random((80, 60, 3), dtype=np.float32))  # smaller image: PyTorch
    assert fake.calls == 1 and not np.allclose(out, 0.25)


@needs_coreml
def test_gpen_upsample_patch_is_exact():
    import torch

    from painite_server.models import gpen_arch
    from painite_server.models.gpen_arch import make_kernel

    orig = gpen_arch.upfirdn2d
    k = make_kernel([1, 3, 3, 1])
    x = torch.randn(1, 4, 9, 11)
    want = [orig(x, k * 4, up=2, pad=(2, 1)), orig(x, k, down=2, pad=(1, 1)), orig(x, k, pad=(2, 2))]
    with coreml._conversion_patches():
        got = [gpen_arch.upfirdn2d(x, k * 4, up=2, pad=(2, 1)), gpen_arch.upfirdn2d(x, k, down=2, pad=(1, 1)),
               gpen_arch.upfirdn2d(x, k, pad=(2, 2))]
    assert gpen_arch.upfirdn2d is orig
    for a, b in zip(want, got, strict=True):
        assert torch.equal(a, b)


@needs_coreml
def test_real_conversion_matches_pytorch_and_preview_matches_export(tmp_path, tiny_esrgan_model):
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    from conftest import make_image
    from painite_server.imaging import pipeline
    from painite_server.imaging.geometry import Rect

    model = tiny_esrgan_model
    model.device = "mps"
    model.tiling = (48, 16)
    model.config["coreml"] = {"compute_units": "gpu", "precision": "fp32"}
    model.prepare_coreml()
    assert model.engine() == "Core ML, gpu, fp32"
    assert list((tmp_path / "coreml").glob("80-fp32-*.mlmodelc"))

    window = np.random.default_rng(1).random((80, 80, 3), dtype=np.float32)
    got = model.run(window)
    with torch.inference_mode():
        x = torch.from_numpy(window.transpose(2, 0, 1).copy())[None].to("mps")
        want = model._net(x).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    assert np.abs(got - want).max() < 2e-3

    src = make_image(tmp_path / "in.tif", 150, 110, fmt="tiff")
    kw = dict(steps=[pipeline.Step(model, {"strength": 0.7})], output={}, work_dir=tmp_path,
              results_dir=tmp_path, tile=48, overlap=16, max_output_pixels=10**9,
              on_progress=lambda p: None, is_canceled=lambda: False)
    full = pipeline.run(src, region=None, job_id="full", **kw)
    prev = pipeline.run(src, region=Rect(48, 48, 96, 62), job_id="prev", **kw)
    import pyvips
    f = pyvips.Image.new_from_file(str(full.path)).numpy()
    p = pyvips.Image.new_from_file(str(prev.path)).numpy()
    assert np.array_equal(p, f[48:110, 48:144])
