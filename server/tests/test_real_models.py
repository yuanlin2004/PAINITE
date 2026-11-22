"""Catalog manifests and the spandrel adapter. The adapter tests build a tiny
ESRGAN network on the CPU, so they need torch + spandrel but no downloads."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from painite_server.models import ADAPTERS
from painite_server.models import manifest as mf
from painite_server.models.spandrel_adapter import validate_config


def test_catalog_has_the_bakeoff_picks():
    cat = mf.catalog()
    assert set(cat) == {
        "denoise/scunet-real-gan@1.0.0",
        "sharpen/realesrgan-x2-down@1.0.0",
        "upscale2x/nomosuni-esrgan-multijpg@1.0.0",
    }
    for m in cat.values():
        assert m.adapter in ADAPTERS and m.default
        assert m.config["weights"] in {f.name for f in m.files}
        assert m.config["context"] >= 64
        validate_config(m.config, m.scale)


@pytest.mark.parametrize("config,scale,msg", [
    ({"model_scale": 1}, 1, "weights"),
    ({"weights": "w", "model_scale": 2}, 1, "2x"),
    ({"weights": "w", "model_scale": 2, "postprocess": [{"type": "area_downscale", "factor": 2}]}, 2, "1x"),
    ({"weights": "w", "context": 4, "postprocess": [{"type": "color_anchor", "sigma": 8}]}, 1, "context"),
    ({"weights": "w", "postprocess": [{"type": "sharpen_more"}]}, 1, "unknown"),
])
def test_validate_config_rejects(config, scale, msg):
    with pytest.raises(ValueError, match=msg):
        validate_config(config, scale)


HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "torchvision", "spandrel"))
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch, torchvision and spandrel not installed")


@pytest.fixture
def tiny_esrgan(tmp_path):
    import torch
    from spandrel.architectures.ESRGAN import ESRGAN

    def build(scale: int):
        torch.manual_seed(0)
        net = ESRGAN(in_nc=3, out_nc=3, num_filters=8, num_blocks=1, scale=scale)
        path = tmp_path / f"tiny_x{scale}.pth"
        torch.save(net.state_dict(), path)
        return path

    return build


def _manifest(op, scale, model_scale, post, weights):
    return mf.Manifest(id=f"{op}/tiny@0.0.1", adapter="spandrel", scale=scale, license={"name": "test"},
                       files=[], config={"weights": weights.name, "model_scale": model_scale, "dtype": "fp32",
                                         "context": 16, "postprocess": post})


@needs_torch
@pytest.mark.parametrize("op,scale,model_scale,post", [
    ("upscale2x", 2, 2, [{"type": "color_anchor", "sigma": 4}]),
    ("sharpen", 1, 2, [{"type": "area_downscale", "factor": 2}, {"type": "color_anchor", "sigma": 2}]),
])
def test_adapter_runs_and_preview_matches_export(tmp_path, tiny_esrgan, op, scale, model_scale, post):
    from painite_server.imaging import pipeline
    from painite_server.imaging.geometry import Rect
    from conftest import make_image

    weights = tiny_esrgan(model_scale)
    model = ADAPTERS["spandrel"](_manifest(op, scale, model_scale, post, weights), weights.parent)
    model.device = "cpu"
    assert model.receptive_padding == 16

    window = np.random.default_rng(0).random((40, 30, 3), dtype=np.float32)
    out = model.run(window)
    assert out.shape == (40 * scale, 30 * scale, 3) and out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1

    src = make_image(tmp_path / "in.tif", 150, 110, fmt="tiff")
    kw = dict(steps=[pipeline.Step(model, {"strength": 0.7} if scale == 1 else {})], output={}, work_dir=tmp_path,
              results_dir=tmp_path, tile=48, overlap=16, max_output_pixels=10**9,
              on_progress=lambda p: None, is_canceled=lambda: False)
    full = pipeline.run(src, region=None, job_id="full", **kw)
    prev = pipeline.run(src, region=Rect(48, 48, 96, 62), job_id="prev", **kw)
    import pyvips
    f = pyvips.Image.new_from_file(str(full.path)).numpy()
    p = pyvips.Image.new_from_file(str(prev.path)).numpy()
    s = scale
    assert np.array_equal(p, f[48 * s:(48 + 62) * s, 48 * s:(48 + 96) * s])


@needs_torch
def test_color_anchor_pins_low_frequencies():
    import torch

    from painite_server.models.spandrel_adapter import color_anchor

    torch.manual_seed(0)
    inp = 0.1 + 0.7 * torch.rand(1, 3, 64, 64)
    drifted = inp + torch.tensor([0.06, -0.03, 0.02]).view(1, 3, 1, 1)  # colour cast, same detail
    fixed = color_anchor(drifted, inp, sigma=4)
    assert (fixed - inp).abs().max() < 1e-5
    assert (drifted - inp).abs().mean() > 0.03
