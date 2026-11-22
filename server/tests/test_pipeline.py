"""Pipelines: several operations in one job."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from conftest import decode, make_image, pixels, sha
from painite_server.imaging import pipeline
from painite_server.imaging.geometry import Rect
from painite_server.models import BilinearUpscale2x, Model


@dataclass
class BoxBlur(Model):
    """3x3 box blur: depends on its neighbors, so wrong tile context would show."""

    receptive_padding: int = 1

    def run(self, tile: np.ndarray) -> np.ndarray:
        p = np.pad(tile, ((1, 1), (1, 1), (0, 0)), mode="edge")
        h, w = tile.shape[:2]
        return sum(p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9


@dataclass
class WindowMean(Model):
    """Mixes in the mean of its whole input window. Its output depends on exactly
    which window each tile gets, so preview and export only match if every step
    tiles on the same grid with the same context."""

    receptive_padding: int = 2

    def run(self, tile: np.ndarray) -> np.ndarray:
        return 0.5 * tile + 0.5 * tile.mean(axis=(0, 1), keepdims=True)


def steps(*ops: str) -> dict:
    out = []
    for op in ops:
        st = {"operation": op}
        if op != "upscale2x":
            st["params"] = {"strength": 0.6}
        out.append(st)
    return {"steps": out}


def test_chained_preview_equals_export_crop(tmp_path):
    src = make_image(tmp_path / "in.tif", 150, 110, fmt="tiff")
    blur = BoxBlur(id="denoise/blur@1", operation="denoise", version="1")
    up = BilinearUpscale2x(id="upscale2x/bilinear@1", operation="upscale2x", version="1")
    mean = WindowMean(id="sharpen/mean@1", operation="sharpen", version="1")
    chain = [pipeline.Step(mean, {"strength": 1.0}), pipeline.Step(blur, {"strength": 1.0}), pipeline.Step(up, {}),
             pipeline.Step(mean, {"strength": 0.5})]
    kw = dict(steps=chain, output={"format": "tiff"}, work_dir=tmp_path, results_dir=tmp_path, tile=32, overlap=4,
              max_output_pixels=10**9, on_progress=lambda p: None, is_canceled=lambda: False)
    full = pipeline.run(src, region=None, job_id="full", **kw)
    assert (full.width, full.height) == (300, 220)
    f = pixels(decode((tmp_path / full.path.name).read_bytes()))
    for region in (Rect(32, 32, 64, 32), Rect(0, 0, 32, 32), Rect(128, 96, 22, 14), Rect(64, 0, 32, 96)):
        prev = pipeline.run(src, region=region, job_id=f"p{region.x}", **kw)
        p = pixels(decode(prev.path.read_bytes()))
        crop = f[region.y * 2:(region.y + region.h) * 2, region.x * 2:(region.x + region.w) * 2]
        assert p.shape == crop.shape
        assert np.array_equal(p, crop), region


@dataclass
class ShapeRecorder(Model):
    """Identity that records the window shape of every call."""

    receptive_padding: int = 4
    shapes: list = None

    def run(self, tile: np.ndarray) -> np.ndarray:
        self.shapes.append(tile.shape[:2])
        return tile


def test_edge_tiles_get_full_size_windows(tmp_path):
    """A short remainder tile at the right/bottom edge is fed a full tile + 2*pad
    window, taking more context from inside the image (window-size-sensitive
    models like Stripformer lost 0.8 dB on narrow edge windows), and previews
    still match exports."""
    src = make_image(tmp_path / "in.tif", 150, 110, fmt="tiff")
    rec = ShapeRecorder(id="unblur_motion/rec@1", operation="unblur_motion", version="1", shapes=[])
    kw = dict(steps=[pipeline.Step(rec, {"strength": 1.0})], output={"format": "tiff"}, work_dir=tmp_path,
              results_dir=tmp_path, tile=32, overlap=4, max_output_pixels=10**9,
              on_progress=lambda p: None, is_canceled=lambda: False)
    pipeline.run(src, region=None, job_id="full", **kw)
    assert set(rec.shapes) == {(40, 40)}  # 32 + 2*4 everywhere, incl. the 22 px and 14 px remainders
    for region in (Rect(140, 100, 10, 10), Rect(0, 96, 150, 14)):
        rec.shapes.clear()
        pipeline.run(src, region=region, job_id=f"p{region.x}{region.y}", **kw)
        assert set(rec.shapes) == {(40, 40)}, region


def test_pipeline_job_over_api(api, images):
    data = images["jpeg"]
    api.upload(data)
    body = {"source_id": "sha256:" + sha(data), "kind": "full"} | steps("denoise", "sharpen", "upscale2x")
    r = api.create(body, key="pipe-full")
    assert r.status_code == 201, r.text
    job = api.wait(r.json()["id"])
    assert job["state"] == "completed", job
    assert job["operation"] == "pipeline"
    assert [s["operation"] for s in job["steps"]] == ["denoise", "sharpen", "upscale2x"]
    assert job["steps"][0]["params"] == {"strength": 0.6} and job["steps"][2]["params"] == {}
    assert all(s["model"] for s in job["steps"])
    assert (job["result"]["width"], job["result"]["height"]) == (400, 300)
    # Same key, same body: same job.
    assert api.create(body, key="pipe-full").json()["id"] == job["id"]


@pytest.mark.parametrize("name", ["tiff8_rot8", "jpeg_rot6"])
def test_pipeline_preview_matches_export(api, images, name):
    data = images[name]
    api.upload(data)
    base = {"source_id": "sha256:" + sha(data)} | steps("denoise", "upscale2x")
    export = api.wait(api.create(base | {"kind": "full", "output": {"format": "tiff"}}, key=f"pe-{name}").json()["id"])
    exp_px = pixels(decode(api.result(export["id"]).content))
    preview = api.wait(api.create(base | {"kind": "preview", "output": {"format": "tiff"},
                                          "region": {"x": 40, "y": 30, "w": 40, "h": 50}}, key=f"pp-{name}").json()["id"])
    assert preview["state"] == "completed", preview
    prev_px = pixels(decode(api.result(preview["id"]).content))
    from painite_server.imaging.geometry import display_rect_to_stored
    src = decode(data)
    orientation = src.get("orientation") if "orientation" in src.get_fields() else 1
    st = display_rect_to_stored(Rect(**preview["region"]), src.width, src.height, orientation)
    crop = exp_px[st.y * 2:(st.y + st.h) * 2, st.x * 2:(st.x + st.w) * 2]
    assert np.array_equal(prev_px, crop)


def test_single_operation_jobs_report_one_step(api, images):
    data = images["jpeg"]
    api.upload(data)
    body = {"source_id": "sha256:" + sha(data), "kind": "full", "operation": "sharpen", "params": {"strength": 0.3}}
    job = api.wait(api.create(body, key="single").json()["id"])
    assert job["operation"] == "sharpen"
    assert job["steps"] == [{"operation": "sharpen", "model": job["model"], "params": {"strength": 0.3}}]


@pytest.mark.parametrize("extra,code", [
    (steps("upscale2x", "upscale2x"), "invalid_params"),
    (steps("denoise", "blur"), "invalid_params"),
    ({"steps": []}, "bad_request"),
    (steps("denoise", "denoise", "denoise", "denoise", "denoise"), "bad_request"),
    (steps("denoise") | {"operation": "denoise"}, "bad_request"),
    ({"steps": [{"operation": "upscale2x", "params": {"strength": 0.5}}]}, "invalid_params"),
    ({}, "bad_request"),
])
def test_pipeline_validation(api, images, extra, code):
    data = images["jpeg"]
    api.upload(data)
    r = api.create({"source_id": "sha256:" + sha(data), "kind": "full"} | extra, key=f"bad-{len(str(extra))}-{code}")
    assert r.status_code in (400, 422), r.text
    assert r.json()["error"]["code"] in (code, "bad_request"), r.text


@pytest.mark.parametrize("scale", [1, 2])
def test_blend_weights_sum_to_one(scale):
    """Tiles cross-fade over `band` px each side of a shared edge; the weights of
    all tiles on an axis add up to exactly 1 everywhere."""
    size, tile, band = 150, 32, 4
    total = np.zeros(size * scale, dtype=np.float32)
    for t in range(0, size, tile):
        lo, hi, w = pipeline._blend_span(t, min(tile, size - t), size, band, 0, size, scale)
        assert (w > 0).all() and (w <= 1).all()
        total[lo:hi] += w
    np.testing.assert_allclose(total, 1, atol=1e-6)
