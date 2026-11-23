"""Face op: alignment geometry, paste-back, and preview == export. Uses a fake
detector and restorer on the CPU, so it needs torch + torchvision but no weights."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import pyvips

from painite_server.imaging import pipeline
from painite_server.imaging.geometry import Rect, display_point_to_stored, display_rect_to_stored

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "torchvision"))
pytestmark = pytest.mark.skipif(not HAVE_TORCH, reason="torch and torchvision not installed")


def test_similarity_transform_recovers_a_known_transform():
    from painite_server.models.face_adapter import FFHQ_TEMPLATE_512, similarity_transform

    ang, scale, t = 0.3, 0.37, np.array([12.5, -40.0])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    src = (FFHQ_TEMPLATE_512 - t) @ np.linalg.inv(scale * rot).T
    a, b = similarity_transform(src, FFHQ_TEMPLATE_512)
    np.testing.assert_allclose(a, scale * rot, atol=1e-9)
    np.testing.assert_allclose(src @ a.T + b, FFHQ_TEMPLATE_512, atol=1e-9)


def test_retinaface_matches_the_facexlib_checkpoint_layout():
    from painite_server.models.face_adapter import _build_retinaface, _priors

    sd = _build_retinaface().state_dict()
    for key in ("body.conv1.weight", "body.layer4.2.bn3.running_var", "fpn.output1.0.weight", "fpn.merge2.1.bias",
                "ssh3.conv7x7_3.1.weight", "ClassHead.0.conv1x1.weight", "BboxHead.2.conv1x1.bias",
                "LandmarkHead.1.conv1x1.weight"):
        assert key in sd
    assert sd["LandmarkHead.0.conv1x1.weight"].shape[0] == 20
    assert sd["ClassHead.0.conv1x1.weight"].shape[0] == 4
    # Two anchors per cell at strides 8, 16, 32.
    assert _priors(64, 96).shape == (2 * (8 * 12 + 4 * 6 + 2 * 3), 4)


def test_crop_mask_is_zero_at_the_border_and_one_inside():
    from painite_server.models.face_adapter import crop_mask

    m = crop_mask(512, 0.05, 0.12)
    assert m[0].max() == 0 and m[:, -1].max() == 0
    assert m[256, 256] == 1 and m[100:412, 100:412].min() == 1


def test_display_point_mapping_agrees_with_rects():
    w, h = 40, 30
    for o in range(1, 9):
        r = display_rect_to_stored(Rect(3, 5, 1, 1), w, h, o)
        assert display_point_to_stored(3, 5, w, h, o) == (r.x, r.y)


# A fake face model --------------------------------------------------------------

# Landmarks of an upright face, 160 px crop square, in displayed pixels.
FACE_DISPLAYED = np.array([[120.0, 110.0], [160.0, 110.0], [140.0, 133.0], [123.0, 152.0], [157.0, 152.0]])


def fake_face_model(face_points=FACE_DISPLAYED, **config):
    from painite_server.models.face_adapter import FaceModel

    class Fake(FaceModel):
        def load(self):
            self.device = "cpu"

        def detect(self, small, full_w, full_h):
            self.detected_on.append((small.shape, full_w, full_h))
            def box(p):
                return np.array([p[:, 0].min() - 15, p[:, 1].min() - 25, p[:, 0].max() + 15, p[:, 1].max() + 15])

            pts = [face_points.copy(), face_points + [400.0, 0.0]]
            return [(0.99, pts[0], box(pts[0])), (0.98, pts[1], box(pts[1]))]

        def _restore(self, crop):
            return 1 - crop  # easy to spot

    cfg = {"detector": "d", "restorer": "r", "face_size": 128, "detect_long_side": 256, "min_face": 8,
           "color_anchor_sigma": 0, **config}
    m = Fake(id="face/fake@1.0.0", operation="face", version="1.0.0", config=cfg)
    m.detected_on = []
    return m


def make_photo(path: Path, width: int, height: int, orientation: int | None = None) -> Path:
    # Smooth content, so resampling through the crop and back is near exact.
    yy, xx = np.mgrid[0:height, 0:width] / 300.0
    arr = np.stack([128 + 90 * np.sin(xx * 3 + yy), 128 + 90 * np.cos(yy * 4 - xx), 60 + 150 * xx / xx.max()], -1)
    arr = np.rint(arr).astype(np.uint8)
    img = pyvips.Image.new_from_array(arr).copy(interpretation="srgb")
    if orientation is not None:
        img.set_type(pyvips.GValue.gint_type, "orientation", orientation)
    img.tiffsave(str(path))
    return path


def render(tmp_path: Path, src: Path, model, region: Rect | None, strength=1.0, job="j", before=None) -> np.ndarray:
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "results").mkdir(exist_ok=True)
    steps = [pipeline.Step(m, {"strength": strength} if m is model else {}) for m in (before or []) + [model]]
    out = pipeline.run(src, steps, region, {"format": "tiff"}, tmp_path / "work",
                       tmp_path / "results", job, 64, 16, 10**9, lambda p: None, lambda: False)
    render.last = out
    return pyvips.Image.new_from_file(str(out.path)).numpy()


def test_face_is_restored_in_place_and_the_rest_is_untouched(tmp_path):
    src = make_photo(tmp_path / "a.tif", 640, 300)
    model = fake_face_model()
    out = render(tmp_path, src, model, None)
    orig = pyvips.Image.new_from_file(str(src)).numpy()
    diff = np.abs(out.astype(int) - orig.astype(int)).max(axis=2)
    # The face centre is inverted, far corners are untouched.
    cy, cx = 130, 140
    np.testing.assert_allclose(out[cy, cx].astype(int), 255 - orig[cy, cx].astype(int), atol=2)
    np.testing.assert_allclose(out[cy, cx + 400].astype(int), 255 - orig[cy, cx + 400].astype(int), atol=2)
    assert diff[:20, :].max() == 0 and diff[:, 632:].max() == 0 and diff[280:, :].max() == 0
    assert model.detected_on[0][1:] == (640, 300)
    assert model.detected_on[0][0][:2] == (120, 256)  # detection ran on the 256 px downscale


def test_strength_zero_is_the_input(tmp_path):
    src = make_photo(tmp_path / "a.tif", 640, 300)
    out = render(tmp_path, src, fake_face_model(), None, strength=0.0)
    np.testing.assert_array_equal(out, pyvips.Image.new_from_file(str(src)).numpy())


@pytest.mark.parametrize("region", [Rect(64, 64, 128, 128), Rect(128, 128, 64, 64), Rect(0, 0, 640, 128),
                                    Rect(448, 64, 192, 64)])
def test_preview_matches_export(tmp_path, region):
    src = make_photo(tmp_path / "a.tif", 640, 300)
    model = fake_face_model()
    full = render(tmp_path, src, model, None, strength=0.7, job="full")
    part = render(tmp_path, src, model, region, strength=0.7, job="prev")
    np.testing.assert_array_equal(part, full[region.y:region.y + region.h, region.x:region.x + region.w])
    assert len(model.detected_on) == 1  # the second job reused the cached detection


def test_rotated_source_is_detected_upright(tmp_path):
    # Stored 300x640 with orientation 6 (displayed 640x300): the fake detector
    # answers in displayed pixels; the paste must land where that face is stored.
    src = make_photo(tmp_path / "r.tif", 300, 640, orientation=6)
    model = fake_face_model()
    out = render(tmp_path, src, model, None)
    orig = pyvips.Image.new_from_file(str(src)).numpy()
    assert model.detected_on[0][1:] == (640, 300)
    sx, sy = display_point_to_stored(140, 130, 300, 640, 6)
    np.testing.assert_allclose(out[int(sy), int(sx)].astype(int), 255 - orig[int(sy), int(sx)].astype(int), atol=2)


def test_large_face_is_shrunk_before_sampling(tmp_path):
    # A face whose crop square is ~4x the restorer size goes through the
    # anti-alias pre-shrink path and still lands in place.
    big = (FACE_DISPLAYED - [140, 130]) * 3 + [150, 150]
    src = make_photo(tmp_path / "b.tif", 900, 300)
    model = fake_face_model(face_points=big)
    out = render(tmp_path, src, model, None)
    orig = pyvips.Image.new_from_file(str(src)).numpy()
    d = out.astype(int) - (255 - orig.astype(int))
    assert np.abs(d[140:160, 140:160]).mean() < 3


def test_size_limits_skip_faces(tmp_path):
    src = make_photo(tmp_path / "a.tif", 640, 300)
    out = render(tmp_path, src, fake_face_model(max_face=50), None)
    np.testing.assert_array_equal(out, pyvips.Image.new_from_file(str(src)).numpy())


@pytest.mark.parametrize("region", [Rect(64, 64, 128, 128), Rect(448, 128, 64, 64)])
def test_face_after_upscale_in_a_pipeline_matches_export(tmp_path, region):
    from painite_server.models import BilinearUpscale2x, IdentityModel

    src = make_photo(tmp_path / "a.tif", 640, 300)
    model = fake_face_model()
    before = [IdentityModel(id="denoise/identity@0.1.0", operation="denoise", version="0.1.0"),
              BilinearUpscale2x(id="upscale2x/bilinear@0.1.0", operation="upscale2x", version="0.1.0")]
    full = render(tmp_path, src, model, None, strength=0.7, job="full", before=before)
    part = render(tmp_path, src, model, region, strength=0.7, job="prev", before=before)
    assert full.shape[:2] == (600, 1280)
    r = Rect(region.x * 2, region.y * 2, region.w * 2, region.h * 2)
    np.testing.assert_array_equal(part, full[r.y:r.y + r.h, r.x:r.x + r.w])
    # The face, detected on the source, lands at twice its source position.
    up = render(tmp_path, src, fake_face_model(), None, strength=0.0, job="up", before=before)
    np.testing.assert_allclose(full[260, 280].astype(int), (255 - up[260, 280].astype(int)) * 0.7
                               + up[260, 280].astype(int) * 0.3, atol=3)


def test_gpen_restorer_loads_from_a_plain_state_dict_and_is_deterministic(tmp_path):
    import torch

    from painite_server.models.face_adapter import FaceModel
    from painite_server.models.gpen_arch import FullGenerator

    torch.manual_seed(0)
    args = {"arch": "gpen", "size": 64, "style_dim": 32, "n_mlp": 2, "channel_multiplier": 1, "narrow": 0.0625}
    net = FullGenerator(64, 32, 2, 1, narrow=0.0625)
    torch.save(net.state_dict(), tmp_path / "gpen.pth")
    from painite_server.models.face_adapter import _build_retinaface

    torch.save(_build_retinaface().state_dict(), tmp_path / "det.pth")
    m = FaceModel(id="face/gpen-tiny@1.0.0", operation="face", version="1.0.0", device="cpu",
                  detector_path=tmp_path / "det.pth", restorer_path=tmp_path / "gpen.pth",
                  config={"detector": "det.pth", "restorer": "gpen.pth", "restorer_args": args, "face_size": 64})
    m.load()
    x = torch.rand(1, 3, 64, 64)
    a, b = m._restore(x), m._restore(x)
    assert a.shape == x.shape and torch.equal(a, b)


def test_job_reports_faces_with_status_in_displayed_pixels(tmp_path):
    src = make_photo(tmp_path / "a.tif", 640, 300)
    render(tmp_path, src, fake_face_model(max_face=200), Rect(0, 0, 64, 64))
    faces = render.last.info["faces"]
    # Both faces are reported even though the preview region touches neither.
    assert [f["status"] for f in faces] == ["enhanced", "enhanced"]
    assert faces[0]["box"] == {"x": 105, "y": 85, "w": 70, "h": 82}
    assert 150 < faces[0]["size"] < 175
    render(tmp_path, src, fake_face_model(max_face=100), None, job="large")
    assert {f["status"] for f in render.last.info["faces"]} == {"too_large"}
    render(tmp_path, src, fake_face_model(min_face=300), None, job="small")
    assert {f["status"] for f in render.last.info["faces"]} == {"too_small"}


def test_rotated_face_boxes_are_reported_upright(tmp_path):
    src = make_photo(tmp_path / "r.tif", 300, 640, orientation=6)
    render(tmp_path, src, fake_face_model(), None)
    # The fake detector answered in displayed pixels; the report is displayed pixels too.
    assert render.last.info["faces"][0]["box"] == {"x": 105, "y": 85, "w": 70, "h": 82}


def test_jobs_without_a_face_step_report_no_faces(tmp_path):
    from painite_server.models import IdentityModel

    src = make_photo(tmp_path / "a.tif", 640, 300)
    render(tmp_path, src, IdentityModel(id="denoise/identity@0.1.0", operation="denoise", version="0.1.0"), None)
    assert render.last.info == {}


# Face-parsing paste mask ------------------------------------------------------

def test_bisenet_matches_the_facexlib_checkpoint_layout():
    from painite_server.models.bisenet_arch import BiSeNet

    sd = BiSeNet(num_class=19).state_dict()
    for key in ("cp.resnet.conv1.weight", "cp.resnet.layer4.1.bn2.running_var", "cp.resnet.layer2.0.downsample.0.weight",
                "cp.arm16.conv_atten.weight", "cp.arm32.bn_atten.bias", "cp.conv_avg.conv.weight",
                "ffm.convblk.conv.weight", "ffm.conv2.weight", "conv_out.conv_out.weight",
                "conv_out16.conv_out.weight", "conv_out32.conv.bn.running_mean"):
        assert key in sd
    assert sd["conv_out.conv_out.weight"].shape == (19, 256, 1, 1)


def test_face_region_drops_detached_skin_and_fills_holes():
    import torch

    from painite_server.models.face_adapter import face_region

    keep = torch.zeros(1, 1, 512, 512)
    yy, xx = torch.meshgrid(torch.arange(512.0), torch.arange(512.0), indexing="ij")
    keep[0, 0][((xx - 256) ** 2 + (yy - 300) ** 2) < 170 ** 2] = 1  # the face
    keep[0, 0, 290:310, 240:270] = 0  # a hole (a shadow the parser missed)
    keep[0, 0, 470:512, 0:120] = 1  # an arm, not touching the face
    r = face_region(keep)
    assert r[0, 0, 300, 256] == 1 and r[0, 0, 300, 400] == 1  # hole filled, face kept
    assert r[0, 0, 490:512, 0:120].max() == 0  # the arm is gone
    assert r[0, 0, 20, 20] == 0
    assert face_region(torch.zeros(1, 1, 512, 512)) is None


def test_soften_grows_and_feathers():
    import torch

    from painite_server.models.face_adapter import soften

    m = torch.zeros(1, 1, 200, 200)
    m[..., 50:150, 50:150] = 1
    s = soften(m, 0.02, 0.01)
    assert s[0, 0, 100, 100] > 0.999 and s[0, 0, 100, 0] < 1e-6
    assert 0.2 < s[0, 0, 100, 154] < 0.8  # the edge moved out by about 4 px
    assert torch.equal(soften(m, 0, 0), m)


class _FakeParser:
    """Labels a disk at the middle of the crop "skin", plus a detached blob at
    the lower left (an arm next to the face), on the 512 parser input."""

    def __call__(self, x):
        import torch

        yy, xx = torch.meshgrid(torch.arange(512.0), torch.arange(512.0), indexing="ij")
        skin = (((xx - 256) ** 2 + (yy - 300) ** 2) < 150 ** 2) | ((xx >= 60) & (xx < 120) & (yy >= 400) & (yy < 470))
        logits = torch.zeros(1, 19, 512, 512)
        logits[0, 1][skin] = 1
        logits[0, 0][~skin] = 1
        return logits


def _crop_to_image(model, crop_pt):
    from painite_server.models.face_adapter import make_face

    f = make_face(FACE_DISPLAYED, 0.99, model.config["face_size"], 640, 300)
    return np.linalg.inv(np.array(f.a)) @ (np.asarray(crop_pt, float) - np.array(f.b))


def test_parse_mask_leaves_a_nearby_arm_alone(tmp_path):
    src = make_photo(tmp_path / "a.tif", 640, 300)
    orig = pyvips.Image.new_from_file(str(src)).numpy().astype(int)

    def with_parser(**config):
        m = fake_face_model(**config)
        load = m.load

        def load_with_parser():
            load()
            m._parser = _FakeParser()

        m.load = load_with_parser
        return m

    model = with_parser()
    out = render(tmp_path, src, model, None, job="parse").astype(int)
    square = render(tmp_path, src, fake_face_model(), None, job="square").astype(int)
    # The arm blob (crop px (22, 109) of 128) is pasted by the square mask, not by the parse mask.
    ax, ay = np.rint(_crop_to_image(model, (90 * 128 / 512, 435 * 128 / 512))).astype(int)
    assert np.abs(square[ay, ax] - orig[ay, ax]).max() > 20
    np.testing.assert_allclose(out[ay, ax], orig[ay, ax], atol=1)
    # The face itself is still restored.
    cx, cy = np.rint(_crop_to_image(model, (64, 75))).astype(int)
    np.testing.assert_allclose(out[cy, cx], 255 - orig[cy, cx], atol=2)
    # Preview == export still holds with the parse mask.
    region = Rect(64, 64, 128, 128)
    part = render(tmp_path, src, with_parser(), region, job="parse-prev").astype(int)
    np.testing.assert_array_equal(part, out[region.y:region.y + region.h, region.x:region.x + region.w])
