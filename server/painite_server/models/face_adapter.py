"""Face enhancement: detect faces on the whole image, align each one to the
FFHQ 512 template, restore the aligned crop, and paste it back with a soft mask.

The paste mask covers only the face itself when the manifest names a face
parser: a BiSeNet face-parsing model labels the restored crop, and only skin,
brows, eyes, glasses, ears, nose and lips connected to the middle of the face
are replaced. Hands, arms, hair, neck, clothes and background inside the
square crop keep their input pixels (otherwise the restorer turns a nearby arm
into blotchy "face skin"). Without a parser the mask is a feathered square, as
in face models before 1.1.0.

Unlike the tiled operations, a face needs the whole face (and detection needs
the whole image), so this model runs once over the full image instead of tile
by tile (`whole_image = True`, see imaging/pipeline.py:_run_whole_image).

Previews stay identical to exports: detection always runs on the same upright
(orientation-applied) downscale of the full image and is cached per source,
and each face's paste is computed over that face's own box in full-image
pixels, whatever region is requested. A preview restores only the faces whose
box touches its region. Faces are pasted in a fixed order, and every face is
restored from the input, not from earlier pastes.

Manifest config:

    "config": {
      "detector": "detection_Resnet50_Final.pth",   # RetinaFace-ResNet50 (facexlib)
      "restorer": "GPEN-BFR-1024.pth",              # spandrel-loadable, or GPEN (below)
      "restorer_args": {"arch": "gpen", "size": 1024},  # or {"weight": 0.8} for CodeFormer
      "dtype": "fp16",                              # restorer dtype: fp16 | bf16 | fp32
      "face_size": 1024,                            # restorer input/output size
      "detect_long_side": 2048,                     # detection runs on this downscale
      "detect_threshold": 0.9,
      "min_face": 24,                               # skip faces smaller than this (px)
      "max_face": 0,                                # skip faces larger than this (0: no limit)
      "mask_inset": 0.05, "mask_feather": 0.12,     # paste mask, as fractions of the crop
      "parser": "parsing_bisenet.pth",              # optional face parser (BiSeNet, facexlib)
      "parse_grow": 0.01, "parse_feather": 0.015,   # parse mask dilation and Gaussian sigma (crop fractions)
      "color_anchor_sigma": 8                       # 0 turns it off
    }

The face size used by min_face / max_face is the side of the aligned crop
square in source pixels (about 2.2x the eye-to-mouth distance).
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import Model

log = logging.getLogger("painite.models")

_load_lock = threading.Lock()

# Five landmarks (left eye, right eye, nose, left mouth corner, right mouth
# corner) of the FFHQ-aligned 512 px face, as used by GFPGAN and CodeFormer.
FFHQ_TEMPLATE_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float64)


def _torch():
    import torch  # noqa: PLC0415 - optional dependency, present in the server image

    return torch


@dataclass(frozen=True)
class Face:
    score: float
    landmarks: tuple  # 5 (x, y) points in full-image pixel coordinates
    # Similarity transform image -> crop: crop = A @ img + b.
    a: tuple
    b: tuple
    size: float  # side of the crop square in image pixels
    box: tuple  # (x0, y0, x1, y1) integer paste box in image pixels, clipped, exclusive end


# Geometry (numpy, deterministic) ---------------------------------------------

def similarity_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares similarity (Umeyama, no reflection) mapping src -> dst.
    Returns (A, b) with dst ≈ A @ src + b."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    rot = u @ np.diag(d) @ vt
    var_s = (s0 ** 2).sum() / len(src)
    scale = (sig * d).sum() / var_s
    a = scale * rot
    b = mu_d - a @ mu_s
    return a, b


def make_face(landmarks: np.ndarray, score: float, face_size: int, width: int, height: int) -> Face:
    template = FFHQ_TEMPLATE_512 * (face_size / 512.0)
    a, b = similarity_transform(landmarks, template)
    a_inv = np.linalg.inv(a)
    corners = np.array([[0, 0], [face_size, 0], [0, face_size], [face_size, face_size]], np.float64)
    img_corners = (corners - b) @ a_inv.T
    x0 = max(int(math.floor(img_corners[:, 0].min())) - 1, 0)
    y0 = max(int(math.floor(img_corners[:, 1].min())) - 1, 0)
    x1 = min(int(math.ceil(img_corners[:, 0].max())) + 2, width)
    y1 = min(int(math.ceil(img_corners[:, 1].max())) + 2, height)
    size = face_size / math.sqrt(abs(np.linalg.det(a)))
    return Face(score=float(score), landmarks=tuple(map(tuple, np.asarray(landmarks).tolist())),
                a=tuple(map(tuple, a.tolist())), b=tuple(b.tolist()), size=float(size), box=(x0, y0, x1, y1))


def crop_mask(face_size: int, inset: float, feather: float) -> np.ndarray:
    """Soft square mask in crop space: 0 within `inset` of the border, rising
    smoothly to 1 over `feather` (both as fractions of the crop side)."""
    c = np.arange(face_size, dtype=np.float64) + 0.5
    d = np.minimum(c, face_size - c)
    t = np.clip((d - inset * face_size) / max(feather * face_size, 1e-6), 0, 1)
    ramp = t * t * (3 - 2 * t)
    return np.minimum(ramp[:, None], ramp[None, :]).astype(np.float32)


# RetinaFace-ResNet50 (facexlib / biubug6 layout, so its weights load as is) --

def _build_retinaface():
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional  # noqa: N806
    from torchvision.models import resnet50  # noqa: PLC0415
    from torchvision.models._utils import IntermediateLayerGetter  # noqa: PLC0415

    def conv_bn(inp, oup, stride=1, leaky=0.0):
        return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup),
                             nn.LeakyReLU(negative_slope=leaky, inplace=True))

    def conv_bn_no_relu(inp, oup, stride=1):
        return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup))

    def conv_bn1x1(inp, oup, stride=1, leaky=0.0):
        return nn.Sequential(nn.Conv2d(inp, oup, 1, stride, 0, bias=False), nn.BatchNorm2d(oup),
                             nn.LeakyReLU(negative_slope=leaky, inplace=True))

    class SSH(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv3X3 = conv_bn_no_relu(cin, cout // 2)
            self.conv5X5_1 = conv_bn(cin, cout // 4)
            self.conv5X5_2 = conv_bn_no_relu(cout // 4, cout // 4)
            self.conv7X7_2 = conv_bn(cout // 4, cout // 4)
            self.conv7x7_3 = conv_bn_no_relu(cout // 4, cout // 4)

        def forward(self, x):
            c3 = self.conv3X3(x)
            c5_1 = self.conv5X5_1(x)
            c5 = self.conv5X5_2(c5_1)
            c7 = self.conv7x7_3(self.conv7X7_2(c5_1))
            return F.relu(torch.cat([c3, c5, c7], dim=1))

    class FPN(nn.Module):
        def __init__(self, cins, cout):
            super().__init__()
            self.output1 = conv_bn1x1(cins[0], cout)
            self.output2 = conv_bn1x1(cins[1], cout)
            self.output3 = conv_bn1x1(cins[2], cout)
            self.merge1 = conv_bn(cout, cout)
            self.merge2 = conv_bn(cout, cout)

        def forward(self, xs):
            o1, o2, o3 = self.output1(xs[0]), self.output2(xs[1]), self.output3(xs[2])
            o2 = self.merge2(o2 + F.interpolate(o3, size=o2.shape[2:], mode="nearest"))
            o1 = self.merge1(o1 + F.interpolate(o2, size=o1.shape[2:], mode="nearest"))
            return [o1, o2, o3]

    class Head(nn.Module):
        def __init__(self, cin, n):
            super().__init__()
            self.n = n
            self.conv1x1 = nn.Conv2d(cin, 2 * n, 1)

        def forward(self, x):
            return self.conv1x1(x).permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.n)

    class RetinaFace(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = IntermediateLayerGetter(resnet50(weights=None), {"layer2": 1, "layer3": 2, "layer4": 3})
            self.fpn = FPN([512, 1024, 2048], 256)
            self.ssh1, self.ssh2, self.ssh3 = SSH(256, 256), SSH(256, 256), SSH(256, 256)
            self.ClassHead = nn.ModuleList([Head(256, 2) for _ in range(3)])
            self.BboxHead = nn.ModuleList([Head(256, 4) for _ in range(3)])
            self.LandmarkHead = nn.ModuleList([Head(256, 10) for _ in range(3)])

        def forward(self, x):
            fpn = self.fpn(list(self.body(x).values()))
            feats = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
            loc = torch.cat([h(f) for h, f in zip(self.BboxHead, feats, strict=True)], 1)
            conf = torch.cat([h(f) for h, f in zip(self.ClassHead, feats, strict=True)], 1)
            ldm = torch.cat([h(f) for h, f in zip(self.LandmarkHead, feats, strict=True)], 1)
            return loc, F.softmax(conf, dim=-1), ldm

    return RetinaFace()


_MIN_SIZES = ((16, 32), (64, 128), (256, 512))
_STEPS = (8, 16, 32)
_VARIANCE = (0.1, 0.2)
_BGR_MEAN = (104.0, 117.0, 123.0)


def _priors(h: int, w: int):
    out = []
    for step, sizes in zip(_STEPS, _MIN_SIZES, strict=True):
        fh, fw = math.ceil(h / step), math.ceil(w / step)
        ys, xs = np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")
        cx = ((xs + 0.5) * step / w).reshape(-1, 1)
        cy = ((ys + 0.5) * step / h).reshape(-1, 1)
        per = [np.concatenate([cx, cy, np.full_like(cx, s / w), np.full_like(cy, s / h)], 1) for s in sizes]
        # Anchor order within a cell: all min_sizes for cell 0, then cell 1, ...
        out.append(np.stack(per, 1).reshape(-1, 4))
    return np.concatenate(out, 0)


# Face-parsing paste mask ----------------------------------------------------

# CelebAMask-HQ classes that the restored face replaces: skin, brows, eyes,
# eyeglasses, ears, earring, nose, mouth, lips. Background, neck, necklace,
# cloth, hair and hat keep the input, and so does anything else the parser
# does not see as face, such as a hand or an arm.
PARSE_KEEP = tuple(range(1, 14))
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# The middle of the face in the 512 template (eyes to mouth), which seeds the
# face region: face pixels not connected to it are left alone.
_CORE_512 = (193, 240, 319, 372)  # x0, y0, x1, y1


def _reconstruct(marker, mask, max_iter: int = 1024):
    """Morphological reconstruction: the parts of binary `mask` (1x1xHxW)
    8-connected to `marker`."""
    torch = _torch()
    F = torch.nn.functional  # noqa: N806
    r = marker * mask
    for i in range(max_iter):
        nxt = F.max_pool2d(r, 3, 1, 1) * mask
        if i % 8 == 7 and torch.equal(nxt, r):
            break
        r = nxt
    return r


def face_region(keep, grid: int = 128):
    """The face in a binary parse mask `keep` (1x1xSxS, template-aligned):
    the face pixels connected to the eyes-to-mouth core, with enclosed holes
    (a shadow or a beard the parser missed) filled. Returns a binary mask the
    size of `keep`, or None when nothing at the core is face."""
    torch = _torch()
    F = torch.nn.functional  # noqa: N806
    size = keep.shape[-1]
    small = (F.interpolate(keep, size=(grid, grid), mode="area") > 0.5).float()
    marker = torch.zeros_like(small)
    x0, y0, x1, y1 = (round(v * grid / 512) for v in _CORE_512)
    marker[..., y0:y1, x0:x1] = 1
    comp = _reconstruct(marker, small)
    if comp.sum() == 0:
        return None
    border = torch.ones_like(small)
    border[..., 1:-1, 1:-1] = 0
    filled = 1 - _reconstruct(border, 1 - comp)
    # Back to full size: the parser's own edge where the component is, plus the holes.
    def up(m):
        return F.interpolate(m, size=(size, size), mode="nearest")
    k = 2 * (size // grid) + 1
    near = F.max_pool2d(up(comp), k, 1, k // 2)
    return torch.clamp(keep * near + up(filled - comp), 0, 1)


def soften(mask, grow: float, sigma: float):
    """Dilate a binary mask (1x1xSxS) by `grow` and blur it with a Gaussian of
    `sigma`, both as fractions of S, so the paste edge falls just outside the face."""
    torch = _torch()
    F = torch.nn.functional  # noqa: N806
    size = mask.shape[-1]
    k = 2 * round(grow * size) + 1
    if k > 1:
        mask = F.max_pool2d(mask, k, 1, k // 2)
    s = sigma * size
    if s > 0:
        r = max(1, int(math.ceil(3 * s)))
        g = torch.exp(-((torch.arange(-r, r + 1, device=mask.device, dtype=mask.dtype) / s) ** 2) / 2)
        g = (g / g.sum()).view(1, 1, 1, -1)
        mask = F.conv2d(F.pad(mask, (r, r, 0, 0), mode="replicate"), g)
        mask = F.conv2d(F.pad(mask, (0, 0, r, r), mode="replicate"), g.transpose(-1, -2))
    return mask


# The face model --------------------------------------------------------------

@dataclass
class FaceModel(Model):
    detector_path: Path = field(default=Path())
    restorer_path: Path = field(default=Path())
    parser_path: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)
    device: str = "auto"  # see devices.py
    whole_image: bool = True
    _det: Any = field(default=None, repr=False)
    _net: Any = field(default=None, repr=False)
    _parser: Any = field(default=None, repr=False)
    _arch: str = field(default="", repr=False)
    _dtype: Any = field(default=None, repr=False)
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _cache_lock: Any = field(default_factory=threading.Lock, repr=False)
    _coreml: Any = field(default=None, repr=False)  # coreml.Runner for the restorer, on a Mac

    # Loading ---------------------------------------------------------------

    def load(self) -> None:
        if self._net is not None:
            return
        with _load_lock:
            if self._net is not None:
                return
            torch = _torch()
            import spandrel  # noqa: PLC0415

            try:
                import spandrel_extra_arches  # noqa: PLC0415 - CodeFormer lives here

                spandrel_extra_arches.install(ignore_duplicates=True)
            except ImportError:
                pass
            from . import devices  # noqa: PLC0415

            self.device = devices.resolve(self.device)
            det = _build_retinaface()
            sd = torch.load(str(self.detector_path), map_location="cpu", weights_only=True)
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
            det.load_state_dict(sd, strict=True)
            self._det = det.to(self.device).eval()
            if self.parser_path is not None:
                from .bisenet_arch import BiSeNet  # noqa: PLC0415

                parser = BiSeNet(num_class=19)
                parser.load_state_dict(torch.load(str(self.parser_path), map_location="cpu", weights_only=True))
                self._parser = parser.to(self.device).eval()

            args = self.config.get("restorer_args", {})
            dtype = self.config.get("dtype", "fp16")
            if args.get("arch") == "gpen":
                # GPEN is not in spandrel; build its generator from the vendored code.
                from .gpen_arch import FullGenerator  # noqa: PLC0415

                net = FullGenerator(int(args.get("size", 1024)), int(args.get("style_dim", 512)),
                                    int(args.get("n_mlp", 8)), int(args.get("channel_multiplier", 2)),
                                    narrow=float(args.get("narrow", 1)))
                net.load_state_dict(torch.load(str(self.restorer_path), map_location="cpu", weights_only=True))
                self._arch, arch_name = "GPEN", "GPEN"
            else:
                desc = spandrel.ModelLoader().load_from_file(str(self.restorer_path))
                net, self._arch, arch_name = desc.model, desc.architecture.id, desc.architecture.name
                if dtype == "fp16" and not desc.supports_half:
                    dtype = "bf16" if desc.supports_bfloat16 else "fp32"
            if self._arch == "CodeFormer":
                dtype = "fp32"  # its codebook lookup fails in fp16 and bf16
            if self.device == "cpu":
                dtype = "fp32"
            self._dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
            net.to(self.device, self._dtype).eval()
            from .spandrel_adapter import configure_determinism  # noqa: PLC0415

            configure_determinism()
            self._net = net
            self._setup_coreml()
            log.info("loaded %s (detector RetinaFace-R50, restorer %s, %s, %s, %s mask%s)", self.id, arch_name,
                     self.device, dtype, "face-parsing" if self._parser is not None else "square",
                     f"; restorer on {self._coreml.describe()}" if self._coreml else "")

    def _setup_coreml(self) -> None:
        from . import coreml  # noqa: PLC0415

        choice = coreml.choose(self._arch, self.config, self.device)
        if choice is not None:
            self._coreml = coreml.Runner(self.id, self._build_restorer_cpu, self.restorer_path.parent / "coreml",
                                         *choice)
            self._coreml.sizes.add(int(self.config.get("face_size", 512)))

    def _build_restorer_cpu(self):
        """The restorer as an fp32 CPU module mapping a [0, 1] crop to its
        [0, 1] restoration (before clamping), for Core ML conversion. GPEN only."""
        torch = _torch()
        from .gpen_arch import FullGenerator  # noqa: PLC0415

        args = self.config.get("restorer_args", {})
        net = FullGenerator(int(args.get("size", 1024)), int(args.get("style_dim", 512)),
                            int(args.get("n_mlp", 8)), int(args.get("channel_multiplier", 2)),
                            narrow=float(args.get("narrow", 1)))
        net.load_state_dict(torch.load(str(self.restorer_path), map_location="cpu", weights_only=True))

        class Restorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = net

            def forward(self, crop):
                y = self.net(crop * 2 - 1)
                y = y[0] if isinstance(y, (tuple, list)) else y
                return (y + 1) / 2

        return Restorer().float().eval()

    def prepare_coreml(self) -> None:
        """Startup hook (see SpandrelModel.prepare_coreml): the restorer runs at the face size."""
        self.load()
        if self._coreml is not None:
            self._coreml.prepare(int(self.config.get("face_size", 512)))

    @property
    def backend(self) -> str:
        """The restorer's engine (part of the tile cache key; models/coreml.py)."""
        return self._coreml.backend if self._coreml is not None else "pytorch"

    def accel_status(self) -> dict | None:
        return self._coreml.status() if self._coreml is not None else None

    def engine(self) -> str:
        if self._coreml is not None and not self._coreml.failed:
            return f"restorer on {self._coreml.describe()}"
        return "PyTorch"

    # Detection -------------------------------------------------------------

    def detect(self, small: np.ndarray, full_w: int, full_h: int) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Faces found in `small`, a downscaled copy of a full_w x full_h image
        (see detection_size). Returns (score, 5x2 landmarks, face box x0 y0 x1 y1),
        both in full-image pixels."""
        self.load()
        torch = _torch()
        cfg = self.config
        dh, dw = small.shape[:2]
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(small.transpose(2, 0, 1)))[None].to(self.device)
            bgr = x[:, [2, 1, 0]] * 255.0 - torch.tensor(_BGR_MEAN, device=x.device).view(1, 3, 1, 1)
            loc, conf, ldm = self._det(bgr)
        # float64 on the CPU: MPS has no float64.
        loc, conf, ldm = (t.cpu().double().numpy() for t in (loc[0], conf[0, :, 1], ldm[0]))
        pri = _priors(dh, dw)
        keep = conf > float(cfg.get("detect_threshold", 0.9))
        loc, conf, ldm, pri = loc[keep], conf[keep], ldm[keep], pri[keep]
        if len(conf) == 0:
            return []
        v0, v1 = _VARIANCE
        centers = pri[:, :2] + loc[:, :2] * v0 * pri[:, 2:]
        sizes = pri[:, 2:] * np.exp(loc[:, 2:] * v1)
        boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], 1) * [dw, dh, dw, dh]
        lms = (pri[:, None, :2] + ldm.reshape(-1, 5, 2) * v0 * pri[:, None, 2:]) * [dw, dh]
        keep = _nms(boxes, conf, float(cfg.get("nms_threshold", 0.4)))
        rx, ry = full_w / dw, full_h / dh
        # Detection pixels -> full-image pixels (pixel centers for landmarks,
        # pixel edges for the face box).
        return [(float(conf[i]), (lms[i] + 0.5) * [rx, ry] - 0.5, boxes[i] * [rx, ry, rx, ry]) for i in keep]

    def detection_size(self, width: int, height: int) -> tuple[int, int]:
        """Size of the downscaled copy that detection runs on."""
        long_side = int(self.config.get("detect_long_side", 2048))
        r = min(1.0, long_side / max(width, height))
        return max(1, round(width * r)), max(1, round(height * r))

    def detections(self, key: Any, small_fn, full_w: int, full_h: int) -> list[tuple[float, np.ndarray]]:
        """detect() with a small cache keyed by source, so preview pans don't re-detect.
        `small_fn()` returns the downscaled image when detection is needed."""
        with self._cache_lock:
            if key is not None and key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        dets = self.detect(small_fn(), full_w, full_h)
        if key is not None:
            with self._cache_lock:
                self._cache[key] = dets
                while len(self._cache) > 16:
                    self._cache.popitem(last=False)
        return dets

    def layout(self, dets, scale: int, width: int, height: int) -> list[Face]:
        """Faces in an image `scale` times the detected one (width x height),
        filtered by size, in a fixed order (top to bottom, left to right)."""
        faces = [f for f, status in self.classify(dets, scale, width, height)
                 if status == "enhanced" and f.box[2] > f.box[0] and f.box[3] > f.box[1]]
        faces.sort(key=lambda f: (f.box[1], f.box[0], f.box[2], f.box[3]))
        return faces

    def classify(self, dets, scale: int, width: int, height: int) -> list[tuple[Face, str]]:
        """Every detected face with what this model does to it: "enhanced",
        "too_small" or "too_large" (by the size of its aligned crop)."""
        cfg = self.config
        face_size = int(cfg.get("face_size", 512))
        min_face, max_face = float(cfg.get("min_face", 24)), float(cfg.get("max_face", 0))
        out = []
        for score, pts, *_ in dets:
            f = make_face((np.asarray(pts) + 0.5) * scale - 0.5, score, face_size, width, height)
            if f.size < min_face:
                status = "too_small"
            elif max_face and f.size > max_face:
                status = "too_large"
            else:
                status = "enhanced"
            out.append((f, status))
        return out

    @staticmethod
    def touching(faces: list[Face], area) -> list[Face]:
        return [f for f in faces if f.box[0] < area.x + area.w and f.box[2] > area.x
                and f.box[1] < area.y + area.h and f.box[3] > area.y]

    @staticmethod
    def input_rect(faces: list[Face], area):
        """The input pixels needed to render `area`: the area plus the whole
        box of every face that touches it."""
        from ..imaging.geometry import Rect  # noqa: PLC0415

        x0, y0, x1, y1 = area.x, area.y, area.x + area.w, area.y + area.h
        for f in FaceModel.touching(faces, area):
            x0, y0 = min(x0, f.box[0]), min(y0, f.box[1])
            x1, y1 = max(x1, f.box[2]), max(y1, f.box[3])
        return Rect(x0, y0, x1 - x0, y1 - y0)

    # Restoration -----------------------------------------------------------

    def _restore(self, crop):
        """crop: 1x3xSxS float32 in [0, 1] on the device. Returns the same shape."""
        torch = _torch()
        if self._coreml is not None and self._arch == "GPEN":
            y = self._coreml.run(crop.float().cpu().numpy())
            if y is not None:
                return torch.from_numpy(y).to(crop.device).clamp(0, 1)
        args = dict(self.config.get("restorer_args", {}))
        x = (crop * 2 - 1).to(self._dtype)
        net = self._net
        with torch.inference_mode():
            if self._arch == "GFPGAN":
                y = net(x, return_rgb=False, randomize_noise=False)[0]
            elif self._arch == "CodeFormer":
                y = net(x, weight=float(args.get("weight", 0.5)), adain=True)[0]
            else:
                y = net(x)
                y = y[0] if isinstance(y, (tuple, list)) else y
        return ((y.float() + 1) / 2).clamp(0, 1)

    def _parse_mask(self, img):
        """Paste mask (1x1xSxS) from face parsing of `img`, a 1x3xSxS crop in
        [0, 1], or None when the parser finds no face at its middle."""
        torch = _torch()
        F = torch.nn.functional  # noqa: N806
        cfg = self.config
        n = img.shape[-1]
        x = img.float()
        if n != 512:
            x = F.interpolate(x, size=(512, 512), mode="area" if n > 512 else "bilinear", align_corners=None)
        mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        with torch.inference_mode():
            labels = self._parser((x - mean) / std).argmax(1, keepdim=True)
            lut = torch.zeros(19, device=x.device)
            lut[list(cfg.get("parse_keep", PARSE_KEEP))] = 1
            region = face_region(lut[labels])
            if region is None:
                return None
            if n != 512:
                region = (F.interpolate(region, size=(n, n), mode="bilinear", align_corners=False) > 0.5).float()
            return soften(region, float(cfg.get("parse_grow", 0.01)), float(cfg.get("parse_feather", 0.015)))

    def _sample(self, img, a_inv, b_inv, out_h, out_w, ox, oy):
        """Bilinear-sample `img` (1xCxHxW) at img_pt = a_inv @ (out_pt + (ox, oy)) + b_inv
        for each output pixel. Zero outside `img`."""
        torch = _torch()
        h, w = img.shape[-2:]
        # The grid is built in float64 on the CPU (MPS has no float64), then moved.
        ys = torch.arange(out_h, dtype=torch.float64) + oy
        xs = torch.arange(out_w, dtype=torch.float64) + ox
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        a = torch.tensor(a_inv, dtype=torch.float64)
        sx = a[0, 0] * gx + a[0, 1] * gy + b_inv[0]
        sy = a[1, 0] * gx + a[1, 1] * gy + b_inv[1]
        grid = torch.stack([2 * sx / max(w - 1, 1) - 1, 2 * sy / max(h - 1, 1) - 1], -1)[None].float().to(img.device)
        return torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def _paste_face(self, window: np.ndarray, origin: tuple[int, int], face: Face):
        """The face's restored pixels and mask over its box, in full-image
        coordinates. Depends only on the source and the face, never on the
        requested region. `window` holds input pixels from `origin` and must
        cover the face's box."""
        torch = _torch()
        F = torch.nn.functional  # noqa: N806
        cfg = self.config
        n = int(cfg.get("face_size", 512))
        a = np.array(face.a)
        b = np.array(face.b)
        a_inv = np.linalg.inv(a)
        # Source patch covering the crop (the face's box), pre-shrunk by an
        # integer factor when the face is larger than the crop, against aliasing.
        x0, y0, x1, y1 = face.box
        wx, wy = origin
        patch = window[y0 - wy:y1 - wy, x0 - wx:x1 - wx]
        patch = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))[None].to(self.device)
        k = max(1, int(face.size / n))
        if k > 1:
            ph, pw = (patch.shape[2] // k) * k, (patch.shape[3] // k) * k
            patch = F.avg_pool2d(patch[:, :, :ph, :pw], k)
        # crop_pt -> img_pt = a_inv @ (crop_pt - b); img_pt -> patch_pt = (img_pt - (x0, y0) - (k-1)/2) / k
        pa = a_inv / k
        pb = (-a_inv @ b - np.array([x0, y0]) - (k - 1) / 2) / k
        crop = self._sample(patch, pa, pb, n, n, 0, 0)
        restored = self._restore(crop)
        sigma = float(cfg.get("color_anchor_sigma", 8))
        if sigma > 0:
            from .spandrel_adapter import color_anchor  # noqa: PLC0415

            restored = color_anchor(restored, crop, sigma)
        mask = torch.from_numpy(crop_mask(n, float(cfg.get("mask_inset", 0.05)),
                                          float(cfg.get("mask_feather", 0.12))))[None, None].to(self.device)
        if self._parser is not None:
            face = self._parse_mask(restored)
            if face is not None:
                mask = mask * face
        # Back to the image: box pixel (u, v) -> img (u + x0, v + y0) -> crop a @ img + b.
        both = torch.cat([restored, mask], 1)
        back = self._sample(both, a, b, y1 - y0, x1 - x0, x0, y0)
        out = back[0, :3].permute(1, 2, 0).cpu().numpy()
        m = back[0, 3].cpu().numpy()[:, :, None]
        return out, m

    def run(self, tile: np.ndarray) -> np.ndarray:  # pragma: no cover - whole-image model
        raise NotImplementedError("face models run through run_area()")

    def run_area(self, window: np.ndarray, origin: tuple[int, int], area, faces: list[Face],
                 strength: float | None = None, on_progress=None, is_canceled=None) -> np.ndarray:
        """Render `area` (full-image pixels). `window` holds the input from
        `origin` and covers input_rect(faces, area). Strength blends each
        restored face with the input: input + strength * (restored - input)."""
        self.load()
        wx, wy = origin
        out = window[area.y - wy:area.y - wy + area.h, area.x - wx:area.x - wx + area.w].copy()
        todo = self.touching(faces, area)
        s = 1.0 if strength is None else float(strength)
        for i, face in enumerate(todo):
            if is_canceled and is_canceled():
                from ..imaging.pipeline import Canceled  # noqa: PLC0415

                raise Canceled()
            vals, m = self._paste_face(window, origin, face)
            x0, y0, x1, y1 = face.box
            ix0, iy0 = max(x0, area.x), max(y0, area.y)
            ix1, iy1 = min(x1, area.x + area.w), min(y1, area.y + area.h)
            sl_box = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
            sl_out = (slice(iy0 - area.y, iy1 - area.y), slice(ix0 - area.x, ix1 - area.x))
            mm = m[sl_box] * s
            out[sl_out] = out[sl_out] * (1 - mm) + vals[sl_box] * mm
            if on_progress:
                on_progress((i + 1) / len(todo))
        if on_progress:
            on_progress(1.0)
        return out


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    keep = []
    x0, y0, x1, y1 = boxes.T
    areas = (x1 - x0) * (y1 - y0)
    while len(order):
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = np.clip(np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]), 0, None)
        ih = np.clip(np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]), 0, None)
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter + 1e-12)
        order = rest[iou <= thresh]
    return keep


def validate_config(config: dict[str, Any]) -> None:
    for key in ("detector", "restorer"):
        if not config.get(key):
            raise ValueError(f"config.{key} is required")
    if "parser" in config and not config["parser"]:
        raise ValueError("config.parser must name a file when set")
    if config.get("dtype", "fp16") not in ("fp16", "bf16", "fp32"):
        raise ValueError("config.dtype must be fp16, bf16 or fp32")
    from . import coreml  # noqa: PLC0415

    coreml.validate_config(config.get("coreml"))


def make(manifest, path: Path) -> FaceModel:
    config = dict(manifest.config)
    validate_config(config)
    return FaceModel(
        id=manifest.id,
        operation=manifest.operation,
        version=manifest.version,
        scale=1,
        detector_path=path / config["detector"],
        restorer_path=path / config["restorer"],
        parser_path=path / config["parser"] if config.get("parser") else None,
        config=config,
    )
