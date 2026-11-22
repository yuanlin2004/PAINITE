"""Face op pipeline: detect on a downscaled copy, 5-point align to the FFHQ template, restore,
inverse-warp and paste back with a soft mask, blended by strength.

Images are HxWx3 uint8 BGR numpy arrays (what the bench reads); warps run on the GPU.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from restorers import FACE_WEIGHTS, Restorer

# FFHQ 512 template for 5 landmarks (left eye, right eye, nose, mouth left, mouth right), as in
# facexlib.utils.face_restoration_helper / GFPGAN / CodeFormer.
FFHQ_512 = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)

# parsenet classes kept in the paste mask (facexlib/GFPGAN MASK_COLORMAP): skin, brows, eyes,
# glasses, ears, earrings, nose, mouth, lips. Background, neck, cloth, hair and hat are excluded.
PARSE_KEEP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


@dataclass
class Face:
    box: np.ndarray  # x0 y0 x1 y1 in full-image px
    score: float
    lm5: np.ndarray  # 5x2 full-image px

    @property
    def eye_dist(self) -> float:
        return float(np.linalg.norm(self.lm5[1] - self.lm5[0]))

    @property
    def size(self) -> float:
        """Side of the 512 template crop in full-image px (template eye distance is 126 px)."""
        return self.eye_dist * 512 / 125.92


class Detector:
    def __init__(self, half=False):
        from facexlib.detection import init_detection_model

        self.net = init_detection_model("retinaface_resnet50", half=half, device="cuda",
                                        model_rootpath=str(FACE_WEIGHTS))

    @torch.inference_mode()
    def __call__(self, img, long_side=2048, conf=0.9, min_eye_dist=0.0) -> list[Face]:
        h, w = img.shape[:2]
        s = min(1.0, long_side / max(h, w)) if long_side else 1.0
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
        dets = self.net.detect_faces(small, conf_threshold=conf)
        faces = [Face(d[:4] / s, float(d[4]), d[5:15].reshape(5, 2) / s) for d in dets]
        return sorted([f for f in faces if f.eye_dist >= min_eye_dist], key=lambda f: -f.eye_dist)


class Parser:
    def __init__(self):
        from facexlib.parsing import init_parsing_model

        self.net = init_parsing_model("parsenet", device="cuda", model_rootpath=str(FACE_WEIGHTS))

    @torch.inference_mode()
    def __call__(self, face512: torch.Tensor) -> torch.Tensor:
        """face512: Bx3x512x512 RGB [0,1]. Returns Bx1x512x512 float mask of face regions."""
        logits = self.net(face512 * 2 - 1)[0]
        lab = logits.argmax(1, keepdim=True)
        keep = torch.zeros(19, device=lab.device)
        keep[PARSE_KEEP] = 1
        return keep[lab].float()


def _affine(lm5, size):
    dst = FFHQ_512 * (size / 512)
    return cv2.estimateAffinePartial2D(lm5.astype(np.float32), dst, method=cv2.LMEDS)[0]


def _grid(m_out_to_in, out_hw, in_hw, device):
    """Sampling grid for F.grid_sample: output px -> input px through 2x3 affine m."""
    h, w = out_hw
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    m = torch.tensor(m_out_to_in, device=device, dtype=torch.float32)
    sx = m[0, 0] * xs + m[0, 1] * ys + m[0, 2]
    sy = m[1, 0] * xs + m[1, 1] * ys + m[1, 2]
    ih, iw = in_hw
    return torch.stack([(sx + 0.5) / iw * 2 - 1, (sy + 0.5) / ih * 2 - 1], -1)[None]


def warp(img_t, m, out_hw, mode="bilinear", padding="reflection"):
    """img_t: 1xCxHxW. m maps input px -> output px; returns 1xCxout_h x out_w."""
    inv = cv2.invertAffineTransform(m)
    g = _grid(inv, out_hw, img_t.shape[-2:], img_t.device)
    return F.grid_sample(img_t, g, mode=mode, padding_mode=padding, align_corners=False)


def template_mask(size, kind, device):
    """Soft paste mask in template space. 'rect' is GFPGAN/facexlib's default (eroded crop
    square, Gaussian feather); 'ellipse' covers the face oval with a wide feather."""
    m = np.zeros((size, size), np.float32)
    if kind == "rect":
        e = int(size * 0.04)
        m[e:-e, e:-e] = 1
        m = cv2.GaussianBlur(m, (0, 0), size * 0.02)
    elif kind == "square":
        # the server adapter's crop_mask(inset=0.05, feather=0.12): smoothstep ramp from the border
        c = np.arange(size, dtype=np.float32) + 0.5
        t = np.clip((np.minimum(c, size - c) - 0.05 * size) / (0.12 * size), 0, 1)
        r = t * t * (3 - 2 * t)
        m = np.minimum(r[:, None], r[None, :])
    elif kind == "ellipse":
        c = (size // 2, int(size * 0.55))
        cv2.ellipse(m, c, (int(size * 0.36), int(size * 0.44)), 0, 0, 360, 1, -1)
        m = cv2.GaussianBlur(m, (0, 0), size * 0.04)
    else:
        raise ValueError(kind)
    return torch.from_numpy(m).to(device)[None, None]


def parse_mask(parser: Parser, restored: torch.Tensor):
    """Face-parsing mask on the restored crop, feathered like GFPGAN's use_parse path."""
    size = restored.shape[-1]
    x = restored if size == 512 else F.interpolate(restored, size=(512, 512), mode="area")
    m = parser(x)
    m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)
    k = max(3, int(size * 0.02) | 1)
    m = -F.max_pool2d(-m, k, 1, k // 2)  # erode a little so the edge sits inside the face
    s = size * 0.02
    r = int(3 * s)
    g = torch.exp(-((torch.arange(-r, r + 1, device=m.device) / s) ** 2) / 2)
    g = (g / g.sum()).view(1, 1, 1, -1)
    m = F.conv2d(F.pad(m, (r, r, 0, 0), mode="replicate"), g)
    m = F.conv2d(F.pad(m, (0, 0, r, r), mode="replicate"), g.transpose(-1, -2))
    return m * template_mask(size, "rect", m.device)  # never reach the crop border


def lowpass(x, sigma):
    r = max(1, int(3 * sigma))
    g = torch.exp(-((torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2) / 2)
    g = (g / g.sum()).view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), g.transpose(-1, -2), groups=x.shape[1])


def _roi(m, size, hw, pad=0):
    """Bounding box (x0, y0, x1, y1) in the full image of the template square, clipped."""
    inv = cv2.invertAffineTransform(m)
    c = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], np.float32) @ inv.T
    x0, y0 = np.floor(c.min(0)).astype(int) - pad
    x1, y1 = np.ceil(c.max(0)).astype(int) + pad
    return max(x0, 0), max(y0, 0), min(x1, hw[1]), min(y1, hw[0])


def _shift(m, x0, y0):
    """Affine full px -> template px, re-based so it takes ROI px starting at (x0, y0)."""
    m = m.copy()
    m[:, 2] += m[:, :2] @ np.array([x0, y0], np.float32)
    return m


def align(full, m, size, ratio):
    """Crop one face into template space. When the face is larger than the template
    (ratio > 1), low-pass the ROI first so the downscaling warp does not alias."""
    hw = full.shape[-2:]
    pad = int(3 * ratio) + 2
    x0, y0, x1, y1 = _roi(m, size, hw, pad)
    roi = full[..., y0:y1, x0:x1]
    if ratio > 1:
        roi = lowpass(roi, 0.5 * ratio)
    return warp(roi, _shift(m, x0, y0), (size, size))


@dataclass
class Options:
    mask: str = "parse"  # rect | square | ellipse | parse
    strength: float = 1.0
    det_long_side: int = 2048
    min_eye_dist: float = 8.0  # px in the full image; smaller faces are left alone
    large_face: str = "restore"  # restore | lowfreq | skip, for faces larger than the restorer's size
    large_face_ratio: float = 1.5  # face size / restorer size above which large_face applies
    color_anchor: float = 0.0  # sigma in template px (512 scale) for a low-freq color anchor, 0 = off


def enhance(img: np.ndarray, restorer: Restorer, det: Detector, parser: Parser | None, opt: Options,
            faces: list[Face] | None = None, batch=8, alpha_out: list | None = None):
    """Returns (output uint8 BGR, faces, per-face info). If alpha_out is a list, the paste
    alpha (HxW float, all faces composited) is appended to it."""
    dev = "cuda"
    if faces is None:
        faces = det(img, opt.det_long_side, min_eye_dist=opt.min_eye_dist)
    full = torch.from_numpy(img[..., ::-1].copy()).to(dev).permute(2, 0, 1)[None].float() / 255
    acc = full.clone()
    alpha = torch.zeros_like(full[:, :1])
    S = restorer.size
    infos = []
    for i in range(0, len(faces), batch):
        chunk = faces[i:i + batch]
        ms = [_affine(f.lm5, S) for f in chunk]
        crops = torch.cat([align(full, m, S, f.size / S) for f, m in zip(chunk, ms)])
        with torch.inference_mode():
            out = restorer(crops)
        if opt.color_anchor:
            sg = opt.color_anchor * S / 512
            out = out - lowpass(out, sg) + lowpass(crops, sg)
        for f, m, crop, o in zip(chunk, ms, crops, out):
            o, crop = o[None], crop[None]
            ratio = f.size / S
            action = "restore"
            if ratio > opt.large_face_ratio:
                action = opt.large_face
            infos.append(dict(eye_dist=f.eye_dist, size=f.size, action=action))
            if action == "skip":
                continue
            if opt.mask == "parse":
                mk = parse_mask(parser, o)
            else:
                mk = template_mask(S, opt.mask, dev)
            x0, y0, x1, y1 = _roi(m, S, img.shape[:2])
            if x1 <= x0 or y1 <= y0:
                continue
            m_back = cv2.invertAffineTransform(_shift(m, x0, y0))  # template px -> ROI px
            hw = (y1 - y0, x1 - x0)
            o_full = warp(o, m_back, hw, padding="border")
            mk_full = warp(mk, m_back, hw, padding="zeros")
            base = acc[..., y0:y1, x0:x1]
            if action == "lowfreq":
                # keep the original's fine detail, take the restorer's tone/shape only:
                # frequencies finer than one restorer pixel come from the original
                sg = max(ratio, 1.0)
                o_full = base - lowpass(base, sg) + lowpass(o_full, sg)
            a = mk_full * opt.strength
            acc[..., y0:y1, x0:x1] = base + a * (o_full - base)
            al = alpha[..., y0:y1, x0:x1]
            alpha[..., y0:y1, x0:x1] = al + a * (1 - al)
    if alpha_out is not None:
        alpha_out.append(alpha[0, 0].cpu().numpy())
    out = (acc[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()[..., ::-1]
    return np.ascontiguousarray(out), faces, infos
