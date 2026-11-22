"""Full-reference metrics for the bake-off. All inputs are HxWx3 float32 sRGB [0,1]."""
import cv2
import numpy as np
import pyiqa
import torch
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

_iqa = {}


def _metric(name):
    if name not in _iqa:
        _iqa[name] = pyiqa.create_metric(name, device="cuda", as_loss=False)
    return _iqa[name]


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].cuda()


def _y(x):
    return 16 / 255 + (65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]) / 255


def psnr_y(out, gt):
    mse = np.mean((_y(out) - _y(gt)) ** 2)
    return float(10 * np.log10(1 / max(mse, 1e-12)))


def ssim_y(out, gt):
    return float(structural_similarity(_y(out), _y(gt), data_range=1.0, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))


@torch.inference_mode()
def lpips(out, gt):
    return float(_metric("lpips")(_t(out), _t(gt)))


@torch.inference_mode()
def dists(out, gt):
    return float(_metric("dists")(_t(out), _t(gt)))


def color_shift(out, gt):
    """Mean CIEDE2000 between heavily low-passed images: isolates casts and saturation
    shifts from detail and noise differences."""
    lp = lambda x: cv2.GaussianBlur(x, (0, 0), 6)
    return float(deltaE_ciede2000(rgb2lab(lp(out)), rgb2lab(lp(gt))).mean())


def _edge_band(gt, width=3):
    g = (_y(gt) * 255).astype(np.uint8)
    e = cv2.Canny(g, 60, 160)
    return cv2.dilate(e, np.ones((2 * width + 1, 2 * width + 1), np.uint8)) > 0


def halo(out, gt):
    """Overshoot beyond the ground truth's local range near strong edges, in 8-bit levels.
    Ringing and oversharpening halos show up as values brighter than the local max or
    darker than the local min of the clean image."""
    yo, yg = _y(out), _y(gt)
    k = np.ones((5, 5), np.uint8)
    hi, lo = cv2.dilate(yg, k), cv2.erode(yg, k)
    over = np.maximum(yo - hi, 0) + np.maximum(lo - yo, 0)
    band = _edge_band(gt)
    return float(over[band].mean() * 255) if band.any() else 0.0


def _hf_energy(y):
    return np.abs(cv2.Laplacian(cv2.GaussianBlur(y, (0, 0), 0.8), cv2.CV_32F))


def detail_ratio(out, gt):
    """High-frequency energy of the output relative to the clean image, measured in
    textured regions. 1.0 is ideal; below means smoothing (detail loss), above means
    residual noise or invented texture."""
    yg, yo = _y(gt).astype(np.float32), _y(out).astype(np.float32)
    eg = _hf_energy(yg)
    tex = cv2.GaussianBlur(eg, (0, 0), 4) > np.percentile(eg, 60)
    return float(_hf_energy(yo)[tex].mean() / max(eg[tex].mean(), 1e-8))


def full_reference(out, gt):
    return dict(psnr=psnr_y(out, gt), ssim=ssim_y(out, gt), lpips=lpips(out, gt), dists=dists(out, gt),
                dE=color_shift(out, gt), halo=halo(out, gt), detail=detail_ratio(out, gt))
