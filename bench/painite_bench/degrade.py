"""Synthetic degradations applied to clean photos so every test has a ground truth."""
import cv2
import numpy as np


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


# Poisson-Gaussian noise in linear sensor space: var = k*x + read^2 (x normalized to 1).
# Values roughly follow published calibrations of full-frame sensors at the given ISO
# (Brooks et al., "Unprocessing Images for Learned Raw Denoising", 2019, ranges).
ISO_LEVELS = {
    "iso3200": dict(k=0.006, read=0.0035),
    "iso12800": dict(k=0.024, read=0.010),
}


def _bilinear_demosaic(mosaic: np.ndarray) -> np.ndarray:
    """RGGB bilinear demosaic of a single-plane float array. Linear, so applying it to a
    noise field gives the spatially and chromatically correlated noise of a real camera."""
    h, w = mosaic.shape
    mr = np.zeros((h, w), np.float32); mr[0::2, 0::2] = 1
    mb = np.zeros((h, w), np.float32); mb[1::2, 1::2] = 1
    mg = 1 - mr - mb
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
    f = lambda m, k: cv2.filter2D(mosaic * m, -1, k, borderType=cv2.BORDER_REFLECT)
    return np.stack([f(mr, k_rb), f(mg, k_g), f(mb, k_rb)], -1)


def camera_noise(img: np.ndarray, level: str, seed: int) -> np.ndarray:
    """img: HxWx3 float32 sRGB in [0,1]. Returns noisy sRGB float32."""
    p = ISO_LEVELS[level]
    rng = np.random.default_rng(seed)
    lin = srgb_to_linear(img).astype(np.float32)
    h, w, _ = lin.shape
    cfa = np.zeros((h, w), np.int64); cfa[0::2, 1::2] = 1; cfa[1::2, 0::2] = 1; cfa[1::2, 1::2] = 2
    site = np.take_along_axis(lin, cfa[..., None], 2)[..., 0]
    sigma = np.sqrt(p["k"] * site + p["read"] ** 2)
    noise = (rng.standard_normal((h, w)).astype(np.float32) * sigma).astype(np.float32)
    noisy = lin + _bilinear_demosaic(noise)
    return linear_to_srgb(noisy).astype(np.float32)


def blur(img: np.ndarray, kind: str, seed: int) -> np.ndarray:
    """Mild softness a photographer would want to 'sharpen': lens/missed-focus blur
    plus a little sensor noise so the deblurrers cannot exploit a noise-free input."""
    rng = np.random.default_rng(seed)
    if kind == "gauss1.2":
        out = cv2.GaussianBlur(img, (0, 0), 1.2)
    elif kind == "defocus2":
        r = 2
        k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(k, (r, r), r, 1, -1)
        out = cv2.filter2D(img, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT)
    elif kind == "motion5":
        k = np.zeros((5, 5), np.float32); k[2, :] = 1
        m = cv2.getRotationMatrix2D((2, 2), float(rng.uniform(0, 180)), 1)
        k = cv2.warpAffine(k, m, (5, 5)); k /= k.sum()
        out = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)
    else:
        raise ValueError(kind)
    out = out + rng.standard_normal(out.shape).astype(np.float32) * (1.5 / 255)
    return np.clip(out, 0, 1).astype(np.float32)


def downscale2x(img: np.ndarray, kind: str, seed: int) -> np.ndarray:
    """'clean': antialiased bicubic /2. 'camera': /2 then mild ISO noise and JPEG q90,
    which is closer to a real photo someone wants to enlarge."""
    h, w, _ = img.shape
    lr = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    if kind == "camera":
        lr = camera_noise(lr, "iso3200", seed) * 0.5 + lr * 0.5  # light noise
        enc = cv2.imencode(".jpg", (np.clip(lr, 0, 1)[..., ::-1] * 255).round().astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        lr = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255
    return np.clip(lr, 0, 1).astype(np.float32)
