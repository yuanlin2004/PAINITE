"""Shared setup: load each installed default model's bare network (fp32, CPU)
plus a realistic input for it."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))
from painite_server.models import default_registry  # noqa: E402

DATA = Path.home() / "Library/Application Support/painite-server"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
PHOTO = Path(os.environ.get("PAINITE_MAC_BENCH_PHOTO", Path.home() / "Downloads/two.jpg"))
OPS = ["denoise", "sharpen", "unblur_motion", "unblur_focus", "upscale2x", "face"]


def photo() -> np.ndarray:
    return np.asarray(Image.open(PHOTO).convert("RGB"), dtype=np.float32) / 255.0


def window(size: int = 640) -> np.ndarray:
    """A size x size crop around the left person's face (tile 512 + 2*64 context)."""
    img = photo()
    cy, cx = 1100, 1050
    return np.ascontiguousarray(img[cy - size // 2: cy + size // 2, cx - size // 2: cx + size // 2])


class Wrapped(torch.nn.Module):
    """[0,1] RGB in -> [0,1] RGB out, the way the server calls it (minus postprocess)."""

    def __init__(self, net, pad_multiple=None):
        super().__init__()
        self.net = net
        self.pad = pad_multiple

    def forward(self, x):
        return self.net(x)


class GpenWrapped(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x * 2 - 1)
        y = y[0] if isinstance(y, (tuple, list)) else y
        return (y + 1) / 2


def load_all(ops=None):
    """{op: (model_obj, bare fp32 cpu module mapping [0,1]->[0,1])}"""
    reg = default_registry(DATA)
    out = {}
    for op in ops or OPS:
        m = reg.get(reg.defaults[op])
        m.device = "cpu"
        m.load()
        if op == "face":
            net = GpenWrapped(m._net)
        elif hasattr(m._net, "model"):  # spandrel descriptor
            net = Wrapped(m._net.model)
        else:
            net = Wrapped(m._net)
        out[op] = (m, net.float().eval())
    return out


def face_crop(face_model) -> np.ndarray:
    """The aligned 1024 crop the face model would restore for the first face in PHOTO."""
    img = photo()
    h, w = img.shape[:2]
    face_model.load()
    sw, sh = face_model.detection_size(w, h)
    small = np.asarray(Image.open(PHOTO).convert("RGB").resize((sw, sh), Image.BILINEAR), np.float32) / 255
    dets = face_model.detect(small, w, h)
    faces = [f for f, _ in face_model.classify(dets, 1, w, h)]
    got = {}
    orig = face_model._restore

    def grab(crop):
        got.setdefault("crop", crop.detach().float().cpu().numpy()[0].transpose(1, 2, 0))
        raise StopIteration

    face_model._restore = grab
    try:
        face_model._paste_face(img, (0, 0), faces[0])
    except StopIteration:
        pass
    face_model._restore = orig
    return got["crop"]


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def timeit(fn, dev, warm=2, n=5):
    for _ in range(warm):
        fn()
    sync(dev)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def psnr(a, b):
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1 / mse)


def maxdiff8(a, b):
    return float(np.abs(np.clip(a, 0, 1) - np.clip(b, 0, 1)).max() * 255)


def save_png(arr, path):
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
