"""Face restorer candidates behind one interface.

Every restorer takes a batch of aligned faces, Bx3xSxS RGB float in [0, 1] on the GPU, where
S = restorer.size (the FFHQ template scaled to S), and returns the same shape. Spandrel's
call_fn is bypassed for GFPGAN and CodeFormer so their extra arguments (noise, fidelity
weight) can be set.
"""
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from painite_bench.config import DATA_DIR, WEIGHTS_DIR  # noqa: E402

FACE_WEIGHTS = WEIGHTS_DIR / "face"
SRC_DIR = DATA_DIR / "src"  # upstream repos cloned for archs spandrel does not cover


def _spandrel(path):
    import spandrel
    import spandrel_extra_arches

    spandrel_extra_arches.install(ignore_duplicates=True)
    return spandrel.ModelLoader().load_from_file(str(path)).model


class Restorer:
    size = 512
    name = ""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GFPGAN(Restorer):
    def __init__(self, file="GFPGANv1.4.pth"):
        self.net = _spandrel(FACE_WEIGHTS / file).cuda().eval()

    @torch.inference_mode()
    def __call__(self, x):
        # randomize_noise=False: fixed noise buffers from the checkpoint, so output is deterministic
        y = self.net(x * 2 - 1, return_rgb=False, randomize_noise=False)[0]
        return ((y + 1) / 2).clamp(0, 1)


class RestoreFormer(Restorer):
    def __init__(self, file):
        self.net = _spandrel(FACE_WEIGHTS / file).cuda().eval()

    @torch.inference_mode()
    def __call__(self, x):
        return ((self.net(x * 2 - 1)[0] + 1) / 2).clamp(0, 1)


class RestoreFormerPP(Restorer):
    """RestoreFormer++ (spandrel detects the checkpoint but not its extra 32 px decoder
    attention), via the upstream arch file, which is plain PyTorch."""

    def __init__(self):
        spec = importlib.util.spec_from_file_location(
            "rfpp_arch", SRC_DIR / "RestoreFormerPlusPlus/RestoreFormer/modules/vqvae/vqvae_arch.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.net = m.VQVAEGANMultiHeadTransformer(head_size=4, ex_multi_scale_num=1)
        sd = torch.load(FACE_WEIGHTS / "RestoreFormer++.ckpt", map_location="cpu", weights_only=False)["state_dict"]
        self.net.load_state_dict({k.removeprefix("vqvae."): v for k, v in sd.items()})
        self.net.cuda().eval()

    @torch.inference_mode()
    def __call__(self, x):
        return ((self.net(x * 2 - 1)[0] + 1) / 2).clamp(0, 1)


class CodeFormer(Restorer):
    def __init__(self, w: float):
        self.w = w
        self.net = _spandrel(FACE_WEIGHTS / "codeformer.pth").cuda().eval()

    @torch.inference_mode()
    def __call__(self, x):
        return ((self.net(x * 2 - 1, weight=self.w)[0] + 1) / 2).clamp(0, 1)


def _gpen_ops():
    """GPEN imports `op` (StyleGAN2 CUDA extensions compiled at import). Provide the
    upstream pure-PyTorch fallbacks instead so nothing is compiled."""
    if "op" in sys.modules:
        return
    from torch import nn

    def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5, device="cpu"):
        return scale * F.leaky_relu(input + bias.view((1, -1) + (1,) * (input.ndim - 2)), negative_slope)

    class FusedLeakyReLU(nn.Module):
        def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5, device="cpu"):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(channel))
            self.negative_slope, self.scale = negative_slope, scale

        def forward(self, input):
            return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0), device="cpu"):
        # upfirdn2d_native from GPEN/face_model/op/upfirdn2d.py, grouped conv instead of reshape
        b, c, h, w = input.shape
        kh, kw = kernel.shape
        x = input
        if up > 1:
            x = x.view(b, c, h, 1, w, 1)
            x = F.pad(x, [0, up - 1, 0, 0, 0, up - 1])
            x = x.view(b, c, h * up, w * up)
        p0, p1 = pad
        x = F.pad(x, [max(p0, 0), max(p1, 0), max(p0, 0), max(p1, 0)])
        x = x[:, :, max(-p0, 0): x.shape[2] - max(-p1, 0), max(-p0, 0): x.shape[3] - max(-p1, 0)]
        k = torch.flip(kernel, [0, 1]).to(x.dtype).view(1, 1, kh, kw).repeat(c, 1, 1, 1)
        x = F.conv2d(x, k, groups=c)
        return x[:, :, ::down, ::down]

    op = types.ModuleType("op")
    op.FusedLeakyReLU, op.fused_leaky_relu, op.upfirdn2d = FusedLeakyReLU, fused_leaky_relu, upfirdn2d
    sys.modules["op"] = op


class GPEN(Restorer):
    def __init__(self, size: int):
        _gpen_ops()
        spec = importlib.util.spec_from_file_location("gpen_model", SRC_DIR / "GPEN/face_model/gpen_model.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.size = size
        self.net = m.FullGenerator(size, 512, 8, 2, narrow=1, device="cpu")
        sd = torch.load(FACE_WEIGHTS / f"GPEN-BFR-{size}.pth", map_location="cpu", weights_only=True)
        self.net.load_state_dict(sd)
        self.net.cuda().eval()

    @torch.inference_mode()
    def __call__(self, x):
        return ((self.net(x * 2 - 1)[0] + 1) / 2).clamp(0, 1)


class Anchored(Restorer):
    """Low-frequency color/tone anchor on the crop, as the server adapter does:
    out - blur(out) + blur(in), σ in crop px at 512 (scaled for other sizes)."""

    def __init__(self, r: Restorer, sigma: float):
        self.r, self.size, self.sigma = r, r.size, sigma * r.size / 512

    def __call__(self, x):
        from torchvision.transforms.functional import gaussian_blur

        y = self.r(x)
        k = int(self.sigma * 3) * 2 + 1
        blur = lambda t: gaussian_blur(t, [k, k], [self.sigma, self.sigma])  # noqa: E731
        return (y - blur(y) + blur(x)).clamp(0, 1)


def load(name: str) -> Restorer:
    """Restorer by bake-off name; a "+anchor<σ>" suffix adds the color anchor."""
    if "+anchor" in name:
        base, sigma = name.split("+anchor")
        r = Anchored(load(base), float(sigma))
        r.name = name
        return r
    if name == "gfpgan-1.4":
        r = GFPGAN()
    elif name == "restoreformer":
        r = RestoreFormer("RestoreFormer.pth")
    elif name == "restoreformer++":
        r = RestoreFormerPP()
    elif name.startswith("codeformer-w"):
        r = CodeFormer(float(name.removeprefix("codeformer-w")))
    elif name.startswith("gpen-"):
        r = GPEN(int(name.removeprefix("gpen-")))
    elif name == "pmrf":
        from pmrf import PMRF
        r = PMRF()
    elif name == "osdface":
        from osdface import OSDFace
        r = OSDFace()
    else:
        raise KeyError(name)
    r.name = name
    return r


def run(r: Restorer, face512: torch.Tensor) -> torch.Tensor:
    """Run on 512 px aligned faces regardless of the restorer's native size (bicubic in,
    area out), for the synthetic set whose ground truth is 512."""
    x = face512
    if r.size != 512:
        x = F.interpolate(x, size=(r.size, r.size), mode="bicubic", align_corners=False).clamp(0, 1)
    y = r(x)
    if r.size != 512:
        y = F.interpolate(y, size=(512, 512), mode="area")
    return y


ALL = ["gfpgan-1.4", "restoreformer", "restoreformer++", "codeformer-w0.5", "codeformer-w0.8",
       "gpen-512", "gpen-1024", "gpen-2048", "pmrf", "osdface"]
