"""PMRF (Ohayon et al., ICLR 2025): posterior-mean rectified flow for blind face restoration.

Uses the upstream arch code (cloned to $PAINITE_BENCH_DIR/src/PMRF) without Lightning. Its
HDiT needs natten neighborhood attention, which has no aarch64/CUDA 13 wheel, so a pure-PyTorch
na2d_qk/na2d_av (same window clamping at borders as natten, dilation 1) stands in for it.
The flow's source noise comes from a fixed-seed generator, so output is deterministic.
"""
import os
import sys
import types

import torch

from restorers import FACE_WEIGHTS, SRC_DIR, Restorer

STEPS = 25  # upstream inference default
NOISE_STD = 0.1  # mmse_noise_std from the checkpoint config


def _window_index(n, k, device):
    start = (torch.arange(n, device=device) - k // 2).clamp(0, n - k)
    return start[:, None] + torch.arange(k, device=device)  # n x k


def na2d_qk(q, k, kernel_size):
    """q, k: N x heads x H x W x E. Returns N x heads x H x W x K*K."""
    h, w = q.shape[2:4]
    ii = _window_index(h, kernel_size, q.device)[:, None, :, None]  # H 1 K 1
    jj = _window_index(w, kernel_size, q.device)[None, :, None, :]  # 1 W 1 K
    kn = k[:, :, ii, jj]  # N nh H W K K E
    return torch.einsum("nahwe,nahwxye->nahwxy", q, kn).flatten(-2)


def na2d_av(a, v, kernel_size):
    h, w = v.shape[2:4]
    ii = _window_index(h, kernel_size, v.device)[:, None, :, None]
    jj = _window_index(w, kernel_size, v.device)[None, :, None, :]
    vn = v[:, :, ii, jj]
    a = a.unflatten(-1, (kernel_size, kernel_size))
    return torch.einsum("nahwxy,nahwxye->nahwe", a, vn)


def _install_natten():
    nat = types.ModuleType("natten")
    nat.has_fused_na = lambda: False
    nat.functional = types.SimpleNamespace(na2d_qk=na2d_qk, na2d_av=na2d_av)
    sys.modules["natten"] = nat


class PMRF(Restorer):
    def __init__(self, steps=STEPS, seed=0):
        os.environ.setdefault("K_DIFFUSION_USE_COMPILE", "0")
        os.environ.setdefault("K_DIFFUSION_USE_FLASH_2", "0")
        _install_natten()
        sys.path.insert(0, str(SRC_DIR / "PMRF"))
        from safetensors.torch import load_file
        from utils.create_arch import create_arch

        self.flow = create_arch("hdit_ImageNet256Sp4", 0)
        self.mmse = create_arch("swinir_L", 0)
        sd = load_file(FACE_WEIGHTS / "PMRF_blind_face.safetensors")
        self.flow.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")})
        self.mmse.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith("mmse_model.")})
        self.flow.cuda().eval()
        self.mmse.cuda().eval()
        self.steps, self.seed = steps, seed

    @torch.inference_mode()
    def __call__(self, y):
        g = torch.Generator(device="cuda").manual_seed(self.seed)
        x = self.mmse(y).clip(0, 1)
        x = x + torch.randn(x.shape, generator=g, device="cuda") * NOISE_STD
        dt = 1.0 / self.steps
        t = torch.ones(x.shape[0], device="cuda")
        for i in range(self.steps):
            x = x + self.flow(x, t * (i / self.steps)).to(x.dtype) * dt
        return x.clip(0, 1)
