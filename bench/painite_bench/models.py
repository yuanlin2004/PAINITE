import cv2
import numpy as np
import spandrel
import spandrel_extra_arches
import torch
from torchvision.transforms.functional import gaussian_blur

from .config import Candidate

spandrel_extra_arches.install()


class Runner:
    """Callable (B,3,h,w) float [0,1] -> (B,3,h*s,w*s) on the GPU."""

    def __init__(self, cand: Candidate, dtype: str = "auto", device: str = "cuda"):
        self.cand = cand
        self.device = device
        self.scale = cand.scale
        if cand.baseline:
            self.dtype = torch.float32
            self.model = None
            return
        d = spandrel.ModelLoader().load_from_file(str(cand.path))
        assert d.scale == cand.model_scale, (cand.id, d.scale)
        if dtype == "auto":
            dtype = "fp16" if d.supports_half else "bf16" if d.supports_bfloat16 else "fp32"
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        d.to(device).eval()
        d.model.to(self.dtype)
        self.desc = d
        self.model = d.model

    @property
    def dtype_name(self) -> str:
        return {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}[self.dtype]

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return baseline(self.cand.baseline, x)
        # autocast covers ops that build fp32 tensors internally (SwinIR's attention mask)
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
            y = self.desc(x.to(self.dtype)).float()  # spandrel pads to the arch's size requirements
        if self.cand.via:
            y = torch.nn.functional.avg_pool2d(y, 2)  # area downscale back to the input size
        return y.clamp_(0, 1)


def _to_np(x: torch.Tensor) -> list[np.ndarray]:
    return [(t.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8) for t in x]


def baseline(name: str, x: torch.Tensor) -> torch.Tensor:
    outs = []
    for im in _to_np(x):
        if name == "nlmeans":
            o = cv2.fastNlMeansDenoisingColored(im, None, 7, 7, 7, 21).astype(np.float32) / 255
        elif name == "unsharp":
            f = im.astype(np.float32) / 255
            o = np.clip(f + 0.8 * (f - cv2.GaussianBlur(f, (0, 0), 1.2)), 0, 1)
        elif name == "lanczos":
            o = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255
        else:
            raise ValueError(name)
        outs.append(torch.from_numpy(np.ascontiguousarray(o)).permute(2, 0, 1))
    return torch.stack(outs).to(x.device)


def color_anchor(out: torch.Tensor, inp: torch.Tensor, sigma: float) -> torch.Tensor:
    """Replace the output's low frequencies with the input's: keeps the model's detail but
    pins overall tone and color to the source, removing the GAN models' small color and
    brightness drift. Works on the full image after tiling. inp is resized to out's size."""
    if inp.shape[-2:] != out.shape[-2:]:
        inp = torch.nn.functional.interpolate(inp, size=out.shape[-2:], mode="bicubic", align_corners=False)
    k = int(sigma * 3) * 2 + 1
    blur = lambda t: gaussian_blur(t, [k, k], [sigma, sigma])
    return (out - blur(out) + blur(inp)).clamp(0, 1)
