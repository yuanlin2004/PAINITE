"""Runs PyTorch image-restoration models through spandrel (which knows the
SCUNet, ESRGAN/RRDBNet, ... architectures from their weights), plus the
bake-off's post-processing steps (bench/BAKEOFF.md).

Manifest config:

    "config": {
      "weights": "scunet_color_real_gan.pth",   # file from the manifest's files
      "model_scale": 1,                          # the network's own scale
      "dtype": "fp16",                           # fp16 | bf16 | fp32
      "context": 64,                             # px of real image context around each tile
      "postprocess": [{"type": "area_downscale", "factor": 2},
                      {"type": "color_anchor", "sigma": 4}]
    }

`run` takes one tile window (tile plus context) and returns the processed
window. Post-processing runs per window, before the pipeline crops the core,
so a preview tile is identical to the same tile in a full export. The color
anchor's blur radius (3σ output px) stays inside the context, so per-window
anchoring matches anchoring the whole image.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import Model, devices

log = logging.getLogger("painite.models")

_DTYPES = ("fp16", "bf16", "fp32")
_load_lock = threading.Lock()


def _torch():
    import torch  # noqa: PLC0415 - optional dependency, present in the server image

    return torch


def configure_determinism() -> None:
    """Same input, same output: previews must match exports bit for bit."""
    torch = _torch()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def area_downscale(x, factor: int):
    torch = _torch()
    return torch.nn.functional.avg_pool2d(x, factor)


def color_anchor(out, inp, sigma: float):
    """Replace the output's low frequencies with the input's:
    out - blur(out) + blur(inp). Keeps the model's detail, pins tone and color
    to the source. `inp` is resized to `out`'s size first (bicubic)."""
    torch = _torch()
    from torchvision.transforms.functional import gaussian_blur  # noqa: PLC0415

    if inp.shape[-2:] != out.shape[-2:]:
        inp = torch.nn.functional.interpolate(inp, size=out.shape[-2:], mode="bicubic", align_corners=False)
    k = int(sigma * 3) * 2 + 1
    return (out - gaussian_blur(out, [k, k], [sigma, sigma]) + gaussian_blur(inp, [k, k], [sigma, sigma])).clamp(0, 1)


def validate_config(config: dict[str, Any], scale: int) -> None:
    if not config.get("weights"):
        raise ValueError("config.weights is required")
    if config.get("dtype", "fp16") not in _DTYPES:
        raise ValueError(f"config.dtype must be one of {_DTYPES}")
    context = int(config.get("context", 64))
    effective = int(config.get("model_scale", scale))
    for step in config.get("postprocess", []):
        kind = step.get("type")
        if kind == "area_downscale":
            effective //= int(step["factor"])
        elif kind == "color_anchor":
            # Blur radius in output px must stay inside the tile's context.
            if 3 * float(step["sigma"]) > context * effective:
                raise ValueError("color_anchor blur radius exceeds the tile context")
        else:
            raise ValueError(f"unknown postprocess step {kind!r}")
    if effective != scale:
        raise ValueError(f"model_scale and postprocess give {effective}x, manifest says {scale}x")


@dataclass
class SpandrelModel(Model):
    weights_path: Path = field(default=Path())
    config: dict[str, Any] = field(default_factory=dict)
    device: str = "auto"  # see devices.py
    _net: Any = field(default=None, repr=False)
    _dtype: Any = field(default=None, repr=False)

    def load(self) -> None:
        """Load weights onto the device. Called lazily by the worker thread."""
        if self._net is not None:
            return
        with _load_lock:
            if self._net is not None:
                return
            torch = _torch()
            import spandrel  # noqa: PLC0415

            self.device = devices.resolve(self.device)
            desc = spandrel.ModelLoader().load_from_file(str(self.weights_path))
            want = int(self.config.get("model_scale", self.scale))
            if desc.scale != want:
                raise ValueError(f"{self.id}: weights are {desc.scale}x, config says {want}x")
            dtype = self.config.get("dtype", "fp16")
            if dtype == "fp16" and not desc.supports_half:
                dtype = "bf16" if desc.supports_bfloat16 else "fp32"
            if self.device == "cpu":
                dtype = "fp32"
            self._dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
            desc.to(self.device).eval()
            desc.model.to(self._dtype)
            configure_determinism()
            self._net = desc
            log.info("loaded %s (%s, %s, %s)", self.id, desc.architecture.name, self.device, dtype)

    def run(self, tile: np.ndarray) -> np.ndarray:
        torch = _torch()
        self.load()
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))[None].to(self.device)
            use_autocast = self._dtype != torch.float32 and self.device == "cuda"
            with torch.autocast(self.device, dtype=self._dtype, enabled=use_autocast):
                y = self._net(x.to(self._dtype)).float()
            y = y.clamp_(0, 1)
            for step in self.config.get("postprocess", []):
                if step["type"] == "area_downscale":
                    y = area_downscale(y, int(step["factor"]))
                elif step["type"] == "color_anchor":
                    y = color_anchor(y, x.float(), float(step["sigma"]))
            return y[0].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)


def make(manifest, path: Path) -> SpandrelModel:
    config = dict(manifest.config)
    validate_config(config, manifest.scale)
    return SpandrelModel(
        id=manifest.id,
        operation=manifest.operation,
        version=manifest.version,
        scale=manifest.scale,
        receptive_padding=int(config.get("context", 64)),
        weights_path=path / config["weights"],
        config=config,
    )
