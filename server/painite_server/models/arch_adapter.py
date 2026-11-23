"""Runs restoration networks that spandrel does not know (the unblur picks,
bench/UNBLUR.md).

PAINITE does not ship these networks' code: their licenses (Stripformer:
non-commercial; LaKDNet: AGPL-3.0) differ from PAINITE's. Like the weights,
the upstream source file is one of the model's manifest files, downloaded from
the pinned upstream commit and SHA-256 checked by `models pull`, and imported
from the model's install directory. `server/third_party/arches/` keeps
reference copies for the tests.

Manifest config:

    "config": {
      "arch": "stripformer",          # a key of ARCHES
      "arch_source": "Stripformer.py",  # optional; ARCHES' file name otherwise
      "weights": "stripformer_RealBlur_J.pth",
      "dtype": "fp16",                # fp16 | bf16 | fp32
      "context": 64,                  # px of real image context around each tile
      "pad_multiple": 8               # optional; the arch's own default otherwise
    }

Each window is reflect-padded up to the network's size multiple, run, and
cropped back, so the output has the window's size. No post-processing: the
color anchor would put the blur back (bench/UNBLUR.md).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import devices
from .spandrel_adapter import SpandrelModel, _load_lock, _torch, configure_determinism

log = logging.getLogger("painite.models")


@dataclass(frozen=True)
class Arch:
    source: str  # upstream file that defines the network, installed with the model
    build: Callable[[Any, dict[str, Any]], Any]  # (source module, config) -> the bare network
    pad_multiple: int
    wrap: Callable[[Any], Any] | None = None  # bare network -> module mapping [0,1] RGB to [0,1] RGB
    supports_half: bool = True


def load_source(path: Path):
    """Import a network definition file from a model's install directory."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path.name} is missing from {path.parent}; reinstall the model with "
            f"`painite-server models pull <id> --accept-license` (it downloads only what is missing)")
    name = f"painite_arch_{path.parent.name}_{path.stem}".replace("@", "_").replace(".", "_").replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def build(arch: str, config: dict[str, Any], source_dir: Path):
    """Build the bare network for `arch` from its source file in `source_dir`."""
    spec = ARCHES[arch]
    module = load_source(source_dir / config.get("arch_source", spec.source))
    return spec.build(module, config)


def _stripformer(module, config: dict[str, Any]):
    return module.Stripformer()


def _centered(net):
    """Stripformer works on images shifted to [-0.5, 0.5]."""
    torch = _torch()

    class Centered(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x - 0.5) + 0.5

    return Centered()


def _lakdnet(module, config: dict[str, Any]):
    kw = {"dim": 32, "num_blocks": [8, 12, 12, 16], "mix_kernel_size": [9, 9, 9, 9]}
    kw.update(config.get("arch_kwargs", {}))
    return module.LaKDNet(inp_channels=3, out_channels=3, **kw)


ARCHES: dict[str, Arch] = {
    "stripformer": Arch("Stripformer.py", _stripformer, pad_multiple=8, wrap=_centered),
    "lakdnet": Arch("LaKDNet.py", _lakdnet, pad_multiple=8),
}


def validate_config(config: dict[str, Any], scale: int) -> None:
    if config.get("arch") not in ARCHES:
        raise ValueError(f"config.arch must be one of {sorted(ARCHES)}")
    if not config.get("weights"):
        raise ValueError("config.weights is required")
    if config.get("dtype", "fp16") not in ("fp16", "bf16", "fp32"):
        raise ValueError("config.dtype must be fp16, bf16 or fp32")
    if config.get("postprocess"):
        raise ValueError("torch_arch models take no postprocess steps")
    if scale != 1:
        raise ValueError("torch_arch models are 1x")


def _state_dict(path: Path) -> dict[str, Any]:
    torch = _torch()
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - older checkpoints pickle extra objects; the file's SHA-256 was checked
        sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    return {k.removeprefix("module."): v for k, v in sd.items()}


def pad_to_multiple(x, multiple: int):
    """Reflect-pad the bottom and right of an NCHW tensor to a multiple of
    `multiple` (replicate where the image is too small to reflect)."""
    torch = _torch()
    h, w = x.shape[-2:]
    ph, pw = -h % multiple, -w % multiple
    if not ph and not pw:
        return x
    mode = "reflect" if ph < h and pw < w else "replicate"
    return torch.nn.functional.pad(x, (0, pw, 0, ph), mode=mode)


@dataclass
class ArchModel(SpandrelModel):
    arch: str = ""
    _pad_multiple: int = field(default=8, repr=False)

    def load(self) -> None:
        if self._net is not None:
            return
        with _load_lock:
            if self._net is not None:
                return
            torch = _torch()
            self.device = devices.resolve(self.device)
            spec = ARCHES[self.arch]
            net = build(self.arch, self.config, self.weights_path.parent)
            net.load_state_dict(_state_dict(self.weights_path), strict=True)
            if spec.wrap is not None:
                net = spec.wrap(net)
            dtype = self.config.get("dtype", "fp16")
            if self.device == "cpu" or (dtype != "fp32" and not spec.supports_half):
                dtype = "fp32"
            self._dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
            net.to(self.device, dtype=self._dtype).eval()
            self._pad_multiple = int(self.config.get("pad_multiple", spec.pad_multiple))
            configure_determinism()
            self._net = net
            log.info("loaded %s (%s, %s, %s)", self.id, self.arch, self.device, dtype)

    def run(self, tile: np.ndarray) -> np.ndarray:
        torch = _torch()
        self.load()
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))[None].to(self.device)
            h, w = x.shape[-2:]
            xp = pad_to_multiple(x.to(self._dtype), self._pad_multiple)
            y = self._net(xp)[..., :h, :w].float().clamp_(0, 1)
            return y[0].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)


def make(manifest, path: Path) -> ArchModel:
    config = dict(manifest.config)
    validate_config(config, manifest.scale)
    return ArchModel(
        id=manifest.id,
        operation=manifest.operation,
        version=manifest.version,
        scale=manifest.scale,
        receptive_padding=int(config.get("context", 64)),
        weights_path=path / config["weights"],
        config=config,
        arch=config["arch"],
    )
