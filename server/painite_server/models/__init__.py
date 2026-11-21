"""Model registry. v1 ships placeholder models only; real models arrive from
the M0 bake-off as adapters with the same interface and a manifest under
/var/lib/painite/models/<op>/<name>@<version>/."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Model:
    id: str
    operation: str
    version: str
    scale: int = 1
    placeholder: bool = False
    # Pixels of context each side of a tile the model needs to avoid seams.
    receptive_padding: int = 0

    def run(self, tile: np.ndarray) -> np.ndarray:  # float32 HxWxC in [0, 1]
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "id": self.id,
            "operation": self.operation,
            "version": self.version,
            "scale": self.scale,
            "placeholder": self.placeholder,
        }


@dataclass
class IdentityModel(Model):
    """Returns its input. Lets the whole job flow run without a GPU."""

    def run(self, tile: np.ndarray) -> np.ndarray:
        return tile


@dataclass
class BilinearUpscale2x(Model):
    """2x bilinear resize (half-pixel centers). Stand-in for the real upscaler."""

    scale: int = 2
    receptive_padding: int = 1

    def run(self, tile: np.ndarray) -> np.ndarray:
        h, w = tile.shape[:2]
        out = _resize_axis(tile, 0, h * 2)
        return _resize_axis(out, 1, w * 2)


def _resize_axis(a: np.ndarray, axis: int, n_out: int) -> np.ndarray:
    n_in = a.shape[axis]
    pos = (np.arange(n_out, dtype=np.float32) + 0.5) * (n_in / n_out) - 0.5
    pos = np.clip(pos, 0, n_in - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n_in - 1)
    frac = (pos - lo).astype(np.float32)
    shape = [1] * a.ndim
    shape[axis] = n_out
    frac = frac.reshape(shape)
    return np.take(a, lo, axis=axis) * (1 - frac) + np.take(a, hi, axis=axis) * frac


@dataclass
class Registry:
    models: dict[str, Model] = field(default_factory=dict)
    defaults: dict[str, str] = field(default_factory=dict)

    def add(self, model: Model, default: bool = False) -> None:
        self.models[model.id] = model
        if default or model.operation not in self.defaults:
            self.defaults[model.operation] = model.id

    def get(self, model_id: str) -> Model | None:
        return self.models.get(model_id)


OPERATION_PARAMS = {
    "denoise": {"strength": {"min": 0.0, "max": 1.0, "default": 0.5}},
    "sharpen": {"strength": {"min": 0.0, "max": 1.0, "default": 0.5}},
    "upscale2x": {},
}


def default_registry() -> Registry:
    reg = Registry()
    reg.add(IdentityModel(id="denoise/identity@0.1.0", operation="denoise", version="0.1.0", placeholder=True))
    reg.add(IdentityModel(id="sharpen/identity@0.1.0", operation="sharpen", version="0.1.0", placeholder=True))
    reg.add(BilinearUpscale2x(id="upscale2x/bilinear@0.1.0", operation="upscale2x", version="0.1.0",
                              placeholder=True))
    return reg
