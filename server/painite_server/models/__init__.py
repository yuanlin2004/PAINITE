"""Model registry. v1 ships placeholder models only; real models arrive from
the M0 bake-off as adapters with the same interface and a manifest under
/var/lib/painite/models/<op>/<name>@<version>/."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .manifest import Manifest

log = logging.getLogger("painite.models")


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
    "unblur": {"strength": {"min": 0.0, "max": 1.0, "default": 1.0}},
    "upscale2x": {},
}


# Adapters turn an installed manifest into a runnable Model. Real adapters
# (NAFNet, Real-ESRGAN, ...) register here once the bake-off picks models.
Adapter = Callable[["Manifest", Path], Model]
ADAPTERS: dict[str, Adapter] = {
    "identity": lambda m, d: IdentityModel(id=m.id, operation=m.operation, version=m.version, placeholder=True),
    "bilinear2x": lambda m, d: BilinearUpscale2x(id=m.id, operation=m.operation, version=m.version, placeholder=True),
    "spandrel": lambda m, d: _spandrel(m, d),
}


def _spandrel(m: "Manifest", d: Path) -> Model:
    from .spandrel_adapter import make  # noqa: PLC0415 - imports torch lazily

    return make(m, d)


def default_registry(data_dir: Path | None = None) -> Registry:
    """Placeholders, then installed models. An installed model becomes its
    operation's default if its manifest says so, or if it is the first
    installed one for that operation."""
    reg = Registry()
    reg.add(IdentityModel(id="denoise/identity@0.1.0", operation="denoise", version="0.1.0", placeholder=True))
    reg.add(IdentityModel(id="sharpen/identity@0.1.0", operation="sharpen", version="0.1.0", placeholder=True))
    reg.add(IdentityModel(id="unblur/identity@0.1.0", operation="unblur", version="0.1.0", placeholder=True))
    reg.add(BilinearUpscale2x(id="upscale2x/bilinear@0.1.0", operation="upscale2x", version="0.1.0",
                              placeholder=True))
    if data_dir is None:
        return reg
    from .manifest import installed  # noqa: PLC0415 - avoid import cycle

    chosen: dict[str, bool] = {}  # operation -> chosen by an explicit default flag
    for manifest, path in installed(data_dir):
        adapter = ADAPTERS.get(manifest.adapter)
        if adapter is None:
            log.warning("model %s needs adapter %r, which this server does not have", manifest.id, manifest.adapter)
            continue
        try:
            model = adapter(manifest, path)
        except Exception:  # noqa: BLE001 - one broken model must not stop the server
            log.exception("could not load model %s", manifest.id)
            continue
        op = manifest.operation
        make_default = (manifest.default and not chosen.get(op)) or op not in chosen
        reg.add(model, default=make_default)
        if make_default:
            chosen[op] = chosen.get(op, False) or manifest.default
        log.info("loaded model %s%s", manifest.id, " (default)" if make_default else "")
    return reg
