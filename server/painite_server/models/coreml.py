"""Core ML acceleration on Apple Silicon Macs.

On a Mac, PyTorch runs the models on the GPU through MPS. Core ML runs the same
networks 1.25x to 2.7x faster (bench/mac/README.md): the ESRGAN models
(sharpen, upscale2x) on the Neural Engine, the others on the GPU. Each model's
network is converted locally the first time it is needed, with coremltools from
the model's own weights, then compiled and kept next to the weights
(`<model dir>/coreml/`), so later starts load it in about a second. Nothing is
downloaded and no converted weights are shipped.

A Core ML model has a fixed input size. The tiled models are converted for the
server's standard window (tile + 2 x context, 640 x 640 by default), which every
tile of an image gets unless the image itself is smaller; such small images run
on PyTorch. Every tile of an image therefore goes through the same engine, and
a preview tile stays identical to the same tile in an export. Face restoration
runs at its fixed face size (1024).

PAINITE_COREML picks it: `auto` (the default) uses Core ML on a Mac when
coremltools is installed and PyTorch would use MPS; `off` keeps everything on
PyTorch. A model's manifest can turn it off or change the choice:

    "config": {..., "coreml": {"compute_units": "gpu", "precision": "fp16"}}
    "config": {..., "coreml": false}

compute_units: neural_engine | gpu | all | cpu; precision: fp16 | fp32.
If a conversion fails, the model logs a warning and stays on PyTorch.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import platform
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("painite.models")

# Bump when the conversion changes in a way that changes the converted model.
CONVERTER_VERSION = 1

# Fastest measured choice per architecture on an M2 Pro (bench/mac/README.md).
# GPEN produces NaNs in fp16 on every Apple engine, so it converts in fp32.
POLICY: dict[str, tuple[str, str]] = {
    "ESRGAN": ("neural_engine", "fp16"),
    "SCUNet": ("gpu", "fp16"),
    "stripformer": ("gpu", "fp16"),
    "lakdnet": ("gpu", "fp16"),
    "GPEN": ("gpu", "fp32"),
}
_UNITS = {"neural_engine": "CPU_AND_NE", "gpu": "CPU_AND_GPU", "all": "ALL", "cpu": "CPU_ONLY"}
_PRECISIONS = ("fp16", "fp32")
MODES = ("auto", "off")


def requested() -> str:
    want = (os.environ.get("PAINITE_COREML") or "auto").strip().lower()
    if want not in MODES:
        log.warning("PAINITE_COREML=%r is not one of %s; using auto", want, MODES)
        return "auto"
    return want


def available() -> bool:
    """Core ML can be used here: an Apple Silicon Mac with coremltools installed."""
    if requested() == "off" or sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import coremltools  # noqa: F401, PLC0415 - optional, installed by install-mac.sh
    except ImportError:
        return False
    return True


def validate_config(value: Any) -> None:
    if value is None or isinstance(value, bool):
        return
    if not isinstance(value, dict):
        raise ValueError("config.coreml must be true, false or an object")
    units = value.get("compute_units")
    if units is not None and units not in _UNITS:
        raise ValueError(f"config.coreml.compute_units must be one of {sorted(_UNITS)}")
    precision = value.get("precision")
    if precision is not None and precision not in _PRECISIONS:
        raise ValueError(f"config.coreml.precision must be one of {_PRECISIONS}")


def choose(arch: str, config: dict[str, Any], device: str) -> tuple[str, str] | None:
    """(compute_units, precision) to run `arch` with, or None to stay on PyTorch.
    Only replaces MPS: a CUDA or CPU server never uses Core ML."""
    value = config.get("coreml")
    if value is False or device != "mps" or not available():
        return None
    units, precision = POLICY.get(arch, (None, None))
    if isinstance(value, dict):
        units = value.get("compute_units", units)
        precision = value.get("precision", precision)
    if units is None:
        return None  # an architecture we have not measured: keep PyTorch unless the manifest asks
    return units, precision or "fp16"


# Conversion ------------------------------------------------------------------

@contextlib.contextmanager
def _conversion_patches():
    """Workarounds needed only while tracing and converting. None of them
    changes what the network computes."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: N812, PLC0415
    from coremltools.converters.mil import Builder as mb  # noqa: PLC0415
    from coremltools.converters.mil.frontend.torch import ops  # noqa: PLC0415

    undo: list[Callable[[], None]] = []

    # coremltools 9 casts 1-element arrays with int(); NumPy >= 2.5 refuses that.
    orig_cast = ops._cast

    def _cast(context, node, dtype, dtype_name):
        x = ops._get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray) and x.val.size == 1:
            context.add(mb.const(val=dtype(x.val.item()), name=node.name), node.name)
            return
        orig_cast(context, node, dtype, dtype_name)

    ops._cast = _cast
    undo.append(lambda: setattr(ops, "_cast", orig_cast))

    # SCUNet builds its shifted-window attention mask with boolean slice
    # assignment, which Core ML lacks: build it eagerly, trace it as a constant.
    try:
        from spandrel.architectures.SCUNet.__arch import SCUNet as scunet  # noqa: PLC0415

        orig_mask = scunet.WMSA.generate_mask

        def generate_mask(self, h, w, p, shift):
            return torch.from_numpy(orig_mask(self, h, w, p, shift).cpu().numpy().copy())

        scunet.WMSA.generate_mask = generate_mask
        undo.append(lambda: setattr(scunet.WMSA, "generate_mask", orig_mask))
    except ImportError:
        pass

    # GPEN's upfirdn2d flips its (symmetric) blur kernel, which Core ML folds
    # into a negative-stride constant, and upsamples through a rank-6 view
    # (Core ML allows rank 5). Same result: no flip, and zero insertion by a
    # stride-`up` transposed 1x1 convolution.
    from . import gpen_arch  # noqa: PLC0415

    orig_up = gpen_arch.upfirdn2d

    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0), device="cpu"):  # noqa: A002
        if not torch.equal(kernel, torch.flip(kernel, [0, 1])):
            return orig_up(input, kernel, up, down, pad, device)
        b, c, h, w = input.shape
        kh, kw = kernel.shape
        x = input
        if up > 1:
            x = F.conv_transpose2d(x, torch.ones(c, 1, 1, 1, dtype=x.dtype), stride=up, groups=c)
            x = F.pad(x, [0, up - 1, 0, up - 1])
        p0, p1 = pad
        x = F.pad(x, [max(p0, 0), max(p1, 0), max(p0, 0), max(p1, 0)])
        x = x[:, :, max(-p0, 0): x.shape[2] - max(-p1, 0), max(-p0, 0): x.shape[3] - max(-p1, 0)]
        k = kernel.to(x.dtype).view(1, 1, kh, kw).repeat(c, 1, 1, 1)
        x = F.conv2d(x, k, groups=c)
        return x[:, :, ::down, ::down]

    gpen_arch.upfirdn2d = upfirdn2d
    undo.append(lambda: setattr(gpen_arch, "upfirdn2d", orig_up))
    try:
        yield
    finally:
        for fn in reversed(undo):
            fn()


_convert_lock = threading.Lock()  # one conversion at a time: each takes GBs of RAM


def convert(module, size: int, precision: str, dest: Path) -> None:
    """Trace `module` (fp32, CPU, NCHW [0,1] -> NCHW) at 1x3xsize x size, convert
    it to an ML program and compile it to `dest` (.mlmodelc)."""
    import coremltools as ct  # noqa: PLC0415
    import torch  # noqa: PLC0415

    with _convert_lock, _conversion_patches():
        x = torch.rand(1, 3, size, size)
        with torch.no_grad():
            traced = torch.jit.trace(module.eval(), x, check_trace=False)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=x.shape)],
            outputs=[ct.TensorType(name="y")],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS14,
            # Don't load it here: loading with the default compute units makes
            # macOS compile it for the Neural Engine too, which takes minutes
            # for attention models that never run there.
            skip_model_load=True,
        )
        tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            pkg = tmp / "model.mlpackage"
            mlmodel.save(str(pkg))
            compiled = Path(ct.models.utils.compile_model(str(pkg), str(tmp / "model.mlmodelc")))
            shutil.rmtree(dest, ignore_errors=True)
            compiled.rename(dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _cache_name(model_id: str, size: int, units: str, precision: str) -> str:
    import coremltools as ct  # noqa: PLC0415

    mac = platform.mac_ver()[0].split(".")[0] or "0"
    key = f"{model_id}|{size}|{precision}|c{CONVERTER_VERSION}|ct{ct.__version__}|macos{mac}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return f"{size}-{precision}-{digest}.mlmodelc"


class Runner:
    """Runs one network through Core ML at the input sizes it was prepared for.

    `build` returns the network as an fp32 CPU torch module mapping a 1x3xHxW
    [0, 1] batch to its output; it is called only when a size has to be
    converted, so the PyTorch weights are not kept in memory for Core ML."""

    def __init__(self, model_id: str, build: Callable[[], Any], cache_dir: Path, units: str, precision: str):
        self.model_id = model_id
        self.build = build
        self.cache_dir = cache_dir
        self.units = units
        self.precision = precision
        self.sizes: set[int] = set()
        self.failed = False
        self._models: dict[int, Any] = {}
        self._lock = threading.RLock()
        self._busy = False  # converting or loading
        self.detail = ""

    def describe(self) -> str:
        return f"Core ML, {self.units.replace('_', ' ')}, {self.precision}"

    @property
    def label(self) -> str:
        return f"coreml-{self.units.replace('_', '-')}-{self.precision}"

    @property
    def backend(self) -> str:
        """What runs the standard window now (part of the tile cache key, like
        models/accel.py): Core ML once its model is loaded, PyTorch before and
        after a failure."""
        return self.label if self._models and not self.failed else "pytorch"

    def status(self) -> dict:
        """For /health, in the shape of accel.Accelerator.status()."""
        if self.failed:
            state = "failed"
        elif self._models:
            state = "active"
        elif self._busy:
            state = "building"
        else:
            state = "idle"
        return {"backend": self.backend, "target": self.label, "state": state, "detail": self.detail,
                "candidates": [self.label], "skipped": [self.label] if self.failed else []}

    def handles(self, shape: tuple[int, ...]) -> bool:
        """True when Core ML runs a 1x3xHxW input of this shape."""
        return (not self.failed and len(shape) == 4 and shape[1] == 3 and shape[2] == shape[3]
                and shape[2] in self.sizes)

    def prepare(self, size: int) -> bool:
        """Make `size` available (convert and compile it if not cached). Returns
        False, and leaves the model on PyTorch, if Core ML cannot run it."""
        with self._lock:
            self.sizes.add(size)
            if self.failed:
                return False
            if size in self._models:
                return True
            self._busy = True
            try:
                self._models[size] = self._load(size)
                return True
            except Exception as e:  # noqa: BLE001 - any failure: PyTorch still works
                log.exception("%s: Core ML conversion failed; using PyTorch (MPS)", self.model_id)
                self.failed = True
                self.detail = f"{type(e).__name__}: {e}"[:300]
                return False
            finally:
                self._busy = False

    def _load(self, size: int):
        import coremltools as ct  # noqa: PLC0415

        path = self.cache_dir / _cache_name(self.model_id, size, self.units, self.precision)
        if not path.is_dir():
            log.info("%s: converting to Core ML at %d px (once; cached in %s)", self.model_id, size, self.cache_dir)
            t0 = time.perf_counter()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            module = self.build()
            convert(module, size, self.precision, path)
            del module
            log.info("%s: converted in %.0f s", self.model_id, time.perf_counter() - t0)
        t0 = time.perf_counter()
        m = ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, _UNITS[self.units]))
        # The first prediction specializes the model for the device; do it now.
        m.predict({"x": np.zeros((1, 3, size, size), np.float32)})
        log.info("%s: Core ML model ready (%s, %d px) in %.1f s", self.model_id, self.describe(), size,
                 time.perf_counter() - t0)
        return m

    def run(self, x: np.ndarray) -> np.ndarray | None:
        """x: float32 1x3xSxS in [0, 1]. Returns the network's output (float32
        NCHW), or None when Core ML does not handle this input (run it on PyTorch)."""
        if not self.handles(x.shape):
            return None
        size = x.shape[2]
        with self._lock:
            if size not in self._models and not self.prepare(size):
                return None
            y = self._models[size].predict({"x": np.ascontiguousarray(x, dtype=np.float32)})["y"]
        return np.asarray(y, dtype=np.float32)
