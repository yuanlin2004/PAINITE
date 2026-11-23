"""Which torch device the models run on.

PAINITE_DEVICE picks it: `auto` (the default) takes an NVIDIA GPU (CUDA), then
an Apple Silicon GPU (MPS), then the CPU. `cuda`, `mps` or `cpu` force one; a
forced GPU that is missing falls back to the CPU with a warning.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess

log = logging.getLogger("painite.models")

DEVICES = ("auto", "cuda", "mps", "cpu")


def _mps_available(torch) -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


def requested() -> str:
    want = (os.environ.get("PAINITE_DEVICE") or "auto").strip().lower()
    if want not in DEVICES:
        log.warning("PAINITE_DEVICE=%r is not one of %s; using auto", want, DEVICES)
        return "auto"
    return want


def resolve(want: str | None = None) -> str:
    """The device to use for `want` (default: PAINITE_DEVICE)."""
    import torch  # noqa: PLC0415 - optional dependency

    want = (want or requested()).lower()
    if want == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available(torch):
            return "mps"
        log.warning("no GPU found (CUDA or Apple MPS); models run on the CPU (slow)")
        return "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA is not available; models run on the CPU (slow)")
        return "cpu"
    if want == "mps" and not _mps_available(torch):
        log.warning("Apple MPS is not available; models run on the CPU (slow)")
        return "cpu"
    return want


def _mac_chip() -> str:
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                             timeout=2, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    return out or f"Apple {platform.machine()}"


def status() -> dict:
    """For /health: is there a GPU, and which one."""
    try:
        import torch  # noqa: PLC0415

        dev = resolve()
        if dev == "cuda":
            return {"available": True, "name": torch.cuda.get_device_name(0)}
        if dev == "mps":
            return {"available": True, "name": f"{_mac_chip()} (MPS)"}
    except Exception:  # noqa: BLE001 - no torch, or a broken driver: report no GPU
        pass
    return {"available": False, "name": None}
