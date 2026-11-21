"""Decode, tile, run the model, encode. One path for previews and full jobs.

Pixels stay in the source's own color space (no conversion to sRGB). The
source ICC profile, EXIF, XMP, IPTC, and orientation tag are copied to the
output; libvips updates EXIF dimension and resolution tags on save.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvips

from ..errors import ApiError
from ..models import Model
from .geometry import Rect

LOADERS = {"jpegload": "jpeg", "jpegload_source": "jpeg", "tiffload": "tiff", "tiffload_source": "tiff"}
FORMAT_EXT = {"jpeg": ".jpg", "tiff": ".tif"}
METADATA_FIELDS = ("icc-profile-data", "exif-data", "xmp-data", "iptc-data", "orientation", "resolution-unit")


class Canceled(Exception):
    pass


@dataclass
class SourceInfo:
    format: str
    width: int
    height: int
    bits: int
    orientation: int
    has_icc: bool

    def as_dict(self) -> dict:
        return {
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "bits": self.bits,
            "orientation": self.orientation,
            "has_icc": self.has_icc,
        }


def _open(path: Path) -> pyvips.Image:
    # access=sequential is not used: tiles read random regions.
    return pyvips.Image.new_from_file(str(path))


def probe(path: Path, max_pixels: int) -> SourceInfo:
    """Read the header and reject unsupported or oversized inputs."""
    try:
        img = _open(path)
        loader = img.get("vips-loader")
    except pyvips.Error as e:
        raise ApiError(415, "unsupported_format", "not a readable JPEG or TIFF image") from e
    fmt = LOADERS.get(loader)
    if fmt is None:
        raise ApiError(415, "unsupported_format", f"unsupported image format ({loader})")
    if img.bands != 3 or img.interpretation not in ("srgb", "rgb16", "rgb"):
        raise ApiError(415, "unsupported_format", "only RGB images without alpha are supported")
    if img.format not in ("uchar", "ushort"):
        raise ApiError(415, "unsupported_format", f"unsupported sample format ({img.format})")
    if img.width * img.height > max_pixels:
        raise ApiError(422, "too_many_pixels", f"image has {img.width * img.height} pixels; limit is {max_pixels}")
    fields = img.get_fields()
    orientation = img.get("orientation") if "orientation" in fields else 1
    return SourceInfo(
        format=fmt,
        width=img.width,
        height=img.height,
        bits=16 if img.format == "ushort" else 8,
        orientation=orientation if 1 <= orientation <= 8 else 1,
        has_icc="icc-profile-data" in fields,
    )


@dataclass
class JobOutput:
    path: Path
    bytes: int
    sha256: str
    format: str
    width: int
    height: int
    timings_ms: dict[str, float]


def run(
    source_path: Path,
    model: Model,
    params: dict,
    region: Rect | None,
    output: dict,
    work_dir: Path,
    results_dir: Path,
    job_id: str,
    tile: int,
    overlap: int,
    max_output_pixels: int,
    on_progress: Callable[[float], None],
    is_canceled: Callable[[], bool],
) -> JobOutput:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    src = _open(source_path)
    maxval = 65535.0 if src.format == "ushort" else 255.0
    area = region or Rect(0, 0, src.width, src.height)
    out_w, out_h = area.w * model.scale, area.h * model.scale
    if out_w * out_h > max_output_pixels:
        raise ApiError(422, "too_many_pixels", f"output would have {out_w * out_h} pixels; limit is {max_output_pixels}")
    # Decode only the area plus the context its edge tiles need.
    pad = max(overlap, model.receptive_padding)
    wx0, wy0 = max(area.x - pad, 0), max(area.y - pad, 0)
    wx1, wy1 = min(area.x + area.w + pad, src.width), min(area.y + area.h + pad, src.height)
    pixels = src.crop(wx0, wy0, wx1 - wx0, wy1 - wy0).numpy().astype(np.float32) / maxval
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    timings["decode"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    local = Rect(area.x - wx0, area.y - wy0, area.w, area.h)
    # Tile grid is anchored at the full image's origin; shift it into the window.
    result = _process_tiles(pixels, local, (wx0, wy0), model, params, tile, pad, on_progress, is_canceled)
    timings["inference"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    fmt = output.get("format") or ("tiff" if src.get("vips-loader").startswith("tiff") else "jpeg")
    out_path = _encode(src, result, model.scale, fmt, int(output.get("jpeg_quality", 92)), work_dir, results_dir, job_id)
    timings["encode"] = (time.perf_counter() - t2) * 1000

    digest = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return JobOutput(
        path=out_path,
        bytes=out_path.stat().st_size,
        sha256=digest.hexdigest(),
        format=fmt,
        width=result.shape[1],
        height=result.shape[0],
        timings_ms=timings,
    )


def _process_tiles(
    pixels: np.ndarray,
    area: Rect,
    origin: tuple[int, int],
    model: Model,
    params: dict,
    tile: int,
    pad: int,
    on_progress: Callable[[float], None],
    is_canceled: Callable[[], bool],
) -> np.ndarray:
    """Run the model tile by tile over `area` (already snapped to the tile grid
    for previews). Each tile is fed with `overlap` pixels of real context from
    the full image on every side and only its core is kept, so a tile's output
    is identical whether it is rendered for a preview or for a full export."""
    height, width = pixels.shape[:2]
    s = model.scale
    strength = params.get("strength")
    out = np.empty((area.h * s, area.w * s, pixels.shape[2]), dtype=np.float32)
    ox0, oy0 = origin
    # Tile starts in window coordinates, aligned to multiples of `tile` in full-image coordinates.
    ys = range(area.y, area.y + area.h, tile)
    xs = range(area.x, area.x + area.w, tile)
    if (area.x + ox0) % tile or (area.y + oy0) % tile:
        raise ValueError("area must start on the tile grid")
    total = len(ys) * len(xs)
    done = 0
    for ty in ys:
        for tx in xs:
            if is_canceled():
                raise Canceled()
            th, tw = min(tile, area.y + area.h - ty), min(tile, area.x + area.w - tx)
            y0, x0 = max(ty - pad, 0), max(tx - pad, 0)
            y1, x1 = min(ty + th + pad, height), min(tx + tw + pad, width)
            window = pixels[y0:y1, x0:x1]
            enhanced = model.run(window)
            if strength is not None and s == 1:
                # DESIGN.md 5.4: out = input + strength * (model(input) - input)
                enhanced = window + float(strength) * (enhanced - window)
            cy, cx = (ty - y0) * s, (tx - x0) * s
            core = enhanced[cy:cy + th * s, cx:cx + tw * s]
            oy, ox = (ty - area.y) * s, (tx - area.x) * s
            out[oy:oy + th * s, ox:ox + tw * s] = core
            done += 1
            on_progress(done / total)
    return out


def _encode(
    src: pyvips.Image,
    result: np.ndarray,
    scale: int,
    fmt: str,
    jpeg_quality: int,
    work_dir: Path,
    results_dir: Path,
    job_id: str,
) -> Path:
    sixteen = fmt == "tiff" and src.format == "ushort"
    maxval = 65535.0 if sixteen else 255.0
    dtype = np.uint16 if sixteen else np.uint8
    arr = np.clip(np.rint(result * maxval), 0, maxval).astype(dtype)
    img = pyvips.Image.new_from_array(arr)
    img = img.copy(
        interpretation="rgb16" if sixteen else "srgb",
        xres=src.xres * scale,
        yres=src.yres * scale,
    )
    fields = src.get_fields()
    for name in METADATA_FIELDS:
        if name in fields:
            img.set_type(src.get_typeof(name), name, src.get(name))

    tmp = work_dir / f"{job_id}.tmp{FORMAT_EXT[fmt]}"
    try:
        if fmt == "jpeg":
            img.jpegsave(str(tmp), Q=jpeg_quality, subsample_mode="off", optimize_coding=True)
        else:
            img.tiffsave(str(tmp), compression="deflate", predictor="horizontal")
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        check = pyvips.Image.new_from_file(str(tmp))
        if (check.width, check.height) != (arr.shape[1], arr.shape[0]):
            raise ApiError(500, "processing_failed", "encoded output failed validation")
        check.avg()  # forces a full decode
        final = results_dir / f"{job_id}{FORMAT_EXT[fmt]}"
        os.replace(tmp, final)
        return final
    finally:
        tmp.unlink(missing_ok=True)
