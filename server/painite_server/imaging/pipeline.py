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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvips

from ..errors import ApiError
from ..models import Model
from .geometry import Rect, display_point_to_stored, displayed_size, stored_rect_to_display

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
    img = pyvips.Image.new_from_file(str(path))
    if img.bands == 1 and img.interpretation in ("b-w", "grey16") and img.format in ("uchar", "ushort"):
        # Grayscale (old black-and-white scans): the models want RGB, so copy the
        # one band to three. A gray ICC profile can't describe RGB pixels, so it
        # is dropped and the result is untagged (read as sRGB).
        img = img.bandjoin([img, img]).copy(interpretation="rgb16" if img.format == "ushort" else "srgb")
        if "icc-profile-data" in img.get_fields():
            img = img.copy()
            img.remove("icc-profile-data")
    return img


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
    info: dict = field(default_factory=dict)  # e.g. {"faces": [...]} from a face step


@dataclass
class Step:
    model: Model
    params: dict


def run(
    source_path: Path,
    steps: list[Step],
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
    """Run `steps` in order over `region` (stored pixels, snapped to the tile
    grid) or the whole image, and encode the result.

    Each step tiles its own input on a grid anchored at that input's origin,
    with real context around every tile, so a preview region reproduces the
    same pixels as a full export even across several steps and an upscale."""
    timings: dict[str, float] = {"decode": 0.0, "inference": 0.0}
    src = _open(source_path)
    maxval = 65535.0 if src.format == "ushort" else 255.0
    sizes = [(src.width, src.height)]
    for st in steps:
        w, h = sizes[-1]
        sizes.append((w * st.model.scale, h * st.model.scale))
    total_scale = sizes[-1][0] // src.width
    area = region or Rect(0, 0, src.width, src.height)
    final = Rect(area.x * total_scale, area.y * total_scale, area.w * total_scale, area.h * total_scale)
    if final.w * final.h > max_output_pixels:
        raise ApiError(422, "too_many_pixels",
                       f"output would have {final.w * final.h} pixels; limit is {max_output_pixels}")

    n = len(steps)

    def decode(need: Rect) -> np.ndarray:
        t0 = time.perf_counter()
        px = src.crop(need.x, need.y, need.w, need.h).numpy().astype(np.float32) / maxval
        if px.ndim == 2:
            px = px[:, :, None]
        timings["decode"] += (time.perf_counter() - t0) * 1000
        return px

    def render(k: int, need: Rect) -> tuple[np.ndarray, Rect]:
        """Pixels of step k's output (k=0 is the source) covering `need`, in that
        output's coordinates. Returns the array and the rect it covers (⊇ need)."""
        if k == 0:
            return decode(need), need
        st = steps[k - 1]
        sc = st.model.scale
        in_w, in_h = sizes[k - 1]
        if st.model.whole_image:
            return render_whole(k, st, need, in_w, in_h)
        # Input pixels whose tiles produce `need`, grown by the blend band (a pixel
        # near a tile edge also needs the neighbouring tile) and then to whole
        # tiles of this step's grid. Pixels within `band` of the area's inner
        # edges are only partly blended, but they are never inside `need`.
        band = overlap // 2
        ix0, iy0 = max(need.x // sc - band, 0), max(need.y // sc - band, 0)
        ix1 = min(-(-(need.x + need.w) // sc) + band, in_w)
        iy1 = min(-(-(need.y + need.h) // sc) + band, in_h)
        area = _snap(Rect(ix0, iy0, ix1 - ix0, iy1 - iy0), tile, in_w, in_h)
        pad = max(overlap, st.model.receptive_padding)
        wx0, wx1 = _span(area.x, area.w, in_w, tile, pad)
        wy0, wy1 = _span(area.y, area.h, in_h, tile, pad)
        want = Rect(wx0, wy0, wx1 - wx0, wy1 - wy0)
        inp, got = render(k - 1, want)
        window = inp[want.y - got.y:want.y - got.y + want.h, want.x - got.x:want.x - got.x + want.w]
        local = Rect(area.x - want.x, area.y - want.y, area.w, area.h)
        t1 = time.perf_counter()

        def step_progress(p: float) -> None:
            on_progress((k - 1 + p) / n)

        out = _process_tiles(window, local, (want.x, want.y), (in_w, in_h), st.model, st.params, tile, pad,
                             band, step_progress, is_canceled)
        timings["inference"] += (time.perf_counter() - t1) * 1000
        return out, Rect(area.x * sc, area.y * sc, area.w * sc, area.h * sc)

    def render_whole(k: int, st: Step, need: Rect, in_w: int, in_h: int) -> tuple[np.ndarray, Rect]:
        """A whole-image step (face). Faces are detected once on the source and
        scaled to this step's input; the step needs `need` plus the full box of
        every face touching it, and nothing else."""
        model = st.model
        dets = _detections(src, source_path, model, maxval, timings)
        faces = model.layout(dets, in_w // src.width, in_w, in_h)
        face_report[:] = _face_report(src, dets, model.classify(dets, in_w // src.width, in_w, in_h))
        want = model.input_rect(faces, need)
        inp, got = render(k - 1, want)
        window = inp[want.y - got.y:want.y - got.y + want.h, want.x - got.x:want.x - got.x + want.w]
        t1 = time.perf_counter()

        def step_progress(p: float) -> None:
            on_progress((k - 1 + p) / n)

        out = model.run_area(window, (want.x, want.y), need, faces, st.params.get("strength"), step_progress,
                             is_canceled)
        timings["inference"] += (time.perf_counter() - t1) * 1000
        return out, need

    face_report: list[dict] = []
    pixels, got = render(n, final)
    result = pixels[final.y - got.y:final.y - got.y + final.h, final.x - got.x:final.x - got.x + final.w]

    t2 = time.perf_counter()
    fmt = output.get("format") or ("tiff" if src.get("vips-loader").startswith("tiff") else "jpeg")
    out_path = _encode(src, np.ascontiguousarray(result), total_scale, fmt, int(output.get("jpeg_quality", 92)),
                       work_dir, results_dir, job_id)
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
        info={"faces": face_report} if any(st.model.whole_image for st in steps) else {},
    )


def _face_report(src: pyvips.Image, dets: list, classified: list) -> list[dict]:
    """Detected faces for clients, in displayed (orientation-applied) source
    pixels like preview regions: the detector's face box, what the face step
    did with it, and the side of its aligned crop at that step's scale."""
    fields = src.get_fields()
    orientation = src.get("orientation") if "orientation" in fields else 1
    orientation = orientation if 1 <= orientation <= 8 else 1
    out = []
    for (score, _pts, box), (face, status) in zip(dets, classified, strict=True):
        x0, y0 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
        x1, y1 = min(int(np.ceil(box[2])), src.width), min(int(np.ceil(box[3])), src.height)
        if x1 <= x0 or y1 <= y0:
            continue
        r = stored_rect_to_display(Rect(x0, y0, x1 - x0, y1 - y0), src.width, src.height, orientation)
        out.append({"box": r.as_dict(), "status": status, "size": round(face.size), "score": round(score, 3)})
    out.sort(key=lambda f: (f["box"]["y"], f["box"]["x"]))
    return out


def _box_to_stored(box, width: int, height: int, orientation: int) -> np.ndarray:
    """A displayed-pixel box (x0, y0, x1, y1, pixel edges) in stored pixels."""
    corners = [display_point_to_stored(x, y, width, height, orientation)
               for x in (box[0], box[2] - 1) for y in (box[1], box[3] - 1)]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return np.array([min(xs), min(ys), max(xs) + 1, max(ys) + 1])


def _window(t: int, core: int, size: int, tile: int, pad: int) -> tuple[int, int]:
    """Model input span [lo, hi) in full-image coordinates for the tile whose
    core is [t, t + core) on an axis of length `size`. Every window is
    tile + 2*pad long where the image allows: a tile at the far edge (often a
    short remainder) gets its missing context from further inside the image
    instead of a narrower window. Attention models (Stripformer) change their
    output with window size, and a narrow edge window cost 0.8 dB on
    RealBlur-J (bench/UNBLUR.md). A function of full-image coordinates only,
    so previews and exports see the same windows."""
    length = tile + 2 * pad
    lo = max(min(t - pad, size - length), 0)
    return lo, min(max(lo + length, t + core + pad), size)


def _span(start: int, length: int, size: int, tile: int, pad: int) -> tuple[int, int]:
    """Input span covering the windows of every tile in [start, start + length)."""
    end = start + length
    lo, _ = _window(start, min(tile, end - start), size, tile, pad)
    last = start + ((length - 1) // tile) * tile
    lo_last, hi = _window(last, end - last, size, tile, pad)
    return min(lo, lo_last), hi


def _detections(src: pyvips.Image, source_path: Path, model: Model, maxval: float,
                timings: dict[str, float]) -> list:
    """Faces in the source, in stored pixels, cached by the model per source.
    Detection runs upright: on an orientation-applied downscale of the whole
    image, so it never depends on the requested region."""
    width, height = src.width, src.height
    fields = src.get_fields()
    orientation = src.get("orientation") if "orientation" in fields else 1
    orientation = orientation if 1 <= orientation <= 8 else 1
    disp_w, disp_h = displayed_size(width, height, orientation)
    dw, dh = model.detection_size(disp_w, disp_h)

    def small() -> np.ndarray:
        t0 = time.perf_counter()
        img = src.autorot() if orientation != 1 else src
        if (dw, dh) != (disp_w, disp_h):
            img = img.resize(dw / disp_w, vscale=dh / disp_h)
        px = img.crop(0, 0, dw, dh).numpy().astype(np.float32) / maxval
        timings["decode"] += (time.perf_counter() - t0) * 1000
        return px[:, :, None] if px.ndim == 2 else px

    st = source_path.stat()
    dets = model.detections((str(source_path), st.st_mtime_ns, st.st_size), small, disp_w, disp_h)
    if orientation != 1:
        dets = [(score, np.array([display_point_to_stored(x, y, width, height, orientation) for x, y in pts]),
                 _box_to_stored(box, width, height, orientation)) for score, pts, box in dets]
    return dets


def _snap(r: Rect, tile: int, width: int, height: int) -> Rect:
    x0 = (r.x // tile) * tile
    y0 = (r.y // tile) * tile
    x1 = min(-(-(r.x + r.w) // tile) * tile, width)
    y1 = min(-(-(r.y + r.h) // tile) * tile, height)
    return Rect(x0, y0, x1 - x0, y1 - y0)


def _process_tiles(
    pixels: np.ndarray,
    area: Rect,
    origin: tuple[int, int],
    full_size: tuple[int, int],
    model: Model,
    params: dict,
    tile: int,
    pad: int,
    band: int,
    on_progress: Callable[[float], None],
    is_canceled: Callable[[], bool],
) -> np.ndarray:
    """Run the model tile by tile over `area` (already snapped to the tile grid
    for previews). Each tile is fed with `overlap` pixels of real context from
    the full image on every side (see `_window`) and only its core is kept, so a
    tile's output is identical whether it is rendered for a preview or for a full
    export. `pixels` must cover every tile's window (see `_span`).

    Neighbouring tiles cross-fade linearly over `band` px on each side of their
    shared edge (weights sum to 1), so models whose output depends on the window
    (Stripformer: seams 2.7x the neighbouring pixel steps without it) leave no
    seams. Tiles are accumulated in a fixed order, so a pixel gets bit-identical
    values in a preview and an export as long as all its tiles are in `area`."""
    s = model.scale
    strength = params.get("strength")
    out = np.zeros((area.h * s, area.w * s, pixels.shape[2]), dtype=np.float32)
    wsum = np.zeros((area.h * s, area.w * s, 1), dtype=np.float32)
    ox0, oy0 = origin
    if band > pad:
        raise ValueError("blend band must not exceed the tile context")
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
            # window in full-image coordinates, then in `pixels` coordinates
            y0, y1 = (v - oy0 for v in _window(ty + oy0, th, full_size[1], tile, pad))
            x0, x1 = (v - ox0 for v in _window(tx + ox0, tw, full_size[0], tile, pad))
            if y0 < 0 or x0 < 0 or y1 > pixels.shape[0] or x1 > pixels.shape[1]:
                raise ValueError("pixels do not cover the tile window")
            window = pixels[y0:y1, x0:x1]
            enhanced = model.run(window)
            if strength is not None and s == 1:
                # DESIGN.md 5.4: out = input + strength * (model(input) - input)
                enhanced = window + float(strength) * (enhanced - window)
            # the tile's core grown by `band` (not past the image or the area),
            # with linear ramps across each shared edge; output coordinates
            ry0, ry1, wy = _blend_span(ty + oy0, th, full_size[1], band, area.y + oy0, area.h, s)
            rx0, rx1, wx = _blend_span(tx + ox0, tw, full_size[0], band, area.x + ox0, area.w, s)
            wt = (wy[:, None] * wx[None, :])[:, :, None]
            ey, ex = ry0 - (y0 + oy0) * s, rx0 - (x0 + ox0) * s  # into `enhanced`
            oy, ox = ry0 - (area.y + oy0) * s, rx0 - (area.x + ox0) * s  # into `out`
            hh, ww = ry1 - ry0, rx1 - rx0
            out[oy:oy + hh, ox:ox + ww] += enhanced[ey:ey + hh, ex:ex + ww] * wt
            wsum[oy:oy + hh, ox:ox + ww] += wt
            done += 1
            on_progress(done / total)
    return out / wsum


def _blend_span(t: int, core: int, size: int, band: int, a0: int, alen: int, s: int):
    """For the tile whose core is [t, t + core) (full-image input coordinates) on
    an axis of length `size`: the output span [lo, hi) it contributes to, clipped
    to the area [a0, a0 + alen), and its weights there. Weights ramp from 0 to 1
    across [edge - band, edge + band) at every edge that is not an image edge, so
    two neighbours' weights sum to 1."""
    c0, c1, b = t * s, (t + core) * s, band * s
    lo = c0 if t == 0 else c0 - b
    hi = c1 if t + core == size else c1 + b
    lo, hi = max(lo, a0 * s), min(hi, (a0 + alen) * s)
    x = np.arange(lo, hi, dtype=np.float32) + 0.5
    w = np.ones(hi - lo, dtype=np.float32)
    if b and t > 0:
        w = np.minimum(w, np.clip((x - (c0 - b)) / (2 * b), 0, 1))
    if b and t + core < size:
        w = np.minimum(w, np.clip(((c1 + b) - x) / (2 * b), 0, 1))
    return lo, hi, w


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
