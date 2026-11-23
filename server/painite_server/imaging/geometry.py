"""Orientation-aware region mapping and tile-grid snapping.

Regions from clients are in displayed (EXIF-orientation-applied) pixels. The
server works in stored pixels and keeps the orientation tag on output.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


def displayed_size(width: int, height: int, orientation: int) -> tuple[int, int]:
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def _display_to_stored(dx: int, dy: int, w: int, h: int, o: int) -> tuple[int, int]:
    return {
        1: (dx, dy),
        2: (w - 1 - dx, dy),
        3: (w - 1 - dx, h - 1 - dy),
        4: (dx, h - 1 - dy),
        5: (dy, dx),
        6: (dy, h - 1 - dx),
        7: (w - 1 - dy, h - 1 - dx),
        8: (w - 1 - dy, dx),
    }[o]


def display_point_to_stored(x: float, y: float, w: int, h: int, o: int) -> tuple[float, float]:
    """A displayed-pixel point (pixel-center coordinates, may be fractional) in stored pixels."""
    return _display_to_stored(x, y, w, h, o)


def _stored_to_display(sx: int, sy: int, w: int, h: int, o: int) -> tuple[int, int]:
    return {
        1: (sx, sy),
        2: (w - 1 - sx, sy),
        3: (w - 1 - sx, h - 1 - sy),
        4: (sx, h - 1 - sy),
        5: (sy, sx),
        6: (h - 1 - sy, sx),
        7: (h - 1 - sy, w - 1 - sx),
        8: (sy, w - 1 - sx),
    }[o]


def _map_rect(r: Rect, fn, w: int, h: int, o: int) -> Rect:
    x0, y0 = fn(r.x, r.y, w, h, o)
    x1, y1 = fn(r.x + r.w - 1, r.y + r.h - 1, w, h, o)
    return Rect(min(x0, x1), min(y0, y1), abs(x1 - x0) + 1, abs(y1 - y0) + 1)


def display_rect_to_stored(r: Rect, width: int, height: int, orientation: int) -> Rect:
    return _map_rect(r, _display_to_stored, width, height, orientation)


def stored_rect_to_display(r: Rect, width: int, height: int, orientation: int) -> Rect:
    return _map_rect(r, _stored_to_display, width, height, orientation)


def clip_rect(r: Rect, width: int, height: int) -> Rect | None:
    x0, y0 = max(r.x, 0), max(r.y, 0)
    x1, y1 = min(r.x + r.w, width), min(r.y + r.h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return Rect(x0, y0, x1 - x0, y1 - y0)


def snap_to_tiles(r: Rect, tile: int, width: int, height: int, band: int = 0) -> Rect:
    """Grow a stored-pixel rect outward to the full image's tile grid, inset by
    the blend band: edges land on `k*tile + band` (left/top) and `k*tile - band`
    (right/bottom), or on the image edge. A pixel within `band` of a tile edge
    is a cross-fade of both tiles, so an edge exactly on the grid would need the
    whole neighbouring tile; inset edges need only the tiles under the region.
    With band 0 this is a plain outward snap to whole tiles."""
    x0, x1 = _snap_axis(r.x, r.x + r.w, tile, width, band)
    y0, y1 = _snap_axis(r.y, r.y + r.h, tile, height, band)
    return Rect(x0, y0, x1 - x0, y1 - y0)


def _snap_axis(lo: int, hi: int, tile: int, size: int, band: int) -> tuple[int, int]:
    k0 = max((lo - band) // tile, 0)
    k1 = -(-(hi + band) // tile)
    a = 0 if k0 == 0 else k0 * tile + band
    b = size if k1 * tile >= size else k1 * tile - band
    return a, b
