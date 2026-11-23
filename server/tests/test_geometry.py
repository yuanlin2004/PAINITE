import itertools

import pytest

from painite_server.imaging.geometry import Rect, display_rect_to_stored, displayed_size, snap_to_tiles, \
    stored_rect_to_display


@pytest.mark.parametrize("orientation", range(1, 9))
def test_round_trip(orientation):
    w, h = 40, 25
    dw, dh = displayed_size(w, h, orientation)
    for x, y in itertools.product(range(0, dw - 5, 7), range(0, dh - 4, 5)):
        r = Rect(x, y, 5, 4)
        s = display_rect_to_stored(r, w, h, orientation)
        assert 0 <= s.x and s.x + s.w <= w and 0 <= s.y and s.y + s.h <= h
        assert stored_rect_to_display(s, w, h, orientation) == r


def test_orientation_6_maps_top_left_to_bottom_left():
    # Orientation 6: stored image is rotated 90° CW for display.
    s = display_rect_to_stored(Rect(0, 0, 1, 1), 40, 25, 6)
    assert (s.x, s.y) == (0, 24)


def test_snap():
    assert snap_to_tiles(Rect(70, 30, 40, 50), 64, 200, 150) == Rect(64, 0, 64, 128)
    assert snap_to_tiles(Rect(130, 140, 5, 5), 64, 150, 150) == Rect(128, 128, 22, 22)


def test_snap_inset_by_blend_band():
    # A region inside 2x2 tiles, at least `band` from their outer edges, needs only those tiles.
    assert snap_to_tiles(Rect(544, 544, 960, 960), 512, 6000, 4000, 32) == Rect(544, 544, 960, 960)
    # A region ending on the grid would need the next tile, so the snap moves to its inset edge.
    assert snap_to_tiles(Rect(512, 512, 1024, 1024), 512, 6000, 4000, 32) == Rect(0, 0, 2016, 2016)
    # Image edges are never inset; nothing past them.
    assert snap_to_tiles(Rect(10, 10, 100, 100), 512, 6000, 4000, 32) == Rect(0, 0, 480, 480)
    assert snap_to_tiles(Rect(5900, 3900, 50, 50), 512, 6000, 4000, 32) == Rect(5664, 3616, 336, 384)
    # Idempotent.
    for r in (Rect(100, 700, 1234, 77), Rect(0, 0, 6000, 4000), Rect(5000, 10, 999, 3989)):
        s = snap_to_tiles(r, 512, 6000, 4000, 32)
        assert snap_to_tiles(s, 512, 6000, 4000, 32) == s
