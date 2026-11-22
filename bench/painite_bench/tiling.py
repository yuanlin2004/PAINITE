"""Tiled inference matching docs/DESIGN.md §5: fixed grid anchored at the origin,
overlap between tiles, feathered (linear ramp) blend across the overlap."""
import torch


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _ramp(n: int, overlap: int, lead: bool, trail: bool, device) -> torch.Tensor:
    w = torch.ones(n, device=device)
    if overlap > 0:
        r = (torch.arange(overlap, device=device, dtype=torch.float32) + 0.5) / overlap
        if lead:
            w[:overlap] = r
        if trail:
            w[-overlap:] = r.flip(0)
    return w


@torch.inference_mode()
def run_tiled(fn, img: torch.Tensor, scale: int = 1, tile: int = 512, overlap: int = 32,
              batch: int = 4) -> torch.Tensor:
    """img: (1,3,H,W) float on the compute device. fn maps (B,3,t,t) -> (B,3,t*s,t*s)."""
    _, c, h, w = img.shape
    ys, xs = tile_starts(h, tile, overlap), tile_starts(w, tile, overlap)
    th, tw = min(tile, h), min(tile, w)
    out = torch.zeros((1, c, h * scale, w * scale), device=img.device, dtype=torch.float32)
    wsum = torch.zeros((1, 1, h * scale, w * scale), device=img.device, dtype=torch.float32)
    coords = [(y, x) for y in ys for x in xs]
    so = overlap * scale
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = torch.cat([img[:, :, y:y + th, x:x + tw] for y, x in chunk])
        res = fn(tiles).float()
        for (y, x), r in zip(chunk, res):
            wy = _ramp(th * scale, so, y > 0, y + th < h, img.device)
            wx = _ramp(tw * scale, so, x > 0, x + tw < w, img.device)
            wt = wy[:, None] * wx[None, :]
            ys_, xs_ = y * scale, x * scale
            out[0, :, ys_:ys_ + th * scale, xs_:xs_ + tw * scale] += r * wt
            wsum[0, :, ys_:ys_ + th * scale, xs_:xs_ + tw * scale] += wt
    return out / wsum


def seam_mask(h: int, w: int, scale: int, tile: int, overlap: int, band: int = 2) -> torch.Tensor:
    """Bool mask of output pixels within `band` px of a tile-overlap boundary."""
    m = torch.zeros(h * scale, w * scale, dtype=torch.bool)
    for starts, dim in ((tile_starts(h, tile, overlap), 0), (tile_starts(w, tile, overlap), 1)):
        for s in starts[1:]:
            for edge in (s, s + overlap):
                e = edge * scale
                sl = slice(max(e - band, 0), e + band)
                if dim == 0:
                    m[sl, :] = True
                else:
                    m[:, sl] = True
    return m
