"""Server-side inference latency and peak GPU memory on the Spark.

  python latency.py [--op denoise] [--models ...] [--sizes 12 24 48]

Measures, per model, with production tiling (512 px tiles, 32 px overlap):
  load_s       weights file -> GPU, first call excluded (cold start component)
  batch        tiles per forward, picked by a quick sweep
  preview p50/p95  a 1024x1024 source-pixel preview region: every full-image grid
                   tile touching it (3x3 tiles for the design's example region)
  full_s       whole image at 12/24/48 MP (input MP; upscale outputs 4x the pixels)
  peak_gb      torch.cuda.max_memory_allocated during the full render
Decode/encode, queueing and network transfer are not included; the server records
those separately (docs/DESIGN.md §9). Results go to results/latency.csv.
"""
import argparse
import csv
import statistics
import time

import cv2
import numpy as np
import torch

from painite_bench import imageio
from painite_bench.config import RESULTS_DIR, TESTSET_DIR, candidates
from painite_bench.models import Runner
from painite_bench.tiling import run_tiled

SIZES = {12: (3000, 4000), 24: (4000, 6000), 48: (5664, 8496)}
TILE, OVERLAP = 512, 32
BUDGET_S = 900  # skip a full render projected to take longer; extrapolate instead


def grid_span(start, length):
    """Slice covering the global tile grid's tiles that intersect [start, start+length)."""
    stride = TILE - OVERLAP
    first = max(0, (start - TILE) // stride + 1)
    last = (start + length - 1) // stride
    return slice(first * stride, last * stride + TILE)


def sync_time(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, r


def pick_batch(run, src):
    x = src[:, :, :1504, :1504]  # 3x3 tiles
    best = None
    for b in (1, 2, 4, 8):
        try:
            run_tiled(run, x, run.scale, TILE, OVERLAP, b)
            t, _ = sync_time(lambda: run_tiled(run, x, run.scale, TILE, OVERLAP, b))
        except torch.OutOfMemoryError:
            break
        if best is None or t < best[1] * 0.95:
            best = (b, t)
    torch.cuda.empty_cache()
    return best[0], best[1] / 9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--sizes", nargs="*", type=int, default=[12, 24, 48])
    ap.add_argument("--preview-runs", type=int, default=10)
    a = ap.parse_args()
    base = imageio.read(TESTSET_DIR / "gt/foliage-0840.png")
    path = RESULTS_DIR / "latency.csv"
    rows = []
    if path.exists():
        with open(path) as f:
            rows = list(csv.DictReader(f))
    # "via" sharpen candidates cost the same as their 2x model plus a cheap pool
    cands = [c for c in candidates(a.op, a.models) if not c.baseline and not c.via]
    rows = [r for r in rows if r["model"] not in {c.id for c in cands} or r["dtype"] != a.dtype]
    for c in cands:
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        run = Runner(c, a.dtype)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t0
        src = torch.from_numpy(cv2.resize(base, SIZES[12][::-1], interpolation=cv2.INTER_CUBIC)
                               ).permute(2, 0, 1)[None].cuda().clamp(0, 1)
        batch, per_tile = pick_batch(run, src)
        # Preview of the design's example region (x=1200, y=800, 1024x1024): run every
        # full-image grid tile that touches it, so the pixels match the export exactly.
        pv = src[:, :, grid_span(800, 1024), grid_span(1200, 1024)]
        runs = a.preview_runs if per_tile * 9 < 20 else 3  # far too slow for previews anyway
        times = [sync_time(lambda: run_tiled(run, pv, run.scale, TILE, OVERLAP, batch))[0]
                 for _ in range(runs)]
        q = statistics.quantiles(times, n=20) if len(times) >= 5 else [max(times)] * 19
        row = dict(model=c.id, dtype=run.dtype_name, batch=batch, load_s=round(load_s, 2),
                   preview_p50=round(statistics.median(times), 3), preview_p95=round(q[18], 3))
        print(f"== {c.id} {run.dtype_name} batch={batch} load={load_s:.1f}s "
              f"preview p50={row['preview_p50']}s p95={row['preview_p95']}s", flush=True)
        for mp in a.sizes:
            h, w = SIZES[mp]
            x = torch.from_numpy(cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1)[None]
            ntiles = (len(range(0, h - TILE, TILE - OVERLAP)) + 1) * (len(range(0, w - TILE, TILE - OVERLAP)) + 1)
            est = per_tile * ntiles
            r = dict(row, mp=mp, tiles=ntiles)
            if est > BUDGET_S:
                r.update(full_s=round(est, 1), measured="extrapolated", peak_gb="")
            else:
                torch.cuda.reset_peak_memory_stats()
                xg = x.cuda().clamp(0, 1)
                t, out = sync_time(lambda: run_tiled(run, xg, run.scale, TILE, OVERLAP, batch).cpu())
                r.update(full_s=round(t, 2), measured="yes",
                         peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
                del xg, out
            r["mp_per_s"] = round(mp / float(r["full_s"]), 3)
            print(f"   {mp} MP: {r['full_s']} s ({r['measured']}) peak {r['peak_gb']} GB", flush=True)
            rows.append(r)
            torch.cuda.empty_cache()
        del run, src
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
