"""Reduced precision vs fp32 for each candidate, on the same tiles (UNBLUR.md, dtype column).

  python dtype_check.py --op unblur [--models ...] [--sets realblurj dpdd shake30-45]

For each model and dtype: PSNR of the low-precision output against the fp32 output
(>= 45 dB is visually identical), max abs difference in 8-bit levels, the fp32 output's
PSNR-Y against GT and the low-precision one's (a broken dtype shows up as a drop), and the
time per image. Writes results/dtype_<op>.csv.
"""
import argparse
import csv
import time

import numpy as np
import torch

import quality
from painite_bench import imageio, metrics
from painite_bench.config import RESULTS_DIR, candidates
from painite_bench.models import Runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="unblur")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--sets", nargs="*", default=["realblurj", "dpdd", "shake30-45"])
    ap.add_argument("--n", type=int, default=2, help="images per set")
    a = ap.parse_args()
    path = RESULTS_DIR / f"dtype_{a.op}.csv"
    rows = []
    if path.exists():
        with open(path) as f:
            rows = list(csv.DictReader(f))
    cands = [c for c in candidates(a.op, a.models) if not c.baseline and not c.precomputed]
    rows = [r for r in rows if r["model"] not in {c.id for c in cands}]
    for c in cands:
        outs = {}
        for dt in ("fp32", "bf16", "fp16"):
            try:
                run = Runner(c, dt)
            except Exception as e:  # noqa: BLE001 - record and move on
                print(c.id, dt, "load failed", e)
                continue
            res = []
            for st in a.sets:
                for n, (name, src, gtp) in enumerate(quality.pairs(f"{a.op}/{st}")):
                    if n >= a.n:
                        break
                    inp, gt = imageio.read(src), imageio.read(gtp)
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    try:
                        out = quality.to_np(quality.tiled(run, quality.to_t(inp), 512, 64))
                    except Exception as e:  # noqa: BLE001
                        print(c.id, dt, "failed", repr(e)[:200])
                        out = None
                    torch.cuda.synchronize()
                    res.append((f"{st}/{name}", out, gt, time.perf_counter() - t0))
            outs[dt] = res
            del run
            torch.cuda.empty_cache()
        ref = outs.get("fp32")
        for dt, res in outs.items():
            for i, (key, out, gt, sec) in enumerate(res):
                r = dict(model=c.id, dtype=dt, image=key, seconds=round(sec, 3))
                if out is None or not np.isfinite(out).all():
                    r.update(vs_fp32_psnr="nan", max_diff="nan", psnr_gt="nan")
                else:
                    o, g = quality.align(out, gt) if key.startswith("realblurj") else (out, gt)
                    r["psnr_gt"] = round(metrics.psnr_y(o, g), 3)
                    if ref and ref[i][1] is not None:
                        d = out - ref[i][1]
                        r["vs_fp32_psnr"] = round(10 * np.log10(1 / max(float((d ** 2).mean()), 1e-12)), 2)
                        r["max_diff"] = round(float(np.abs(d).max() * 255), 1)
                rows.append(r)
            sel = [r for r in rows if r["model"] == c.id and r["dtype"] == dt]
            f = lambda k: np.mean([float(r[k]) for r in sel if r.get(k) not in (None, "", "nan")] or [np.nan])
            print(f"{c.id:36s} {dt}  vs fp32 {f('vs_fp32_psnr'):6.2f} dB  psnr_gt {f('psnr_gt'):.2f}  "
                  f"{f('seconds'):.2f}s", flush=True)
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
