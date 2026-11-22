"""Run candidates over the test set with the production tiling and score them.

  python quality.py --op denoise [--models nafnet-sidd-w64 ...] [--no-save]

Writes results/quality_<op>.csv (one row per model x input) and, unless --no-save,
outputs under $PAINITE_BENCH_DIR/out/<op>/<model>/<set>/ for visual review.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from painite_bench import imageio, metrics
from painite_bench.config import OUT_DIR, RESULTS_DIR, TESTSET_DIR, candidates
from painite_bench.models import Runner, color_anchor
from painite_bench.tiling import run_tiled, seam_mask

SETS = {
    "denoise": ["denoise/iso3200", "denoise/iso12800", "denoise/sidd", "denoise/polyu"],
    # gt as input checks what "sharpen" does to an already sharp photo (halos, crunch).
    "sharpen": ["sharpen/gauss1.2", "sharpen/defocus2", "sharpen/motion5", "gt"],
    "upscale2x": ["upscale2x/clean", "upscale2x/camera"],
}


def pairs(sub: str):
    if sub in ("denoise/sidd", "denoise/polyu"):
        for f in sorted((TESTSET_DIR / sub / "noisy").glob("*.png")):
            yield f"{sub.split('/')[1]}-{f.stem}", f, TESTSET_DIR / sub / "gt" / f.name
    else:
        for f in sorted((TESTSET_DIR / sub).glob("*.png")):
            yield f.stem, f, TESTSET_DIR / "gt" / f.name


def to_t(im):
    return torch.from_numpy(im).permute(2, 0, 1)[None].cuda()


def to_np(t):
    return t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


def seam_score(run: Runner, img: np.ndarray, tile=512, overlap=32):
    """Tiled output vs the same model run on the whole crop in one pass."""
    crop = img[:1024, :1024]
    x = to_t(crop)
    whole = run(x)
    tiled = run_tiled(run, x, run.scale, tile, overlap)
    d = (tiled - whole).abs().mean(1)[0].cpu()
    m = seam_mask(crop.shape[0], crop.shape[1], run.scale, tile, overlap)
    mse = float((d ** 2).mean())
    return dict(seam_psnr=10 * np.log10(1 / max(mse, 1e-12)), seam_max=float(d.max() * 255),
                seam_band=float(d[m].mean() * 255), seam_ratio=float(d[m].mean() / max(d[~m].mean(), 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=SETS)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--sets", nargs="*", help="subset of the op's test sets")
    ap.add_argument("--limit", type=int, default=0, help="images per set (0 = all)")
    a = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"quality_{a.op}.csv"
    rows = []
    if path.exists():  # keep other models' rows, replace the ones being rerun
        with open(path) as f:
            rows = list(csv.DictReader(f))
    sets = a.sets or SETS[a.op]
    cands = candidates(a.op, a.models)
    if a.op != "upscale2x" and not a.models:
        cands = [None] + cands  # score the degraded input itself
    ids = {c.id if c else "input" for c in cands}
    rows = [r for r in rows if not (r["model"] in ids and r["set"] in sets)]
    for c in cands:
        run = Runner(c, a.dtype) if c else None
        mid = c.id if c else "input"
        print(f"== {mid} ({run.dtype_name if run else '-'})", flush=True)
        for sub in sets:
            for n, (name, src, gtp) in enumerate(pairs(sub)):
                if a.limit and n >= a.limit:
                    break
                inp, gt = imageio.read(src), imageio.read(gtp)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                if run:
                    x = to_t(inp)
                    y = run_tiled(run, x, run.scale, overlap=a.overlap)
                    out = to_np(color_anchor(y, x, c.anchor) if c.anchor else y)
                else:
                    out = inp
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                r = dict(model=mid, set=sub, image=name, dtype=run.dtype_name if run else "",
                         seconds=round(dt, 3), **{k: round(v, 5) for k, v in metrics.full_reference(out, gt).items()})
                if run and sub == SETS[a.op][0] and n < 4:
                    r.update({k: round(v, 5) for k, v in seam_score(run, inp).items()})
                rows.append(r)
                if run and not a.no_save:
                    d = OUT_DIR / a.op / mid.split("/")[1] / sub.replace("/", "_")
                    d.mkdir(parents=True, exist_ok=True)
                    imageio.write(d / f"{name}.png", out)
            done = [r for r in rows if r["model"] == mid and r["set"] == sub]
            print(f"  {sub:22s} psnr {np.mean([float(r['psnr']) for r in done]):.2f}  "
                  f"lpips {np.mean([float(r['lpips']) for r in done]):.4f}  "
                  f"s/img {np.mean([float(r['seconds']) for r in done]):.2f}", flush=True)
        del run
        torch.cuda.empty_cache()
        save(path, rows)  # after each model, so a crash keeps finished models


def save(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
