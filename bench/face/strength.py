"""Strength sweep on the synthetic set, from the outputs face/quality.py saved:
out = input + s * (restored - input), s in {0.25, 0.5, 0.75, 1}.

  python face/strength.py --models gpen-1024+anchor8 codeformer-w0.8

Writes results/face/strength_synthetic.csv and prints means per model, level and strength.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from painite_bench import metrics  # noqa: E402
from painite_bench.config import OUT_DIR, RESULTS_DIR, TESTSET_DIR  # noqa: E402
from quality import landmarks  # noqa: E402  (before restorers puts bench/ ahead on sys.path)
from identity import ArcFace  # noqa: E402

SYN = TESTSET_DIR / "face/synthetic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["gpen-1024+anchor8", "codeformer-w0.8"])
    ap.add_argument("-n", type=int, default=100)
    a = ap.parse_args()
    names = sorted(p.name for p in (SYN / "gt").glob("*.png"))[: a.n]
    gts = {n: cv2.imread(str(SYN / "gt" / n)) for n in names}
    lms = landmarks(gts)
    arc = ArcFace()
    rows = []
    for m in a.models:
        for level in ("mild", "heavy"):
            for n in names:
                x = cv2.imread(str(SYN / level / n)).astype(np.float32)
                y = cv2.imread(str(OUT_DIR / "face/synthetic" / m / level / n)).astype(np.float32)
                for s in (0.25, 0.5, 0.75, 1.0):
                    o = np.clip(x + s * (y - x), 0, 255).round().astype(np.uint8)
                    of, gf = (o[..., ::-1] / 255.0).astype(np.float32), (gts[n][..., ::-1] / 255.0).astype(np.float32)
                    rows.append(dict(model=m, level=level, strength=s, image=n, psnr=metrics.psnr_y(of, gf),
                                     lpips=metrics.lpips(of, gf), dists=metrics.dists(of, gf),
                                     identity=arc.similarity(o, gts[n], lms[n])))
            for s in (0.25, 0.5, 0.75, 1.0):
                sel = [r for r in rows if r["model"] == m and r["level"] == level and r["strength"] == s]
                print(f"{m:20s} {level:5s} s={s:<4} " + " ".join(
                    f"{k}={np.mean([r[k] for r in sel]):.4f}" for k in ("psnr", "lpips", "dists", "identity")), flush=True)
    with open(RESULTS_DIR / "face/strength_synthetic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
