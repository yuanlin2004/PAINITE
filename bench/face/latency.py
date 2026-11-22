"""Face op latency on the bench GPU: detection on 24 MP, per-face restore, full 24 MP renders with
1 and 10 faces, a 1024x1024 preview region, and peak GPU memory. Image already decoded; host<->
device copies included (same convention as bench/latency.py).

  python face/latency.py [--models ...] [--dtypes fp32 fp16]

Writes results/face/latency.csv.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from painite_bench.config import RESULTS_DIR, TESTSET_DIR  # noqa: E402
import pipeline as P  # noqa: E402
import restorers  # noqa: E402

REAL = TESTSET_DIR / "face/real"
CSV = RESULTS_DIR / "face/latency.csv"


def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


class Cast:
    """Run a restorer in reduced precision (weights and input cast, output back to fp32)."""

    def __init__(self, r, dtype):
        self.r, self.size, self.name = r, r.size, r.name
        self.dtype = dtype
        for attr in ("net", "flow", "mmse"):
            if hasattr(r, attr):
                getattr(r, attr).to(dtype)

    def __call__(self, x):
        with torch.autocast("cuda", enabled=False):
            return self.r(x.to(self.dtype)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=restorers.ALL)
    ap.add_argument("--dtypes", nargs="*", default=["fp32", "fp16"])
    a = ap.parse_args()

    one = cv2.imread(str(REAL / "portrait_marinescu.jpg"))  # 6000x4000, one large face
    grp = cv2.imread(str(REAL / "group_usa_team.jpg"))  # 6000x4000, 14 faces
    assert one.shape[:2] == grp.shape[:2] == (4000, 6000)
    det, parser = P.Detector(), P.Parser()
    rows = []

    for ls in (1024, 2048, 0):
        t = timed(lambda: det(grp, ls))
        rows.append(dict(what=f"detect 24MP long side {ls or 'full'}", model="retinaface-r50", dtype="fp32", sec=t,
                         faces=len(det(grp, ls)), peak_gb=""))
        print(rows[-1], flush=True)
    f1 = det(one, 2048)[:1]
    f10 = det(grp, 2048)[:10]
    assert len(f10) == 10
    t = timed(lambda: parser(torch.rand(1, 3, 512, 512, device="cuda")))
    rows.append(dict(what="parse mask per face", model="parsenet", dtype="fp32", sec=t, faces=1, peak_gb=""))

    # preview: 1024x1024 region around the third face of the group photo; detection is cached
    # per source, so a preview restores only the faces touching the region
    cx, cy = f10[2].box[:2].astype(int)
    rx0, ry0 = max(cx - 384, 0), max(cy - 384, 0)
    region = (rx0, ry0, rx0 + 1024, ry0 + 1024)
    touch = [f for f in f10 if f.box[0] < region[2] and f.box[2] > region[0] and f.box[1] < region[3] and f.box[3] > region[1]]

    for m in a.models:
        base = restorers.load(m)
        for dt in a.dtypes:
            if dt == "fp16" and m in ("osdface",):
                r = base  # already fp16
            elif dt == "fp32" and m == "osdface":
                continue
            elif dt == "fp16":
                try:
                    r = Cast(restorers.load(m), torch.float16)
                    y = r(torch.rand(1, 3, r.size, r.size, device="cuda"))
                    if not torch.isfinite(y).all():
                        raise RuntimeError("non-finite output")
                except Exception as e:  # noqa: BLE001
                    print(m, "fp16 failed:", e)
                    continue
            else:
                r = base
            x = torch.rand(1, 3, r.size, r.size, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            t_face = timed(lambda: r(x))
            for what, img, faces in (("24MP, 1 face", one, f1), ("24MP, 10 faces", grp, f10)):
                opt = P.Options(mask="parse")
                t = timed(lambda: P.enhance(img, r, det, parser, opt, faces=det(img, 2048)[:len(faces)]), reps=2)
                rows.append(dict(what=what + " (incl. detection)", model=m, dtype=dt, sec=t, faces=len(faces),
                                 peak_gb=""))
            reg = grp[region[1]:region[3], region[0]:region[2]]
            shifted = [P.Face(f.box - [region[0], region[1]] * 2, f.score, f.lm5 - [region[0], region[1]]) for f in touch]
            t = timed(lambda: P.enhance(reg, r, det, parser, P.Options(mask="parse"), faces=shifted))
            rows.append(dict(what="preview 1024x1024 region (detection cached)", model=m, dtype=dt, sec=t,
                             faces=len(touch), peak_gb=""))
            rows.append(dict(what="restore one face", model=m, dtype=dt, sec=t_face, faces=1,
                             peak_gb=torch.cuda.max_memory_allocated() / 1e9))
            for row in rows[-4:]:
                print(row, flush=True)
        del base
        torch.cuda.empty_cache()

    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
