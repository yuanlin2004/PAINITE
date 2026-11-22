"""Synthetic face set: restore aligned 512 px degraded faces, score against ground truth.

  python face/quality.py [--models gfpgan-1.4 ...] [--levels mild heavy] [-n 100]

Writes results/face/quality_synthetic.csv (per image) and outputs under
$PAINITE_BENCH_DIR/out/face/synthetic/<model>/<level>/. "input" is the degraded face itself.
Identity is ArcFace cosine to the ground truth (higher is better; same person is usually > 0.5).
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
from painite_bench import metrics  # noqa: E402
from painite_bench.config import OUT_DIR, RESULTS_DIR, TESTSET_DIR  # noqa: E402
import restorers  # noqa: E402
from identity import ArcFace  # noqa: E402

SYN = TESTSET_DIR / "face/synthetic"
CSV = RESULTS_DIR / "face/quality_synthetic.csv"
FIELDS = ["model", "level", "image", "psnr", "ssim", "lpips", "dists", "identity", "musiq", "clipiqa", "sec"]


def _t(bgr):
    return torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1] / 255.0)).float().permute(2, 0, 1)[None].cuda()


def _bgr(t):
    return (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8)


def landmarks(gts):
    from facexlib.detection import init_detection_model

    det = init_detection_model("retinaface_resnet50", device="cuda", model_rootpath=str(restorers.FACE_WEIGHTS))
    out = {}
    with torch.inference_mode():
        for name, g in gts.items():
            b = det.detect_faces(g, 0.5)
            b = b[np.argmax(b[:, 4])]
            out[name] = b[5:15].reshape(5, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["input"] + restorers.ALL)
    ap.add_argument("--levels", nargs="*", default=["mild", "heavy"])
    ap.add_argument("-n", type=int, default=100)
    a = ap.parse_args()

    names = sorted(p.name for p in (SYN / "gt").glob("*.png"))[: a.n]
    gts = {n: cv2.imread(str(SYN / "gt" / n)) for n in names}
    lms = landmarks(gts)
    arc = ArcFace()
    import pyiqa

    nr = {k: pyiqa.create_metric(k, device="cuda") for k in ("musiq", "clipiqa")}

    CSV.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if CSV.exists():
        with open(CSV) as f:
            old = [r for r in csv.DictReader(f) if r["model"] not in a.models or r["level"] not in a.levels]
    rows = []
    for m in a.models:
        r = None if m == "input" else restorers.load(m)
        for level in a.levels:
            od = OUT_DIR / "face/synthetic" / m / level
            od.mkdir(parents=True, exist_ok=True)
            for n in names:
                lq = cv2.imread(str(SYN / level / n))
                x = _t(lq)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                y = x if r is None else restorers.run(r, x)
                torch.cuda.synchronize()
                sec = time.perf_counter() - t0
                out = _bgr(y)
                cv2.imwrite(str(od / n), out)
                o, g = out[..., ::-1] / 255.0, gts[n][..., ::-1] / 255.0
                with torch.inference_mode():
                    yt = _t(out)
                    row = dict(model=m, level=level, image=n, psnr=metrics.psnr_y(o, g), ssim=metrics.ssim_y(o, g),
                               lpips=metrics.lpips(o.astype(np.float32), g.astype(np.float32)),
                               dists=metrics.dists(o.astype(np.float32), g.astype(np.float32)),
                               identity=arc.similarity(out, gts[n], lms[n]),
                               musiq=float(nr["musiq"](yt)), clipiqa=float(nr["clipiqa"](yt)), sec=sec)
                rows.append(row)
            sel = [x for x in rows if x["model"] == m and x["level"] == level]
            print(f"{m:18s} {level:6s} " + " ".join(
                f"{k}={np.mean([x[k] for x in sel]):.4f}" for k in FIELDS[3:]), flush=True)
        del r
        torch.cuda.empty_cache()
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(old + rows)


if __name__ == "__main__":
    main()
