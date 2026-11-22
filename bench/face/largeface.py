"""Large faces: what restoring at the model's size does to a face that is bigger in the photo.

Ground truth is CelebA-HQ (native 1024 px) area-resized to a face size L in {512, 768, 1024},
i.e. 1x, 1.5x and 2x a 512 restorer. Two inputs at size L: "soft" (blur σ 1.0 px, noise σ 2,
JPEG q90, an ordinary slightly soft face at that size) and "clean" (JPEG q95 only). Policies,
all scored at L against the ground truth:

  input              untouched
  <m>                resample to the restorer's size, restore, resample back (pipeline default)
  <m> strength 0.5   50/50 blend of that with the input
  <m> lowfreq        (L > restorer size) only frequencies coarser than one restorer pixel
                     (σ = L / size) come from the restorer; finer detail stays from the input

  python face/largeface.py [-n 50] [--sizes 512 768 1024]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from painite_bench import metrics  # noqa: E402
from painite_bench.config import DATA_DIR, RESULTS_DIR  # noqa: E402
import pipeline as P  # noqa: E402
import restorers  # noqa: E402
from identity import ArcFace  # noqa: E402

MODELS = ["gfpgan-1.4", "restoreformer++", "codeformer-w0.8", "gpen-512", "gpen-1024", "gpen-1024+anchor8"]
CSV = RESULTS_DIR / "face/quality_largeface.csv"


def degrade(gt, kind, seed):
    rng = np.random.default_rng(seed)
    x = gt.astype(np.float32)
    q = 95
    if kind == "soft":
        x = cv2.GaussianBlur(x, (0, 0), 1.0) + rng.normal(0, 2, x.shape)
        q = 90
    ok, b = cv2.imencode(".jpg", np.clip(x, 0, 255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def _half(bgr):
    return cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_AREA)  # ArcFace input without aliasing


def _t(bgr):
    return torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1] / 255.0)).float().permute(2, 0, 1)[None].cuda()


def _bgr(t):
    return (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8)


def _resize(t, n):
    if t.shape[-1] == n:
        return t
    if t.shape[-1] > n:
        return F.interpolate(t, size=(n, n), mode="area")
    return F.interpolate(t, size=(n, n), mode="bicubic", align_corners=False).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--sizes", nargs="*", type=int, default=[512, 768, 1024])
    a = ap.parse_args()
    import pyarrow.parquet as pq

    rows_pq = pq.read_table(DATA_DIR / "data/face/celebahq_val.parquet", columns=["image"]).slice(0, a.n).to_pylist()
    full = [cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR) for r in rows_pq]
    det, arc = P.Detector(), ArcFace()
    lms = [det(g, 0)[0].lm5 / 2 for g in full]  # in the 512 px copy that ArcFace sees
    rows = []
    for L in a.sizes:
        gts = [cv2.resize(g, (L, L), interpolation=cv2.INTER_AREA) if L != 1024 else g for g in full]
        inputs = {k: [degrade(g, k, i) for i, g in enumerate(gts)] for k in ("soft", "clean")}

        def score(policy, kind, i, out):
            o, g = (out[..., ::-1] / 255.0).astype(np.float32), (gts[i][..., ::-1] / 255.0).astype(np.float32)
            rows.append(dict(size=L, policy=policy, input=kind, image=i, psnr=metrics.psnr_y(o, g),
                             lpips=metrics.lpips(o, g), dists=metrics.dists(o, g), detail=metrics.detail_ratio(o, g),
                             identity=arc.similarity(_half(out), _half(gts[i]), lms[i])))

        for kind in inputs:
            for i, x in enumerate(inputs[kind]):
                score("input", kind, i, x)
        for m in a.models:
            r = restorers.load(m)
            for kind in inputs:
                for i, x in enumerate(inputs[kind]):
                    xt = _t(x)
                    with torch.inference_mode():
                        y = _resize(r(_resize(xt, r.size)), L)
                    score(m, kind, i, _bgr(y))
                    score(f"{m} strength 0.5", kind, i, _bgr((xt + y) / 2))
                    if L > r.size:
                        sg = L / r.size
                        score(f"{m} lowfreq", kind, i, _bgr(xt - P.lowpass(xt, sg) + P.lowpass(y, sg)))
            del r
            torch.cuda.empty_cache()
            print(L, m, flush=True)

    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keys = ("psnr", "lpips", "dists", "detail", "identity")
    for L in a.sizes:
        for kind in ("soft", "clean"):
            print(f"\nface {L} px, {kind}")
            sel0 = [r_ for r_ in rows if r_["size"] == L and r_["input"] == kind]
            for pol in dict.fromkeys(r_["policy"] for r_ in sel0):
                sel = [r_ for r_ in sel0 if r_["policy"] == pol]
                print(f"  {pol:30s} " + " ".join(f"{k}={np.mean([s[k] for s in sel]):.4f}" for k in keys))


if __name__ == "__main__":
    main()
