"""Paste-mask comparison on real photos: square (server adapter default), rect (GFPGAN/facexlib
default), feathered ellipse, and face-parsing (parsenet) masks, for one restorer.

  python face/masks.py --model codeformer-w0.8

Saves input | square | rect | ellipse | parse crops (1.9x the face box, so hair, ears and beard
edges are in frame) to $PAINITE_BENCH_DIR/out/face/masks/<model>/, and prints a seam score per
mask: the mean gradient-magnitude change (output vs input) in a thin band along the mask's
transition, over the gradient change inside the face. Lower means the paste edge is less visible.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from painite_bench.config import OUT_DIR, TESTSET_DIR  # noqa: E402
import pipeline as P  # noqa: E402
import restorers  # noqa: E402
from real import face_crop, label  # noqa: E402

PHOTOS = ["portrait_horridge", "portrait_garcia_alix", "portrait_richard", "half_bahner", "half_rice",
          "half_selfie", "event_coty", "event_silyan", "group_it_cast", "group_sxsw", "group_lawyers"]
MASKS = ["square", "rect", "ellipse", "parse"]


def grad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codeformer-w0.8")
    a = ap.parse_args()
    det, parser = P.Detector(), P.Parser()
    r = restorers.load(a.model)
    od = OUT_DIR / "face/masks" / a.model
    od.mkdir(parents=True, exist_ok=True)
    seam = {m: [] for m in MASKS}
    for p in PHOTOS:
        img = cv2.imread(str(TESTSET_DIR / "face/real" / f"{p}.jpg"))
        faces = [f for f in det(img, 2048) if f.eye_dist >= 24][:3]
        outs, alphas = {}, {}
        for m in MASKS:
            al = []
            outs[m], _, _ = P.enhance(img, r, det, parser, P.Options(mask=m, large_face="restore"), faces=faces,
                                      alpha_out=al)
            alphas[m] = al[0]
        gi = grad(img)
        for k, f in enumerate(faces):
            x0, y0, x1, y1 = f.box.astype(int)
            w = x1 - x0
            sl = (slice(max(y0 - w, 0), y1 + w), slice(max(x0 - w, 0), x1 + w))
            for m in MASKS:
                al = alphas[m][sl]
                band = (al > 0.2) & (al < 0.8)
                inner = al > 0.95
                d = np.abs(grad(outs[m])[sl] - gi[sl])
                if band.any() and inner.any():
                    seam[m].append(d[band].mean() / max(d[inner].mean(), 1e-6))
            tiles = [label(face_crop(img, f, 512), "input")] + [label(face_crop(outs[m], f, 512), m) for m in MASKS]
            # wider view so hair/ears/shoulders are visible
            big = P.Face(np.array([x0 - w * 0.35, y0 - w * 0.35, x1 + w * 0.35, y1 + w * 0.35]), f.score, f.lm5)
            wide = [label(face_crop(img, big, 512), "input")] + [label(face_crop(outs[m], big, 512), m) for m in MASKS]
            cv2.imwrite(str(od / f"{p}_f{k}.jpg"), np.vstack([np.hstack(tiles), np.hstack(wide)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(p, len(faces), flush=True)
    for m in MASKS:
        print(f"{m:8s} seam ratio {np.mean(seam[m]):.3f} (n={len(seam[m])})")


if __name__ == "__main__":
    main()
