"""Real photos: full pipeline (detect on long side 2048, align, restore, parse-mask paste) on the
Commons set, no-reference scores on face crops, identity drift vs the input face, and
side-by-side crops for visual review.

  python face/real.py [--models ...] [--mask parse] [--tag default]

Writes results/face/quality_real.csv and $PAINITE_BENCH_DIR/out/face/real/<tag>/compare/*.jpg
(input | model ... for the largest, middle and smallest face of each photo).
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from painite_bench.config import OUT_DIR, RESULTS_DIR, TESTSET_DIR  # noqa: E402
import pipeline as P  # noqa: E402
import restorers  # noqa: E402
from identity import ArcFace  # noqa: E402

REAL = TESTSET_DIR / "face/real"
CSV = RESULTS_DIR / "face/quality_real.csv"
MIN_SCORED_EYE = 16  # px; smaller faces are restored but too small for crop scores


def face_crop(img, f: P.Face, size=384):
    """Square crop around the face box (1.6x its size), resized to `size`."""
    x0, y0, x1, y1 = f.box
    c = ((x0 + x1) / 2, (y0 + y1) / 2)
    s = max(x1 - x0, y1 - y0) * 1.6
    m = np.array([[size / s, 0, size / 2 - c[0] * size / s], [0, size / s, size / 2 - c[1] * size / s]], np.float32)
    if s > size:
        k = 0.5 * s / size
        roi_pad = int(s)
        X0, Y0 = max(int(c[0] - roi_pad), 0), max(int(c[1] - roi_pad), 0)
        roi = cv2.GaussianBlur(img[Y0:int(c[1] + roi_pad), X0:int(c[0] + roi_pad)], (0, 0), k)
        m[:, 2] += m[:, :2] @ np.array([X0, Y0], np.float32)
        return cv2.warpAffine(roi, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)


def arc_crop(img, f: P.Face):
    """112 px ArcFace crop, pre-blurred when the face is large so it does not alias."""
    s = f.eye_dist / 35.0
    x0, y0, x1, y1 = f.box.astype(int)
    pad = int(max(x1 - x0, y1 - y0))
    X0, Y0 = max(x0 - pad, 0), max(y0 - pad, 0)
    roi = img[Y0:y1 + pad, X0:x1 + pad]
    if s > 1.5:
        roi = cv2.GaussianBlur(roi, (0, 0), 0.5 * s)
    return ArcFace.crop(roi, f.lm5 - np.array([X0, Y0], np.float32))


def label(im, text):
    im = im.copy()
    cv2.putText(im, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=restorers.ALL)
    ap.add_argument("--photos", nargs="*")
    ap.add_argument("--mask", default="parse")
    ap.add_argument("--large-face", default="restore")
    ap.add_argument("--tag", default="default")
    a = ap.parse_args()

    photos = sorted(p.stem for p in REAL.glob("*.jpg") if not a.photos or p.stem in a.photos)
    det, parser, arc = P.Detector(), P.Parser(), ArcFace()
    import pyiqa

    nr = {k: pyiqa.create_metric(k, device="cuda") for k in ("musiq", "clipiqa")}

    def nrs(bgr):
        t = torch.from_numpy(bgr[..., ::-1] / 255.0).float().permute(2, 0, 1)[None].cuda()
        with torch.inference_mode():
            return {k: float(m(t)) for k, m in nr.items()}

    imgs = {p: cv2.imread(str(REAL / f"{p}.jpg")) for p in photos}
    faces = {p: det(imgs[p], 2048, min_eye_dist=8) for p in photos}
    for p in photos:
        print(f"{p:22s} {len(faces[p])} faces, eye dist {[round(f.eye_dist) for f in faces[p]]}", flush=True)

    rows, crops = [], {}
    base = {}
    for p in photos:
        for i, f in enumerate(faces[p]):
            if f.eye_dist >= MIN_SCORED_EYE:
                c = face_crop(imgs[p], f)
                crops[(p, i, "input")] = c
                base[(p, i)] = dict(nrs(c), emb=arc.embed([arc_crop(imgs[p], f)])[0])
                rows.append(dict(model="input", photo=p, face=i, eye_dist=f.eye_dist, size=f.size,
                                 musiq=base[(p, i)]["musiq"], clipiqa=base[(p, i)]["clipiqa"], identity=1.0))
    opt = P.Options(mask=a.mask, large_face=a.large_face)
    for m in a.models:
        r = restorers.load(m)
        for p in photos:
            try:
                out, _, _ = P.enhance(imgs[p], r, det, parser, opt, faces=faces[p])
            except Exception as e:  # noqa: BLE001 - keep the other models' results
                print(f"{m} failed on {p}: {e!r}", flush=True)
                continue
            for i, f in enumerate(faces[p]):
                if f.eye_dist < MIN_SCORED_EYE:
                    continue
                c = face_crop(out, f)
                crops[(p, i, m)] = c
                s = nrs(c)
                e = arc.embed([arc_crop(out, f)])[0]
                rows.append(dict(model=m, photo=p, face=i, eye_dist=f.eye_dist, size=f.size, musiq=s["musiq"],
                                 clipiqa=s["clipiqa"], identity=float(e @ base[(p, i)]["emb"])))
        sel = [x for x in rows if x["model"] == m]
        print(f"{m:18s} faces={len(sel)} " + " ".join(f"{k}={np.mean([x[k] for x in sel]):.4f}"
                                                   for k in ("musiq", "clipiqa", "identity")), flush=True)
        del r
        torch.cuda.empty_cache()

    cd = OUT_DIR / "face/real" / a.tag / "compare"
    cd.mkdir(parents=True, exist_ok=True)
    for p in photos:
        idx = [i for i, f in enumerate(faces[p]) if f.eye_dist >= MIN_SCORED_EYE]
        for k in sorted({idx[0], idx[len(idx) // 2], idx[-1]} if idx else []):
            tiles = [label(crops[(p, k, n)], n) for n in ["input"] + a.models if (p, k, n) in crops]
            cv2.imwrite(str(cd / f"{p}_f{k}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92])

    CSV.parent.mkdir(parents=True, exist_ok=True)
    out_csv = CSV if a.tag == "default" else CSV.with_name(f"quality_real_{a.tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
