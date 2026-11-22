"""Side-by-side crop sheets of saved outputs, one PNG per test set (for visual review).

  python crops.py --op unblur --models evssm-realblurj mlwnet-realblurj ... --sets realblurj kohler \
      --out ~/git/my/unblur-crops [--n 4] [--size 320]

Rows are images, columns are input | models... | GT (when there is one). The crop is the
size x size window with the most edge detail in the reference (GT, else input), so it
lands on text, faces or texture rather than sky.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from painite_bench import imageio
from painite_bench.config import OUT_DIR, TESTSET_DIR


def best_window(ref: np.ndarray, size: int) -> tuple[int, int]:
    g = cv2.cvtColor((ref * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    e = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F))
    s = cv2.boxFilter(e, -1, (size, size), normalize=True)
    h, w = g.shape
    m = size // 2
    s = s[m:h - m, m:w - m] if h > size and w > size else s
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return (int(y), int(x)) if h > size and w > size else (0, 0)


def label(tile: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((26, tile.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text[:42], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="unblur")
    ap.add_argument("--models", nargs="+", required=True, help="output dir names under out/<op>/")
    ap.add_argument("--labels", nargs="*", help="column titles (default: model names)")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--images", nargs="*", help="image stems to use (default: first --n with outputs)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    labels = a.labels or a.models
    for st in a.sets:
        base = TESTSET_DIR / a.op / st
        names = a.images or [f.stem for f in sorted((base / "blur").glob("*.png"))]
        rows = []
        for name in names:
            outs = []
            for m in a.models:
                d = OUT_DIR / a.op / m
                f = next((p for p in (d / f"{a.op}_{st}" / f"{name}.png", d / st / f"{name}.png") if p.exists()), None)
                outs.append(imageio.read(f) if f else None)
            if all(o is None for o in outs):
                continue
            inp = imageio.read(base / "blur" / f"{name}.png")
            gtp = base / "gt" / f"{name}.png"
            gt = imageio.read(gtp) if gtp.exists() and st != "sharp" else None
            y, x = best_window(gt if gt is not None else inp, a.size)
            cut = lambda im: (np.clip(im[y:y + a.size, x:x + a.size], 0, 1) * 255).round().astype(np.uint8)
            blank = np.full((a.size, a.size, 3), 128, np.uint8)
            cols = [label(cut(inp), f"input  {name}")]
            cols += [label(cut(o) if o is not None else blank, lab) for o, lab in zip(outs, labels)]
            if gt is not None:
                cols.append(label(cut(gt), "ground truth"))
            gap = np.full((a.size + 26, 4, 3), 255, np.uint8)
            rows.append(np.hstack([c for col in cols for c in (col, gap)]))
            if len(rows) >= a.n:
                break
        if rows:
            w = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
            sheet = np.vstack(rows)
            cv2.imwrite(str(a.out / f"{st}.png"), sheet[..., ::-1])
            print(a.out / f"{st}.png", sheet.shape)


if __name__ == "__main__":
    main()
