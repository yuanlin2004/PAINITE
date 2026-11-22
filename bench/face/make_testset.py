"""Face bake-off test sets under $PAINITE_BENCH_DIR/testset/face.

  synthetic/gt/NNN.png      100 aligned 512 px CelebA-HQ faces (validation split, first 100)
  synthetic/mild/NNN.png    photographer case: blur σ 0.5–1.5, 2x down/up, light noise, JPEG q75–90
  synthetic/heavy/NNN.png   CelebA-Test style (GFPGAN): blur σ 0.1–10, ↓0.8–8, noise 0–20, JPEG 60–100

  python face/make_testset.py --parquet ~/painite-bench/data/face/celebahq_val.parquet

The parquet is the validation split of the HF dataset mattymchen/celeba-hq (1024 px JPEG).
"""
import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from painite_bench.config import DATA_DIR, TESTSET_DIR  # noqa: E402


def _jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def degrade(gt: np.ndarray, level: str, seed: int) -> np.ndarray:
    """gt: 512x512x3 uint8 BGR. Returns uint8 BGR at the same size."""
    rng = np.random.default_rng(seed)
    x = gt.astype(np.float32)
    h, w = x.shape[:2]
    if level == "mild":
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
        x = cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        x = x + rng.normal(0, rng.uniform(1, 4), x.shape)
        x = _jpeg(np.clip(x, 0, 255).round().astype(np.uint8), rng.uniform(75, 90)).astype(np.float32)
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    elif level == "heavy":
        # GFPGAN's degradation for CelebA-Test: blur -> downsample -> noise -> JPEG -> resize back
        x = cv2.GaussianBlur(x, (41, 41), rng.uniform(0.1, 10))
        s = rng.uniform(0.8, 8)
        x = cv2.resize(x, (int(w / s), int(h / s)), interpolation=cv2.INTER_LINEAR)
        x = x + rng.normal(0, rng.uniform(0, 20), x.shape)
        x = _jpeg(np.clip(x, 0, 255).round().astype(np.uint8), rng.uniform(60, 100)).astype(np.float32)
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(level)
    return np.clip(x, 0, 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(DATA_DIR / "data/face/celebahq_val.parquet"))
    ap.add_argument("-n", type=int, default=100)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    rows = pq.read_table(a.parquet, columns=["image"]).slice(0, a.n).to_pylist()
    out = TESTSET_DIR / "face/synthetic"
    for d in ("gt", "mild", "heavy"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        im = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        gt = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"gt/{i:03d}.png"), gt)
        for j, level in enumerate(("mild", "heavy")):
            cv2.imwrite(str(out / f"{level}/{i:03d}.png"), degrade(gt, level, seed=1000 * j + i))
    print(f"{len(rows)} faces -> {out}")


if __name__ == "__main__":
    main()
