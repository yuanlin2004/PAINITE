"""Build the unblur test set (bench/UNBLUR.md). Sources are downloaded on the bench machine,
never committed (research-use licenses); see README.md for where to get them.

Layout under $PAINITE_BENCH_DIR/testset/unblur/<set>/{blur,gt}/<name>.png:
  shake15-25, shake30-45     DIV2K picks with random camera-shake PSFs, in linear light,
                             + light noise and JPEG q90 (degrade.UNBLUR_KINDS)
  defocus4, defocus8, defocus12   same, anti-aliased defocus disks of radius r px
  sharp                      the clean photos themselves (blur = gt): what unblur does to sharp input
  realblurj                  RealBlur-J test, real camera shake with GT (subset)
  dpdd                       DPDD test, real defocus (Canon 5D IV), centre view (subset)
  realdof                    RealDOF, real defocus from a different camera (no model trained on it)
  kohler                     Kohler et al. ECCV12 real camera shake (robot-replayed hand-shake
                             trajectories on a real camera), blurred only: visual check
"""
import argparse
import json
import random
import zlib
from pathlib import Path

import numpy as np

from painite_bench import degrade, imageio
from painite_bench.config import DATA_DIR, TESTSET_DIR
from make_testset import IMAGES

OUT = TESTSET_DIR / "unblur"
SRC = DATA_DIR / "data/deblur"


def put(set_, name, blur, gt=None):
    for sub, im in (("blur", blur), ("gt", gt)):
        if im is None:
            continue
        d = OUT / set_ / sub
        d.mkdir(parents=True, exist_ok=True)
        if isinstance(im, Path):
            im = imageio.read(im)
        imageio.write(d / f"{name}.png", im)


def synthetic(meta):
    for cls, ids in IMAGES.items():
        for i in ids:
            name = f"{cls}-{i}"
            gt = imageio.read(TESTSET_DIR / "gt" / f"{name}.png")
            put("sharp", name, gt, gt)
            for kind, make in degrade.UNBLUR_KINDS.items():
                seed = zlib.crc32(f"{kind}/{name}".encode())
                k = make(np.random.default_rng(seed))
                put(kind, name, degrade.apply_blur(gt, k, seed), gt)
                meta[f"{kind}/{name}"] = {"kernel_px": k.shape[0]}
            print(name, flush=True)


def paired(set_, blur_dir: Path, gt_dir: Path, n: int, pattern="*.png", gt_name=lambda f: f.name):
    files = sorted(blur_dir.glob(pattern))
    random.Random(0).shuffle(files)
    for f in sorted(files[:n] if n else files):
        put(set_, f.stem, f, gt_dir / gt_name(f))
    print(set_, min(n, len(files)) if n else len(files), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["synthetic", "realblurj", "dpdd", "realdof", "kohler"])
    ap.add_argument("--n-realblurj", type=int, default=80)
    ap.add_argument("--n-dpdd", type=int, default=40)
    a = ap.parse_args()
    meta = {}
    if "synthetic" in a.sets:
        synthetic(meta)
        (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    if "realblurj" in a.sets:
        r = SRC / "realblurj/test/RealBlur_J"
        paired("realblurj", r / "input", r / "target", a.n_realblurj)
    if "dpdd" in a.sets:
        r = SRC / "dpdd/test"
        paired("dpdd", r / "inputC", r / "target", a.n_dpdd)
    if "realdof" in a.sets and (SRC / "RealDOF").exists():
        r = SRC / "RealDOF"
        paired("realdof", r / "source", r / "target", 0, "*.png")
    if "kohler" in a.sets and (SRC / "kohler").exists():
        # 4 scenes x 12 recorded shakes; keep 3 shakes per scene (small, medium, large)
        for f in sorted((SRC / "kohler").glob("Blurry*_*.png")):
            if int(f.stem.split("_")[1]) in (1, 6, 10):
                put("kohler", f.stem, f)
        print("kohler", flush=True)


if __name__ == "__main__":
    main()
