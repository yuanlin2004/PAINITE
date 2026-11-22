"""Build the bake-off test set from DIV2K validation HR photos (CC BY-NC-SA research
use, downloaded on the bench machine, not committed) plus SIDD real-noise crops.

Layout under $PAINITE_BENCH_DIR/testset/:
  gt/<name>.png                     clean reference (even dimensions)
  denoise/<level>/<name>.png        synthetic camera noise (iso3200, iso12800)
  denoise/sidd/{noisy,gt}/<n>.png   real smartphone noise, 64 crops of 256 px
  denoise/polyu/{noisy,gt}/<n>.png  real DSLR high-ISO JPEGs (PolyU), 100 crops of 512 px
  sharpen/<kind>/<name>.png         gauss1.2, defocus2, motion5 (+ light noise)
  upscale2x/<kind>/<name>.png       half-size input: clean, camera (noise + JPEG q90)
"""
import argparse
import json
import random
import zlib
from pathlib import Path

from painite_bench import degrade, imageio
from painite_bench.config import DATA_DIR, TESTSET_DIR

# DIV2K validation picks, by the PRD's content classes.
IMAGES = {
    "skin": ["0855", "0838", "0813", "0804"],
    "foliage": ["0840", "0849", "0876", "0899"],
    "text": ["0831", "0891", "0867"],
    "edges": ["0845", "0846", "0812", "0824"],
    "texture": ["0805", "0853"],
    "lowlight": ["0814"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--div2k", default=DATA_DIR / "data/DIV2K_valid_HR", type=Path)
    ap.add_argument("--sidd", default=DATA_DIR / "data/sidd_val/val/SIDD", type=Path)
    ap.add_argument("--polyu", default=DATA_DIR / "data/polyu/CroppedImages", type=Path)
    ap.add_argument("--only-real", action="store_true", help="just (re)build the SIDD/PolyU sets")
    a = ap.parse_args()
    if a.only_real:
        return real_sets(a)
    meta = {}
    for cls, ids in IMAGES.items():
        for i in ids:
            name = f"{cls}-{i}"
            img = imageio.read(a.div2k / f"{i}.png")
            h, w, _ = img.shape
            img = img[: h // 4 * 4, : w // 4 * 4]
            seed = zlib.crc32(name.encode())
            out = {"gt": img}
            for lvl in degrade.ISO_LEVELS:
                out[f"denoise/{lvl}"] = degrade.camera_noise(img, lvl, seed)
            for k in ("gauss1.2", "defocus2", "motion5"):
                out[f"sharpen/{k}"] = degrade.blur(img, k, seed)
            for k in ("clean", "camera"):
                out[f"upscale2x/{k}"] = degrade.downscale2x(img, k, seed)
            for sub, im in out.items():
                d = TESTSET_DIR / sub
                d.mkdir(parents=True, exist_ok=True)
                imageio.write(d / f"{name}.png", im)
            meta[name] = {"class": cls, "source": f"DIV2K_valid_HR/{i}.png", "size": list(img.shape[:2])}
            print(name, img.shape)
    real_sets(a)
    (TESTSET_DIR / "meta.json").write_text(json.dumps(meta, indent=1))


def real_sets(a):
    files = sorted((a.sidd / "input_crops").glob("*.png"))
    random.Random(0).shuffle(files)
    for sub in ("noisy", "gt"):
        (TESTSET_DIR / "denoise/sidd" / sub).mkdir(parents=True, exist_ok=True)
    for f in files[:64]:
        for sub, src in (("noisy", "input_crops"), ("gt", "target_crops")):
            (TESTSET_DIR / "denoise/sidd" / sub / f.name).write_bytes((a.sidd / src / f.name).read_bytes())
    # PolyU: camera JPEGs at high ISO; ground truth is the mean of ~500 frames.
    for sub in ("noisy", "gt"):
        (TESTSET_DIR / "denoise/polyu" / sub).mkdir(parents=True, exist_ok=True)
    for f in sorted(a.polyu.glob("*_real.JPG")):
        stem = f.name.removesuffix("_real.JPG")
        imageio.write(TESTSET_DIR / "denoise/polyu/noisy" / f"{stem}.png", imageio.read(f))
        imageio.write(TESTSET_DIR / "denoise/polyu/gt" / f"{stem}.png", imageio.read(a.polyu / f"{stem}_mean.JPG"))


if __name__ == "__main__":
    main()
