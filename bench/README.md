# bench: model bake-off and latency benchmarks

M0 model selection for denoise, sharpen and 2× upscale (docs/DESIGN.md §8, §10), and the
unblur bake-off for real motion blur and defocus ([`UNBLUR.md`](UNBLUR.md)).
Runs on spark1 (DGX Spark, GB10, aarch64, CUDA 13). The decision and its evidence are in
[`BAKEOFF.md`](BAKEOFF.md); raw numbers are in `results/`.

## Setup

```sh
cd bench
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
python fetch.py                     # weights -> $PAINITE_BENCH_DIR/weights (default ~/painite-bench), sha256-checked
```

Test-set sources (downloaded on the bench machine, never committed; research-use licenses):

- DIV2K validation HR, 100 photos: `https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip` → `$PAINITE_BENCH_DIR/data/DIV2K_valid_HR/`
- SIDD validation crops (Restormer's packaging): `gdown 1Fw6Ey1R-nCHN9WEpxv0MnMqxij-ECQYJ`, unzip to `$PAINITE_BENCH_DIR/data/sidd_val/`
- PolyU real-world noisy images: `git clone https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset $PAINITE_BENCH_DIR/data/polyu`

Unblur test-set sources (`$PAINITE_BENCH_DIR/data/deblur/`, see `make_unblur_testset.py`):

- RealBlur-J test (Restormer's packaging): `gdown 1Rb1DhhXmX7IXfilQ-zL9aGjQfAAvQTrW` → `realblurj/`
- DPDD test: `gdown 1dDWUQ_D93XGtcywoUcZE1HOXCV4EuLyw` → `dpdd/`
- RealDOF: `https://www.dropbox.com/s/arox1aixvg67fw5/RealDOF.zip?dl=1` → `RealDOF/{source,target}`
- Köhler et al. blurry images: `http://web.archive.org/web/20240825195707/http://people.kyb.tuebingen.mpg.de/rolfk/BenchmarkECCV2012/BlurryImages.zip` → `kohler/`

## Run

```sh
python make_testset.py              # 18 photos x degradations + SIDD and PolyU real-noise crops
python quality.py --op denoise      # also: sharpen, upscale2x; --models <name> to rerun one
python latency.py                   # preview + 12/24/48 MP full renders, peak memory
python summarize.py                 # results/summary.md
```

## Method

**Test set.** 18 DIV2K photos picked by the PRD's content classes: skin (4), foliage (4), text (3),
hard edges/architecture (4), fine texture (2), low light (1). Each gets synthetic degradations so
every score has a clean reference:

| Operation | Inputs |
|---|---|
| Denoise | Poisson-Gaussian sensor noise added in linear light on an RGGB mosaic and bilinearly demosaicked, so the noise is signal-dependent and chroma-correlated like a real camera's (`iso3200`, `iso12800`); plus real noise: 64 smartphone crops from SIDD and 100 DSLR high-ISO crops from PolyU (Canon 5D II, 80D, 600D, Nikon D800, Sony A7 II; camera JPEGs, ground truth is a ~500-frame mean). No candidate was trained on PolyU, so it is the fairest real-noise test. |
| Sharpen | Gaussian σ=1.2, defocus disk r=2, 5 px motion, each with light noise; plus the clean photo itself, to see what "sharpen" does to an already sharp image. |
| 2× upscale | Half-size input made with area downsampling (`clean`), and the same with light noise and JPEG q90 (`camera`). |

**Tiling.** Every model runs through the production tiling from the design: 512 px tiles,
32 px overlap, linear feathered blend, grid anchored at the origin (`painite_bench/tiling.py`).

**Metrics** (`painite_bench/metrics.py`), against the clean image:

- PSNR-Y, SSIM: pixel fidelity.
- LPIPS, DISTS: perceptual distance; these track "looks right" better than PSNR, especially for GAN outputs.
- ΔE00 after a σ=6 blur: color casts and saturation shifts, separated from detail.
- Halo: mean overshoot (8-bit levels) beyond the clean image's 5×5 local min/max inside a band around strong edges. Captures ringing and sharpening halos.
- Detail ratio: high-frequency energy of the output over the clean image, in textured regions. Under 1 is smoothing (detail loss), over 1 is leftover noise or invented texture.
- Tiling consistency: the tiled output vs one pass over a 1024 px crop. A seam/interior ratio near 1 means differences are not concentrated at tile boundaries (no visible seams).

**Latency** (`latency.py`) is server-side inference only, image already decoded and on the GPU,
including host↔device copies of the full image. Decode, encode, queueing and Wi-Fi transfer are
excluded on purpose; the server records those per job and the end-to-end benchmark reports them
separately. A preview is the design's example 1024×1024 region, processed as every full-image
grid tile that touches it (3×3 tiles), so it matches the export pixel for pixel.

**Reference tools.** The PRD asks for comparison against existing tools (Lightroom Denoise,
Topaz). Those do not run on Linux, so classical baselines stand in here (OpenCV non-local means,
unsharp mask, Lanczos). A side-by-side against Lightroom on real high-ISO raws is a follow-up on the Mac.

## Face bake-off

Face restoration (the "face" op) has its own harness in `face/`. The decision is in
[`FACE_BAKEOFF.md`](FACE_BAKEOFF.md) and the tables are in `results/face/`.

```sh
uv pip install -r face/requirements.txt
python fetch.py --op face && python fetch.py --op face-aux    # weights -> $PAINITE_BENCH_DIR/weights/face
# Upstream arch code for the models spandrel does not load (read before running):
git clone --depth 1 https://github.com/yangxy/GPEN                        $PAINITE_BENCH_DIR/src/GPEN
git clone --depth 1 https://github.com/wzhouxiff/RestoreFormerPlusPlus    $PAINITE_BENCH_DIR/src/RestoreFormerPlusPlus
git clone --depth 1 https://github.com/ohayonguy/PMRF                     $PAINITE_BENCH_DIR/src/PMRF
git clone --depth 1 https://github.com/jkwang28/OSDFace                   $PAINITE_BENCH_DIR/src/OSDFace
unzip $PAINITE_BENCH_DIR/weights/face/buffalo_l.zip -d $PAINITE_BENCH_DIR/weights/face/buffalo_l
# Test data (never committed): CelebA-HQ validation split from HF mattymchen/celeba-hq
curl -L -o $PAINITE_BENCH_DIR/data/face/celebahq_val.parquet \
  https://huggingface.co/datasets/mattymchen/celeba-hq/resolve/main/data/validation-00000-of-00001-a47eeed86041f01b.parquet

python face/make_testset.py         # 100 aligned faces x {mild, heavy}
python face/fetch_real.py           # 19 Commons photos listed in face/real_photos.txt (+ sources.csv)
python face/quality.py              # synthetic: PSNR, LPIPS, DISTS, ArcFace identity, MUSIQ, CLIP-IQA
python face/real.py                 # real photos: full pipeline, no-reference scores, identity drift, crops
python face/largeface.py            # face-size policy
python face/strength.py             # strength sweep (from quality.py outputs)
python face/masks.py --model gpen-1024+anchor8
python face/latency.py
python face/adapter_check.py        # server face_adapter vs bench pipeline
python face/summarize.py            # results/face/summary.md
```
