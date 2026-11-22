# Unblur bake-off: results and picks

Run on spark1 (DGX Spark, GB10, aarch64, CUDA 13, PyTorch 2.14) on 2026-11-21 and 2026-11-22.
"Unblur" fixes real motion blur (camera shake, subject motion) and out-of-focus photos. It's separate
from v1's "sharpen" (Real-ESRGAN x2 then shrink, see [BAKEOFF.md](BAKEOFF.md)), which only fixes mild
softness. Models were chosen on quality first, then speed. Licenses are recorded but did not
filter the choice. Full tables are in [results/unblur_summary.md](results/unblur_summary.md)
(shortlist) and [results/unblur_screen_summary.md](results/unblur_screen_summary.md) (all 27
variants). Per-image numbers are in `results/quality_unblur*.csv`.

## Picks

No single model is good at both kinds of blur, so unblur needs two models and a mode switch.

| Mode | Model | dtype | 12 MP | 24 MP | 48 MP | Preview (1024² region) | Peak GPU mem |
|---|---|---|---|---|---|---|---|
| Motion / shake | **Stripformer, RealBlur-J weights** (`stripformer-realblurj`) | fp16 | 14.5 s | 27.9 s | 56.4 s | 2.6 s | 2.6 GB |
| Out of focus | **LaKDNet-S, LFDOF+DPDD weights** (`lakdnet-defocus-lfdpdd-s`) | fp16 | 27.2 s | 54.9 s | 104 s | 5.0 s | 2.2 GB |
| (fast defocus alternate) | DRBNet, LFDOF+DPDD weights | fp16 | 2.6 s | 5.2 s | 8.0 s | 0.3 s | 2.1 GB |

Server-side inference only, 512 px tiles with 64 px overlap, the same method as BAKEOFF.md.
Manifests are in [`manifests/unblur/`](manifests/unblur/).
No color anchor and no other post-processing (see "Pipeline findings" below).

Why these:

- **Stripformer-RealBlurJ** ties the best model on real camera shake: on RealBlur-J (80 images) it
  scores 35.03 dB and LPIPS 0.088, against 29.88 / 0.222 for the input. It is the only model that
  clearly helps on large synthetic shake (+1.75 dB at 15–25 px). Among the motion models it
  degrades most gracefully on other blur (best LPIPS on RealDOF: 0.373, against 0.524 for the
  input). It is safe on sharp photos, and it is 2.5–9× faster than the other RealBlur-trained
  models.
- **LaKDNet-S (LFDOF+DPDD)** is the best defocus model on RealDOF, a camera no candidate was
  trained on: 27.50 dB / LPIPS 0.245, against 23.85 / 0.524 for the input. It also wins every
  synthetic disk size (r = 4, 8, 12). On DPDD it is within 0.4 dB of the best DPDD-only models,
  and it changes sharp photos the least of the defocus models. Its 24 MP time (55 s) and preview
  time (5.0 s) are at the PRD limits, so **DRBNet** is the fallback if it misses them in the
  server: it is 10× faster and ~0.4 dB behind on RealDOF (27.08 / 0.265).
- **How to choose the mode.** Expose it as a user option ("Motion" / "Focus"). An automatic choice
  from blur anisotropy doesn't work reliably. The median directional-gradient anisotropy is 1.46
  for RealBlur-J and 1.45 for RealDOF, because scene structure swamps the blur signal. A learned
  blur-type classifier is the follow-up if auto is needed. A wrong choice is close to a no-op,
  not damage:

  | Wrong mode on real blur | PSNR-Y in → out | LPIPS in → out |
  |---|---|---|
  | Motion model on defocus (RealDOF) | 23.85 → 24.75 | 0.524 → 0.373 |
  | Motion model on defocus (DPDD) | 25.44 → 25.19 | 0.352 → 0.317 |
  | Defocus model on shake (RealBlur-J) | 29.88 → 29.58 | 0.222 → 0.193 |

## Server adapter changes needed

Neither pick loads in spandrel 0.4.2, so the server needs two vendored architectures.

1. **Stripformer.** Vendor `models/Stripformer.py` from the pinned commit in the manifest. It is
   pure PyTorch with no extra dependencies, about 400 lines. The network works on `x - 0.5` and
   returns a residual in that space, so the adapter feeds `x - 0.5`, adds `0.5` back and clamps.
   Reflect-pad to a multiple of 8, then crop.
2. **LaKDNet.** Vendor `models/LaKDNet.py`. It is pure PyTorch and needs only `einops`. Construct
   it with `dim=32, num_blocks=[8,12,12,16], mix_kernel_size=[9,9,9,9]` for defocus. The motion
   checkpoints use 7. Reflect-pad to a multiple of 8.
3. **Weights.** Stripformer's weights exist only on Google Drive, so mirror the file and verify it
   by sha256. LaKDNet's are on a direct MPI URL.
4. **Tiling.** Unchanged: 512/64, batch 1. Bigger tiles don't help (next section).
5. **dtype.** fp16 for both picks: within 72–76 dB of fp32, and about 40% faster than fp32.
6. **No postprocess.** Don't apply the color anchor the other ops use.
7. **Licenses.** Stripformer is non-commercial only. LaKDNet and DRBNet are AGPL-3.0. Both are
   recorded in the manifests. By Yuan's rule they don't filter the choice, but they matter if
   PAINITE ships commercially. The best permissive alternatives are FFTformer-RealBlurJ (MIT, but
   4–8× slower) for motion and Restormer-defocus (non-commercial) or GRL-B (MIT, slow) for
   defocus.

## Test sets

All under `$PAINITE_BENCH_DIR/testset/unblur/` and built by `make_unblur_testset.py`. Nothing is
committed (research-use licenses).

| Set | What | Images (screening / final) |
|---|---|---|
| `realblurj` | RealBlur-J test: real camera shake, ground truth (GT) through a beam splitter. Scored after the benchmark's ECC homography alignment. | 4 / 80 |
| `dpdd` | DPDD test: real defocus, Canon 5D IV, center view | 4 / 40 |
| `realdof` | RealDOF: real defocus, different camera, no candidate trained on it | 4 / 50 |
| `kohler` | Köhler et al.: real recorded hand-shake replayed on a camera, no GT (visual only) | 12 |
| `shake15-25`, `shake30-45` | The 18 DIV2K photos with random hand-shake point-spread functions (PSFs) of 15–45 px, convolved in linear light, + noise σ = 2 levels + JPEG q90 | 4 / 9 |
| `defocus4/8/12` | Same photos, anti-aliased defocus disks with r = 4, 8, 12 px | 4 / 9 |
| `sharp` | The clean photos as input: what unblur does to a sharp image | 4 / 9 |

The shake PSFs come from a smooth random-heading walk with variable speed and an occasional
tremor reversal (`degrade.shake_trajectory`). They look like the recorded Levin and Köhler
kernels. Lai et al.'s real set is no longer downloadable (404), and no real phone shots were
available, so Köhler is the no-GT real-shake set.

## Results (final run, shortlist)

PSNR-Y / LPIPS, means over the images listed. The bold model is the pick for that column's blur
type.

| Model | RealBlur-J (80) | DPDD (40) | RealDOF (50) | shake 15–25 | shake 30–45 | defocus r4 | r8 | r12 | sharp (PSNR / LPIPS) |
|---|---|---|---|---|---|---|---|---|---|
| input | 29.88 / 0.222 | 25.44 / 0.352 | 23.85 / 0.524 | 20.64 / 0.445 | 19.74 / 0.487 | 23.63 / 0.429 | 21.41 / 0.578 | 20.32 / 0.647 | — |
| **Stripformer-RBJ** | 35.03 / **0.088** | 25.19 / 0.317 | 24.75 / 0.373 | **22.39 / 0.383** | **20.11** / 0.460 | 25.45 / 0.346 | 22.83 / 0.476 | 21.66 / 0.539 | 34.10 / 0.072 |
| FFTformer-RBJ | 35.04 / 0.088 | 25.33 / 0.339 | 24.30 / 0.459 | 20.83 / 0.397 | 19.83 / 0.454 | 24.85 / 0.336 | 22.10 / 0.510 | 20.84 / 0.619 | 33.52 / 0.077 |
| UFPNet-RBJ | **35.65** / 0.090 | 25.24 / 0.342 | 24.90 / 0.415 | 21.34 / 0.395 | 19.83 / 0.451 | 25.59 / 0.327 | 22.91 / 0.448 | 21.27 / 0.545 | 34.07 / 0.080 |
| Restormer-motion (GoPro) | 31.70 / 0.147 | 25.54 / 0.322 | 23.82 / 0.476 | 19.97 / 0.412 | 19.38 / 0.473 | 23.60 / 0.383 | 21.34 / 0.540 | 20.25 / 0.615 | **42.87 / 0.015** |
| **LaKDNet-S (LFDOF+DPDD)** | 29.58 / 0.193 | 27.74 / 0.152 | **27.50 / 0.245** | 20.49 / 0.388 | 19.59 / **0.442** | 26.23 / **0.204** | **24.95 / 0.312** | **23.40 / 0.418** | 35.53 / 0.038 |
| DRBNet (LFDOF+DPDD) | 29.24 / 0.197 | 27.11 / 0.188 | 27.08 / 0.265 | 20.43 / 0.401 | 19.63 / 0.456 | **26.25** / 0.239 | 24.44 / 0.348 | 23.28 / 0.431 | 34.22 / 0.050 |
| LaKDNet-L (DPDD) | 29.51 / 0.196 | **28.12 / 0.150** | 26.75 / 0.259 | 20.32 / 0.418 | 19.48 / 0.462 | 25.53 / 0.238 | 23.95 / 0.365 | 22.58 / 0.450 | 30.00 / 0.043 |
| Restormer-defocus (DPDD) | 29.89 / 0.199 | 28.00 / 0.161 | 26.64 / 0.278 | 19.87 / 0.430 | 19.03 / 0.473 | 24.69 / 0.267 | 23.77 / 0.357 | 22.09 / 0.481 | 29.13 / 0.044 |

Seconds per image and DISTS, halo and detail-ratio columns are in
[results/unblur_summary.md](results/unblur_summary.md).

What the numbers say:

- **Training data matters more than architecture.** On the full RealBlur-J set, the three
  RealBlur-trained finalists are 3.3–4.0 dB above Restormer-motion (GoPro). On the 4 screening
  images, the GoPro-trained models score 34.0–35.0 dB, and the RealBlur-trained ones (Stripformer,
  UFPNet, EVSSM, LoFormer, MLWNet, FFTformer) reach 35.9–39.2.
- **RealBlur-J flatters RealBlur-trained models.** It is their own test split, so it measures
  in-distribution quality. The generalization tests are synthetic shake, Köhler and the
  cross-blur results. Stripformer is the most consistent across them.
- **Large blur is still mostly unsolved.** At 30–45 px shake or r = 12 defocus, the best regression
  model recovers 0.4 dB (shake) to 3 dB (defocus) of a ~20 dB input. Expect "noticeably better",
  not "sharp", on severe camera shake (see `kohler.png`, row 2).
- **Defocus models need defocus training.** Light-field data (LFDOF) generalizes better than DPDD
  alone. The two LFDOF+DPDD models (LaKDNet-S, DRBNet) beat the DPDD-only ones on RealDOF and on
  every synthetic disk, even though the DPDD-only ones win on DPDD itself.

### Screening (all candidates, 4 images per set)

27 variants. The full table is [results/unblur_screen_summary.md](results/unblur_screen_summary.md).
Dropped for cause:

| Model | Why dropped |
|---|---|
| EVSSM-RealBlurJ (CVPR 2025) | Best on RealBlur-J (39.18 vs Stripformer 35.85 on the same 4 images), but below the input on synthetic shake (19.99 vs 20.50) and ~9× slower than Stripformer (≈42 s per image vs 4.9 s in the dtype run; ~14 min at 48 MP, estimated). It also needs mamba-ssm, which had to be built from source with patches for CUDA 13 and GB10 (below). |
| MLWNet-RealBlurJ / GoPro | 37.72 on RealBlur-J, but **numerical blow-up on sharp, high-detail content**: activations grow ~10⁴× through the deepest encoder stage. Sharp-photo PSNR is 23.1 (RealBlur-J weights) and 9.3 (GoPro weights). fp32, plain and TLC variants all do it. Real blurry photos have in-focus regions, so this is disqualifying. |
| ID-Blau FFTformer / Restormer (RealBlur-J) | Fine on RealBlur-J, but they collapse to 11–17 dB on bright daylight content, with saturated color blotches. Same result in fp32, so it isn't a precision problem. |
| NAFNet GoPro (±TLC) | Re-checked the old "color blotches" finding. It is **not** a tiling, fp16 or padding artifact: fp16 matches fp32 to 76 dB, and the Test-time Local Converter (TLC) doesn't fix it (sharp-photo PSNR 26.8 plain, 22.8 with TLC). It is the same instability as MLWNet (also a NAFNet family) on content GoPro training never showed it. Also weak on real shake (34.75). |
| FFTformer-RealBlurJ | Good (ties Stripformer on RealBlur-J), but 4–8× slower: 434 s at 48 MP in bf16, and fp16 is broken. It was the strongest on the 4-image synthetic screen, but the lead vanished on 9 images. |
| UFPNet-RealBlurJ | +0.6 dB over Stripformer on RealBlur-J, but worse on everything else and 2.5× slower (24 MP 121 s fp32). |
| GRL-B (RealBlur-J, GoPro, DPDD) | Never best on a real set, and the slowest family: ~110 s per 2.7 MP image. |
| LoFormer-L, LaKDNet motion, MPRNet, Restormer-motion, NAFNet-REDS | Dominated by Stripformer on every real set. |
| Uformer-B, AdaRevD, XYScanNet, DeblurDiNAT, IFAN, NRKNet | Not run: weights only on SharePoint or Baidu, GoPro-only weights, or dependencies (natten 0.14, causal-conv1d) with no aarch64/CUDA 13 build. Higher-ranked candidates in the same families were tested. |
| v1 "sharpen" pick (Real-ESRGAN x2 ↓2) | As expected, doesn't undo real blur: RealBlur-J 32.52 vs input 32.79, DPDD 22.14 vs 22.26. |

### Generative reference: HYPIR (one-step SD2.1 restorer)

HYPIR ran with its own tiler (512 latent patches, stride 256, empty prompt, upscale 1) on the
4 screening images per set. That's about 30 s per 2 MP image on GB10.

- It is the only model that makes heavy blur look sharp. It has the best LPIPS on both shake sets
  and on defocus r12 (shake 30–45: 0.337 vs best regression 0.364), and the best DISTS on every
  synthetic set except r4. LaKDNet still beats it on r4 and r8 LPIPS.
- It does so by **inventing detail**. PSNR falls well below the input everywhere (shake 30–45:
  16.5 vs 19.8). It rewrites already-sharp photos (19.6 dB), changes object shapes, adds fake
  lettering to signs, and leaves real defocused text blurry while inventing hair texture. See
  `realdof.png` and `shake15-25.png`. Its low frequencies match the input (32–34 dB after
  low-pass), so the color anchor doesn't help (+0.1 dB).
- It isn't a pick. It could be an explicit "Extreme (generative)" option later, with a warning
  about faces and text. Its license is also non-commercial.

## Pipeline findings

1. **Tile size and context don't matter for these models.** Tiling at 512/64, 1024/128 and the
   whole image in one pass agree within ±0.5 dB, in no consistent direction. LaKDNet and DRBNet
   agree to 0.04 dB on RealDOF and defocus r12 (`results/quality_unblur_context.csv`). Large blur
   isn't held back by the 512 px tile. The models simply don't undo blur that large. Keep 512/64.
2. **No color anchor for unblur.** The low-frequency anchor that helps the other ops hurts here.
   Its σ = 4–8 px sits inside the blur scale, so it puts the blur back. On the first 12 images
   per set, Stripformer on RealBlur-J drops 34.42 → 33.80 dB with σ = 8, and LaKDNet on RealDOF
   drops 27.49 → 26.40. It only helps on already-sharp input.
3. **dtype** (`results/dtype_unblur.csv`, fp16 and bf16 vs fp32 on the same tiles): fp16 is within
   70–77 dB of fp32 for every shortlisted model except FFTformer, where fp16 output is broken
   (~10 dB, so use bf16 there). bf16 is within 52–58 dB everywhere. In the latency runs, fp16 is
   about 40% faster than fp32 for the three vendored models measured: Stripformer 24 MP 49.8 s →
   27.9 s, LaKDNet 96.4 s → 54.9 s, DRBNet 9.0 s → 5.2 s.
4. **Batch 1.** `latency.py`'s batch sweep picked batch 1 for every model. The vendored adapters
   run one tile at a time anyway, because EVSSM only accepts batch 1.

## Caveats and follow-ups

- **No real phone shots.** Köhler (real recorded shake) is the only real no-GT motion set. Before
  release, run 10–20 real blurry photos from Yuan's phone and camera through both modes.
- **Mixed blur** (shake + defocus in one photo) wasn't tested. Running the defocus model then the
  motion model is a cheap experiment.
- **Auto mode** needs a learned blur-type classifier. The anisotropy heuristic overlaps too much
  on real photos.
- **EVSSM's mamba-ssm build** (for reproducing the screening): build mamba-ssm 2.2.2 from the git
  tag with `-std=c++20`, gencode for `sm_121` only, and `cub::LaneId()` / `cub::CTA_SYNC()`
  replaced by `threadIdx.x % 32` / `__syncthreads()`, which CCCL 3 removed.
  `painite_bench/arches.py` imports only `selective_scan_interface`, because `mamba_ssm/__init__`
  needs an old transformers. A pure-PyTorch scan fallback is included but takes ~90 s per tile.
- HYPIR used a separate venv (diffusers 0.32, transformers 4.49, peft 0.14) and the
  `sd2-community/stable-diffusion-2-1-base` mirror, since the official repo is gated now.

## Reproduce

```sh
cd bench && source .venv/bin/activate
uv pip install PyWavelets kornia einops   # for the upstream archs
python fetch.py --op unblur --src          # weights (sha256-checked) + pinned upstream arch repos
python make_unblur_testset.py              # needs the downloads listed in make_unblur_testset.py
python quality.py --op unblur --overlap 64 --limit 4 --csv quality_unblur_screen.csv   # screening
./run_unblur.sh                            # shortlist, context study, dtype, Köhler renders, latency
python summarize_unblur.py
python crops.py --op unblur --sets realdof --models lakdnet-defocus-lfdpdd-s drbnet-lfdpdd --out ../../unblur-crops
```
