# M0 model bake-off: results and picks

Run on spark1 (DGX Spark, GB10, aarch64, CUDA 13, PyTorch 2.14) on 2026-11-21.
Method, test set and metrics are in [README.md](README.md). Full tables are in
[results/summary.md](results/summary.md), and per-image numbers are in `results/*.csv`.
Models were chosen on quality and speed only. Licenses are recorded in the manifests.

## Picks

| Operation | Model | Post-processing | 12 MP | 24 MP | 48 MP | Preview (1024² region) | Peak GPU mem |
|---|---|---|---|---|---|---|---|
| Denoise | **SCUNet real, GAN variant** (`scunet_color_real_gan`) | color anchor σ=8 | 10.5 s | 16.8 s | 35.6 s | 1.3 s | 2.0 GB |
| Sharpen | **Real-ESRGAN x2plus, then area ↓2** | color anchor σ=4 | 16.5 s | 21.7 s | 60.0 s | 1.7 s | 5.8 GB |
| 2× upscale | **2xNomosUni ESRGAN multijpg** | color anchor σ=8 | 16.6 s | 32.0 s | 61.0 s | 1.7 s | 5.8 GB |

The times are server-side inference only. The image is already decoded and on the GPU, host↔device
copies are included, and the timings use 512 px tiles with 32 px overlap. Decode, encode, queue and
Wi-Fi transfer are not included; the server records those per job. All three picks meet the PRD
targets with room to spare: a 24 MP render well under 60 s, and a preview well under 5 s.
Manifests are in [`manifests/`](manifests/).

Two pipeline changes came out of the bake-off and apply to all three picks:

1. **Use a 64 px tile overlap, not 32.** At 32 px, the convolutional picks show slightly larger
   differences along tile boundaries than inside tiles: a seam/interior ratio of 2 to 3, and a mean
   difference in the seam band of 0.3 to 0.7 levels out of 255. At 64 px, the seam-band difference
   roughly halves. It costs about 14% more tiles at 48 MP (247 vs 216). The preview-to-export match
   is unaffected, because previews run the same grid tiles.
2. **Color anchor** (`painite_bench/models.py:color_anchor`). After tiling, the output's low
   frequencies are replaced with the input's: `out − blur(out) + blur(input)`. GAN restorers drift
   slightly in tone and color, and this pins the overall tone and color to the source while keeping
   the model's detail. PSNR improved for every pick, and LPIPS barely moved (it got slightly worse only on synthetic ISO 12800):

   | Pick, test set | PSNR-Y without → with anchor | LPIPS without → with |
   |---|---|---|
   | Denoise, PolyU real DSLR | 40.35 → 41.26 | 0.0544 → 0.0534 |
   | Denoise, synthetic ISO 12800 | 32.27 → 33.31 | 0.1462 → 0.1526 |
   | Upscale, clean | 32.37 → 32.94 | 0.0591 → 0.0589 |
   | Upscale, camera (noise + JPEG) | 30.21 → 31.35 | 0.1534 → 0.1519 |
   | Sharpen, Gaussian blur | 32.02 → 32.61 | 0.1270 → 0.1253 |

   (The ΔE00 column improves by construction, since it measures the same low frequencies, so PSNR
   and LPIPS are the evidence here.)

## Denoise

The fairest test is PolyU: 100 crops of real high-ISO camera JPEGs from Canon 5D II, 80D and 600D,
Nikon D800, and Sony A7 II bodies. None of the candidates was trained on it.

| Model | PolyU PSNR | PolyU LPIPS | PolyU DISTS | ISO 12800 PSNR | ISO 12800 LPIPS | 48 MP |
|---|---|---|---|---|---|---|
| noisy input | 38.58 | 0.198 | 0.116 | 27.90 | 0.506 | |
| **SCUNet real GAN** | 40.35 | **0.054** | **0.086** | 32.27 | **0.146** | 36 s |
| SCUNet real PSNR | **41.44** | 0.062 | 0.112 | **33.74** | 0.181 | 36 s |
| Restormer real | 40.30 | 0.099 | 0.087 | 32.05 | 0.224 | 71 s |
| NAFNet SIDD | 38.71 | 0.108 | 0.101 | 28.91 | 0.335 | 34 s |
| 1xDeNoise RealPLKSR | 31.47 | 0.195 | 0.182 | 26.84 | 0.315 | 86 s |
| OpenCV NL-means | 39.92 | 0.103 | 0.117 | 28.67 | 0.427 | CPU |

- The two SCUNet variants lead every camera-noise test. The GAN variant keeps fabric, foliage and
  skin texture that the PSNR variant smooths away, and it has the best perceptual scores on every
  set. That holds in side-by-side crops too, so it's the pick. The PSNR variant is the
  "smoothest" alternate, at the same speed and with the same manifest shape.
- NAFNet-SIDD and Restormer win only on SIDD, the smartphone set they were trained on (42.9 and
  42.5 dB vs SCUNet's 37.5). On DSLR noise, NAFNet over-smooths (detail ratio 0.53) and shifts color
  (ΔE 4.9 at ISO 12800).
- RealPLKSR adds halos (2.0) and lowers PSNR below the noisy input. It's dropped.

## Sharpen

None of the deblurring candidates helps with ordinary lens softness. The motion deblurrers are
trained on GoPro or REDS video blur. On Gaussian or defocus softness they score at or below the
blurred input, and NAFNet-GoPro adds color blotches (it's not a loading error: fp32 gives the same
result, and spandrel detects the architecture correctly). Running a real-world 2× super-resolution model and
area-downscaling back does much better, because those models are trained to undo exactly this kind
of softness:

| Model | Gauss σ1.2 PSNR / LPIPS | Defocus r2 PSNR / LPIPS | Motion 5px PSNR / LPIPS | Already-sharp photo PSNR / halo | 48 MP |
|---|---|---|---|---|---|
| blurred input | 29.98 / 0.269 | 30.06 / 0.256 | 30.12 / 0.229 | — | |
| **Real-ESRGAN x2 ↓2 + anchor** | **32.61 / 0.125** | **31.28 / 0.139** | 29.46 / **0.162** | 35.5 / 0.26 | 60 s |
| SwinIR-M real x2 ↓2 | 32.64 / 0.119 | 29.73 / 0.149 | 28.40 / 0.166 | 35.8 / 0.34 | 384 s |
| NomosUni x2 ↓2 | 29.69 / 0.284 | 29.80 / 0.269 | 29.84 / 0.236 | 40.3 / 0.01 | 61 s |
| Restormer motion | 29.01 / 0.257 | 29.36 / 0.244 | **31.57** / 0.180 | 42.2 / 0.22 | 80 s |
| Restormer defocus | 29.16 / 0.179 | 28.96 / 0.180 | 27.63 / 0.214 | 29.8 / 0.67 | 83 s |
| NAFNet REDS | 28.14 / 0.290 | 28.30 / 0.281 | 29.37 / 0.246 | 34.3 / 0.25 | 23 s |
| NAFNet GoPro | 25.53 / 0.339 | 25.57 / 0.320 | 27.16 / 0.255 | 26.4 / 5.56 | 26 s |
| Unsharp mask | 31.33 / 0.212 | 30.91 / 0.202 | 30.47 / 0.195 | 32.7 / 2.11 | CPU |

- Real-ESRGAN x2 ↓2 with the anchor is the pick. It ties SwinIR on quality at a sixth of the time,
  and has a fifth of unsharp masking's halo on already-sharp photos.
- Restormer-motion is still the best on true motion blur. If camera-shake sharpening becomes a
  feature, it's the candidate for that, but it isn't v1's "sharpen".
- NomosUni barely changes the image when used as a sharpener: it upscales cleanly but doesn't add
  acutance.

## 2× upscale

| Model | Clean PSNR / LPIPS / DISTS | Camera (noise + JPEG) PSNR / LPIPS / DISTS | Halo (clean) | 12 MP | 48 MP |
|---|---|---|---|---|---|
| **NomosUni ESRGAN + anchor** | **32.94 / 0.059 / 0.035** | **31.35** / 0.152 / 0.085 | 0.04 | 17 s | 61 s |
| SwinIR-M real GAN | 30.61 / 0.142 / 0.080 | 30.02 / **0.144 / 0.079** | 0.58 | 112 s | 384 s |
| SwinIR-M classical | 32.25 / 0.075 / 0.058 | 30.38 / 0.269 / 0.123 | 0.74 | 110 s | 365 s |
| HAT (ImageNet pretrain) | 31.80 / 0.081 / 0.059 | 30.16 / 0.272 / 0.125 | 1.02 | 396 s | ~40 min* |
| HAT-L (ImageNet pretrain) | 31.78 / 0.081 / 0.060 | 30.16 / 0.274 / 0.126 | 1.04 | ~24 min* | ~82 min* |
| Real-ESRGAN x2plus | 29.52 / 0.168 / 0.089 | 29.14 / 0.175 / 0.093 | 0.58 | 16 s | 60 s |
| Lanczos | 32.31 / 0.192 / 0.094 | 30.96 / 0.355 / 0.144 | 0.05 | | |

\* extrapolated from per-tile throughput.

- NomosUni is the pick. It has the best clean-input scores by a wide margin, almost no halos, and
  is competitive on noisy JPEGs. It's also 6× faster than SwinIR.
- The classical-SR transformers (SwinIR classical, HAT, HAT-L) sharpen sensor noise and JPEG blocks
  on camera-like inputs (LPIPS 0.27). On the Spark they're also very slow: HAT-L takes about 13 s
  per 512 px tile. The windowed attention is limited by memory bandwidth on GB10's LPDDR5X (most of
  the time goes to elementwise kernels on the attention tensors, not matmuls), and `torch.compile`
  made it slower, not faster. HAT-L doesn't pay for itself in quality either.
- **DRCT** is not in the table: its authors publish only ×4 weights, and no ×2 checkpoint exists.
- SwinIR-M real GAN is the alternate if upscaling noisy inputs matters more than speed.

## Caveats and follow-ups

- **No commercial reference yet.** Lightroom Denoise and Topaz don't run on Linux, so classical
  filters stand in for "existing tools". Next step: run 5 to 10 real high-ISO raws through Lightroom
  Denoise on the Mac, run the same renders through the picks, and compare side by side.
- **Most test photos are synthetic degradations.** PolyU is the real-noise check for denoise. Real
  soft or enlarged photos have no ground truth, so sharpen and upscale need a visual pass on real
  photographer images before release.
- **Tile context dependence.** NAFNet and Restormer use global channel attention, so a tiled output
  differs from a single pass by up to 57 to 113 levels in places (spread evenly, not at seams). The
  picks don't have this problem: their maximum differences are 7 to 32 levels, and their mean
  differences in the seam band are under 1 level.
- The latency table uses 32 px overlap. With the recommended 64 px, expect about 14% more time at 48 MP.
- Peak memory is the PyTorch allocator's peak. On GB10 that memory is unified with the host's
  128 GB, so even 48 MP to 192 MP upscales fit easily.
