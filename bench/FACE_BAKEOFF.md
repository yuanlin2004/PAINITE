# Face bake-off: results and picks

Run on spark1 (DGX Spark, GB10, aarch64, CUDA 13, PyTorch 2.14) on 2026-11-21. Scripts are in
[`face/`](face/), full tables are in [results/face/summary.md](results/face/summary.md), and
per-image numbers are in `results/face/*.csv`. Models were chosen on quality and speed only.
Licenses are recorded in the manifests and in `candidates.toml`.

The face op restores faces in ordinary photographer photos: portraits, group shots and events.
The pipeline detects faces on a downscaled copy of the full image, aligns each one to the FFHQ
template with 5 landmarks, restores the crop, then inverse-warps it and pastes it back with a soft
mask. Strength blends the pasted face with the original.

## Picks

| | Pick | Why |
|---|---|---|
| Restorer | **GPEN-BFR-1024** (yangxy/GPEN, 1024 px), bf16 | Best identity preservation everywhere a photographer would use it, best LPIPS and PSNR on mild degradation, and the only one that beats the untouched input at 512–768 px faces |
| Detector | **RetinaFace-ResNet50** (facexlib weights), on a copy with long side **2048**, threshold 0.9 | 14 of 14 faces in a 24 MP team photo, vs 11 at 1024 and 13 at full resolution; deterministic; 0.14 s |
| Paste mask | **Feathered square** (the adapter's `crop_mask`, inset 0.05, feather 0.12) | No visible seams with GPEN, no extra model. Face parsing gave no visible gain (details below) |
| Color anchor | **On**, σ = 16 crop px at 1024 (8 at 512 scale) | +0.75 dB PSNR on the synthetic set and +0.007 identity on real photos, at no cost |
| Face size policy | **min_face 128, max_face 1024** (crop side in source px) | Below 128, every model invents a different person. Above 1024, the original already has more true detail than any restorer |
| Default strength | **0.5** | On mild degradation, s = 0.5 beats s = 1 on PSNR, DISTS and identity, at nearly the same LPIPS |
| CodeFormer w | Not the pick. If CodeFormer is used, **w = 0.8** (identity 0.909 vs 0.809 at w = 0.5) | |

Latency on spark1, server-side inference only (image decoded, host↔device copies included):

| | GPEN-1024 fp32 | GPEN-1024 bf16 |
|---|---|---|
| Detection, 24 MP (long side 2048) | 0.14 s | — |
| Restore one face | 0.105 s | 0.056 s (0.063 s in the server image) |
| 24 MP, 1 face (detect + restore + paste) | 0.75 s | |
| 24 MP, 10 faces | 2.7 s | |
| Preview, 1024×1024 region (detection cached) | 0.20 s | |
| Peak GPU memory, one face | 1.9 GB | 1.2 GB |

The 10-face figure uses the bench pipeline, which batches 8 crops at a time (peak 12 GB). The
server adapter restores one face at a time, so its peak is the one-face figure. fp16 overflows in
GPEN-1024's StyleGAN2 decoder (non-finite output), so use bf16, not fp16. bf16 differs from fp32
by 0.4 levels on average (max 17 of 255) and is half the time.

The manifests are in [`manifests/face/`](manifests/face/): the pick and a spandrel-loadable
runner-up (CodeFormer w = 0.8) that the current server adapter can run today.

## Candidates

| Candidate | Status |
|---|---|
| GFPGAN v1.4 | spandrel. Run with `randomize_noise=False`, otherwise output is random |
| RestoreFormer | spandrel |
| RestoreFormer++ | Spandrel detects the checkpoint but builds the wrong arch: it lacks the extra 32 px decoder attention. Ran via the upstream `vqvae_arch.py`, which is plain PyTorch (`face/restorers.py:RestoreFormerPP`) |
| CodeFormer w = 0.5, 0.8 | spandrel_extra_arches |
| GPEN-BFR-512 / 1024 / 2048 | Upstream `gpen_model.py` with its CUDA ops replaced by their pure-PyTorch fallbacks (`face/restorers.py:_gpen_ops`). The 1024 and 2048 weights are only on ModelScope now: the original Aliyun links return AccessDenied |
| PMRF (2024) | Upstream HDiT and SwinIR, without Lightning. natten has no aarch64/CUDA 13 wheel, so `face/pmrf.py` implements `na2d_qk`/`na2d_av` in PyTorch. 25 flow steps, fixed seed |
| OSDFace (CVPR 2025, one-step diffusion) | Upstream encoder, LoRA merged into the SD 2.1-base UNet (`face/osdface.py`), fp16, VAE posterior mean so output is deterministic. Weights come from the authors' Google Drive |
| DiffBIR v2 face | Not run. OSDFace covers "newer diffusion restorer", and DiffBIR needs a 5 GB SD checkpoint plus its own sampler stack for a slow reference. Time-boxed out |

## Test sets

1. **Synthetic, with ground truth.** The first 100 faces of the CelebA-HQ validation split
   (HF `mattymchen/celeba-hq`), area-downscaled to 512 px (`face/make_testset.py`). None of the
   candidates trained on CelebA-HQ; they trained on FFHQ.
   - **Mild** (the photographer case): Gaussian blur σ 0.5–1.5, 2× down/up, noise σ 1–4, JPEG q75–90.
   - **Heavy** (CelebA-Test style, GFPGAN's recipe): blur σ 0.1–10, downscale 0.8–8×, noise 0–20, JPEG 60–100.
2. **Real photos.** 19 Wikimedia Commons photos, 4–6.7k px wide, ISO 100–1600, all CC licensed.
   There are 5 large-face portraits (face crops 935–2750 px), 3 half-body, 4 events and 7 group
   shots with 5–31 faces; 127 faces have eyes at least 16 px apart. The list is in
   [`face/real_photos.txt`](face/real_photos.txt), and `face/fetch_real.py` downloads them and
   records the URL, license, author and ISO in `sources.csv` next to them.
3. **Large faces.** CelebA-HQ at its native 1024 px, area-resized to face sizes of 512, 768 and
   1024 px (1×, 1.5× and 2× a 512 restorer). There are two inputs at each size: "soft" (blur
   σ 1 px, noise σ 2, JPEG q90) and "clean" (JPEG q95 only). See `face/largeface.py`.

**Identity** is the cosine similarity of ArcFace embeddings (insightface buffalo_l `w600k_r50`),
with both faces cropped using the same landmarks. On synthetic faces it's measured against the
ground truth; on real photos, against the same face in the input. On real photos, 1.0 means
unchanged, and that's not automatically good: a restorer has to change something. The question
is whether it changes who the person is. Typical same-person pairs score above 0.5, but a
photographer notices drift long before that, so treat a drop from 0.97 to 0.90 as significant.

## Restorer

Synthetic set, 100 faces (higher is better for PSNR and identity, lower for LPIPS and DISTS):

| Model | Mild PSNR | Mild LPIPS | Mild DISTS | **Mild identity** | Heavy LPIPS | Heavy identity | Restore one face |
|---|---|---|---|---|---|---|---|
| degraded input | 33.58 | 0.236 | 0.151 | 0.983 | 0.600 | 0.737 | |
| **GPEN-1024 + anchor** | **31.56** | **0.134** | 0.131 | **0.970** | 0.271 | 0.666 | 0.105 s |
| GPEN-1024 | 30.81 | 0.136 | 0.133 | 0.968 | 0.265 | 0.657 | 0.105 s |
| GPEN-2048 | 30.55 | 0.154 | 0.140 | 0.966 | 0.313 | 0.630 | 0.20 s |
| GPEN-512 | 28.28 | 0.167 | 0.141 | 0.955 | 0.261 | 0.668 | 0.058 s |
| PMRF | 30.47 | 0.150 | **0.115** | 0.949 | 0.224 | **0.702** | 4.6 s |
| RestoreFormer++ | 28.15 | 0.170 | 0.132 | 0.910 | 0.221 | 0.699 | 0.23 s |
| CodeFormer w = 0.8 | 30.44 | 0.147 | 0.137 | 0.909 | **0.210** | 0.640 | 0.107 s |
| GFPGAN 1.4 | 30.53 | 0.145 | 0.148 | 0.876 | 0.230 | 0.634 | 0.056 s |
| CodeFormer w = 0.5 | 29.27 | 0.162 | 0.150 | 0.809 | 0.216 | 0.585 | 0.107 s |
| OSDFace | 20.68 | 0.289 | 0.202 | 0.801 | 0.303 | 0.588 | 0.21 s |
| RestoreFormer | 26.59 | 0.183 | 0.146 | 0.779 | 0.229 | 0.637 | 0.12 s |

Real photos, 127 faces, full pipeline:

| Model | MUSIQ | CLIP-IQA | Identity, all | crops 128–256 px | 256–768 px | ≥ 768 px |
|---|---|---|---|---|---|---|
| input | 46.9 | 0.486 | 1 | | | |
| **GPEN-1024 + anchor** | 61.0 | 0.566 | **0.973** | **0.971** | **0.995** | **0.979** |
| GPEN-2048 | 61.2 | 0.548 | 0.969 | | | |
| GPEN-1024 | 61.0 | 0.566 | 0.966 | | | |
| PMRF | 55.1 | 0.520 | 0.952 | | | |
| GPEN-512 | 61.7 | 0.577 | 0.940 | | | |
| CodeFormer w = 0.8 + anchor | 59.4 | 0.570 | 0.941 | | | |
| RestoreFormer++ | 56.5 | 0.542 | 0.920 | | | |
| CodeFormer w = 0.8 | 59.7 | 0.576 | 0.917 | | | |
| GFPGAN 1.4 | 60.3 | 0.568 | 0.897 | | | |
| OSDFace | **65.7** | **0.656** | 0.798 | | | |

(Per-size breakdowns for every model are in [summary.md](results/face/summary.md).)

- **GPEN-1024 is the pick.** It keeps the person's identity better than anything else on mild and
  real faces (0.970 synthetic, 0.973 real vs 0.917 for CodeFormer w = 0.8 and 0.897 for GFPGAN). It
  has the best mild LPIPS and PSNR, and it's fast. Restoring at 1024 px also means the typical
  250–700 px face in a 24 MP photo is restored at or above its own resolution, which matters for
  the large-face results below.
- GFPGAN and CodeFormer act like a beauty filter: they smooth skin, shrink pores and slightly
  reshape features. That shows up as their identity drop, and you can see it in
  [real_midsize_faces.png](results/face/real_midsize_faces.png) (columns: input, GFPGAN,
  CodeFormer 0.8, PMRF, GPEN-1024 + anchor).
- **PMRF** is the best on heavy damage (identity 0.702) and has the best mild DISTS. At 4.6 s per
  face it costs 46 s for a 10-face group photo. It's worth revisiting as an "old photo / heavy"
  mode, not as the default.
- **OSDFace** gets the highest no-reference scores and the worst identity: it invents texture,
  glasses and contrast (PSNR 20.7 on mild). That's the opposite of what photographers want.
- **GPEN-2048** isn't better than 1024 at any size we tested and takes twice the time and memory.
- On heavy damage, CodeFormer w = 0.8 has the best LPIPS, but GPEN-1024's identity there (0.666) is
  still level with CodeFormer's (0.640), and heavy damage isn't the target use.

## Face size policy

**Small faces.** Below a crop of about 128 px (eyes under ~32 px apart), every model invents a
different person, because there isn't enough signal. Real identity on crops under 96 px: GFPGAN
0.47, GPEN-512 0.46, GPEN-1024 0.73. There were no faces between 96 and 128 px; at 128–192 px
GPEN-1024 is back at 0.947. In group photos, the smallest faces are usually also out of focus on
purpose (shallow depth of field), and "restoring" them to sharp faces looks wrong. See
[real_tiny_face.png](results/face/real_tiny_face.png): the same 77 px bokeh face becomes three
different men, one of them wearing glasses. **min_face = 128.**

**Large faces.** Large-face test, soft input (σ 1 px at the face's own size), LPIPS / identity:

| Face size | input | GPEN-1024, s = 1 | GPEN-1024, s = 0.5 | CodeFormer@512, s = 1 | CodeFormer@512 low-freq only |
|---|---|---|---|---|---|
| 512 | 0.128 / 0.991 | 0.105 / 0.982 | **0.087 / 0.990** | 0.141 / 0.913 | |
| 768 | 0.095 / 0.996 | 0.130 / 0.990 | **0.085** / 0.995 | 0.186 / 0.913 | 0.151 / 0.928 |
| 1024 | **0.085** / 0.998 | 0.174 / 0.991 | 0.107 / 0.996 | 0.226 / 0.909 | 0.154 / 0.926 |

- At a fixed blur in pixels, a bigger face is relatively sharper, so there's less to restore and
  more real detail to lose. Restoring at 512 and upscaling is clearly harmful above 512 px. Even
  "low-frequency only" (keep the original's detail above the restorer's Nyquist and take only
  coarse shape and tone from the restorer) is worse than the input, because the restorer's
  coarse changes are themselves the identity drift.
- GPEN-1024 at s = 0.5 still helps at 768 px (0.085 vs 0.095) and is a net loss at 1024 px
  (0.107 vs 0.085), although identity barely moves (0.996).
- On the portraits (face crops 935–2750 px), every model replaces real pores, stubble and wrinkles
  with generic skin. See [large_face_priest.png](results/face/large_face_priest.png) (input,
  GFPGAN, CodeFormer, GPEN-1024; the face crop is 2270 px).
- **max_face = 1024**: leave larger faces untouched. A later refinement could fade strength from
  768 to 1024 px instead of a hard cutoff; that needs a small adapter change.

## Paste mask

I compared four masks on 21 faces in 11 photos (`face/masks.py`, crops in
`$PAINITE_BENCH_DIR/out/face/masks/`):

- **square**: the adapter's `crop_mask`, inset 0.05, smoothstep feather 0.12.
- **rect**: facexlib/GFPGAN's default, an eroded square with a Gaussian feather.
- **ellipse**: a feathered face oval.
- **parse**: a parsenet face-parsing mask (skin, brows, eyes, ears, nose, mouth; no hair, neck or background), eroded and feathered.

The seam score compares the change in gradient along the mask's transition band with the change
inside the face; lower means a less visible edge:

| Mask | Seam ratio with GPEN-1024 | Seam ratio with CodeFormer 0.8 |
|---|---|---|
| rect | **0.47** | **0.49** |
| square | 0.51 | 0.53 |
| ellipse | 0.54 | 0.57 |
| parse | 0.64 | 0.68 |

At 1:1 on hair, ear and beard edges, none of the masks shows a visible seam with GPEN-1024.
**rect** reaches almost to the crop border. With CodeFormer, the restorer's junk at the crop edge
then shows through in the hair (the top-left of a sports portrait). **parse** keeps the original
hair and background, which is its point. But its transition runs right along the jaw and hairline,
where it scores worst, and it adds an 85 MB model plus a facexlib/opencv dependency that doesn't
install cleanly in the server image (below). **Pick: the adapter's square mask.** Parsing is the
fallback if a future restorer hallucinates hair or background.

## Strength

`out = input + s × (restored − input)` inside the mask, from `face/strength.py`:

| GPEN-1024 + anchor | s = 0.25 | s = 0.5 | s = 0.75 | s = 1 |
|---|---|---|---|---|
| Mild: PSNR / LPIPS / DISTS / identity | 33.98 / 0.168 / 0.121 / 0.984 | 33.61 / 0.131 / **0.108** / 0.981 | 32.70 / **0.126** / 0.117 / 0.977 | 31.56 / 0.134 / 0.131 / 0.970 |
| Heavy: LPIPS / identity | 0.450 / 0.731 | 0.339 / 0.700 | 0.285 / 0.678 | **0.271** / 0.666 |

**Default 0.5.** Strength means how much of the restored face replaces the original: 0 leaves the
photo untouched, 1 is the full restorer output. 0.5 is the best balance on the photographer case.
Users with genuinely damaged faces (old scans, heavy blur) should raise it to 1.

## Detector

- RetinaFace-R50 on the 24 MP team photo (6000×4000): 11 faces at long side 1024, **14 at 2048**,
  14 at 4096, and 13 at full resolution (the full-resolution pass misses one large face because
  the anchors are tuned for smaller faces). Detection times: 0.03, **0.14**, 0.85 and 1.4 s.
- It's deterministic: two runs give identical boxes and landmarks (difference 0.0 px). All 100
  CelebA-HQ ground-truth faces were detected.
- Smallest face found at 2048: eye distance 17 px in the full image. With min_face = 128 (eye
  distance ≈ 32 px), detection at 2048 has margin to spare.

## Server adapter check

`face/adapter_check.py` runs the server's `FaceModel`
(`server/painite_server/models/face_adapter.py`) and the bench pipeline with the same settings
(CodeFormer w = 0.8, square mask, anchor σ 8, fp32) on 5 photos with 27 faces:

- **Same faces.** The landmarks agree to within 1.8 px. The difference comes from the downscale
  used for detection.
- **Same pixels, nearly.** Inside the face boxes, the mean difference is 1.0–1.8 levels (p99: 8–16)
  and PSNR is 37–42 dB. The two implementations are equivalent.
- **Bug: CodeFormer with `dtype: fp16` crashes.** CodeFormer doesn't support half precision, so
  the adapter falls back to bf16, and `CodeFormer.quantize.get_codebook_feat` then multiplies a
  float32 one-hot by bf16 embeddings and fails. Either run CodeFormer in fp32 (the runner-up
  manifest says fp32) or cast the codebook lookup.

**What the adapter needs for GPEN (spandrel can't load it):**

1. The GPEN generator: `FullGenerator` and its StyleGAN2 `Generator`, `ConvLayer`, `StyledConv`,
   `ModulatedConv2d`, `EqualLinear`, `ToRGB`, `NoiseInjection` and `Blur` from yangxy/GPEN
   `face_model/gpen_model.py`. That's about 750 lines of plain PyTorch. Its `op` package compiles
   CUDA extensions at import, so replace it with the pure-PyTorch `fused_leaky_relu` and
   `upfirdn2d` used in `face/restorers.py:_gpen_ops`. Nothing gets compiled, and it runs on the
   server image's NGC torch 2.9 in bf16 at 0.063 s per face (tested in a throwaway container from
   `painite-server:latest`).
2. Build it with `FullGenerator(1024, 512, 8, channel_multiplier=2, narrow=1)` and
   `load_state_dict(strict=True)` on `GPEN-BFR-1024.pth`, which is a plain state dict. Feed it
   input in [−1, 1]; output `[0]` is also in [−1, 1]. It's deterministic: the noise inputs come
   from the encoder features, not from `randn`.
3. `restorer_args.arch = "gpen"` selects it; `size`, `channel_multiplier` and `narrow` are
   constructor arguments. `face_size: 1024` makes the existing align/paste code work unchanged.
4. Set `color_anchor_sigma` in crop pixels: 16 at face_size 1024 matches σ 8 at 512.
5. `default_strength` is a new key (0.5); the adapter doesn't read it yet.
6. Licensing: yangxy/GPEN has no license file; the ModelScope model card says Apache-2.0.
   Vendoring `gpen_model.py` into the server needs a decision (recorded, not filtered).

## Packages

The pick needs **no packages beyond the server image**: torch and numpy for GPEN, and the
adapter's native RetinaFace port. facexlib isn't needed.

facexlib was checked anyway, for a parse-mask variant. `pip install facexlib==0.3.0` in
`painite-server:latest` (NGC 25.10, torch 2.9.0a0, numpy 2.1) succeeds and leaves torch in place,
but it pulls in `opencv-python`, whose import fails there (`libGL.so.1` missing). Using it would
need `opencv-python-headless` plus `pip install --no-deps facexlib filterpy`, or `libgl1` via apt.

Bench-only extras are in [`face/requirements.txt`](face/requirements.txt): facexlib (detection
and parsing in the harness), insightface plus onnxruntime (ArcFace metric), pyarrow (CelebA-HQ
parquet), and diffusers plus pytorch-lightning (OSDFace).

## Caveats and follow-ups

- The ArcFace identity on a mildly degraded input (0.983) is higher than any restorer's. ArcFace
  works at 112 px and barely sees mild softness. The metric measures drift, not improvement;
  LPIPS, DISTS and PSNR against the ground truth carry the improvement signal.
- CelebA-HQ ground truth is itself partly upscaled from CelebA, so "clean 1024" isn't true 1024 px
  detail. The large-face conclusions agree with the real portraits, though.
- Real photos have no ground truth. Their identity score is drift from the input, and the
  no-reference scores (MUSIQ, CLIP-IQA) reward invented sharpness, as OSDFace shows.
- A visual review by a photographer on their own portraits and event shots should precede release,
  in particular for skin texture at 256–768 px faces with s = 0.5.
- Possible later modes: PMRF for heavy damage or old photos (slow), and a max_face fade instead of
  a hard cutoff.
