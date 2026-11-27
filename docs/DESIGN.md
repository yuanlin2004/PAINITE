# PAINITE design

**Status** — Draft for review · **Covers** — First release · **Requirements** — [PRD](PRD.md)

This document turns the PRD into an architecture, a service contract, and a build order. Decisions marked **Proposed** are recommendations waiting for review. Items marked **Spike** need a prototype before they can be settled.

## 1. Architecture

```
 ┌───────────────────────┐        ┌──────────────────────────────────────────────┐
 │ PAINITE.app (macOS)   │        │ painite-server on DGX Spark (spark1)         │
 │  SwiftUI, Lightroom   │──┐     │                                              │
 │  "Edit in" handoff    │  │     │  ┌──────────┐   ┌───────────┐   ┌─────────┐  │
 └───────────────────────┘  │HTTPS│  │ API      │──▶│ Job queue │──▶│ GPU     │  │
                            ├────▶│  │ (FastAPI)│   │ (SQLite)  │   │ worker  │  │
 ┌───────────────────────┐  │     │  └──────────┘   └───────────┘   └─────────┘  │
 │ painite CLI           │──┘     │        │ image I/O, color, metadata, tiling  │
 │  macOS + Linux        │        │        ▼                                     │
 └───────────────────────┘        │  /var/lib/painite (sources, work, results)   │
                                  └──────────────────────────────────────────────┘
```

**Proposed: the server owns all image decoding, color handling, metadata, and encoding.** Clients upload the original file bytes and download a finished file. This gives one processing path for every mode (a PRD requirement), keeps the CLI small and identical across platforms, and means color and metadata fidelity is implemented and tested once.

The Mac app does its own decoding only for on-screen display of the original. The enhanced preview it shows comes from the server.

## 2. Components and technology

| Component | Proposed stack | Why |
| --- | --- | --- |
| Inference server | Python 3.12, FastAPI + Uvicorn, PyTorch (CUDA, ARM64) in an NVIDIA NGC container | Model candidates ship as PyTorch; NGC containers support the Spark's GB10 (ARM64 + Blackwell). PyTorch is the reference path; TensorRT engines and `torch.compile` are an optional, checked acceleration on CUDA (section 5). |
| Image I/O on server | pyvips (libvips) for decode/encode and ICC, tifffile for 16-bit TIFF edge cases, exiv2 (via pyexiv2) for EXIF/IPTC/XMP | libvips handles large images in bounded memory and preserves ICC; exiv2 gives field-level metadata control. |
| Job store | SQLite (WAL) in the server's data directory | One machine, one active job; no external database to run. |
| CLI | Go, single static binary | Cross-compiles to darwin/arm64, linux/amd64, linux/arm64 from one toolchain; easy `.deb` packaging; no runtime to install. |
| Mac app | Swift 6, SwiftUI with AppKit where needed, Core Image / ImageIO for display | Native color management (ColorSync) for the before/after view; native file-open handoff for Lightroom. |
| Shared contract | OpenAPI 3.1 spec in `api/openapi.yaml` | Generates Go and Swift clients and server validation from one source. |

## 3. Repository layout

One monorepo, so the contract, server, and clients change together.

```
PAINITE/
  api/openapi.yaml            # the job contract (source of truth)
  server/                     # painite-server (Python)
    painite_server/
      api/                    # HTTP routes, auth
      jobs/                   # queue, scheduler, retention sweeper
      imaging/                # decode, color, metadata, tiling, encode
      models/                 # model registry and adapters (denoise, sharpen, upscale)
    tests/
    Dockerfile
  cli/                        # painite CLI (Go)
  mac/                        # PAINITE.app (Xcode project)
  bench/                      # model bake-off and latency benchmarks
  testdata/                   # small fidelity fixtures (tagged/untagged JPEG, rotated, 16-bit wide-gamut TIFF)
  docs/                       # PRD, design, logos
```

## 4. Service contract (v1)

All routes are under `/v1`, over HTTPS, with a bearer token. JSON for control, raw bytes for images.

| Method and path | Purpose |
| --- | --- |
| `GET /health` | Liveness, GPU status, queue depth, each model's inference backend. No auth required beyond TLS. |
| `GET /capabilities` | Operations, parameter ranges, model IDs and versions, input/output limits, supported formats. |
| `PUT /sources/{sha256}` | Upload an original. Content-addressed by its SHA-256, so a re-upload is a no-op. Returns `source_id`. |
| `POST /jobs` | Create a preview or full job. Requires an `Idempotency-Key` header (the PRD's request key). |
| `GET /jobs/{id}` | Status: `queued`, `running`, `completed`, `failed`, `canceled`, with progress and error code. |
| `POST /jobs/{id}/cancel` | Cancel a queued or running job. |
| `GET /jobs/{id}/result` | Download the output. Supports HTTP `Range` so interrupted downloads resume. |
| `DELETE /jobs/{id}/result` | Delete the output early. |

Job request:

```json
{
  "source_id": "sha256:…",
  "kind": "preview | full",
  "operation": "denoise | sharpen | upscale2x",
  "params": { "strength": 0.6 },
  "model": "denoise/nafnet-sidd@1.0.0",
  "region": { "x": 1200, "y": 800, "w": 1024, "h": 1024 },
  "output": { "format": "jpeg | tiff", "jpeg_quality": 92 }
}
```

- `model` is pinned by the client. Preview and export send the same value, which guarantees they use the same model version. If it is no longer installed, the server returns `model_unavailable` and the app asks the user to regenerate the preview (PRD).
- `region` is only for previews, in source pixels of the displayed (orientation-applied) image. The server maps it to stored pixels and adds tile padding so the preview matches the same region of a full export.
- Errors use stable codes (`unsupported_format`, `too_many_pixels`, `upload_too_large`, `model_unavailable`, `result_unavailable`, `unauthorized`, `queue_full`, …). The CLI's JSON output reuses them.
- Idempotency: the key is scoped to the token's owner. Same key and same body returns the existing job, during the job's life and for 24 hours after it ends, even if its result was deleted early. Same key with a different body returns `409`.

## 5. Server internals

**Scheduling.** One GPU worker runs one job at a time. The queue is bounded (start at 32). When the worker picks the next job, queued previews go first, then full jobs in arrival order. A newer preview from the same client for the same source supersedes older queued previews, which are canceled automatically. Concurrency above one only changes after benchmarks show it helps throughput without hurting preview latency.

**Processing pipeline.**
1. Decode with libvips. Keep the stored pixel orientation; record the EXIF orientation tag.
2. Convert to float32 linear-ish working values in the source's own color space. Images are not converted to sRGB, so same-color-space output and the ICC profile are retained. Untagged inputs are treated as sRGB (**Proposed**, the PRD asks for this assumption to be stated).
3. Tile with overlap (start at 512 px tiles, 32 px overlap, feathered blend) so 48 MP inputs fit predictably in memory and preview crops reproduce the export exactly. Preview regions are snapped to the tile grid of the full image, which makes preview-to-export differences zero by construction, apart from floating-point nondeterminism (deterministic CUDA kernels are enabled).
4. Run the model. **Strength** for denoise and sharpen is a blend: `out = input + strength × (model(input) − input)`. One model run per tile serves every strength, so the app can cache the model output and re-blend for slider changes. Upscale has no strength in v1.
   **Face enhancement** is the exception to tiling, because a restorer needs the whole face and detection needs the whole image. The server detects faces once per source on an upright (orientation-applied) downscale of the full image, aligns each face to the FFHQ 512 template, restores the aligned crop, and pastes it back with a soft mask; pixels outside face masks are untouched. A preview restores only the faces whose box touches its region, and each face's paste is computed over that face's own box in full-image pixels, so preview and export still match exactly. Strength blends each restored face with the original. Model choice and the face-size policy are in `bench/FACE_BAKEOFF.md`.
   **Accelerated backends** (CUDA only; `server/painite_server/models/accel.py`). A model's manifest lists the backends to try in order (`config.accel`: TensorRT FP16, `torch.compile`; PyTorch is always the last resort). The catalog asks for TensorRT for denoise, sharpen, upscale2x and unblur_focus. For unblur_motion it asks for TensorRT first, then `torch.compile`. Stripformer is built as two TensorRT engines split after its last decoder attention block (`split_after`): as one FP16 engine, TensorRT on a GB10 fuses those blocks with the full-resolution conv tail and computes it wrong (24 dB, no error), while split it is correct (60.8 dB on the check) and 1.2x faster than `torch.compile`. The face model stays on PyTorch. Engines are built on the device in a background subprocess the first time a model loads, under a time limit (never BF16), and cached in `<data_dir>/engines/`, keyed by the weights' SHA-256, model config, tile-window shape, precision, PyTorch/TensorRT versions and GPU, so an upgrade rebuilds them once. Before use, every engine is checked against PyTorch FP32 on four fixed synthetic tiles: it must be finite, reach 50 dB PSNR (or come within 6 dB of today's PyTorch FP16 path where that path itself is lower), and keep its max difference within twice that path's. A failure is logged and recorded, and the next backend in the list is tried; on later starts recorded failures are skipped. Failures that may not repeat (out of GPU memory, allocation errors, the time limit) are retried on the next start, two builds in all; `PAINITE_ACCEL_RETRY=1` retries everything once, and deleting `<data_dir>/engines/` starts over. PyTorch also serves while engines build, for windows of other shapes, and after any run-time error. A finished engine switches in between jobs, never during one. The backend is part of the tile cache key, so a preview made on PyTorch is never stitched into an export made by TensorRT. `PAINITE_ACCEL=off` turns this off; `PAINITE_ACCEL=tensorrt` or `=compile` allows only one backend. On a DGX Spark a 24 MP export is 1.3–2.0x faster, and output stays within 58–67 dB PSNR of the PyTorch path.
5. Encode to the requested format at the source's bit depth (16-bit TIFF stays 16-bit), embed the source ICC profile, copy EXIF/IPTC/XMP, keep the orientation tag, and update pixel-dimension tags (and resolution for 2×).
6. Write to a temp file, validate by re-opening it, then atomically rename into `results/`. Only then mark the job completed.

**Storage and retention.** `/var/lib/painite/{sources,work,results,db}`. Results are kept 24 hours after completion. Sources and working files are deleted when their last job ends. A sweeper runs every minute and on startup (crash cleanup), and the server refuses new uploads with `disk_full` above a configured cap.

**Security.** `painite-server admin init` creates a private CA and a server certificate for the Spark's LAN name. `painite-server token create --name studio-mac` prints a token; only its hash is stored, and `token revoke` disables it. Clients pin the CA fingerprint on first configuration. Every job, source, and result is owned by a token; other tokens get `404`. Logs record job IDs, sizes, and timings, never pixels, file names, or tokens.

**Deployment.** One Docker image based on an NGC PyTorch container, run with `--gpus all` and a systemd unit on spark1. Model weights live in a versioned directory (`/var/lib/painite/models/<op>/<name>@<version>/` with a `manifest.json` and SHA-256), so several versions can be installed side by side. PAINITE does not ship weights: each model's manifest names its upstream download URL, expected SHA-256, and license, and `painite-server models pull <model>` downloads it on the operator's machine after showing that license. The image ships only the configuration and pipeline code.

## 6. CLI

```
painite enhance --op denoise --strength 0.6 -o out/ IMG_001.jpg IMG_002.tif
painite enhance --op upscale2x --json --overwrite -o big.tif small.tif
painite config set endpoint https://spark1.local:8443
painite status
```

- All validation (formats, collisions, overwrite, source-as-destination) runs before any upload. Collisions exit `2` even with `--overwrite`.
- Files are processed through the server's queue; the CLI keeps up to 4 uploads in flight and polls job status.
- `--json` prints one result per input in argument order on stdout. Progress and diagnostics go to stderr. Exit `0` all succeeded, `1` any failed, `2` invalid invocation.
- Writes go to a temp file in the destination directory, then rename, so failed writes leave existing destinations intact.
- Releases: `.deb` for amd64 and arm64 built with GoReleaser plus SHA-256 checksums; a signed, notarized binary for macOS.

## 7. Mac app

- **Open and preview.** On open, the app uploads the source in the background (content-addressed, so reopening is free) and requests a preview for the visible region at the current settings. Before/after views share one zoom and pan state. Each new preview request gets a sequence number; results older than the latest are discarded.
- **Export.** Sends a `full` job with the same model ID and parameters, shows progress, supports cancel, and keeps the source for retry.
- **Lightroom handoff.** Registers as an editor for JPEG and TIFF and handles `application(_:open:)` for cold and warm launch. **Save** is enabled only when the handoff file is verified writable and its format is supported; otherwise **Save As**. Direct save writes a temp file beside the handoff copy and atomically replaces it.
- **Spike (first, before app UI work):** a minimal app that accepts the Lightroom 10.0 handoff and logs format, bit depth, ICC profile, path, and writability for an edited RAW and an edited JPEG from the cloud library, cold and warm. Its results finalize the format matrix and the Save rules.

## 8. Model selection

Run a bake-off on spark1 under `bench/` before committing to models. Large diffusion-based restorers are left out of v1 because of the preview latency target. Pick the best open model for each operation on quality and performance alone. License does not limit the choice, because operators download weights themselves (see Deployment). Candidates:

| Operation | Candidates to evaluate |
| --- | --- |
| Denoise | NAFNet (SIDD), Restormer (real denoising), SCUNet, and variants trained on larger datasets |
| Sharpen | NAFNet / Restormer deblurring variants, and a classical detail-enhance baseline |
| 2× upscale | Real-ESRGAN x2, SwinIR, HAT / HAT-L, DRCT |

Score each on the PRD's test set (high-ISO renders, skin, foliage, text, hard edges) for detail loss, halos, color shifts, and tile seams, against reference outputs from existing tools, and record latency and peak memory at 12, 24, and 48 MP on the Spark. Record each chosen model's license and download source in its manifest. Redistribution terms are resolved before release, not during selection (see open question 5).

## 9. Testing

- **Fidelity fixtures** in `testdata/`, checked automatically by the server tests: tagged and untagged JPEG, non-default EXIF orientation, wide-gamut 16-bit TIFF. Assertions cover ICC bytes, bit depth, dimensions, orientation, and preserved metadata fields.
- **Contract tests** run the CLI against a server that uses an identity "model", so the whole job flow is tested without a GPU in CI.
- **Preview-export match test**: render a preview region and a full export and compare the matching pixels against the agreed tolerance.
- **Failure tests**: kill the server mid-job, fill the disk, cancel during download, and compare source hashes before and after.
- **Benchmarks** in `bench/` produce the latency table in the PRD's acceptance section. Clients reach spark1 over Wi-Fi (the test client mm2014 sees 7 to 88 ms round trips), so every job records server-side timings (queue, decode, inference, encode) in its status, and the benchmark reports them separately from client-measured upload and download time.

## 10. Build order

| Milestone | Delivers | Exit check |
| --- | --- | --- |
| M0 Spikes | Model bake-off on spark1; Lightroom handoff spike on the Mac | Models chosen and manifests written; handoff format matrix recorded |
| M1 Server + CLI | Contract, server with identity model then real models, CLI, `.deb` builds | CLI acceptance checks from the PRD pass on all three platforms |
| M2 Mac app | Standalone open, preview, export | Standalone GUI check passes; preview p95 measured |
| M3 Lightroom | Edit in, Save / Save As, Finish / Browse | Lightroom round-trip checks pass cold and warm |
| M4 Hardening | Security, retention, failure recovery, benchmarks | Full PRD release-acceptance table passes |

M0's two spikes are independent and can run in parallel. M1 can start on the identity model before M0 finishes.

## 11. Open questions

1. Does spark1 need to be reachable outside the LAN? The design assumes LAN-only (Wi-Fi) with TLS.
2. Is a 2× upscale with denoise in one pass wanted, or is each operation strictly separate in v1? The design keeps them separate, as the PRD states.
3. Default JPEG quality (proposed 92, 4:4:4 chroma) and which EXIF/IPTC fields to drop, if any (proposed: keep all, drop embedded thumbnails and regenerate none).
4. macOS distribution: Developer ID with notarization outside the App Store (assumed), since Lightroom handoff and arbitrary file writes are simpler outside the sandbox.
5. Model licensing. Models are chosen on quality and performance alone, and users download weights directly. Before release, confirm that each chosen model's license allows this use, and whether runtime dependencies bundled in the image need their own notices.
