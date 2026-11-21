# PAINITE product requirements

**Status** — Draft for review · **Target** — First release

## Product goal

PAINITE helps photographers reduce noise, improve detail, and enlarge photos for delivery or print. Photographers can work in a standalone Mac app, automate processing from the terminal, or send a photo from Adobe Lightroom desktop and bring the enhanced result back into the same editing workflow.

The macOS app handles file selection, previews, and saving. The CLI runs independently on macOS and Linux (x86-64 and ARM64). Both clients use the same inference service over the network. NVIDIA DGX Spark is the initial deployment target. [NVIDIA DGX Spark system overview](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html)

## Proposed first-release scope

| Area | Requirement |
| --- | --- |
| Enhancement | Offer denoise, sharpen, and upscale. Provide an adjustable strength for denoise and sharpen and a 2× upscale option. Offer each operation independently. |
| Still-image formats | Read and write SDR RGB JPEG and TIFF, including 16-bit-per-channel TIFF. Validate the actual Lightroom handoff format before finalizing the compatibility matrix. |
| Color and metadata | Render a color-managed preview and retain the embedded ICC profile for same-color-space output. Keep 16-bit TIFF output at 16 bits per channel, preserve supported EXIF/IPTC fields, and ensure output orientation and dimension tags match the pixels. |
| Originals | Write standalone results to separate paths. For Lightroom, replace only a verified writable handoff copy when the user explicitly saves the result. |
| macOS | Require macOS 27 or later on Apple silicon (ARM64) for the desktop app, Lightroom integration, and CLI. |
| Linux | Support the standalone CLI on Ubuntu 24.04 or later on both x86-64 and ARM64. |
| Inference server | Deploy the inference service separately on NVIDIA DGX Spark. |

Direct camera RAW development, HDR, layered files, video, and on-device inference are outside the proposed first release. Lightroom-rendered copies of RAW photos are in scope once the handoff is validated.

## Standalone Mac app

1. The photographer configures the inference endpoint and can check its connection and model availability from the app.
2. They open a supported photo from Finder or the app, choose an operation, and adjust the available settings.
3. They compare the before and after views at a matched zoom and position. A newer preview request supersedes older results so the displayed preview reflects the current settings.
4. They choose a destination and export the full-resolution result using the previewed settings. The app shows progress, supports cancellation, and keeps the source available for retry if processing or saving fails.

Preview and export use the same operation, parameters, and model version. If that version is unavailable at export, ask the user to regenerate the preview. At 100% zoom the source is shown at native pixel scale. For 2× upscale, match the same source region on both sides and scale the enhanced view to the same framing for comparison. During design, fix the benchmark crop dimensions in source pixels and define an acceptable tolerance for differences between the crop preview and the matching export region, including tile boundaries.

## Standalone CLI

1. A script or operator specifies one or more supported input paths, an output path or directory, the operation and settings, and the inference endpoint through configuration or flags.
2. The CLI runs on macOS and headless Linux without interactive prompts, a desktop environment, Lightroom, or a local GPU. The same commands, options, and output behavior are available on all supported platforms. It validates paths and settings before submitting work and reports per-file progress and outcomes.
3. Use exit code `0` when every file succeeds, `1` when any file fails, and `2` for invalid invocations, including two inputs that resolve to the same output path. Reject such collisions before submitting any jobs, even with overwrite enabled. Existing destinations require explicit overwrite permission, and source files cannot be output destinations.
4. Provide an opt-in JSON result on stdout with one entry per input in argument order, including input path, intended output path, status, and a stable error code on failure. Send progress and human-readable diagnostics to stderr. A mixed-result batch keeps successfully written outputs.

Distribute separate Ubuntu `.deb` packages for `amd64` and `arm64` as release downloads with checksums. Build and test both on Ubuntu 24.04. For each subsequent Ubuntu LTS, verify installation and a single-file job on both architectures before adding that version to the tested matrix. Defer an APT repository until automatic updates are needed.

## Adobe Lightroom desktop external editor

Use Adobe Lightroom desktop 10.0 as the initial compatibility baseline on macOS 27. Record the exact tested build and rerun the cloud-library round trip before declaring support for newer builds. Adobe's workflow opens a photo in a third-party app through **Edit in**, then brings the saved result back when the photographer selects **Finish** in Lightroom. Adobe also documents a **Browse** option for linking a separately saved file. [Adobe Lightroom desktop release notes](https://helpx.adobe.com/lightroom/desktop/introduction/release-notes.html) · [Adobe Lightroom Help — Edit in other apps](https://helpx.adobe.com/lightroom/desktop/edit-photos/edit-in-other-apps.html)

1. The photographer opens a single photo in Lightroom's Grid or Detail view, selects **Edit in**, and chooses PAINITE. The app accepts the macOS file-open handoff on both cold launch and while already running.
2. PAINITE opens the supplied image and provides the same enhancement controls and preview as standalone mode. It treats the supplied rendered pixels as the input, including any adjustments already baked into them.
3. Enable direct save only when PAINITE can reliably identify a writable editing copy, preserve its format, and Lightroom detects the update. On **Save**, write a complete, readable image to the same handoff path in the same format. Otherwise, offer **Save As**.
4. The photographer selects **Finish** in Lightroom to return the directly saved result. For a separately saved result, the photographer uses **Browse** in Lightroom to select the output.

Prototype the Lightroom handoff before finalizing the Mac app's file-format and save requirements. Use an edited camera RAW photo and an edited JPEG from the cloud library with PAINITE closed and already running. Record the delivered format, bit depth, ICC profile, baked-in adjustments, and whether Lightroom creates a separate writable copy.

The minimum supported round trip is **Edit in → Save As → Browse** for both test inputs. If direct save passes verification, document the exact supported cases and test that **Finish** detects the write. Verify that saving, canceling, and a failed write preserve the original in every supported case. If the minimum round trip fails, resolve the compatibility issue or revisit release scope.

The first release targets the cloud library in Adobe Lightroom desktop. Lightroom Classic is outside this scope. Integration with Lightroom desktop's **Local** tab is deferred. [Adobe Lightroom Help — Edit locally stored photos](https://helpx.adobe.com/lightroom/desktop/add-import-and-capture-photos/access-photos.html)

## Inference and file handling

- The service exposes endpoint health, supported operations, and model versions. The desktop app and CLI use a common job contract for upload, settings, status, cancellation, and result download. Matching inputs, settings, and model versions use the same processing path in all modes.
- The photographer or studio operator runs DGX Spark on a private network and provisions and revokes client credentials. Authenticate clients, encrypt image transfers, and allow only the job owner to view job status, download or delete results, or cancel jobs through the service. Keep photo contents and credentials out of service logs.
- Keep outputs for 24 hours from successful completion. Let the owner delete an output sooner. Return a clear result-unavailable error if a download is requested or resumed after deletion or expiry. Delete uploads and working files when jobs complete, fail, or are canceled. Clean up orphaned files after a crash and enforce a disk usage cap.
- Report queued, running, completed, failed, and canceled jobs. Start with one active inference job per DGX Spark and a bounded queue. Prioritize pending previews when selecting the next job. Raise concurrency only if benchmarks show better throughput without worsening preview latency.
- Benchmark with provisional limits of 48 megapixels per input and 192 megapixels per output for 2× upscale. Enforce separate caps on upload bytes and decoded pixels. Give clients actionable errors for unsupported inputs, unavailable models, and connection failures.
- Expect Wi-Fi between clients and DGX Spark. Uploads and downloads must tolerate brief drops and variable latency: retry with backoff, resume downloads, and avoid re-uploading a photo the server already has.
- After a network interruption, allow clients to query or resume known jobs by ID. Deduplicate submission retries by job owner and request key while the job is active and for 24 hours after it reaches a terminal state, even if its output is deleted early. A failed or canceled write must leave the source, any existing destination, and the Lightroom handoff intact. Publish a completed output only after it has been fully written and validated.

## Release acceptance

Record the exact macOS, Ubuntu, and Lightroom builds used for acceptance. The baseline matrix covers the Mac app and CLI on Apple silicon with macOS 27, Lightroom desktop 10.0 on that Mac, and headless Ubuntu 24.04 on x86-64 and ARM64. Clients reach DGX Spark over the studio's Wi-Fi network, which is the baseline network for acceptance. Record the Wi-Fi standard, band, and measured throughput between each client and the Spark.

| Check | Pass condition |
| --- | --- |
| Standalone GUI | With Lightroom closed, process and export a photo through each operation. Verify the preview uses the chosen settings and the exported image opens. |
| Standalone CLI | On each baseline CLI platform, run a single-file job, a mixed-result batch, and a same-name collision. Verify output paths, JSON results, and exit codes. Run the macOS checks with both desktop apps closed and install the release packages for the Ubuntu checks. |
| Lightroom round trip | Pass the minimum round trip on cold and warm launches for edited cloud-library RAW and JPEG inputs. Verify direct save for each advertised case and confirm source preservation on success, cancellation, and write failure. |
| Image fidelity | Test tagged and untagged JPEG, a JPEG with non-default EXIF orientation, and a wide-gamut 16-bit TIFF. Check color interpretation, bit depth, output orientation and dimensions, and metadata. Before implementation, specify the untagged-input color assumption, JPEG encoding defaults, and EXIF/IPTC fields to preserve. Verify 2× upscale doubles each dimension and that matching preview and export regions meet the agreed tolerance. |
| Failure recovery and ownership | Exercise disconnects, cancellation, write failure, and full disk. Compare source-file hashes before and after and verify existing destinations survive failed writes. Use two credentials to verify job isolation. Check retries within the deduplication window, early deletion, and download attempts after expiry. |
| Performance and capacity | Measure median and p95 end-to-end latency on DGX Spark at 12, 24, and 48 megapixels. With warm models, an empty queue, and clients on the baseline Wi-Fi network, aim for p95 under 5 seconds for a 100% crop preview of an already uploaded photo and under 60 seconds for a 24 megapixel full render. Report server processing time separately from upload, download, and queue time, so Wi-Fi variation can be told apart from server performance. Track 2× upscale separately. Record cold starts, preview latency with an upscale already running, transfer and queue time, peak memory, and scratch space. Use benchmark results to set pass/fail latency thresholds and final image limits before release. |

## Design-phase decision

Select the models for denoise, sharpen, and 2× upscale during design. On DGX Spark, compare candidate outputs with reference outputs from existing enhancement tools. Use rendered high-ISO images and examples with skin, foliage, text, and hard edges. Check detail loss, halos, color shifts, and tile seams. Include capacity and latency in the selection. Choose the best open models on quality and performance; licensing does not limit the choice. If weights can't be redistributed, users download them directly and PAINITE provides the configuration and pipeline. Confirm the terms for the chosen weights and runtime dependencies before release.
