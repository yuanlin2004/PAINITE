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
| Platform | Ship a macOS desktop app, a standalone CLI for macOS and Linux (x86-64 and ARM64), and a separately deployed DGX Spark inference service. |

Direct camera RAW development, HDR, layered files, video, and on-device inference are outside the proposed first release. Lightroom-rendered copies of RAW photos are in scope once the handoff is validated.

## Standalone Mac app

1. The photographer configures the inference endpoint and can check its connection and model availability from the app.
2. They open a supported photo from Finder or the app, choose an operation, and adjust the available settings.
3. They compare the before and after views at a matched zoom and position. A newer preview request supersedes older results so the displayed preview reflects the current settings.
4. They choose a destination and export the full-resolution result using the previewed settings. The app shows progress, supports cancellation, and keeps the source available for retry if processing or saving fails.

## Standalone CLI

1. A script or operator specifies one or more supported input paths, an output path or directory, the operation and settings, and the inference endpoint through configuration or flags.
2. The CLI runs on macOS and headless Linux without interactive prompts, a desktop environment, Lightroom, or a local GPU. The same commands, options, and output behavior are available on all supported platforms. It validates paths and settings before submitting work and reports per-file progress and outcomes.
3. The CLI provides stable exit codes for complete success, failed processing, and invalid invocation. A mixed-result batch exits nonzero, identifies failed inputs, and keeps successfully written outputs. Existing destinations require explicit overwrite permission, and source files cannot be output destinations.

## Adobe Lightroom desktop external editor

The integration targets the current cloud-connected Adobe Lightroom desktop app on macOS. Adobe's workflow opens a photo in a third-party app through **Edit in**, then brings the saved result back when the photographer selects **Finish** in Lightroom. Adobe also documents a **Browse** option for linking a separately saved file. [Adobe Lightroom Help — Edit in other apps](https://helpx.adobe.com/lightroom/desktop/edit-photos/edit-in-other-apps.html)

1. The photographer opens a single photo in Lightroom's Grid or Detail view, selects **Edit in**, and chooses PAINITE. The app accepts the macOS file-open handoff on both cold launch and while already running.
2. PAINITE opens the supplied image and provides the same enhancement controls and preview as standalone mode. It treats the supplied rendered pixels as the input, including any adjustments already baked into them.
3. On Save, PAINITE writes a complete, readable image to the same handoff path and in the same format when the file is verified as a writable editing copy. The app offers **Save As** when it cannot verify the copy, preserve the format, or write back safely.
4. The photographer selects **Finish** in Lightroom to return the directly saved result. For a separately saved result, the photographer uses **Browse** in Lightroom to select the output.

Validate the delivered format, bit depth, ICC profile, baked-in adjustments, and writable-copy behavior with an edited camera RAW photo and an edited JPEG on the supported Lightroom desktop version. Verify that Lightroom recognizes the save strategy and that the original photo is preserved. Single-photo **Edit in** from the cloud library is the initial target. Local-tab handoffs are a compatibility question.

## Inference and file handling

- The service exposes endpoint health, supported operations, and model versions. The desktop app and CLI use a common job contract for upload, settings, status, cancellation, and result download. Matching inputs, settings, and model versions use the same processing path in all modes.
- Authenticate clients and encrypt image transfers. Keep photo contents and credentials out of service logs. Process images on the configured inference host and expire temporary uploads and results after a documented retention window.
- Report queued, running, completed, failed, and canceled jobs. Bound queue depth and image sizes to protect the server from overload. Give clients actionable errors for unsupported inputs, unavailable models, and connection failures.
- After a network interruption, allow clients to query or resume known jobs by ID. Retrying a submission after an ambiguous response must avoid duplicate processing. A failed or canceled write must leave the source, any existing destination, and the Lightroom handoff intact. Publish a completed output only after it has been fully written and validated.

## Release acceptance

| Check | Pass condition |
| --- | --- |
| Standalone GUI | With Lightroom closed, process and export a photo through each operation. Verify the preview uses the chosen settings and the exported image opens. |
| Standalone CLI | Run the same single-file command and mixed-result batch on macOS with both desktop apps closed and on headless Linux on x86-64 and ARM64. Use the same DGX Spark endpoint and verify output paths, per-file results, and exit status on each platform. |
| Lightroom round trip | Send an edited RAW photo and an edited JPEG through **Edit in** on the supported Lightroom desktop version. Verify **Finish** returns the saved result, **Browse** can link a separately saved result, and the source in Lightroom is preserved. |
| Image fidelity | Check orientation, color, bit depth, and metadata with representative JPEG and 16-bit TIFF samples. Verify 2× upscale doubles each dimension and maintains aspect ratio. Review high-ISO images, fine textures, and hard edges for noise reduction, retained detail, and objectionable artifacts. |
| Failure recovery | Exercise loss of connectivity, cancellation, and file write errors. Verify the error identifies the failed step, existing files remain readable, and a retry can complete. |
| Performance | Record median and 95th-percentile preview and full-render times on DGX Spark for representative 12, 24, and 48 megapixel photos. Measure transfer, queue, and processing time to set release limits and latency targets. |

## Open decisions

- Supported macOS versions and CPU architectures, supported Linux distributions and CLI packaging, and supported Lightroom desktop versions. Lightroom's Local tab support is also open.
- Lightroom handoff format and direct-save detection, including how the app identifies a writable copy reliably.
- Model selection and licensing, image size limits, concurrent jobs, latency targets, and temporary file retention.
