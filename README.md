<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/painite-logo-dark.svg">
  <img src="docs/logo/painite-logo.svg" alt="PAINITE — Photo Brilliance" width="440" height="140">
</picture>

# PAINITE

PAINITE is an AI photo enhancement project with a macOS app and a standalone CLI for macOS and Linux. Both send photos to a separate inference server for processing. NVIDIA DGX Spark is the initial server target.

## Planned workflows

| Mode | Workflow |
| --- | --- |
| Standalone Mac app | On macOS 27 or later (Apple silicon), open a photo, preview the enhancement against the original, and export the result. |
| Standalone CLI | Enhance a single photo or batch on macOS 27 or later (Apple silicon) or Ubuntu 24.04 or later (x86-64 and ARM64). |
| Adobe Lightroom desktop | On a supported Mac, open a photo in PAINITE through **Edit in**, save the enhancement, then select **Finish** in Lightroom to bring it back. |

The first release proposes AI noise reduction, sharpening, and upscaling across all three workflows. The project is in the requirements stage.

[Product requirements document](docs/PRD.md) · [Design](docs/DESIGN.md)
