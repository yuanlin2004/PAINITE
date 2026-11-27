<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/painite-logo-dark.svg">
  <img src="docs/logo/painite-logo.svg" alt="PAINITE — Photo Brilliance" width="440" height="140">
</picture>

# PAINITE

PAINITE is an AI photo enhancement project with a macOS app, a standalone CLI for macOS and Linux, and an MCP server so AI agents can use it. Both send photos to a separate inference server for processing. NVIDIA DGX Spark is the initial server target. The server is also tested on an x86-64 Ubuntu PC with a GeForce RTX 4070 (12 GB), which needs NVIDIA driver 580 or newer, and runs natively on a Mac with Apple silicon (M2 Pro tested).

**[User guide](https://yuanlin2004.github.io/PAINITE/)** · **[Download](https://github.com/yuanlin2004/PAINITE/releases/latest)** · [Release notes](docs/release-notes/v1.3.0.md)

## Workflows

| Mode | Workflow |
| --- | --- |
| Standalone Mac app | On macOS 14 or later (Apple silicon), open a photo, preview the enhancement against the original, and export the result. |
| Standalone CLI | Enhance a single photo or batch on macOS 14 or later (Apple silicon) or Ubuntu 24.04 or later (x86-64 and ARM64). |
| AI agents (MCP) | Claude Code, Claude Desktop or another MCP client runs `painite mcp` to check the server, preview enhancements and enhance photos ([guide](https://yuanlin2004.github.io/PAINITE/agents.html)). |
| Adobe Lightroom desktop | On a supported Mac, open a photo in PAINITE through **Edit in**, save the enhancement, then select **Finish** in Lightroom to bring it back. |

Version 1.3 has six enhancements: denoise, sharpen, unblur (motion and focus), face restoration and 2× upscale. Up to four of them can run in one pass.

## Repository

| Path | What |
| --- | --- |
| [`api/openapi.yaml`](api/openapi.yaml) | Service contract shared by the server and clients |
| [`server/`](server/README.md) | Inference server (Python, FastAPI, PyTorch) and its installer |
| [`cli/`](cli/README.md) | `painite` command line (Go) for macOS and Linux, including the `painite mcp` server for AI agents |
| [`mac/`](mac/README.md) | Mac app (Swift) |
| [`site/`](site/build.py) | User guide, published to GitHub Pages |
| [`bench/`](bench/) | Model bake-offs that chose each model |

[Product requirements document](docs/PRD.md) · [Design](docs/DESIGN.md) · [Server performance report](docs/report/2026-11-27-server-performance-dgx-spark-rtx4070-m2pro.md)

## License

PAINITE is © 2026 Yuan Lin and licensed under the [Apache License 2.0](LICENSE). If you redistribute it or build on it, include the [NOTICE](NOTICE) file. Model weights are not part of PAINITE. Each model has its own license, and some allow only non-commercial use ([details](https://yuanlin2004.github.io/PAINITE/help.html#licenses)).
