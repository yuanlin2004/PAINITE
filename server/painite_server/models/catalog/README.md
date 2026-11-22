Model manifests shipped with the server. Each `*.json` file names a model ID
(`<operation>/<name>@<version>`), the adapter that runs it, the upstream
download URL and SHA-256 of every weight file, and the license. Operators
install one with `painite-server models pull <id>`.

This directory is empty until the model bake-off (docs/DESIGN.md section 8)
picks models and their adapters land. A manifest looks like:

```json
{
  "id": "denoise/nafnet-sidd@1.0.0",
  "adapter": "nafnet",
  "description": "NAFNet-width64 trained on SIDD",
  "license": {"name": "MIT", "url": "https://github.com/megvii-research/NAFNet/blob/main/LICENSE"},
  "files": [
    {"name": "NAFNet-SIDD-width64.pth", "url": "https://…", "sha256": "…", "bytes": 464000000}
  ],
  "config": {"width": 64},
  "default": true
}
```
