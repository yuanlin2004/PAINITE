# Mac server speed-up benchmarks (MPS dtypes, Core ML, tile size)

Evaluation only; nothing here changes the server. Measured on an M2 Pro (16 GB),
November 2026. Write-up: the "Mac server speedups" doc in the project.

Models come from an installed Mac server (`~/Library/Application Support/painite-server`).
Test photo: `PAINITE_MAC_BENCH_PHOTO` (default `~/Downloads/two.jpg`), a 640 px window
around a face for the tiled models and an aligned 1024 px face crop for GPEN.

Two environments:

- the server's venv (PyTorch 2.15) for `bench_mps.py` and `tiles_mps.py`
- a venv with `torch==2.7.0 torchvision==0.22.0 spandrel==0.4.2 einops coremltools==9.0
  pillow "numpy<2.3" fastapi pydantic` for `bench_coreml.py` (coremltools 9.0 fails
  to convert under PyTorch 2.15)

```
python bench_mps.py [op ...]        # MPS fp32/fp16/bf16 time + PSNR vs fp32; run first (saves inputs)
python bench_coreml.py [op ...] [--precision fp16|fp32] [--units ALL CPU_AND_GPU CPU_AND_NE] [--size N]
python tiles_mps.py [op ...]        # MPS fp16 seconds per output megapixel vs tile size
```

`bench_coreml.py` patches two things for conversion only (math unchanged): SCUNet's
attention mask is built eagerly as a constant, and GPEN's upfirdn2d skips the flip of
its symmetric kernel and upsamples with a transposed 1x1 conv instead of a rank-6 view.
Raw numbers are in `results/`.
