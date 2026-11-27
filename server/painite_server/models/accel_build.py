"""Build and check one TensorRT engine, in its own process (models/accel.py).

    python -m painite_server.models.accel_build --model-dir <install dir> \
        --shape 1,3,640,640 --precision fp16 --typing weak --out <engine.pt2> --record <record.json>

Loads the model with the server's own adapter, exports the bare network at the
given shape, builds the engine with Torch-TensorRT, saves it, loads it back
from disk and runs the golden-tile check against PyTorch FP32. Writes the
record (status ok or failed, with the check's numbers) either way. A separate
process so the server can kill a build that runs past its time limit, and so a
failed build cannot leave the server's dynamo state broken.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
import traceback
from pathlib import Path

log = logging.getLogger("painite.models")


def export_prep(net, shape) -> list[str]:
    """Export-friendly changes that leave the weights and the math alone.

    SCUNet builds its shifted-window attention mask with in-place slice
    assignment on every call (`generate_mask`), which export rejects
    (aten.fill_ on a slice). The mask depends only on the static shape, so it
    is computed once here and handed to the trace as a constant."""
    import torch  # noqa: PLC0415

    notes = []
    blocks = [m for m in net.modules() if hasattr(m, "generate_mask")]
    for m in blocks:
        cache: dict = {}
        orig = m.generate_mask

        def cached(h, w, p, shift, _orig=orig, _cache=cache):
            key = (int(h), int(w), int(p), int(shift))
            if key not in _cache:
                with torch.inference_mode(False):
                    _cache[key] = _orig(*key).clone()
            return _cache[key]

        m.generate_mask = cached
    if blocks:
        with torch.no_grad():
            net(torch.zeros(shape, device="cuda", dtype=next(net.parameters()).dtype))  # fill the mask caches
        notes.append(f"generate_mask precomputed for {len(blocks)} blocks")
    return notes


FENCE = "torch.ops.painite.fence.default"


def register_fence(torch) -> None:
    """`painite::fence`, an identity op TensorRT has no converter for: placed on
    a tensor, it runs in PyTorch and so splits the network into separate engines
    there (models/accel.py, `split_after`). Must be registered before an engine
    that uses it is built or loaded."""
    if hasattr(torch.ops.painite, "fence"):
        return

    @torch.library.custom_op("painite::fence", mutates_args=(), schema="(Tensor x) -> Tensor")
    def fence(x):
        return x.clone()

    @fence.register_fake
    def _(x):
        return torch.empty_like(x)


def split_at(torch, net, paths) -> list[str]:
    """Fence the output of each named submodule of `net`."""
    if not paths:
        return []
    register_fence(torch)
    fence = torch.ops.painite.fence

    def hook(_mod, _inp, out):
        return tuple(fence(t) for t in out) if isinstance(out, tuple) else fence(out)

    for p in paths:
        net.get_submodule(p).register_forward_hook(hook)
    return [f"engines split after {', '.join(paths)}"]


def build(model, shape: tuple, precision: str, typing: str, split_after: tuple = ()):
    """Torch-TensorRT program for the model's bare network. Returns (module, input dtype, notes)."""
    import torch  # noqa: PLC0415
    import torch_tensorrt  # noqa: PLC0415

    net = copy.deepcopy(model.accel_module())
    kw = dict(ir="dynamo", min_block_size=1, truncate_double=True, workspace_size=8 << 30,
              cache_built_engines=False, reuse_cached_engines=False)
    split_notes = split_at(torch, net, split_after)
    if split_after:
        kw["torch_executed_ops"] = {FENCE}
    if typing == "strong":
        # The network cast to the precision in PyTorch; TensorRT keeps every tensor's type as traced.
        dt = torch.float16 if precision == "fp16" else torch.float32
        net = net.to(dt).eval()
        notes = export_prep(net, shape) + split_notes
        trt = torch_tensorrt.compile(net, inputs=[torch_tensorrt.Input(shape, dtype=dt)],
                                     use_explicit_typing=True, **kw)
        return trt, dt, notes
    net = net.float().eval()
    notes = export_prep(net, shape) + split_notes
    precisions = {torch.float32} if precision == "fp32" else {torch.float16, torch.float32}
    trt = torch_tensorrt.compile(net, inputs=[torch_tensorrt.Input(shape, dtype=torch.float32)],
                                 enabled_precisions=precisions, **kw)
    return trt, torch.float32, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--typing", choices=("weak", "strong"), default="weak")
    ap.add_argument("--split-after", default="", help="comma-separated submodules whose output ends an engine")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--record", required=True, type=Path)
    ap.add_argument("--min-psnr", type=float, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from . import ADAPTERS  # noqa: PLC0415
    from . import accel  # noqa: PLC0415
    from .manifest import Manifest  # noqa: PLC0415

    shape = tuple(int(v) for v in args.shape.split(","))
    split = tuple(p for p in args.split_after.split(",") if p)
    manifest = Manifest.load(args.model_dir / "manifest.json")
    spec = {"backend": "tensorrt", "precision": args.precision, "typing": args.typing}
    if split:
        spec["split_after"] = list(split)
    record = {"model": manifest.id, "spec": spec, "shape": list(shape), "at": time.time()}
    try:
        import torch  # noqa: PLC0415
        import torch_tensorrt  # noqa: PLC0415

        model = ADAPTERS[manifest.adapter](manifest, args.model_dir)
        model.device = "cuda"
        model.load()
        t0 = time.perf_counter()
        trt, in_dtype, notes = build(model, shape, args.precision, args.typing, split)
        record["build_s"] = round(time.perf_counter() - t0, 1)
        record["notes"] = notes
        x = torch.rand(shape, device="cuda", dtype=in_dtype)
        torch_tensorrt.save(trt, str(args.out), output_format="exported_program", arg_inputs=[x])
        del trt
        record["engine_mb"] = round(args.out.stat().st_size / 2**20, 1)
        record["input_dtype"] = str(in_dtype).replace("torch.", "")
        log.info("built %s in %.1f s (%.0f MB)", manifest.id, record["build_s"], record["engine_mb"])

        # Check the engine as the server will run it: loaded back from disk.
        mod = torch.export.load(str(args.out)).module()

        def fn(t):
            y = mod(t.to(in_dtype))
            return (y[0] if isinstance(y, (tuple, list)) else y).float()

        net = model.accel_module()
        ref = copy.deepcopy(net).float()
        checked = accel.check(torch, fn, ref, model.accel_call(net), shape)
        if args.min_psnr is not None:
            ok, why = accel.verdict(checked["candidate"], checked["pytorch"], args.min_psnr)
            checked["passed"], checked["reason"] = ok, why
        record["validation"] = checked
        record["status"] = "ok" if checked["passed"] else "failed"
        record["reason"] = checked["reason"]
        log.info("check %s: %s", "passed" if checked["passed"] else "FAILED", checked["reason"])
    except Exception as e:  # noqa: BLE001 - record the failure for the server
        traceback.print_exc()
        record["status"] = "failed"
        record["reason"] = f"build error: {type(e).__name__}: {str(e)[:500]}"
    if record["status"] != "ok":
        args.out.unlink(missing_ok=True)
    accel.write_record(args.record, record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
