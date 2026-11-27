"""Convert each model to Core ML (mlprogram) at its fixed input size, then time
it on ALL / CPU_AND_GPU / CPU_AND_NE and compare with the MPS fp32 output.

  python bench_coreml.py [op ...] [--size N] [--precision fp16|fp32]
"""
import argparse
import json
import time
import traceback

import coremltools as ct
import numpy as np
import torch

from common import window, OUT, load_all, maxdiff8, psnr, save_png

ap = argparse.ArgumentParser()
ap.add_argument("ops", nargs="*")
ap.add_argument("--size", type=int, default=0, help="override input size (tiled models)")
ap.add_argument("--precision", default="fp16")
ap.add_argument("--units", nargs="*", default=["ALL", "CPU_AND_GPU", "CPU_AND_NE"])
a = ap.parse_args()


def patch_for_coreml():
    """Conversion-only workarounds; the math is unchanged."""
    import torch.nn.functional as F
    from painite_server.models import gpen_arch
    from spandrel.architectures.SCUNet.__arch import SCUNet as sc

    def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0), device="cpu"):
        # GPEN's blur kernels are symmetric, so the flip is a no-op; Core ML
        # const-folds torch.flip into a negative-stride array and fails.
        assert torch.equal(kernel, torch.flip(kernel, [0, 1]))
        b, c, h, w = input.shape
        kh, kw = kernel.shape
        x = input
        if up > 1:
            # zero-insertion upsample without a rank-6 view (Core ML max rank 5)
            one = torch.ones(c, 1, 1, 1, dtype=x.dtype)
            x = F.conv_transpose2d(x, one, stride=up, groups=c)
            x = F.pad(x, [0, up - 1, 0, up - 1])
        p0, p1 = pad
        x = F.pad(x, [max(p0, 0), max(p1, 0), max(p0, 0), max(p1, 0)])
        x = x[:, :, max(-p0, 0): x.shape[2] - max(-p1, 0), max(-p0, 0): x.shape[3] - max(-p1, 0)]
        k = kernel.to(x.dtype).view(1, 1, kh, kw).repeat(c, 1, 1, 1)
        x = F.conv2d(x, k, groups=c)
        return x[:, :, ::down, ::down]

    gpen_arch.upfirdn2d = upfirdn2d
    orig = sc.WMSA.generate_mask

    def generate_mask(self, h, w, p, shift):
        # Built eagerly, then a constant: Core ML has no bool slice-assign.
        with torch.jit._disable_tracing() if hasattr(torch.jit, "_disable_tracing") else torch.no_grad():
            m = orig(self, h, w, p, shift).cpu().numpy()
        return torch.from_numpy(m.copy())

    sc.WMSA.generate_mask = generate_mask


patch_for_coreml()
models = load_all(a.ops or None)
res = {}
for op, (m, net) in models.items():
    if a.ops and op not in a.ops:
        continue
    inp = np.load(OUT / f"{op}_input.npy")
    if a.size and op != "face":
        c = (inp.shape[0] - a.size) // 2
        inp = inp[c:c + a.size, c:c + a.size] if a.size <= inp.shape[0] else window(a.size)
    ref = np.load(OUT / f"{op}_mps_fp32.npy") if not a.size else None
    x = torch.from_numpy(inp.transpose(2, 0, 1).copy())[None]
    tag = f"{op}_{a.precision}" + (f"_{a.size}" if a.size else "")
    r = {}
    pkg = OUT / f"{tag}.mlpackage"
    try:
        t0 = time.perf_counter()
        if not pkg.exists():
            with torch.no_grad():
                traced = torch.jit.trace(net, x, check_trace=False)
            mlm = ct.convert(
                traced,
                inputs=[ct.TensorType(name="x", shape=x.shape)],
                outputs=[ct.TensorType(name="y")],
                convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16 if a.precision == "fp16" else ct.precision.FLOAT32,
                minimum_deployment_target=ct.target.macOS14,
            )
            mlm.save(str(pkg))
        r["convert_s"] = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        r["convert_error"] = f"{type(e).__name__}: {e}"[:600]
        traceback.print_exc()
        res[tag] = r
        print(tag, r, flush=True)
        continue
    feed = {"x": x.numpy()}
    for unit in a.units:
        try:
            t0 = time.perf_counter()
            mm = ct.models.MLModel(str(pkg), compute_units=getattr(ct.ComputeUnit, unit))
            load_s = time.perf_counter() - t0
            y = mm.predict(feed)["y"]  # first call (compile/specialize)
            first = time.perf_counter() - t0
            ts = []
            for _ in range(4):
                t1 = time.perf_counter()
                y = mm.predict(feed)["y"]
                ts.append(time.perf_counter() - t1)
            y = np.asarray(y, np.float32)[0].transpose(1, 2, 0)
            d = {"s": float(np.median(ts)), "load_s": load_s, "first_s": first,
                 "nan": bool(np.isnan(y).any())}
            if ref is not None:
                d["psnr_vs_fp32"] = psnr(np.nan_to_num(y), ref)
                d["max8_vs_fp32"] = maxdiff8(np.nan_to_num(y), ref)
                np.save(OUT / f"{tag}_{unit}.npy", y)
                save_png(np.nan_to_num(y), OUT / f"{tag}_{unit}.png")
            r[unit] = d
        except Exception as e:  # noqa: BLE001
            r[unit] = {"error": f"{type(e).__name__}: {e}"[:400]}
        print(tag, unit, r[unit], flush=True)
    res[tag] = r
    path = OUT / "coreml.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(res)
    path.write_text(json.dumps(old, indent=1))
