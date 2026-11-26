"""MPS fp32 / fp16 / bf16 timing and quality per model. Saves outputs to out/."""
import copy
import json
import sys

import numpy as np
import torch

from common import OUT, face_crop, load_all, maxdiff8, psnr, save_png, timeit, window

ops = sys.argv[1:] or None
models = load_all(ops)
res = {}
for op, (m, net) in models.items():
    if ops and op not in ops:
        continue
    inp = face_crop(m) if op == "face" else window(640)
    np.save(OUT / f"{op}_input.npy", inp)
    save_png(inp, OUT / f"{op}_input.png")
    x = torch.from_numpy(inp.transpose(2, 0, 1).copy())[None]
    outs, r = {}, {}
    for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        try:
            n = copy.deepcopy(net).to("mps", dt).eval()
            xm = x.to("mps", dt)

            def f():
                with torch.inference_mode():
                    return n(xm)

            t = timeit(f, "mps")
            with torch.inference_mode():
                y = n(xm).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            outs[name] = y
            np.save(OUT / f"{op}_mps_{name}.npy", y)
            save_png(y, OUT / f"{op}_mps_{name}.png")
            r[name] = {"s": t}
            del n
            torch.mps.empty_cache()
        except Exception as e:  # noqa: BLE001
            r[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(op, name, r[name], flush=True)
    for name in outs:
        if name != "fp32" and "fp32" in outs:
            r[name]["psnr_vs_fp32"] = psnr(outs[name], outs["fp32"])
            r[name]["max8_vs_fp32"] = maxdiff8(outs[name], outs["fp32"])
            r[name]["nan"] = bool(np.isnan(outs[name]).any())
    r["input_shape"] = list(inp.shape)
    res[op] = r
    print(op, json.dumps(r), flush=True)
path = OUT / "mps.json"
old = json.loads(path.read_text()) if path.exists() else {}
old.update(res)
path.write_text(json.dumps(old, indent=1))
