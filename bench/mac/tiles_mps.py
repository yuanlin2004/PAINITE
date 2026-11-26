"""MPS fp16 (production dtype) time per tile vs tile size; context 64 px each side.
Reports seconds per megapixel of finished (core) output area."""
import json
import sys

import numpy as np
import torch

from common import OUT, load_all, photo, timeit

ops = sys.argv[1:] or ["denoise", "sharpen", "unblur_motion", "unblur_focus", "upscale2x"]
img = photo()
res = {}
for op, (m, net) in load_all(ops).items():
    dt = torch.float16
    n = net.to("mps", dt).eval()
    r = {}
    for tile in (256, 384, 512, 768, 1024):
        w = tile + 128
        x = torch.from_numpy(img[600:600 + w, 600:600 + w].transpose(2, 0, 1).copy())[None].to("mps", dt)
        try:
            def f():
                with torch.inference_mode():
                    return n(x)
            t = timeit(f, "mps", warm=1, n=3)
            r[tile] = {"s_per_tile": t, "s_per_core_mp": t / (tile * tile / 1e6)}
        except Exception as e:  # noqa: BLE001
            r[tile] = {"error": f"{type(e).__name__}: {e}"[:200]}
        print(op, tile, r[tile], flush=True)
        torch.mps.empty_cache()
    res[op] = r
    del n
    torch.mps.empty_cache()
(OUT / "tiles_mps.json").write_text(json.dumps(res, indent=1))
