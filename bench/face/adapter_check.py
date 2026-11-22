"""Run a spandrel-loadable pick through the server's face adapter
(server/painite_server/models/face_adapter.py) and compare with the bench pipeline on the
real photos: same faces found, and how far the pasted pixels differ.

  python face/adapter_check.py --restorer codeformer.pth --weight 0.8 --model codeformer-w0.8

The bench side is run with the adapter's settings (square mask inset 0.05 / feather 0.12,
color anchor σ 8, detection at 2048, threshold 0.9) so differences come from implementation,
not settings.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "server"))
from painite_bench.config import TESTSET_DIR  # noqa: E402
from painite_server.imaging.geometry import Rect  # noqa: E402
from painite_server.models.face_adapter import FaceModel  # noqa: E402
import pipeline as P  # noqa: E402
import restorers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorer", default="codeformer.pth")
    ap.add_argument("--weight", type=float, default=0.8)
    ap.add_argument("--model", default="codeformer-w0.8")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--photos", nargs="*", default=["portrait_horridge", "half_bahner", "event_coty",
                                                     "group_usa_team", "group_lawyers"])
    a = ap.parse_args()
    cfg = dict(detector="detection_Resnet50_Final.pth", restorer=a.restorer, restorer_args={"weight": a.weight},
               dtype=a.dtype, face_size=512, detect_long_side=2048, detect_threshold=0.9, min_face=24,
               max_face=0, mask_inset=0.05, mask_feather=0.12, color_anchor_sigma=8)
    fm = FaceModel(id="face/check@0", operation="face", version="0", detector_path=restorers.FACE_WEIGHTS / cfg["detector"],
                   restorer_path=restorers.FACE_WEIGHTS / cfg["restorer"], config=cfg)
    det, r = P.Detector(), restorers.load(a.model)
    opt = P.Options(mask="square", color_anchor=8, min_eye_dist=24 * 125.92 / 512)
    for p in a.photos:
        bgr = cv2.imread(str(TESTSET_DIR / "face/real" / f"{p}.jpg"))
        h, w = bgr.shape[:2]
        rgb = bgr[..., ::-1].astype(np.float32) / 255
        dw, dh = fm.detection_size(w, h)
        small = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        faces = fm.layout(fm.detect(small, w, h), 1, w, h)
        srv = fm.run_area(rgb, (0, 0), Rect(0, 0, w, h), faces)
        srv = (np.clip(srv, 0, 1)[..., ::-1] * 255).round().astype(np.uint8)
        bench, bfaces, _ = P.enhance(bgr, r, det, None, opt)
        m = np.zeros((h, w), bool)
        for f in faces:
            x0, y0, x1, y1 = f.box
            m[y0:y1, x0:x1] = True
        d = np.abs(srv.astype(np.int16) - bench.astype(np.int16))[m]
        lm_err = [min(np.abs(np.array(f.landmarks) - b.lm5).max() for b in bfaces) for f in faces] if bfaces else []
        mse = np.mean(d.astype(np.float64) ** 2)
        print(f"{p:20s} faces server {len(faces)} bench {len(bfaces)}  max landmark diff {max(lm_err, default=0):.2f} px"
              f"  face-box pixels: mean |diff| {d.mean():.2f}, p99 {np.percentile(d, 99):.0f}, "
              f"PSNR {10 * np.log10(255 ** 2 / max(mse, 1e-9)):.1f} dB", flush=True)
        if p == a.photos[0]:
            f = faces[0]
            x0, y0, x1, y1 = f.box
            out = Path(restorers.DATA_DIR) / "out/face/adapter_check"
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{p}.jpg"), np.hstack([bgr[y0:y1, x0:x1], srv[y0:y1, x0:x1], bench[y0:y1, x0:x1]]))


if __name__ == "__main__":
    main()
