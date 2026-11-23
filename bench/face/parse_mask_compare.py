"""Before/after for the face paste mask, through the server's own pipeline:
the feathered square of face/gpen-bfr-1024@1.0.0 against the face-parsing
mask of @1.1.0, on photos with a hand or an arm near a face.

  python face/parse_mask_compare.py --model-dir <data_dir>/models/face/gpen-bfr-1024@1.1.0 \
      --out results/face/parse_mask photo.jpg photo2.jpg@x,y,w,h ...

`@x,y,w,h` frames that region (upright pixels) instead of each face's box.
Writes <name>.jpg per photo (rows: faces or the framed region; columns:
input | square mask | face parsing) and prints, per row, the mean change
outside the face-parsing mask (where the square mask altered arms, hands,
hair and background).
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))
import pyvips  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from painite_server.imaging import pipeline  # noqa: E402
from painite_server.models import face_adapter  # noqa: E402
from painite_server.models.manifest import Manifest  # noqa: E402

ROW_H = 560


def render(src: Path, model, strength: float, work: Path, job: str):
    (work / "w").mkdir(exist_ok=True)
    (work / "r").mkdir(exist_ok=True)
    out = pipeline.run(src, [pipeline.Step(model, {"strength": strength})], None, {"format": "tiff"},
                       work / "w", work / "r", job, 512, 32, 10**10, lambda p: None, lambda: False)
    return pyvips.Image.new_from_file(str(out.path)).numpy(), out.info.get("faces", [])


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True, help="installed face/gpen-bfr-1024@1.1.0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--max-faces", type=int, default=3)
    ap.add_argument("photos", nargs="+")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    manifest = Manifest.load(a.model_dir / "manifest.json")
    parsed = face_adapter.make(manifest, a.model_dir)
    square = face_adapter.make(manifest, a.model_dir)
    square.parser_path = None
    assert parsed.parser_path is not None, "the manifest names no parser"
    f_lab, f_cap = font(26), font(18)

    for spec in a.photos:
        path, _, frame = spec.partition("@")
        path = Path(path)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "in.tif"
            img = pyvips.Image.new_from_file(str(path)).autorot()
            if img.bands == 1:
                img = img.colourspace("srgb")
            img[:3].copy(interpretation="srgb").tiffsave(str(src))
            inp = pyvips.Image.new_from_file(str(src)).numpy()
            sq, faces = render(src, square, a.strength, td, "square")
            pa, _ = render(src, parsed, a.strength, td, "parse")
        h, w = inp.shape[:2]
        if frame:
            x, y, fw, fh = (int(v) for v in frame.split(","))
            boxes = [(x, y, x + fw, y + fh)]
        else:
            enh = sorted((f for f in faces if f["status"] == "enhanced"), key=lambda f: -f["box"]["w"])
            boxes = []
            for f in enh[:a.max_faces]:
                b = f["box"]
                cx, cy, s = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2, 1.6 * max(b["w"], b["h"])
                boxes.append((int(max(cx - s, 0)), int(max(cy - s, 0)), int(min(cx + s, w)), int(min(cy + s, h))))
        print(f"{path.name}: {len(faces)} faces ({sum(f['status'] == 'enhanced' for f in faces)} enhanced)")
        rows = []
        for x0, y0, x1, y1 in boxes:
            crops = [im[y0:y1, x0:x1] for im in (inp, sq, pa)]
            # Pixels the square mask changed that face parsing kept: arms, hands, hair, background.
            dsq = np.abs(crops[1].astype(int) - crops[0]).max(-1)
            dpa = np.abs(crops[2].astype(int) - crops[0]).max(-1)
            outside = dpa <= 1
            print(f"  region {x0},{y0},{x1 - x0}x{y1 - y0}: mean change outside the face, "
                  f"square {dsq[outside].mean():.2f} vs parse {dpa[outside].mean():.2f} (8-bit levels)")
            sc = ROW_H / (y1 - y0)
            tiles = [Image.fromarray(c).resize((max(1, round((x1 - x0) * sc)), ROW_H), Image.LANCZOS) for c in crops]
            row = Image.new("RGB", (sum(t.width for t in tiles) + 16 * 2, ROW_H), "white")
            xo = 0
            for t in tiles:
                row.paste(t, (xo, 0))
                xo += t.width + 16
            rows.append(row)
        if not rows:
            print("  no enhanced faces")
            continue
        head = 48
        width = max(r.width for r in rows)
        sheet = Image.new("RGB", (width, head + sum(r.height + 16 for r in rows) + 30), "white")
        d = ImageDraw.Draw(sheet)
        tw = rows[0].width // 3
        for i, label in enumerate(("Input", "Before: square paste (1.0.0)", "After: face parsing (1.1.0)")):
            d.text((i * (tw + 5) + 8, 10), label, fill="black", font=f_lab)
        y = head
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height + 16
        d.text((8, y), f"{path.name}, face strength {a.strength}", fill="#555", font=f_cap)
        out = a.out / f"{path.stem}.jpg"
        sheet.save(out, quality=90)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
