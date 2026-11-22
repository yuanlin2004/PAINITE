"""Download candidate weights into $PAINITE_BENCH_DIR/weights and verify sha256.
Also prints where to get the test-set sources (DIV2K validation HR, SIDD validation).

  python fetch.py [--op denoise] [--models ...]
"""
import argparse
import hashlib
import subprocess
import urllib.request
import zipfile

from painite_bench import arches
from painite_bench.config import WEIGHTS_DIR, candidates


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--src", action="store_true", help="also clone the pinned upstream arch repos (unblur)")
    a = ap.parse_args()
    if a.src:
        for name, (url, commit) in arches.REPOS.items():
            d = arches.SRC / name
            if not d.exists():
                subprocess.run(["git", "clone", "-q", url, str(d)], check=True)
            subprocess.run(["git", "-C", str(d), "checkout", "-q", commit], check=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for c in candidates(a.op, a.models):
        if c.baseline:
            continue
        if not c.path.exists():
            print(f"fetch {c.id} <- {c.url or 'gdrive:' + c.gdrive}")
            tmp = c.path.with_suffix(".part")
            c.path.parent.mkdir(parents=True, exist_ok=True)
            gid = c.gdrive or (c.url.removeprefix("gdrive:") if c.url.startswith("gdrive:") else None)
            if gid:
                subprocess.run(["gdown", "-q", gid, "-O", str(tmp)], check=True)
            else:
                urllib.request.urlretrieve(c.url, tmp)
            if c.member:  # weights shipped inside a zip
                with zipfile.ZipFile(tmp) as z, open(c.path, "wb") as f:
                    f.write(z.read(c.member))
                tmp.unlink()
            else:
                tmp.rename(c.path)
        got = sha256(c.path)
        status = "ok" if got == c.sha256 else f"MISMATCH (got {got})"
        print(f"{c.id:40s} {status}")


if __name__ == "__main__":
    main()
