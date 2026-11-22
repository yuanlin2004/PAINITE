"""Download candidate weights into $PAINITE_BENCH_DIR/weights and verify sha256.
Also prints where to get the test-set sources (DIV2K validation HR, SIDD validation).

  python fetch.py [--op denoise] [--models ...]
"""
import argparse
import hashlib
import subprocess
import urllib.request

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
    a = ap.parse_args()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for c in candidates(a.op, a.models):
        if c.baseline:
            continue
        if not c.path.exists():
            print(f"fetch {c.id} <- {c.url}")
            tmp = c.path.with_suffix(".part")
            if c.url.startswith("gdrive:"):
                subprocess.run(["gdown", "-q", c.url.removeprefix("gdrive:"), "-O", str(tmp)], check=True)
            else:
                urllib.request.urlretrieve(c.url, tmp)
            tmp.rename(c.path)
        got = sha256(c.path)
        status = "ok" if got == c.sha256 else f"MISMATCH (got {got})"
        print(f"{c.id:40s} {status}")


if __name__ == "__main__":
    main()
