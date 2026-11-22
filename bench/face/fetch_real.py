"""Download the real-photo set listed in real_photos.txt from Wikimedia Commons into
$PAINITE_BENCH_DIR/testset/face/real/<name>.jpg and write sources.csv (title, URL, license,
author, size, camera ISO) next to them.

  python face/fetch_real.py
"""
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from painite_bench.config import TESTSET_DIR  # noqa: E402

UA = {"User-Agent": "PAINITE-bench/0.1 (https://github.com/yuanlin2004/PAINITE)"}
API = "https://commons.wikimedia.org/w/api.php?"


def _get(url):
    for attempt in range(6):  # Commons rate-limits bursts (HTTP 429)
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(15 * (attempt + 1))


def info(title):
    p = dict(action="query", format="json", titles=title, prop="imageinfo",
             iiprop="url|size|extmetadata|commonmetadata")
    r = json.loads(_get(API + urllib.parse.urlencode(p)))
    return next(iter(r["query"]["pages"].values()))["imageinfo"][0]


def main():
    out = TESTSET_DIR / "face/real"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in open(Path(__file__).with_name("real_photos.txt")):
        if not line.strip() or line.startswith("#"):
            continue
        name, cls, title = [s.strip() for s in line.split("|")]
        ii = info(title)
        dst = out / f"{name}.jpg"
        if not dst.exists():
            dst.write_bytes(_get(ii["url"]))
            time.sleep(3)
        em, cm = ii["extmetadata"], {m["name"]: m["value"] for m in ii.get("commonmetadata", [])}
        artist = re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", "")).strip()
        rows.append(dict(name=name, cls=cls, title=title, url=ii["descriptionurl"], license=em.get("LicenseShortName", {}).get("value"),
                         author=artist, width=ii["width"], height=ii["height"], iso=cm.get("ISOSpeedRatings", "")))
        print(f"{name:22s} {ii['width']}x{ii['height']} ISO {rows[-1]['iso']} {rows[-1]['license']}")
    with open(out / "sources.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
