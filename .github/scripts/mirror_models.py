"""Copy every catalog model file into the `models-v1` GitHub release.

Run by .github/workflows/mirror-models.yml. Downloads each file named in
server/painite_server/models/catalog/*.json from its upstream URL, checks its
SHA-256, and uploads it as <operation>.<name>-<version>.<file> (the name
painite_server.models.manifest.Manifest.mirror_name gives). Files already in
the release are skipped. Needs GH_TOKEN and the gh CLI.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import urllib.request

TAG = "models-v1"
ROOT = pathlib.Path(__file__).resolve().parents[2]
CATALOG = ROOT / "server" / "painite_server" / "models" / "catalog"
THIRD_PARTY = ROOT / "server" / "third_party" / "arches"


def gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], check=check, text=True, capture_output=True)


def release_notes(manifests: list[dict]) -> str:
    rows = "\n".join(
        f"| `{m['id']}` | [{m['license']['name']}]({m['license'].get('url', '')}) | "
        + ", ".join(f"[source]({f['url']})" for f in m["files"]) + " |"
        for m in manifests)
    return f"""Copies of the model files that `painite-server models pull` installs, kept so PAINITE keeps
working if an upstream download goes away. The server downloads from here first and falls
back to the upstream URL; every file is checked against the SHA-256 in its manifest.

**These files are not part of PAINITE and are not covered by its Apache-2.0 license.** Each
is redistributed unchanged under its authors' license, below. Some allow only
non-commercial use. The LICENSE-* files here are the full texts for the two network
definitions.

| Model | License | Upstream |
| --- | --- | --- |
{rows}
"""


def main() -> int:
    manifests = [json.loads(p.read_text()) for p in sorted(CATALOG.glob("*.json"))]
    notes = release_notes(manifests)
    if gh("release", "view", TAG, check=False).returncode != 0:
        gh("release", "create", TAG, "--title", "Model mirror (models-v1)", "--notes", notes,
           "--latest=false", "--target", "main")
    else:
        gh("release", "edit", TAG, "--notes", notes)
    have = {a["name"] for a in json.loads(gh("release", "view", TAG, "--json", "assets").stdout)["assets"]}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for lic in sorted(THIRD_PARTY.glob("LICENSE-*")):
            if lic.name not in have:
                gh("release", "upload", TAG, str(lic))
                print("uploaded", lic.name)
        for m in manifests:
            op, rest = m["id"].split("/", 1)
            name, version = rest.split("@", 1)
            for f in m["files"]:
                asset = f"{op}.{name}-{version}.{f['name']}"
                if asset in have:
                    print("have", asset)
                    continue
                path = tmp / asset
                print("downloading", f["url"], flush=True)
                req = urllib.request.Request(f["url"], headers={"User-Agent": "painite-mirror"})
                h = hashlib.sha256()
                with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out:
                    while chunk := resp.read(1 << 20):
                        h.update(chunk)
                        out.write(chunk)
                if h.hexdigest() != f["sha256"]:
                    print(f"error: {asset}: SHA-256 {h.hexdigest()}, expected {f['sha256']}", file=sys.stderr)
                    return 1
                gh("release", "upload", TAG, str(path))
                path.unlink()
                print("uploaded", asset, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
