"""Model manifests, installation (`painite-server models pull`), and loading.

PAINITE does not ship weights (DESIGN.md section 5, Deployment). Each model
has a manifest naming its upstream download URLs, expected SHA-256 values, and
license. The operator pulls the weights on their own machine after accepting
the license. Installed models live side by side, one version per directory:

    <data_dir>/models/<operation>/<name>@<version>/manifest.json + files
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

log = logging.getLogger("painite.models")

ID_RE = re.compile(r"^(denoise|sharpen|unblur|upscale2x)/([a-z0-9][a-z0-9._-]*)@([0-9A-Za-z.+-]+)$")
MANIFEST = "manifest.json"


class ManifestError(ValueError):
    pass


@dataclass
class ModelFile:
    name: str
    url: str
    sha256: str
    bytes: int | None = None


@dataclass
class Manifest:
    id: str
    adapter: str
    scale: int
    license: dict[str, str]
    files: list[ModelFile] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    default: bool = False
    description: str = ""

    @property
    def operation(self) -> str:
        return self.id.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.id.split("/", 1)[1].split("@", 1)[0]

    @property
    def version(self) -> str:
        return self.id.rsplit("@", 1)[1]

    def install_dir(self, data_dir: Path) -> Path:
        return models_dir(data_dir) / self.operation / f"{self.name}@{self.version}"

    @classmethod
    def parse(cls, data: dict[str, Any]) -> Manifest:
        try:
            mid = data["id"]
            if not ID_RE.match(mid):
                raise ManifestError(f"bad model id {mid!r} (want <operation>/<name>@<version>)")
            files = []
            for f in data.get("files", []):
                if "/" in f["name"] or f["name"] in ("", ".", "..", MANIFEST):
                    raise ManifestError(f"bad file name {f['name']!r}")
                if not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]):
                    raise ManifestError(f"bad sha256 for {f['name']}")
                files.append(ModelFile(name=f["name"], url=f["url"], sha256=f["sha256"], bytes=f.get("bytes")))
            lic = data["license"]
            if not lic.get("name"):
                raise ManifestError("license.name is required")
            scale = int(data.get("scale", 2 if mid.startswith("upscale2x/") else 1))
            return cls(id=mid, adapter=data["adapter"], scale=scale, license=lic, files=files,
                       config=data.get("config", {}), default=bool(data.get("default", False)),
                       description=data.get("description", ""))
        except KeyError as e:
            raise ManifestError(f"manifest is missing {e.args[0]!r}") from e

    @classmethod
    def load(cls, path: Path) -> Manifest:
        try:
            return cls.parse(json.loads(path.read_text()))
        except json.JSONDecodeError as e:
            raise ManifestError(f"{path}: {e}") from e

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "adapter": self.adapter,
            "scale": self.scale,
            "description": self.description,
            "license": self.license,
            "files": [{"name": f.name, "url": f.url, "sha256": f.sha256, "bytes": f.bytes} for f in self.files],
            "config": self.config,
            "default": self.default,
        }


def models_dir(data_dir: Path) -> Path:
    return data_dir / "models"


def catalog() -> dict[str, Manifest]:
    """Manifests shipped with the server (models known to work with an adapter)."""
    out: dict[str, Manifest] = {}
    for entry in sorted(resources.files("painite_server.models").joinpath("catalog").iterdir(),
                        key=lambda e: e.name):
        if entry.name.endswith(".json"):
            m = Manifest.parse(json.loads(entry.read_text()))
            out[m.id] = m
    return out


def installed(data_dir: Path) -> list[tuple[Manifest, Path]]:
    """Installed models whose manifest and files are present. Files are
    size-checked here; `models verify` checks hashes."""
    out = []
    root = models_dir(data_dir)
    for mpath in sorted(root.glob(f"*/*/{MANIFEST}")):
        d = mpath.parent
        try:
            m = Manifest.load(mpath)
        except ManifestError as e:
            log.warning("skipping %s: %s", d, e)
            continue
        missing = [f.name for f in m.files if not (d / f.name).is_file()
                   or (f.bytes is not None and (d / f.name).stat().st_size != f.bytes)]
        if missing:
            log.warning("skipping %s: missing or truncated files %s", m.id, missing)
            continue
        out.append((m, d))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, progress: Callable[[int, int | None], None] | None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "painite-server"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        total = resp.headers.get("Content-Length")
        total_n = int(total) if total else None
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total_n)


def pull(m: Manifest, data_dir: Path, progress: Callable[[str, int, int | None], None] | None = None) -> Path:
    """Download and verify every file, then write the manifest last so a
    partial install is never loaded."""
    final = m.install_dir(data_dir)
    staging = final.with_name(final.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        for f in m.files:
            target = staging / f.name
            existing = final / f.name
            if existing.is_file() and _sha256(existing) == f.sha256:
                shutil.copy2(existing, target)
                continue
            _download(f.url, target, (lambda d, t, n=f.name: progress(n, d, t)) if progress else None)
            got = _sha256(target)
            if got != f.sha256:
                raise ManifestError(f"{f.name}: SHA-256 mismatch (expected {f.sha256}, got {got})")
        (staging / MANIFEST).write_text(json.dumps(m.to_json(), indent=2) + "\n")
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
        return final
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify(m: Manifest, d: Path) -> list[str]:
    """Return a list of problems (empty if the install is intact)."""
    problems = []
    for f in m.files:
        p = d / f.name
        if not p.is_file():
            problems.append(f"{f.name}: missing")
        elif _sha256(p) != f.sha256:
            problems.append(f"{f.name}: SHA-256 mismatch")
    return problems


def remove(model_id: str, data_dir: Path) -> bool:
    match = ID_RE.match(model_id)
    if not match:
        raise ManifestError(f"bad model id {model_id!r}")
    op, name, version = match.groups()
    d = models_dir(data_dir) / op / f"{name}@{version}"
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True
