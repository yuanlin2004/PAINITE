import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PAINITE_BENCH_DIR", Path.home() / "painite-bench"))
WEIGHTS_DIR = DATA_DIR / "weights"
TESTSET_DIR = DATA_DIR / "testset"
OUT_DIR = DATA_DIR / "out"
RESULTS_DIR = BENCH_DIR / "results"


@dataclass
class Candidate:
    id: str
    op: str
    file: str | None = None
    url: str | None = None
    upstream: str | None = None
    sha256: str | None = None
    license: str | None = None
    baseline: str | None = None
    via: str | None = None  # sharpen by a 2x model's output area-downscaled back to 1x
    anchor: float = 0  # sigma (output px) of the low-frequency color/tone anchor, 0 = off

    @property
    def path(self) -> Path:
        return WEIGHTS_DIR / self.file

    @property
    def scale(self) -> int:
        return 2 if self.op == "upscale2x" else 1

    @property
    def model_scale(self) -> int:
        return 2 if self.via else self.scale


def candidates(op: str | None = None, ids: list[str] | None = None) -> list[Candidate]:
    with open(BENCH_DIR / "candidates.toml", "rb") as f:
        cs = [Candidate(**m) for m in tomllib.load(f)["model"]]
    by_id = {c.id: c for c in cs}
    for c in cs:
        if c.via:
            src = by_id[c.via]
            if src.via is None and src.file is None:
                raise ValueError(c.id)
            c.file, c.url, c.upstream, c.sha256, c.license = src.file, src.url, src.upstream, src.sha256, src.license
    if op:
        cs = [c for c in cs if c.op == op]
    if ids:
        cs = [c for c in cs if c.id in ids or c.id.split("/")[1] in ids]
    return cs
