"""Server settings, read from PAINITE_* environment variables."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def default_data_dir() -> Path:
    """/var/lib/painite on Linux; on a Mac, where the server runs as the user,
    ~/Library/Application Support/painite-server."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "painite-server"
    return Path("/var/lib/painite")


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("PAINITE_DATA_DIR") or default_data_dir()))
    max_upload_bytes: int = field(default_factory=lambda: _env_int("PAINITE_MAX_UPLOAD_BYTES", 512 * 1024 * 1024))
    max_input_pixels: int = field(default_factory=lambda: _env_int("PAINITE_MAX_INPUT_PIXELS", 48_000_000))
    max_output_pixels: int = field(default_factory=lambda: _env_int("PAINITE_MAX_OUTPUT_PIXELS", 192_000_000))
    queue_capacity: int = field(default_factory=lambda: _env_int("PAINITE_QUEUE_CAPACITY", 32))
    result_retention_s: int = field(default_factory=lambda: _env_int("PAINITE_RESULT_RETENTION_S", 24 * 3600))
    idempotency_window_s: int = field(default_factory=lambda: _env_int("PAINITE_IDEMPOTENCY_WINDOW_S", 24 * 3600))
    # Sources with no active job are kept this long after last use, so a client
    # can upload and then create jobs (and the Mac app can request several
    # previews) without re-uploading. See DESIGN.md section 5.
    source_idle_s: int = field(default_factory=lambda: _env_int("PAINITE_SOURCE_IDLE_S", 15 * 60))
    disk_cap_bytes: int = field(default_factory=lambda: _env_int("PAINITE_DISK_CAP_BYTES", 200 * 1024**3))
    sweep_interval_s: int = field(default_factory=lambda: _env_int("PAINITE_SWEEP_INTERVAL_S", 60))
    tile_size: int = field(default_factory=lambda: _env_int("PAINITE_TILE_SIZE", 512))
    tile_overlap: int = field(default_factory=lambda: _env_int("PAINITE_TILE_OVERLAP", 64))

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "painite.sqlite3"

    def ensure_dirs(self) -> None:
        for d in (self.sources_dir, self.work_dir, self.results_dir, self.db_path.parent):
            d.mkdir(parents=True, exist_ok=True)
