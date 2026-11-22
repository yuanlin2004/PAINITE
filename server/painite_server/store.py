"""SQLite job store (WAL). One connection guarded by a lock; the server has
one API process and one GPU worker thread, so contention is low."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tokens (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  hash TEXT NOT NULL UNIQUE,
  created_at REAL NOT NULL,
  revoked_at REAL
);
CREATE TABLE IF NOT EXISTS sources (
  owner TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  format TEXT NOT NULL,
  bits INTEGER NOT NULL,
  orientation INTEGER NOT NULL,
  has_icc INTEGER NOT NULL,
  created_at REAL NOT NULL,
  last_used_at REAL NOT NULL,
  PRIMARY KEY (owner, sha256)
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  owner TEXT NOT NULL,
  idem_key TEXT NOT NULL,
  body_hash TEXT NOT NULL,
  kind TEXT NOT NULL,
  operation TEXT NOT NULL,
  model TEXT NOT NULL,
  source_sha TEXT NOT NULL,
  params TEXT NOT NULL,
  region TEXT,
  output TEXT NOT NULL,
  state TEXT NOT NULL,
  progress REAL NOT NULL DEFAULT 0,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  error_code TEXT,
  error_message TEXT,
  created_at REAL NOT NULL,
  started_at REAL,
  finished_at REAL,
  timings TEXT NOT NULL DEFAULT '{}',
  result_path TEXT,
  result_bytes INTEGER,
  result_sha TEXT,
  result_format TEXT,
  result_width INTEGER,
  result_height INTEGER,
  result_expires_at REAL,
  result_deleted INTEGER NOT NULL DEFAULT 0,
  steps TEXT,
  UNIQUE (owner, idem_key)
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state, kind, created_at);
"""

TERMINAL = ("completed", "failed", "canceled")
ACTIVE = ("queued", "running")

_CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"


def new_id(prefix: str) -> str:
    """Prefix + 26 base32 chars: 48-bit ms timestamp then 80 random bits (ULID-like)."""
    n = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[n & 31])
        n >>= 5
    return prefix + "".join(reversed(chars))


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class Token:
    id: str
    name: str
    created_at: float
    revoked_at: float | None


class Store:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        # Databases created before pipelines lack the steps column.
        cols = {r["name"] for r in self._db.execute("PRAGMA table_info(jobs)")}
        if "steps" not in cols:
            self._db.execute("ALTER TABLE jobs ADD COLUMN steps TEXT")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _exec(self, sql: str, args: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, args)

    def _one(self, sql: str, args: tuple | dict = ()) -> dict | None:
        with self._lock:
            row = self._db.execute(sql, args).fetchone()
        return dict(row) if row else None

    def _all(self, sql: str, args: tuple | dict = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    # Tokens -----------------------------------------------------------------

    def create_token(self, name: str) -> tuple[Token, str]:
        secret = "pnt_" + secrets.token_urlsafe(32)
        tok = Token(id=new_id("tok_"), name=name, created_at=time.time(), revoked_at=None)
        try:
            self._exec(
                "INSERT INTO tokens (id, name, hash, created_at) VALUES (?, ?, ?, ?)",
                (tok.id, name, hash_token(secret), tok.created_at),
            )
        except sqlite3.IntegrityError as e:
            raise ValueError(f"a token named {name!r} already exists") from e
        return tok, secret

    def list_tokens(self) -> list[Token]:
        rows = self._all("SELECT id, name, created_at, revoked_at FROM tokens ORDER BY created_at")
        return [Token(**r) for r in rows]

    def revoke_token(self, name: str) -> bool:
        cur = self._exec(
            "UPDATE tokens SET revoked_at = ? WHERE name = ? AND revoked_at IS NULL", (time.time(), name)
        )
        return cur.rowcount > 0

    def owner_for_token(self, secret: str) -> str | None:
        row = self._one("SELECT id FROM tokens WHERE hash = ? AND revoked_at IS NULL", (hash_token(secret),))
        return row["id"] if row else None

    # Sources ----------------------------------------------------------------

    def get_source(self, owner: str, sha: str) -> dict | None:
        return self._one("SELECT * FROM sources WHERE owner = ? AND sha256 = ?", (owner, sha))

    def put_source(self, owner: str, sha: str, info: dict[str, Any]) -> None:
        now = time.time()
        self._exec(
            """INSERT INTO sources (owner, sha256, bytes, width, height, format, bits, orientation, has_icc,
                                    created_at, last_used_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (owner, sha256) DO UPDATE SET last_used_at = excluded.last_used_at""",
            (owner, sha, info["bytes"], info["width"], info["height"], info["format"], info["bits"],
             info["orientation"], int(info["has_icc"]), now, now),
        )

    def touch_source(self, owner: str, sha: str) -> None:
        self._exec("UPDATE sources SET last_used_at = ? WHERE owner = ? AND sha256 = ?", (time.time(), owner, sha))

    def idle_sources(self, older_than: float) -> list[dict]:
        """Sources not used since `older_than` and not referenced by an active job."""
        return self._all(
            """SELECT s.owner, s.sha256 FROM sources s
               WHERE s.last_used_at < ?
                 AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.owner = s.owner AND j.source_sha = s.sha256
                                 AND j.state IN ('queued', 'running'))""",
            (older_than,),
        )

    def all_sources(self) -> list[dict]:
        return self._all("SELECT owner, sha256 FROM sources")

    def delete_source(self, owner: str, sha: str) -> None:
        self._exec("DELETE FROM sources WHERE owner = ? AND sha256 = ?", (owner, sha))

    # Jobs -------------------------------------------------------------------

    def job_by_key(self, owner: str, key: str) -> dict | None:
        return self._one("SELECT * FROM jobs WHERE owner = ? AND idem_key = ?", (owner, key))

    def get_job(self, owner: str | None, job_id: str) -> dict | None:
        if owner is None:
            return self._one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return self._one("SELECT * FROM jobs WHERE id = ? AND owner = ?", (job_id, owner))

    def count_active(self) -> dict[str, int]:
        rows = self._all("SELECT state, COUNT(*) AS n FROM jobs WHERE state IN ('queued','running') GROUP BY state")
        counts = {"queued": 0, "running": 0}
        for r in rows:
            counts[r["state"]] = r["n"]
        return counts

    def insert_job(self, job: dict[str, Any]) -> None:
        cols = ", ".join(job)
        marks = ", ".join(f":{k}" for k in job)
        self._exec(f"INSERT INTO jobs ({cols}) VALUES ({marks})", job)

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k} = :{k}" for k in fields)
        self._exec(f"UPDATE jobs SET {sets} WHERE id = :_id", {**fields, "_id": job_id})

    def supersede_previews(self, owner: str, source_sha: str, keep_id: str) -> int:
        cur = self._exec(
            """UPDATE jobs SET state = 'canceled', finished_at = ?, error_code = 'canceled',
                               error_message = 'superseded by a newer preview'
               WHERE owner = ? AND source_sha = ? AND kind = 'preview' AND state = 'queued' AND id != ?""",
            (time.time(), owner, source_sha, keep_id),
        )
        return cur.rowcount

    def next_queued(self) -> dict | None:
        """Previews first, then full jobs, each in arrival order."""
        return self._one(
            """SELECT * FROM jobs WHERE state = 'queued'
               ORDER BY CASE kind WHEN 'preview' THEN 0 ELSE 1 END, created_at LIMIT 1"""
        )

    def claim(self, job_id: str) -> bool:
        cur = self._exec(
            "UPDATE jobs SET state = 'running', started_at = ? WHERE id = ? AND state = 'queued'",
            (time.time(), job_id),
        )
        return cur.rowcount == 1

    def requeue_running(self) -> int:
        """After a crash, running jobs go back to the queue (processing is idempotent)."""
        cur = self._exec("UPDATE jobs SET state = 'queued', started_at = NULL, progress = 0 WHERE state = 'running'")
        return cur.rowcount

    def expired_results(self, now: float) -> list[dict]:
        return self._all(
            "SELECT id, result_path FROM jobs WHERE result_path IS NOT NULL AND result_deleted = 0 "
            "AND result_expires_at < ?",
            (now,),
        )

    def live_result_paths(self) -> set[str]:
        rows = self._all("SELECT result_path FROM jobs WHERE result_path IS NOT NULL AND result_deleted = 0")
        return {r["result_path"] for r in rows}

    def purge_old_jobs(self, finished_before: float) -> int:
        cur = self._exec(
            "DELETE FROM jobs WHERE state IN ('completed','failed','canceled') AND finished_at < ? "
            "AND (result_deleted = 1 OR result_path IS NULL)",
            (finished_before,),
        )
        return cur.rowcount


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
