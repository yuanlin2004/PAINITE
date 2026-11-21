from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
import pyvips
from fastapi.testclient import TestClient
from PIL import ImageCms

from painite_server.app import create_app
from painite_server.config import Settings


def srgb_icc() -> bytes:
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    return profile.tobytes()


def make_image(path: Path, width: int, height: int, *, fmt: str, bits: int = 8, icc: bool = True,
               orientation: int | None = None, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([xx / max(width - 1, 1), yy / max(height - 1, 1), (xx + yy) / (width + height)], axis=-1)
    base = np.clip(base + rng.normal(0, 0.03, base.shape), 0, 1)
    maxval = 65535 if bits == 16 else 255
    arr = np.rint(base * maxval).astype(np.uint16 if bits == 16 else np.uint8)
    img = pyvips.Image.new_from_array(arr).copy(interpretation="rgb16" if bits == 16 else "srgb")
    if icc:
        img.set_type(pyvips.GValue.blob_type, "icc-profile-data", srgb_icc())
    if orientation is not None:
        img.set_type(pyvips.GValue.gint_type, "orientation", orientation)
    if fmt == "jpeg":
        img.jpegsave(str(path), Q=95)
    elif fmt == "tiff":
        img.tiffsave(str(path))
    else:
        img.pngsave(str(path))
    return path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Api:
    def __init__(self, client: TestClient, token: str):
        self.c = client
        self.h = {"Authorization": f"Bearer {token}"}

    def upload(self, data: bytes):
        return self.c.put(f"/v1/sources/{sha(data)}", content=data, headers=self.h)

    def create(self, body: dict, key: str):
        return self.c.post("/v1/jobs", json=body, headers={**self.h, "Idempotency-Key": key if key == "short" else "test-key-" + key})

    def wait(self, job_id: str, timeout: float = 30) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            job = self.c.get(f"/v1/jobs/{job_id}", headers=self.h).json()
            if job["state"] in ("completed", "failed", "canceled"):
                return job
            time.sleep(0.05)
        raise AssertionError(f"job {job_id} did not finish")

    def result(self, job_id: str, **headers):
        return self.c.get(f"/v1/jobs/{job_id}/result", headers={**self.h, **headers})


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.tile_size = 64
    s.tile_overlap = 8
    s.sweep_interval_s = 3600
    return s


@pytest.fixture
def client(settings: Settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def tokens(client: TestClient):
    store = client.app.state.store
    _, a = store.create_token("a")
    _, b = store.create_token("b")
    return a, b


@pytest.fixture
def api(client, tokens) -> Api:
    return Api(client, tokens[0])


@pytest.fixture
def other(client, tokens) -> Api:
    return Api(client, tokens[1])


def decode(data: bytes) -> pyvips.Image:
    return pyvips.Image.new_from_buffer(data, "")


def pixels(img: pyvips.Image) -> np.ndarray:
    return img.numpy().astype(np.int64)


@pytest.fixture
def images(tmp_path: Path) -> dict[str, bytes]:
    d = tmp_path / "img"
    d.mkdir()
    return {
        "jpeg": make_image(d / "a.jpg", 200, 150, fmt="jpeg").read_bytes(),
        "jpeg_untagged": make_image(d / "b.jpg", 120, 90, fmt="jpeg", icc=False, seed=1).read_bytes(),
        "jpeg_rot6": make_image(d / "c.jpg", 200, 130, fmt="jpeg", orientation=6, seed=2).read_bytes(),
        "tiff16": make_image(d / "d.tif", 190, 170, fmt="tiff", bits=16, seed=3).read_bytes(),
        "tiff8_rot8": make_image(d / "e.tif", 150, 100, fmt="tiff", orientation=8, seed=4).read_bytes(),
        "png": make_image(d / "f.png", 32, 32, fmt="png").read_bytes(),
    }


