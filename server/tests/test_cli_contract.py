"""Contract tests: the Go CLI against a live server with placeholder models
(DESIGN.md section 9). Skipped when Go is not installed."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from conftest import make_image
from painite_server import tls
from painite_server.app import create_app

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("go") is None, reason="go not installed")


@pytest.fixture(scope="module")
def cli_bin(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bin") / "painite"
    subprocess.run(["go", "build", "-o", str(out), "./cmd/painite"], cwd=REPO / "cli", check=True)
    return out


def _start(settings, use_tls: bool):
    app = create_app(settings)
    _, token = app.state.store.create_token("cli")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ssl_args = {}
    if use_tls:
        tls.init(settings.data_dir, ["localhost"], ["127.0.0.1"])
        p = tls.paths(settings.data_dir)
        ssl_args = {"ssl_certfile": str(p["server_cert"]), "ssl_keyfile": str(p["server_key"])}
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", **ssl_args))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    return srv, t, f"{'https' if use_tls else 'http'}://localhost:{port}", token


@pytest.fixture
def server(settings):
    srv, t, url, token = _start(settings, use_tls=False)
    yield url, token
    srv.should_exit = True
    t.join(timeout=10)


@pytest.fixture
def tls_server(settings):
    srv, t, url, token = _start(settings, use_tls=True)
    yield url, token, settings
    srv.should_exit = True
    t.join(timeout=10)


def painite(cli_bin, server, tmp_path, *args, endpoint_env=True):
    endpoint, token = server[:2]
    env = {"PAINITE_CONFIG": str(tmp_path / "cfg" / "config.json"), "PAINITE_TOKEN": token, "PATH": "/usr/bin:/bin"}
    if endpoint_env:
        env["PAINITE_ENDPOINT"] = endpoint
    return subprocess.run([str(cli_bin), *args], capture_output=True, text=True, env=env, timeout=120)


def test_single_file(cli_bin, server, tmp_path):
    src = make_image(tmp_path / "in.jpg", 300, 200, fmt="jpeg")
    out = tmp_path / "out.tif"
    r = painite(cli_bin, server, tmp_path, "enhance", "--op", "denoise", "--strength", "0.6", "--json", "-o",
                str(out), str(src))
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)["results"]
    assert [x["status"] for x in res] == ["succeeded"]
    assert res[0]["output"] == str(out) and res[0]["model"] == "denoise/identity@0.1.0"
    assert out.read_bytes()[:2] in (b"II", b"MM")
    assert out.stat().st_mode & 0o777 == 0o644
    assert "done" in r.stderr
    # Re-running without --overwrite is an invalid invocation, and the file is untouched.
    before = out.read_bytes()
    r = painite(cli_bin, server, tmp_path, "enhance", "--op", "denoise", "-o", str(out), str(src))
    assert r.returncode == 2 and out.read_bytes() == before


def test_mixed_batch_keeps_successes(cli_bin, server, tmp_path):
    good = make_image(tmp_path / "good.jpg", 120, 80, fmt="jpeg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is not a jpeg")
    good2 = make_image(tmp_path / "good2.tif", 90, 70, fmt="tiff", bits=16)
    outdir = tmp_path / "out"
    r = painite(cli_bin, server, tmp_path, "enhance", "--op", "upscale2x", "--json", "-o", str(outdir),
                str(good), str(bad), str(good2))
    assert r.returncode == 1, r.stderr
    res = json.loads(r.stdout)["results"]
    assert [x["input"] for x in res] == [str(good), str(bad), str(good2)]
    assert [x["status"] for x in res] == ["succeeded", "failed", "succeeded"]
    assert res[1]["error"]["code"] == "unsupported_format"
    assert (outdir / "good.jpg").exists() and (outdir / "good2.tif").exists()
    assert not (outdir / "bad.jpg").exists()
    assert not list(outdir.glob(".*part"))


def test_collision_submits_nothing(cli_bin, server, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = make_image(tmp_path / "a" / "img.jpg", 50, 50, fmt="jpeg")
    b = make_image(tmp_path / "b" / "img.jpg", 50, 50, fmt="jpeg")
    r = painite(cli_bin, server, tmp_path, "enhance", "--op", "sharpen", "--overwrite", "--json", "-o",
                str(tmp_path / "out"), str(a), str(b))
    assert r.returncode == 2
    res = json.loads(r.stdout)["results"]
    assert [x["error"]["code"] for x in res] == ["output_collision", "output_collision"]
    assert not (tmp_path / "out").exists()


def test_pipeline(cli_bin, server, tmp_path):
    src = make_image(tmp_path / "in.jpg", 120, 80, fmt="jpeg")
    out = tmp_path / "out.tif"
    r = painite(cli_bin, server, tmp_path, "enhance", "--op", "denoise:0.6,sharpen,upscale2x", "--strength", "0.3",
                "--json", "-o", str(out), str(src))
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)["results"][0]
    assert res["status"] == "succeeded"
    assert res["model"] == "denoise/identity@0.1.0+sharpen/identity@0.1.0+upscale2x/bilinear@0.1.0"
    import pyvips
    img = pyvips.Image.new_from_file(str(out))
    assert (img.width, img.height) == (240, 160)
    r = painite(cli_bin, server, tmp_path, "enhance", "--op", "upscale2x,upscale2x", "-o", str(tmp_path / "x.tif"),
                str(src))
    assert r.returncode == 2 and "upscale at most once" in r.stderr


def test_invalid_flags(cli_bin, server, tmp_path):
    src = make_image(tmp_path / "in.jpg", 50, 50, fmt="jpeg")
    r = painite(cli_bin, server, tmp_path, "enhance", "--op", "upscale2x", "--strength", "0.5", "-o",
                str(tmp_path / "o.jpg"), str(src))
    assert r.returncode == 2 and "strength" in r.stderr


def test_status(cli_bin, server, tmp_path):
    r = painite(cli_bin, server, tmp_path, "status", "--json")
    assert r.returncode == 0, r.stderr
    body = json.loads(r.stdout)
    assert body["health"]["status"] == "ok"
    assert any(m["id"] == "upscale2x/bilinear@0.1.0" for m in body["capabilities"]["models"])


def test_bad_token(cli_bin, server, tmp_path):
    src = make_image(tmp_path / "in.jpg", 50, 50, fmt="jpeg")
    endpoint, _ = server
    r = painite(cli_bin, (endpoint, "wrong"), tmp_path, "enhance", "--op", "denoise", "--json", "-o",
                str(tmp_path / "o.jpg"), str(src))
    assert r.returncode == 1
    assert json.loads(r.stdout)["results"][0]["error"]["code"] == "unauthorized"


def test_tls_trust_flow(cli_bin, tls_server, tmp_path):
    url, token, settings = tls_server
    fp = tls.fingerprint(tls.paths(settings.data_dir)["ca_cert"].read_bytes())
    run = lambda *a: painite(cli_bin, tls_server, tmp_path, *a, endpoint_env=False)  # noqa: E731
    assert run("config", "set", "endpoint", url).returncode == 0

    # Untrusted until pinned, with an actionable error.
    r = run("status")
    assert r.returncode == 1 and "painite trust" in r.stderr

    # Without --fingerprint: prints what the server presents, trusts nothing.
    r = run("trust")
    assert r.returncode == 1 and r.stdout.strip() == fp
    r = run("trust", "--fingerprint", "SHA256:" + "0" * 64)
    assert r.returncode == 1 and "mismatch" in r.stderr

    r = run("trust", "--fingerprint", fp.upper().replace("SHA256:", ""))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "cfg" / "ca.pem").exists()
    assert run("status").returncode == 0

    src = make_image(tmp_path / "in.jpg", 100, 80, fmt="jpeg")
    r = run("enhance", "--op", "sharpen", "-o", str(tmp_path / "out.jpg"), str(src))
    assert r.returncode == 0, r.stderr
