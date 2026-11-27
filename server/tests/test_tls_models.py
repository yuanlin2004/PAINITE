from __future__ import annotations

import hashlib
import json
import ssl
from pathlib import Path

import pytest
from cryptography import x509

from painite_server import tls
from painite_server.__main__ import main
from painite_server.models import default_registry
from painite_server.models import manifest as mf


def test_admin_init_creates_ca_and_server_cert(tmp_path, capsys):
    data = tmp_path / "d"
    assert main(["--data-dir", str(data), "admin", "init", "--hostname", "spark1.local", "--ip", "192.168.29.160"]) == 0
    fp = capsys.readouterr().out.strip()
    p = tls.paths(data)
    assert fp == tls.fingerprint(p["ca_cert"].read_bytes())
    assert p["server_key"].stat().st_mode & 0o777 == 0o600
    cert = x509.load_pem_x509_certificate(p["server_cert"].read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.DNSName) == ["spark1.local"]
    assert [str(i) for i in san.get_values_for_type(x509.IPAddress)] == ["192.168.29.160"]
    # The server cert chains to the CA.
    ctx = ssl.create_default_context(cafile=str(p["ca_cert"]))
    ctx.load_cert_chain(p["server_cert"], p["server_key"])

    # Re-running keeps the CA (clients stay pinned) and reissues the server cert.
    old_server = p["server_cert"].read_bytes()
    assert main(["--data-dir", str(data), "admin", "init", "--hostname", "other.local"]) == 0
    assert capsys.readouterr().out.strip() == fp
    assert p["server_cert"].read_bytes() != old_server
    assert main(["--data-dir", str(data), "admin", "fingerprint"]) == 0
    assert capsys.readouterr().out.strip() == fp


def test_ca_endpoint(client, settings):
    assert client.get("/v1/ca.pem").json()["error"]["code"] == "not_found"
    tls.init(settings.data_dir, ["localhost"], [])
    r = client.get("/v1/ca.pem")
    assert r.status_code == 200
    assert r.content == tls.paths(settings.data_dir)["ca_cert"].read_bytes()


def _write_manifest(tmp_path: Path, model_id: str, adapter: str, default: bool = False,
                    payload: bytes = b"weights") -> Path:
    w = tmp_path / f"{hashlib.sha1(model_id.encode()).hexdigest()}.bin"
    w.write_bytes(payload)
    m = {
        "id": model_id,
        "adapter": adapter,
        "license": {"name": "Test-1.0", "url": "https://example.invalid/license"},
        "files": [{"name": "w.bin", "url": w.as_uri(), "sha256": hashlib.sha256(payload).hexdigest(),
                   "bytes": len(payload)}],
        "default": default,
    }
    path = tmp_path / f"{w.stem}.json"
    path.write_text(json.dumps(m))
    return path


def test_models_pull_requires_license_and_verifies(tmp_path, capsys):
    data = tmp_path / "d"
    man = _write_manifest(tmp_path, "denoise/test-net@1.0.0", "identity")
    assert main(["--data-dir", str(data), "models", "pull", "--manifest", str(man)]) == 1
    assert "Test-1.0" in capsys.readouterr().err
    assert not (data / "models" / "denoise").exists()

    assert main(["--data-dir", str(data), "models", "pull", "--manifest", str(man), "--accept-license"]) == 0
    d = data / "models" / "denoise" / "test-net@1.0.0"
    assert (d / "w.bin").read_bytes() == b"weights" and (d / "manifest.json").exists()
    assert main(["--data-dir", str(data), "models", "verify"]) == 0
    assert "ok" in capsys.readouterr().out

    (d / "w.bin").write_bytes(b"weightz")
    assert main(["--data-dir", str(data), "models", "verify"]) == 1
    assert main(["--data-dir", str(data), "models", "remove", "denoise/test-net@1.0.0"]) == 0
    assert not d.exists()


def test_pull_rejects_bad_hash_and_leaves_nothing(tmp_path):
    data = tmp_path / "d"
    man_path = _write_manifest(tmp_path, "denoise/bad@1.0.0", "identity")
    m = mf.Manifest.load(man_path)
    m.files[0].sha256 = "0" * 64
    with pytest.raises(mf.ManifestError):
        mf.pull(m, data)
    assert not list((data / "models").rglob("w.bin"))
    assert not list((data / "models").rglob("*.partial"))


def test_pull_prefers_the_mirror_and_falls_back_to_upstream(tmp_path):
    m = mf.Manifest.load(_write_manifest(tmp_path, "denoise/mirrored@1.0.0", "identity"))
    f = m.files[0]
    assert m.mirror_name(f) == "denoise.mirrored-1.0.0.w.bin"
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    assert mf.sources(m, f, mirror.as_uri() + "/") == [mirror.as_uri() + "/denoise.mirrored-1.0.0.w.bin", f.url]
    assert mf.sources(m, f, "") == [f.url]

    # Mirror has the file: upstream is not needed.
    (mirror / m.mirror_name(f)).write_bytes(b"weights")
    upstream = Path(f.url.removeprefix("file://"))
    upstream.unlink()
    d = mf.pull(m, tmp_path / "a", mirror=mirror.as_uri() + "/")
    assert (d / "w.bin").read_bytes() == b"weights"

    # Mirror serves the wrong bytes or nothing: upstream is used.
    upstream.write_bytes(b"weights")
    (mirror / m.mirror_name(f)).write_bytes(b"tampered")
    assert (mf.pull(m, tmp_path / "b", mirror=mirror.as_uri() + "/") / "w.bin").read_bytes() == b"weights"
    (mirror / m.mirror_name(f)).unlink()
    assert (mf.pull(m, tmp_path / "c", mirror=mirror.as_uri() + "/") / "w.bin").read_bytes() == b"weights"


def test_pull_rejects_unknown_adapter(tmp_path, capsys):
    man = _write_manifest(tmp_path, "denoise/future@1.0.0", "nafnet")
    assert main(["--data-dir", str(tmp_path / "d"), "models", "pull", "--manifest", str(man),
                 "--accept-license"]) == 1
    assert "adapter" in capsys.readouterr().err


def test_bad_manifests():
    with pytest.raises(mf.ManifestError):
        mf.Manifest.parse({"id": "denoise/x", "adapter": "identity", "license": {"name": "x"}})
    with pytest.raises(mf.ManifestError):
        mf.Manifest.parse({"id": "denoise/x@1", "adapter": "identity", "license": {"name": "x"},
                           "files": [{"name": "../evil", "url": "file:///x", "sha256": "0" * 64}]})
    with pytest.raises(mf.ManifestError):
        mf.Manifest.parse({"id": "denoise/x@1", "adapter": "identity"})


def test_registry_loads_installed_models_and_defaults(tmp_path):
    data = tmp_path / "d"
    for mid, default in [("denoise/aaa@1.0.0", False), ("denoise/bbb@2.0.0", True), ("upscale2x/up@1.0.0", False)]:
        m = mf.Manifest.load(_write_manifest(tmp_path, mid, "bilinear2x" if "upscale" in mid else "identity",
                                             default=default))
        mf.pull(m, data)
    reg = default_registry(data)
    assert reg.defaults["denoise"] == "denoise/bbb@2.0.0"
    assert reg.defaults["upscale2x"] == "upscale2x/up@1.0.0"
    assert reg.defaults["sharpen"] == "sharpen/identity@0.1.0"
    assert reg.get("denoise/aaa@1.0.0") is not None  # older versions stay pinnable
    assert reg.get("upscale2x/up@1.0.0").scale == 2


def test_newest_default_version_wins(tmp_path):
    from painite_server.models import _version_key

    assert _version_key("1.10.0") > _version_key("1.9.2") > _version_key("1.1.0") > _version_key("1.0.0")
    data = tmp_path / "d"
    for mid in ("denoise/aaa@1.0.0", "denoise/x@1.10.0", "denoise/x@1.9.0"):
        mf.pull(mf.Manifest.load(_write_manifest(tmp_path, mid, "identity", default=True)), data)
    assert default_registry(data).defaults["denoise"] == "denoise/x@1.10.0"


def test_catalog_is_valid():
    for m in mf.catalog().values():
        assert m.files, m.id


def test_token_create_output(tmp_path, capsys):
    assert main(["--data-dir", str(tmp_path / "d"), "token", "create", "--name", "mac"]) == 0
    out = capsys.readouterr()
    assert out.out.strip().startswith("pnt_")
    assert "tok_" not in out.err and "pnt_" in out.err


def test_placeholder_operations(tmp_path):
    from painite_server.models import IdentityModel

    reg = default_registry(tmp_path / "empty")
    assert reg.placeholder_operations() == sorted(reg.defaults)
    reg.add(IdentityModel(id="denoise/real@1.0.0", operation="denoise", version="1.0.0"), default=True)
    assert "denoise" not in reg.placeholder_operations()
    assert "sharpen" in reg.placeholder_operations()
