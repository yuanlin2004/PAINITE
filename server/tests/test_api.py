from __future__ import annotations

import numpy as np
import pytest

from conftest import decode, pixels, sha


def full(source: bytes, op: str = "denoise", **extra) -> dict:
    body = {"source_id": "sha256:" + sha(source), "kind": "full", "operation": op}
    body.update(extra)
    return body


def test_health_needs_no_auth(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["queue"]["capacity"] == 32


def test_auth_required_and_revocation(client, api):
    assert client.get("/v1/capabilities").json()["error"]["code"] == "unauthorized"
    assert client.get("/v1/capabilities", headers={"Authorization": "Bearer nope"}).status_code == 401
    r = client.get("/v1/capabilities", headers=api.h)
    assert r.status_code == 200
    caps = r.json()
    assert {o["name"] for o in caps["operations"]} == {"denoise", "sharpen", "upscale2x"}
    assert all(m["placeholder"] for m in caps["models"])
    client.app.state.store.revoke_token("a")
    assert client.get("/v1/capabilities", headers=api.h).status_code == 401


def test_upload_is_content_addressed(api, images):
    data = images["jpeg"]
    assert api.c.head(f"/v1/sources/{sha(data)}", headers=api.h).status_code == 404
    r = api.upload(data)
    assert r.status_code == 200, r.text
    src = r.json()
    assert src == {
        "source_id": "sha256:" + sha(data), "bytes": len(data), "width": 200, "height": 150, "format": "jpeg",
        "bits_per_channel": 8, "orientation": 1, "has_icc": True,
    }
    assert api.upload(data).json() == src
    assert api.c.head(f"/v1/sources/{sha(data)}", headers=api.h).status_code == 200


def test_upload_errors(api, images):
    data = images["jpeg"]
    r = api.c.put(f"/v1/sources/{'0' * 64}", content=data, headers=api.h)
    assert (r.status_code, r.json()["error"]["code"]) == (400, "digest_mismatch")
    r = api.upload(images["png"])
    assert (r.status_code, r.json()["error"]["code"]) == (415, "unsupported_format")
    r = api.upload(b"not an image")
    assert r.json()["error"]["code"] == "unsupported_format"


def test_too_many_pixels(client, api, images):
    client.app.state.settings.max_input_pixels = 1000
    r = api.upload(images["jpeg"])
    assert (r.status_code, r.json()["error"]["code"]) == (422, "too_many_pixels")


def test_upload_too_large(client, api, images):
    client.app.state.settings.max_upload_bytes = 100
    r = api.upload(images["jpeg"])
    assert (r.status_code, r.json()["error"]["code"]) == (413, "upload_too_large")


def test_full_job_preserves_icc_orientation_and_size(api, images):
    data = images["jpeg_rot6"]
    api.upload(data)
    r = api.create(full(data, params={"strength": 0.6}), key="key-rot6-1")
    assert r.status_code == 201, r.text
    job = api.wait(r.json()["id"])
    assert job["state"] == "completed", job
    assert job["model"] == "denoise/identity@0.1.0"
    assert set(job["timings_ms"]) >= {"queue", "decode", "inference", "encode", "total"}
    res = api.result(job["id"])
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert res.headers["x-painite-sha256"] == sha(res.content) == job["result"]["sha256"]
    out, src = decode(res.content), decode(data)
    assert (out.width, out.height) == (src.width, src.height)
    assert out.get("orientation") == 6
    assert out.get("icc-profile-data") == src.get("icc-profile-data")
    # Identity model: only JPEG re-encoding differs.
    assert np.abs(pixels(out) - pixels(src)).mean() < 6  # identity model; JPEG re-encode of a noisy image


def test_untagged_jpeg_stays_untagged(api, images):
    data = images["jpeg_untagged"]
    api.upload(data)
    job = api.wait(api.create(full(data), key="untagged-1").json()["id"])
    out = decode(api.result(job["id"]).content)
    assert "icc-profile-data" not in out.get_fields()


def test_tiff16_identity_is_lossless(api, images):
    data = images["tiff16"]
    api.upload(data)
    job = api.wait(api.create(full(data, "sharpen", params={"strength": 1.0}), key="tiff16-1").json()["id"])
    assert job["state"] == "completed", job
    out, src = decode(api.result(job["id"]).content), decode(data)
    assert out.format == "ushort"
    assert out.get("icc-profile-data") == src.get("icc-profile-data")
    assert np.array_equal(pixels(out), pixels(src))


def test_tiff_to_jpeg_conversion(api, images):
    data = images["tiff16"]
    api.upload(data)
    job = api.wait(api.create(full(data, output={"format": "jpeg", "jpeg_quality": 90}), key="tiff2jpeg").json()["id"])
    res = api.result(job["id"])
    assert res.headers["content-type"] == "image/jpeg"
    assert decode(res.content).format == "uchar"


def test_upscale_doubles_dimensions(api, images):
    data = images["tiff8_rot8"]
    api.upload(data)
    job = api.wait(api.create(full(data, "upscale2x"), key="up-1").json()["id"])
    assert job["state"] == "completed", job
    out = decode(api.result(job["id"]).content)
    assert (out.width, out.height) == (300, 200)
    assert out.get("orientation") == 8
    assert job["result"]["width"] == 300


def test_upscale_rejects_strength(api, images):
    data = images["jpeg"]
    api.upload(data)
    r = api.create(full(data, "upscale2x", params={"strength": 0.5}), key="up-strength")
    assert (r.status_code, r.json()["error"]["code"]) == (400, "invalid_params")


@pytest.mark.parametrize("name,op", [("tiff8_rot8", "upscale2x"), ("tiff16", "denoise")])
def test_preview_matches_export_region(api, images, name, op):
    data = images[name]
    api.upload(data)
    src = decode(data)
    params = {} if op == "upscale2x" else {"params": {"strength": 0.7}}
    export = api.wait(api.create(full(data, op, **params), key=f"exp-{name}").json()["id"])
    exp_px = pixels(decode(api.result(export["id"]).content))

    body = full(data, op, **params) | {"kind": "preview", "region": {"x": 40, "y": 30, "w": 40, "h": 50}}
    preview = api.wait(api.create(body, key=f"prev-{name}").json()["id"])
    assert preview["state"] == "completed", preview
    region = preview["region"]
    # Snapped outward to the 64 px tile grid (in displayed coordinates for rotated images).
    assert region["x"] <= 40 and region["y"] <= 30
    assert region["x"] + region["w"] >= 80 and region["y"] + region["h"] >= 80
    prev_px = pixels(decode(api.result(preview["id"]).content))

    # Map the displayed region back to stored pixels to locate it in the export.
    from painite_server.imaging.geometry import Rect, display_rect_to_stored
    orientation = src.get("orientation") if "orientation" in src.get_fields() else 1
    st = display_rect_to_stored(Rect(**region), src.width, src.height, orientation)
    s = 2 if op == "upscale2x" else 1
    crop = exp_px[st.y * s:(st.y + st.h) * s, st.x * s:(st.x + st.w) * s]
    assert prev_px.shape == crop.shape
    assert np.array_equal(prev_px, crop)


def test_newer_preview_supersedes_queued_one(client, api, images):
    data = images["jpeg"]
    api.upload(data)
    client.app.state.worker.stop()  # keep jobs queued
    body = full(data) | {"kind": "preview", "region": {"x": 0, "y": 0, "w": 10, "h": 10}}
    first = api.create(body, key="prev-a").json()
    second = api.create(body | {"params": {"strength": 0.2}}, key="prev-b").json()
    assert api.c.get(f"/v1/jobs/{first['id']}", headers=api.h).json()["state"] == "canceled"
    assert api.c.get(f"/v1/jobs/{second['id']}", headers=api.h).json()["state"] == "queued"


def test_idempotency(api, images):
    data = images["jpeg"]
    api.upload(data)
    a = api.create(full(data), key="same-key-1")
    b = api.create(full(data), key="same-key-1")
    assert a.status_code == 201 and b.status_code == 200
    assert a.json()["id"] == b.json()["id"]
    c = api.create(full(data, params={"strength": 0.1}), key="same-key-1")
    assert (c.status_code, c.json()["error"]["code"]) == (409, "idempotency_conflict")
    assert api.create(full(data), key="short").json()["error"]["code"] == "bad_request"


def test_isolation_between_tokens(api, other, images):
    data = images["jpeg"]
    api.upload(data)
    # The other token cannot use, see, cancel, or download a's objects.
    assert other.create(full(data), key="other-key-1").status_code == 404
    job = api.wait(api.create(full(data), key="iso-1").json()["id"])
    for method, path in [("GET", f"/v1/jobs/{job['id']}"), ("GET", f"/v1/jobs/{job['id']}/result"),
                         ("POST", f"/v1/jobs/{job['id']}/cancel"), ("DELETE", f"/v1/jobs/{job['id']}/result")]:
        assert other.c.request(method, path, headers=other.h).status_code == 404
    # The same idempotency key under another token is a separate job.
    other.upload(data)
    assert other.create(full(data), key="iso-1").json()["id"] != job["id"]


def test_range_download_and_early_delete(api, images):
    data = images["tiff16"]
    api.upload(data)
    job = api.wait(api.create(full(data), key="range-1").json()["id"])
    whole = api.result(job["id"]).content
    part = api.result(job["id"], Range="bytes=100-")
    assert part.status_code == 206
    assert part.content == whole[100:]
    assert api.c.delete(f"/v1/jobs/{job['id']}/result", headers=api.h).status_code == 204
    r = api.result(job["id"])
    assert (r.status_code, r.json()["error"]["code"]) == (410, "result_unavailable")
    # Idempotent retry still returns the job after early deletion.
    again = api.create(full(data), key="range-1").json()
    assert again["id"] == job["id"] and again["result"]["deleted"] is True


def test_cancel_queued_job(client, api, images):
    data = images["jpeg"]
    api.upload(data)
    client.app.state.worker.stop()
    job = api.create(full(data), key="cancel-1").json()
    assert api.result(job["id"]).json()["error"]["code"] == "result_not_ready"
    r = api.c.post(f"/v1/jobs/{job['id']}/cancel", headers=api.h).json()
    assert r["state"] == "canceled" and r["error"]["code"] == "canceled"


def test_queue_full(client, api, images):
    data = images["jpeg"]
    api.upload(data)
    client.app.state.worker.stop()
    client.app.state.settings.queue_capacity = 1
    assert api.create(full(data), key="queue-1").status_code == 201
    r = api.create(full(data, params={"strength": 0.3}), key="queue-2")
    assert (r.status_code, r.json()["error"]["code"]) == (503, "queue_full")


def test_unknown_model(api, images):
    data = images["jpeg"]
    api.upload(data)
    r = api.create(full(data, model="denoise/nafnet@9"), key="model-1")
    assert (r.status_code, r.json()["error"]["code"]) == (409, "model_unavailable")


def test_sweeper_expires_results_and_idle_sources(client, api, images, settings):
    data = images["jpeg"]
    api.upload(data)
    job = api.wait(api.create(full(data), key="sweep-1").json()["id"])
    settings.result_retention_s = -1
    settings.source_idle_s = -1
    store = client.app.state.store
    store.update_job(job["id"], result_expires_at=0)
    client.app.state.worker.sweep()
    assert api.result(job["id"]).json()["error"]["code"] == "result_unavailable"
    assert api.c.head(f"/v1/sources/{sha(data)}", headers=api.h).status_code == 404
