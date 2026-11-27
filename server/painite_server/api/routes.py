"""HTTP routes for /v1. See api/openapi.yaml."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .. import __version__
from ..errors import ApiError, not_found
from ..imaging import pipeline
from ..imaging.geometry import Rect, clip_rect, display_rect_to_stored, displayed_size, snap_to_tiles, \
    stored_rect_to_display
from ..jobs.worker import disk_usage, source_path
from ..models import OPERATION_PARAMS
from ..store import dumps, new_id

router = APIRouter(prefix="/v1")

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CONTENT_TYPES = {"jpeg": "image/jpeg", "tiff": "image/tiff"}


# Auth ------------------------------------------------------------------------

def current_owner(request: Request, authorization: Annotated[str | None, Header()] = None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise ApiError(401, "unauthorized", "missing bearer token")
    presented = authorization[7:].strip()
    owner = request.app.state.store.owner_for_token(presented)
    if owner is None and presented.startswith("tok_"):
        raise ApiError(401, "unauthorized",
                       "that is a token id, not the token; use the pnt_ secret printed by `token create`")
    if owner is None:
        raise ApiError(401, "unauthorized", "invalid or revoked token")
    return owner


Owner = Annotated[str, Depends(current_owner)]


# Models ----------------------------------------------------------------------

class RegionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(ge=1)
    h: int = Field(ge=1)


class OutputIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["jpeg", "tiff"] | None = None
    jpeg_quality: int = Field(default=92, ge=1, le=100)


class ParamsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strength: float | None = Field(default=None, ge=0, le=1)


class StepIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: str
    params: ParamsIn = Field(default_factory=ParamsIn)
    model: str | None = None


MAX_STEPS = 4


class JobRequest(BaseModel):
    """Either one `operation` (with `params` and `model`), or a pipeline of `steps`
    run in order in one job."""

    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    kind: Literal["preview", "full"]
    operation: str | None = None
    params: ParamsIn | None = None
    model: str | None = None
    steps: list[StepIn] | None = Field(default=None, min_length=1, max_length=MAX_STEPS)
    region: RegionIn | None = None
    output: OutputIn = Field(default_factory=OutputIn)

    def step_list(self) -> list[StepIn]:
        if self.steps is not None:
            if self.operation is not None or self.params is not None or self.model is not None:
                raise ApiError(400, "bad_request", "give either operation/params/model or steps, not both")
            return self.steps
        if self.operation is None:
            raise ApiError(400, "bad_request", "operation or steps is required")
        return [StepIn(operation=self.operation, params=self.params or ParamsIn(), model=self.model)]


# Helpers ---------------------------------------------------------------------

def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def job_view(job: dict, store) -> dict:
    region = None
    if job["region"]:
        src = store.get_source(job["owner"], job["source_sha"])
        r = Rect(**json.loads(job["region"]))
        region = stored_rect_to_display(r, src["width"], src["height"], src["orientation"]).as_dict() if src \
            else r.as_dict()
    result = None
    if job["result_path"]:
        result = {
            "bytes": job["result_bytes"],
            "sha256": job["result_sha"],
            "format": job["result_format"],
            "width": job["result_width"],
            "height": job["result_height"],
            "expires_at": _iso(job["result_expires_at"]),
            "deleted": bool(job["result_deleted"]),
        }
    error = {"code": job["error_code"], "message": job["error_message"]} if job["error_code"] else None
    return {
        "id": job["id"],
        "state": job["state"],
        "kind": job["kind"],
        "operation": job["operation"],
        "model": job["model"],
        "source_id": "sha256:" + job["source_sha"],
        "params": json.loads(job["params"]),
        "steps": job_steps(job),
        "region": region,
        "progress": job["progress"],
        "created_at": _iso(job["created_at"]),
        "started_at": _iso(job["started_at"]),
        "finished_at": _iso(job["finished_at"]),
        "result": result,
        "timings_ms": json.loads(job["timings"]),
        "faces": json.loads(job["info"]).get("faces") if job.get("info") else None,
        "error": error,
    }


def job_steps(job: dict) -> list[dict]:
    """The job's steps; jobs stored before pipelines have one step in the legacy columns."""
    if job.get("steps"):
        return json.loads(job["steps"])
    return [{"operation": job["operation"], "model": job["model"], "params": json.loads(job["params"])}]


def _gpu_status() -> dict:
    from ..models import devices  # noqa: PLC0415

    return devices.status()


# Routes ----------------------------------------------------------------------

@router.get("/health")
def health(request: Request) -> dict:
    state = request.app.state
    if not hasattr(state, "gpu"):
        state.gpu = _gpu_status()
    counts = state.store.count_active()
    return {
        "status": "ok",
        "version": __version__,
        "gpu": state.gpu,
        "queue": {**counts, "capacity": state.settings.queue_capacity},
        "accel": _accel_status(state.registry),
    }


def _accel_status(reg) -> dict:
    """Each model's backend: what runs now, what it is building toward (models/accel.py)."""
    from ..models import accel  # noqa: PLC0415

    models = {}
    for model_id, m in reg.models.items():
        st = m.accel_status()
        if st is not None:
            models[model_id] = st
    return {"mode": accel.mode_label(), "models": models}


@router.get("/ca.pem")
def ca_certificate(request: Request) -> Response:
    """The private CA certificate, so clients can pin it by fingerprint
    (`painite trust`). Public data; no auth."""
    from ..tls import paths  # noqa: PLC0415

    ca = paths(request.app.state.settings.data_dir)["ca_cert"]
    if not ca.exists():
        raise not_found("this server has no private CA; run `painite-server admin init`")
    return Response(ca.read_bytes(), media_type="application/x-pem-file")


@router.get("/capabilities")
def capabilities(request: Request, owner: Owner) -> dict:
    s = request.app.state.settings
    reg = request.app.state.registry
    return {
        "operations": [
            {"name": op, "default_model": reg.defaults[op], "params": OPERATION_PARAMS[op]}
            for op in OPERATION_PARAMS if op in reg.defaults
        ],
        "models": [m.describe() for m in reg.models.values()],
        "limits": {
            "max_upload_bytes": s.max_upload_bytes,
            "max_input_pixels": s.max_input_pixels,
            "max_output_pixels": s.max_output_pixels,
            "queue_capacity": s.queue_capacity,
            "result_retention_seconds": s.result_retention_s,
            "tile_size": s.tile_size,
            "tile_band": s.tile_overlap // 2,
        },
        "formats": {"input": ["jpeg", "tiff"], "output": ["jpeg", "tiff"]},
    }


def _check_sha(sha256: str) -> None:
    if not SHA_RE.match(sha256):
        raise ApiError(400, "bad_request", "source digest must be 64 lowercase hex characters")


@router.head("/sources/{sha256}")
def head_source(sha256: str, request: Request, owner: Owner) -> Response:
    _check_sha(sha256)
    store = request.app.state.store
    if store.get_source(owner, sha256) is None:
        return Response(status_code=404)
    store.touch_source(owner, sha256)
    return Response(status_code=200)


def _source_view(sha: str, row: dict) -> dict:
    return {
        "source_id": "sha256:" + sha,
        "bytes": row["bytes"],
        "width": row["width"],
        "height": row["height"],
        "format": row["format"],
        "bits_per_channel": row["bits"],
        "orientation": row["orientation"],
        "has_icc": bool(row["has_icc"]),
    }


@router.put("/sources/{sha256}")
async def put_source(sha256: str, request: Request, owner: Owner) -> dict:
    _check_sha(sha256)
    state = request.app.state
    s = state.settings
    existing = state.store.get_source(owner, sha256)
    if existing is not None:
        state.store.touch_source(owner, sha256)
        return _source_view(sha256, existing)

    declared = request.headers.get("content-length")
    if declared and int(declared) > s.max_upload_bytes:
        raise ApiError(413, "upload_too_large", f"upload exceeds {s.max_upload_bytes} bytes")
    if disk_usage(s) + int(declared or 0) > s.disk_cap_bytes:
        raise ApiError(507, "disk_full", "server storage is full; try again later")

    tmp = s.work_dir / f"upload-{new_id('')}.part"
    digest = hashlib.sha256()
    size = 0
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > s.max_upload_bytes:
                    raise ApiError(413, "upload_too_large", f"upload exceeds {s.max_upload_bytes} bytes")
                digest.update(chunk)
                f.write(chunk)
        if digest.hexdigest() != sha256:
            raise ApiError(400, "digest_mismatch", "uploaded bytes do not match the SHA-256 in the URL")
        info = pipeline.probe(tmp, s.max_input_pixels)
        dest = source_path(s, owner, sha256)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    row = {"bytes": size, **info.as_dict()}
    state.store.put_source(owner, sha256, row)
    return _source_view(sha256, row)


@router.post("/jobs")
def create_job(
    body: JobRequest,
    request: Request,
    response: Response,
    owner: Owner,
    idempotency_key: Annotated[str | None, Header()] = None,
) -> dict:
    state = request.app.state
    s, store, reg = state.settings, state.store, state.registry
    if not idempotency_key or not (8 <= len(idempotency_key) <= 128):
        raise ApiError(400, "bad_request", "Idempotency-Key header (8 to 128 characters) is required")

    body_hash = hashlib.sha256(dumps(body.model_dump(mode="json")).encode()).hexdigest()
    existing = store.job_by_key(owner, idempotency_key)
    if existing is not None:
        if existing["body_hash"] != body_hash:
            raise ApiError(409, "idempotency_conflict", "this Idempotency-Key was used with a different request")
        response.status_code = 200
        return job_view(existing, store)

    steps = []
    for i, st in enumerate(body.step_list()):
        where = f"step {i + 1}: " if body.steps is not None else ""
        if st.operation not in OPERATION_PARAMS:
            raise ApiError(400, "invalid_params", f"{where}unknown operation {st.operation!r}")
        model_id = st.model or reg.defaults.get(st.operation)
        model = reg.get(model_id) if model_id else None
        if model is None:
            raise ApiError(409, "model_unavailable", f"{where}model {model_id} is not installed")
        if model.operation != st.operation:
            raise ApiError(400, "invalid_params", f"{where}model {model_id} does not perform {st.operation}")
        params: dict = {}
        if "strength" in OPERATION_PARAMS[st.operation]:
            default = OPERATION_PARAMS[st.operation]["strength"]["default"]
            params["strength"] = st.params.strength if st.params.strength is not None else default
        elif st.params.strength is not None:
            raise ApiError(400, "invalid_params", f"{where}{st.operation} does not take a strength")
        steps.append({"operation": st.operation, "model": model.id, "params": params, "scale": model.scale})
    scale = 1
    for st in steps:
        scale *= st["scale"]
    if sum(1 for st in steps if st["scale"] > 1) > 1:
        raise ApiError(400, "invalid_params", "a pipeline can upscale at most once")

    sha = body.source_id.removeprefix("sha256:")
    src = store.get_source(owner, sha)
    if src is None or not source_path(s, owner, sha).exists():
        raise ApiError(404, "not_found", "source not found; upload it first")

    region = None
    if body.kind == "preview":
        if body.region is None:
            raise ApiError(400, "bad_request", "preview jobs need a region")
        dw, dh = displayed_size(src["width"], src["height"], src["orientation"])
        r = clip_rect(Rect(**body.region.model_dump()), dw, dh)
        if r is None:
            raise ApiError(400, "bad_request", "region is outside the image")
        stored = display_rect_to_stored(r, src["width"], src["height"], src["orientation"])
        region = snap_to_tiles(stored, s.tile_size, src["width"], src["height"], s.tile_overlap // 2)
    elif body.region is not None:
        raise ApiError(400, "bad_request", "full jobs do not take a region")

    area = region.w * region.h if region else src["width"] * src["height"]
    if area * scale ** 2 > s.max_output_pixels:
        raise ApiError(422, "too_many_pixels", "output would exceed the server's output pixel limit")

    counts = store.count_active()
    if counts["queued"] >= s.queue_capacity:
        raise ApiError(503, "queue_full", "the job queue is full; retry later")

    now = time.time()
    job = {
        "id": new_id("job_"),
        "owner": owner,
        "idem_key": idempotency_key,
        "body_hash": body_hash,
        "kind": body.kind,
        "operation": steps[0]["operation"] if len(steps) == 1 else "pipeline",
        "model": "+".join(st["model"] for st in steps),
        "source_sha": sha,
        "params": dumps(steps[0]["params"] if len(steps) == 1 else {}),
        "steps": dumps([{k: st[k] for k in ("operation", "model", "params")} for st in steps]),
        "region": dumps(region.as_dict()) if region else None,
        "output": dumps(body.output.model_dump(mode="json", exclude_none=True)),
        "state": "queued",
        "created_at": now,
    }
    store.insert_job(job)
    store.touch_source(owner, sha)
    if body.kind == "preview":
        store.supersede_previews(owner, sha, job["id"])
    state.worker.notify()
    response.status_code = 201
    return job_view(store.get_job(owner, job["id"]), store)


def _owned_job(request: Request, owner: str, job_id: str) -> dict:
    job = request.app.state.store.get_job(owner, job_id)
    if job is None:
        raise not_found("job not found")
    return job


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request, owner: Owner) -> dict:
    return job_view(_owned_job(request, owner, job_id), request.app.state.store)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request, owner: Owner) -> dict:
    store = request.app.state.store
    job = _owned_job(request, owner, job_id)
    if job["state"] == "queued":
        store.update_job(job_id, state="canceled", cancel_requested=1, finished_at=time.time(),
                         error_code="canceled", error_message="canceled by client")
    elif job["state"] == "running":
        store.update_job(job_id, cancel_requested=1)
    return job_view(store.get_job(owner, job_id), store)


@router.get("/jobs/{job_id}/result")
def get_result(job_id: str, request: Request, owner: Owner):
    job = _owned_job(request, owner, job_id)
    if job["state"] in ("queued", "running"):
        raise ApiError(409, "result_not_ready", f"job is {job['state']}")
    if job["state"] != "completed" or job["result_deleted"] or not job["result_path"] \
            or not os.path.exists(job["result_path"]):
        raise ApiError(410, "result_unavailable", "the result was deleted, expired, or never produced")
    return FileResponse(
        job["result_path"],
        media_type=CONTENT_TYPES[job["result_format"]],
        headers={"X-Painite-SHA256": job["result_sha"], "Cache-Control": "no-store"},
    )


@router.delete("/jobs/{job_id}/result", status_code=204)
def delete_result(job_id: str, request: Request, owner: Owner) -> Response:
    job = _owned_job(request, owner, job_id)
    if job["result_path"] and not job["result_deleted"]:
        os.unlink(job["result_path"]) if os.path.exists(job["result_path"]) else None
        request.app.state.store.update_job(job_id, result_deleted=1)
    return Response(status_code=204)
