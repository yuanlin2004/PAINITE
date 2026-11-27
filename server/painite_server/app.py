"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .api.routes import router
from .config import Settings
from .errors import ApiError
from .jobs.worker import Worker
from .models import Registry, default_registry
from .store import Store

log = logging.getLogger("painite")


def create_app(settings: Settings | None = None, registry: Registry | None = None, start_worker: bool = True) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    store = Store(settings.db_path)
    registry = registry or default_registry(settings.data_dir, settings.tile_size, settings.tile_overlap)
    worker = Worker(settings, store, registry)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        requeued = store.requeue_running()
        if requeued:
            log.warning("requeued %d jobs interrupted by a restart", requeued)
        missing = registry.placeholder_operations()
        if missing:
            log.warning("no model installed for %s: placeholders answer instead and return the photo unchanged "
                        "(upscale2x: a plain resize). Run `painite-server models pull --all --accept-license` "
                        "with this data directory (%s), then restart.", ", ".join(missing), settings.data_dir)
        worker.sweep(startup=True)
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    app = FastAPI(title="PAINITE inference service", version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.store = store
    app.state.registry = registry
    app.state.worker = worker
    app.include_router(router)

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        code = "invalid_params" if "params" in where else "bad_request"
        return JSONResponse({"error": {"code": code, "message": f"{where}: {first.get('msg', 'invalid request')}"}},
                            status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "unauthorized", 404: "not_found"}.get(exc.status_code, "bad_request")
        return JSONResponse({"error": {"code": code, "message": str(exc.detail)}}, status_code=exc.status_code)

    return app
