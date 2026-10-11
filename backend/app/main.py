"""
FiniRAG — FastAPI Application Entry Point.

Mounts:
  - REST API at /api/v1/
  - Static files at /static/
  - HTML frontend at /
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.app.api.auth import require_api_key
from backend.app.core.config import settings
from backend.app.core.database import init_db, close_db
from backend.app.core.logging import logger
from backend.app.api.routes.documents import public_router as health_router
from backend.app.api.routes.documents import router as documents_router
from backend.app.api.routes.qa import router as qa_router

# ─────────────────────────────────────────────────────────────────────
# App Lifecycle
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + best-effort stale-job recovery. Shutdown: close DB."""
    logger.info("Starting FiniRAG Document Intelligence Platform")
    # Fail closed in production (audit finding #11): an unauthenticated
    # public deployment exposes full extracted financial records. When
    # ENVIRONMENT=production and no API_KEYS are configured, refuse to
    # boot rather than serve the API open. Local dev (ENVIRONMENT unset
    # or anything else) keeps the open default.
    from backend.app.api.auth import auth_enabled

    if settings.is_production and not auth_enabled():
        raise RuntimeError(
            "Refusing to boot: ENVIRONMENT=production but no API_KEYS are "
            "configured. Set API_KEYS in the environment before exposing "
            "this service publicly — the API serves extracted financial "
            "records and must not run open.")
    await init_db()
    try:
        from backend.app.workers.indexing_worker import recover_stale_jobs

        await recover_stale_jobs()
    except Exception as e:
        logger.warning("Startup stale recovery skipped", extra={
            "error_type": type(e).__name__})
    yield
    await close_db()
    logger.info("FiniRAG shutdown complete")


# ─────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FiniRAG — Document Intelligence Platform",
    description="Upload financial documents → Extract structured data → Validate arithmetic",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — same-origin only by default: the shipped dashboard is served by
# this same app and needs no cross-origin access at all. Set ALLOWED_ORIGINS
# (comma-separated) only for a separately hosted UI. Credentials stay off so
# a wildcard can never combine with cookies/auth.
_allowed = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(qa_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Keep request-level failures in the same public error envelope."""
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    code = detail.get("code", "HTTP_ERROR")
    message = detail.get("message", str(exc.detail))
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": message}},
        headers=getattr(exc, "headers", None),
    )


@app.middleware("http")
async def body_cap_middleware(request: Request, call_next):
    """Ingress request-body cap for the upload route (audit finding #12).

    Runs BEFORE routing/multipart parsing: FastAPI/Starlette parses the
    multipart body inside the route handler, so the handler-level bounded
    read alone cannot be the first defense against huge request bodies or
    multipart-spooling exhaustion. When Content-Length declares more than
    the file limit plus multipart framing overhead, reject with 413
    without ever parsing the body.

    Limits of this layer (stated, not hidden): requests WITHOUT a
    Content-Length (chunked encoding) cannot be pre-sized — those still
    rely on the route's bounded read (limit+1 bytes) plus the decoded
    image/PDF pixel caps. A reverse proxy `client_max_body_size`-style
    limit remains the ideal outermost defense.
    """
    if request.method == "POST" and request.url.path == "/api/v1/documents/process":
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > settings.max_file_size_bytes + 1024 * 1024:
                    return JSONResponse(
                        status_code=413,
                        content={"error": {
                            "code": "FILE_TOO_LARGE",
                            "message": f"Uploads are limited to "
                                       f"{settings.MAX_FILE_SIZE_MB} MB."}},
                    )
            except ValueError:
                pass  # malformed header: route-level bounded read decides
    return await call_next(request)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach a request id for log/response correlation. Never fails the request."""
    import uuid as _uuid

    request_id = request.headers.get("x-request-id") or str(_uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """RED metrics per route: rate, errors, duration. Never fails the request."""
    import time

    from backend.app.core.metrics import (
        HTTP_LATENCY, HTTP_REQUESTS, metrics,
    )

    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        # Record the failure, then let the global handler produce the response.
        route = _route_template(request)
        metrics.inc(HTTP_REQUESTS, {"method": request.method, "route": route,
                                    "status": "500"})
        metrics.observe(HTTP_LATENCY, time.perf_counter() - start,
                        {"method": request.method, "route": route})
        raise
    route = _route_template(request)
    metrics.inc(HTTP_REQUESTS, {"method": request.method, "route": route,
                                "status": str(status)})
    metrics.observe(HTTP_LATENCY, time.perf_counter() - start,
                    {"method": request.method, "route": route})
    return response


def _route_template(request: Request) -> str:
    """The route *template* (e.g. /api/v1/documents/{document_name}), never the
    raw path — keeps label cardinality bounded."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or "unmatched"

# Static files + templates
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None


# ─────────────────────────────────────────────────────────────────────
# Global Exception Handler (Trap P14)
# ─────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler. Returns generic error to client.
    NEVER exposes stack traces (Correction #10, Trap P14).
    """
    logger.error("Unhandled exception", extra={
        "error_type": type(exc).__name__,
        "path": str(request.url.path),
    })
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred. Please try again.",
            }
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Frontend Routes
# ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main frontend page."""
    if templates:
        return templates.TemplateResponse(request, "index.html")
    return HTMLResponse("<h1>FiniRAG — Document Intelligence Platform</h1>")


@app.get("/metrics")
async def metrics_endpoint(format: str = "prometheus",
                           _auth: None = Depends(require_api_key)):
    """Operational metrics. Prometheus text by default (`?format=json` for JSON).

    Contains no secrets, document text or financial values — only counts,
    latencies and enumerated labels. Gated like the rest of the API when
    API_KEYS is configured."""
    from fastapi.responses import PlainTextResponse

    from backend.app.core.metrics import metrics

    if format == "json":
        return JSONResponse(metrics.snapshot())
    return PlainTextResponse(metrics.render_prometheus(),
                             media_type="text/plain; version=0.0.4")
