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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.app.core.config import settings
from backend.app.core.database import init_db, close_db
from backend.app.core.logging import logger
from backend.app.api.routes.documents import router as documents_router

# ─────────────────────────────────────────────────────────────────────
# App Lifecycle
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB. Shutdown: close DB."""
    logger.info("Starting FiniRAG Document Intelligence Platform")
    await init_db()
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

# CORS — wildcard origin cannot be combined with credentials; same-origin
# frontend needs no CORS, so keep credentials off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(documents_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Keep request-level failures in the same public error envelope."""
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    code = detail.get("code", "HTTP_ERROR")
    message = detail.get("message", str(exc.detail))
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": message}},
    )

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
