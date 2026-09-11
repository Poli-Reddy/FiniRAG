"""Deployment smoke checks that do not require provider calls.

Run after installing requirements and before starting Uvicorn:

    python scripts/render_smoke_check.py

It verifies that the application imports, the expected routes exist, the
frontend assets are present, and the configured file/page limits are valid.
It deliberately does not connect to PostgreSQL or call an AI provider.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.app.core.config import settings
from backend.app.main import app


EXPECTED_ROUTES = {
    "/",
    "/docs",
    "/api/v1/documents/process",
    "/api/v1/documents",
    "/api/v1/documents/{document_name}",
    "/api/v1/health",
}


def main() -> None:
    routes = {route.path for route in app.routes}
    missing = EXPECTED_ROUTES - routes
    if missing:
        raise SystemExit(f"missing routes: {sorted(missing)}")
    if settings.MAX_FILE_SIZE_MB <= 0 or settings.MAX_PAGES <= 0:
        raise SystemExit("invalid upload limits")
    for relative in (
        "frontend/templates/index.html",
        "frontend/static/js/app.js",
        "frontend/static/css/style.css",
        "backend/requirements.txt",
        "render.yaml",
        "runtime.txt",
    ):
        if not (ROOT / relative).is_file():
            raise SystemExit(f"missing deployment file: {relative}")
    print("render smoke check passed")


if __name__ == "__main__":
    main()
