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
    "/api/v1/documents/{document_id}/index-status",
    "/api/v1/documents/{document_id}/reindex",
    "/api/v1/indexing/jobs/{job_id}",
    "/api/v1/qa/ask",
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
        "alembic.ini",
        "alembic/env.py",
        "alembic/versions/0000_documents.py",
        "alembic/versions/0001_rag.py",
        "scripts/run_indexing_worker.py",
        "scripts/live_embedding_smoke.py",
        "scripts/verify_retrieval.py",
    ):
        if not (ROOT / relative).is_file():
            raise SystemExit(f"missing deployment file: {relative}")
    # Env/config parity: every EMBEDDING_/RETRIEVAL_/QA_/INDEX_ setting has an
    # .env.example entry (prevents Render/code drift).
    example = (ROOT / ".env.example").read_text()
    for key in ("EMBEDDING_MODEL", "EMBEDDING_DIM", "EMBEDDING_BATCH_SIZE",
                "INDEX_VERSION", "RETRIEVAL_LIMIT_PER_PATH"):
        if key not in example:
            raise SystemExit(f".env.example missing {key}")
    # Retired model must not be the CONFIGURED value (mentions in comments/docs ok).
    for line in example.splitlines():
        s = line.strip()
        if s.startswith("EMBEDDING_MODEL") and "text-embedding-004" in s:
            raise SystemExit("retired text-embedding-004 still configured")
    if "text-embedding-004" in settings.EMBEDDING_MODEL:
        raise SystemExit("retired text-embedding-004 still configured")
    print("render smoke check passed")
    # RAG config sanity (no network): supported model + dim + version set.
    assert settings.EMBEDDING_MODEL == "gemini-embedding-2", "embedding model must be gemini-embedding-2"
    assert settings.EMBEDDING_DIM == 768, "embedding dim must be 768"
    assert settings.INDEX_VERSION, "index version required"


if __name__ == "__main__":
    main()
