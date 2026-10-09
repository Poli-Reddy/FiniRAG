"""Live-provider RAG evaluation — REQUIRES GEMINI_API_KEY + PostgreSQL.

Runs the same question set with GeminiEmbeddingProvider against PostgreSQL
(native pgvector + FTS, strict — any silent fallback fails the run).
Synthetic seed rows are rolled back; production rows are never touched.
Writes evaluation/rag_live_report.json (kept separate from the offline run).

Usage: $env:PYTHONPATH='.'; python evaluation/rag_live_evaluation.py
Without credentials it exits with a clear skip message (never fabricates).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")


async def main() -> dict:
    from backend.app.core.config import settings

    has_key = bool(os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY)
    db_url = os.getenv("DATABASE_URL") or settings.DATABASE_URL
    if not has_key:
        return _skip("GEMINI_API_KEY not set — live semantic evaluation requires credentials.")
    if not db_url.startswith(("postgresql", "postgres")):
        return _skip("DATABASE_URL is not PostgreSQL — live vector/FTS needs PostgreSQL+pgvector.")

    from sqlalchemy.ext.asyncio import create_async_engine

    from backend.app.providers.embedding_provider import GeminiEmbeddingProvider
    from backend.app.services.embedding_service import EmbeddingService
    from evaluation.rag_evaluation import main as run_eval

    provider = GeminiEmbeddingProvider()
    engine = create_async_engine(db_url)
    try:
        return await run_eval(
            provider=provider,
            engine=engine,
            out=Path(__file__).resolve().parent / "rag_live_report.json",
            live=True,
            strict_retrieval=True,
        )
    finally:
        await engine.dispose()


def _skip(reason: str) -> dict:
    result = {"status": "SKIPPED", "reason": reason + " See evaluation/rag_report.json for the offline run."}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    asyncio.run(main())
