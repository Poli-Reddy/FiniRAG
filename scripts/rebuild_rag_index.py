#!/usr/bin/env python
"""
Rebuild the RAG index (document_chunks) from imported documents.

Uses the existing IndexingService and configured embedding provider.
Processes documents in batches with bounded retries and safe recovery.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import normalize_database_url, settings
from backend.app.models.document import Document
from backend.app.models.rag import DocumentChunk, IndexingJob
from backend.app.services.indexing_service import IndexingService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.providers.embedding_provider import GeminiEmbeddingProvider, HashEmbeddingProvider


def _db_host(db_url: str) -> str:
    """Hostname of a database URL, lowercased, for production guards."""
    try:
        return (db_url.split("@", 1)[1].split("/", 1)[0].split(":")[0] or "").lower()
    except Exception:
        return ""


def _is_production_host(db_url: str) -> bool:
    """True unless the URL is clearly local/test.

    Fail-closed default: hash embeddings are only ever permitted against
    localhost-class hosts. Anything else (Render, any remote host) is
    treated as production.
    """
    host = _db_host(db_url)
    return host not in ("localhost", "127.0.0.1", "::1", "test", "")


async def rebuild_index(
    session: AsyncSession,
    limit: int | None = None,
    only_missing: bool = True,
    force_reindex: bool = False,
    dry_run: bool = False,
    allow_hash_embeddings: bool = False,
    db_url: str | None = None,
) -> dict[str, Any]:
    """Rebuild document_chunks for all imported documents."""

    # Get documents that need indexing
    query = select(Document).where(Document.processing_status == "PASS")
    if limit:
        query = query.limit(limit)

    docs = list((await session.execute(query)).scalars().all())

    if not docs:
        return {"status": "NO_DOCUMENTS", "message": "No PASS-status documents found"}

    # Check which already have active chunks
    stats = {
        "total_documents": len(docs),
        "already_indexed": 0,
        "indexed": 0,
        "failed": 0,
        "skipped": 0,
        "details": [],
    }

    # Embedding provider — FAIL CLOSED (audit finding #3). Hash vectors
    # are non-semantic; writing them into a production chunks table makes
    # RAG return nonsense while health checks still report "embeddings
    # exist". Real Gemini embeddings, or stop with the data preserved.
    # The guard inspects the URL this session actually targets, not just
    # ambient settings.
    guard_url = normalize_database_url(
        db_url or os.environ.get("DATABASE_URL") or settings.DATABASE_URL)
    if settings.GEMINI_API_KEY or settings.GEMINI_API_KEYS:
        try:
            embedding_provider = GeminiEmbeddingProvider()
            await embedding_provider.validate()
            print("Using GeminiEmbeddingProvider (real embeddings)")
        except Exception as e:
            print(f"GeminiEmbeddingProvider unavailable ({e}).")
            print("Indexing ABORTED: imported financial data is preserved, "
                  "no hash vectors were written. Fix credentials/quota and "
                  "re-run; pass --allow-hash-embeddings ONLY for a local "
                  "offline test database.")
            return {"status": "ABORTED_NO_EMBEDDINGS",
                    "message": f"Gemini unavailable: {e}",
                    "total_documents": len(docs),
                    "already_indexed": 0, "indexed": 0,
                    "failed": 0, "skipped": 0, "details": []}
    elif allow_hash_embeddings and not _is_production_host(guard_url):
        print("Using HashEmbeddingProvider (OFFLINE TEST ONLY, local DB).")
        embedding_provider = HashEmbeddingProvider()
    else:
        print("No Gemini API key configured. Indexing ABORTED: refusing to "
              "write non-semantic hash vectors into document_chunks. "
              "Set GEMINI_API_KEY(S) and re-run; --allow-hash-embeddings "
              "is accepted ONLY against a localhost test database.")
        return {"status": "ABORTED_NO_EMBEDDINGS",
                "message": "no Gemini credentials",
                "total_documents": len(docs),
                "already_indexed": 0, "indexed": 0,
                "failed": 0, "skipped": 0, "details": []}

    embedding_service = EmbeddingService(embedding_provider)
    indexing_service = IndexingService(session, embeddings=embedding_service)

    for doc in docs:
        # Check existing active chunks
        existing_chunks = await session.execute(
            select(func.count(DocumentChunk.id)).where(
                DocumentChunk.document_id == doc.id,
                DocumentChunk.is_active == True,  # noqa: E712
            )
        )
        chunk_count = existing_chunks.scalar() or 0

        if chunk_count > 0 and only_missing and not force_reindex:
            stats["already_indexed"] += 1
            stats["details"].append({
                "document_id": str(doc.id),
                "document_name": doc.document_name,
                "status": "SKIPPED",
                "reason": f"Already has {chunk_count} active chunks",
            })
            continue

        # Create or reuse indexing job — NEVER in dry-run: ensure_job
        # inserts a PENDING row, which is a write (audit finding #2).
        if dry_run:
            from backend.app.repositories.rag_repository import IndexingJobRepository

            existing_job = await IndexingJobRepository(session).get_active_for_document(doc.id)
            stats["details"].append({
                "document_id": str(doc.id),
                "document_name": doc.document_name,
                "status": "WOULD_INDEX",
                "job_id": str(existing_job.id) if existing_job is not None else None,
                "existing_chunks": chunk_count,
            })
            continue

        job = await indexing_service.ensure_job(doc.id)

        try:
            print(f"Indexing {doc.document_name} ({doc.id})...")
            result = await indexing_service.process_job(job)

            if result.get("status") == "COMPLETED":
                stats["indexed"] += 1
                stats["details"].append({
                    "document_id": str(doc.id),
                    "document_name": doc.document_name,
                    "status": "INDEXED",
                    "job_id": str(job.id),
                    "chunks_created": result.get("chunks_created", 0),
                })
            else:
                stats["failed"] += 1
                stats["details"].append({
                    "document_id": str(doc.id),
                    "document_name": doc.document_name,
                    "status": "FAILED",
                    "job_id": str(job.id),
                    "error": result.get("error", "Unknown"),
                })

        except Exception as e:
            stats["failed"] += 1
            stats["details"].append({
                "document_id": str(doc.id),
                "document_name": doc.document_name,
                "status": "ERROR",
                "error": str(e),
            })

    return stats


async def main(argv: list[str] | None = None):
    """Entry point. `argv` is injectable so the orchestrator and tests call
    this directly — no sys.argv simulation (audit finding #2)."""
    parser = argparse.ArgumentParser(description="Rebuild RAG index from imported documents")
    parser.add_argument("--limit", type=int, help="Limit number of documents to process")
    parser.add_argument("--force", action="store_true", help="Force reindex even if chunks exist")
    parser.add_argument("--all", dest="only_missing", action="store_false",
                        help="Process all PASS documents, not just those missing chunks")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be indexed without writing")
    parser.add_argument("--confirm", action="store_true", help="Confirm actual indexing (required for non-dry-run)")
    parser.add_argument("--allow-hash-embeddings", action="store_true",
                        help="Permit non-semantic hash vectors — OFFLINE LOCAL TEST "
                             "DATABASES ONLY. Refused against any remote host.")

    args = parser.parse_args(argv)

    db_url = normalize_database_url(
        os.environ.get("DATABASE_URL") or settings.DATABASE_URL)
    if not db_url or "user:pass@host" in db_url:
        print("ERROR: DATABASE_URL not configured")
        return 1

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        stats = await rebuild_index(
            session,
            limit=args.limit,
            only_missing=args.only_missing,
            force_reindex=args.force,
            dry_run=args.dry_run,
            allow_hash_embeddings=args.allow_hash_embeddings,
            db_url=db_url,
        )

        if stats.get("status") == "ABORTED_NO_EMBEDDINGS":
            print(f"\nABORTED: {stats.get('message')}")
            print("Imported financial data is untouched. Resolve embeddings and re-run.")
            await engine.dispose()
            return 2

        print("\n=== REBUILD INDEX REPORT ===")
        for key, val in stats.items():
            if key != "details":
                print(f"  {key}: {val}")

        if stats.get("details"):
            print("\nDetails:")
            for d in stats["details"]:
                print(f"  {d['status']:12} {d.get('document_name', '')[:40]:40} {d.get('document_id', '')[:8]}...")

        if not args.dry_run:
            if not args.confirm:
                print("\nERROR: Non-dry-run requires --confirm flag")
                return 1
            try:
                await session.commit()
                print("\nChanges committed.")
            except Exception as e:
                await session.rollback()
                print(f"\nERROR: Commit failed: {e}")
                return 1

    await engine.dispose()
    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))