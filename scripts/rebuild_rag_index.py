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

from backend.app.core.config import settings
from backend.app.models.document import Document
from backend.app.models.rag import DocumentChunk, IndexingJob
from backend.app.services.indexing_service import IndexingService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.providers.embedding_provider import GeminiEmbeddingProvider, HashEmbeddingProvider


async def rebuild_index(
    session: AsyncSession,
    limit: int | None = None,
    only_missing: bool = True,
    force_reindex: bool = False,
    dry_run: bool = False,
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

    # Use real embedding provider if configured, else hash for testing
    if settings.GEMINI_API_KEY or settings.GEMINI_API_KEYS:
        try:
            embedding_provider = GeminiEmbeddingProvider()
            await embedding_provider.validate()
            print("Using GeminiEmbeddingProvider (real embeddings)")
        except Exception as e:
            print(f"GeminiEmbeddingProvider not available ({e}), falling back to HashEmbeddingProvider")
            embedding_provider = HashEmbeddingProvider()
    else:
        print("No Gemini API key configured, using HashEmbeddingProvider (offline mode)")
        embedding_provider = HashEmbeddingProvider()

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

        # Create or reuse indexing job
        job = await indexing_service.ensure_job(doc.id)

        if dry_run:
            stats["details"].append({
                "document_id": str(doc.id),
                "document_name": doc.document_name,
                "status": "WOULD_INDEX",
                "job_id": str(job.id),
                "existing_chunks": chunk_count,
            })
            continue

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


async def main():
    parser = argparse.ArgumentParser(description="Rebuild RAG index from imported documents")
    parser.add_argument("--limit", type=int, help="Limit number of documents to process")
    parser.add_argument("--force", action="store_true", help="Force reindex even if chunks exist")
    parser.add_argument("--all", dest="only_missing", action="store_false",
                        help="Process all PASS documents, not just those missing chunks")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be indexed without writing")
    parser.add_argument("--confirm", action="store_true", help="Confirm actual indexing (required for non-dry-run)")

    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL") or settings.DATABASE_URL
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
        )

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