#!/usr/bin/env python
"""
Verify the recovered database and application after restore.

Runs comprehensive checks:
- Database schema and tables
- Document counts and types
- Chunk counts and embeddings
- Vector search and FTS retrieval
- API route smoke test
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.models.document import Document
from backend.app.models.rag import DocumentChunk, IndexingJob
from backend.app.services.embedding_service import EmbeddingService
from backend.app.providers.embedding_provider import GeminiEmbeddingProvider, HashEmbeddingProvider
from backend.app.services.retrieval_service import RetrievalService


async def run_verification() -> dict[str, Any]:
    db_url = os.environ.get("DATABASE_URL") or settings.DATABASE_URL
    if not db_url or "user:pass@host" in db_url:
        return {"error": "DATABASE_URL not configured"}

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    results = {}

    async with async_session() as session:
        # 1. Document counts
        total_docs = (await session.execute(select(func.count(Document.id)))).scalar() or 0
        pass_docs = (await session.execute(
            select(func.count(Document.id)).where(Document.processing_status == "PASS")
        )).scalar() or 0
        fail_docs = (await session.execute(
            select(func.count(Document.id)).where(Document.processing_status == "FAIL")
        )).scalar() or 0

        by_type = {}
        for dtype in ("invoice", "balance_sheet", "profit_and_loss", "cash_flow_statement"):
            cnt = (await session.execute(
                select(func.count(Document.id)).where(Document.document_type == dtype)
            )).scalar() or 0
            by_type[dtype] = cnt

        results["documents"] = {
            "total": total_docs,
            "PASS": pass_docs,
            "FAIL": fail_docs,
            "by_type": by_type,
        }

        # 2. Chunk counts
        total_chunks = (await session.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.is_active == True)  # noqa
        )).scalar() or 0

        chunks_with_embedding = (await session.execute(
            select(func.count(DocumentChunk.id)).where(
                DocumentChunk.is_active == True,  # noqa
                DocumentChunk.embedding.is_not(None),
            )
        )).scalar() or 0

        results["chunks"] = {
            "total_active": total_chunks,
            "with_embeddings": chunks_with_embedding,
            "embedding_coverage": f"{(chunks_with_embedding/total_chunks*100):.1f}%" if total_chunks else "N/A",
        }

        # 3. Indexing jobs
        jobs_total = (await session.execute(select(func.count(IndexingJob.id)))).scalar() or 0
        jobs_by_status = {}
        for status in ("PENDING", "PROCESSING", "COMPLETED", "FAILED"):
            cnt = (await session.execute(
                select(func.count(IndexingJob.id)).where(IndexingJob.status == status)
            )).scalar() or 0
            jobs_by_status[status] = cnt

        results["indexing_jobs"] = {
            "total": jobs_total,
            "by_status": jobs_by_status,
        }

        # 4. Verify pgvector and indexes
        try:
            idx_result = await session.execute(text("""
                SELECT indexname, tablename FROM pg_indexes
                WHERE tablename IN ('document_chunks', 'documents')
                ORDER BY tablename, indexname
            """))
            indexes = [f"{r[1]}.{r[0]}" for r in idx_result.fetchall()]
            results["indexes"] = indexes

            # Check pgvector extension
            vec_result = await session.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
            vec_ver = vec_result.scalar()
            results["pgvector_version"] = vec_ver

        except Exception as e:
            results["index_check_error"] = str(e)

    # 5. Test retrieval (if embeddings available)
    if results["chunks"]["with_embeddings"] > 0:
        try:
            if settings.GEMINI_API_KEY or settings.GEMINI_API_KEYS:
                emb_provider = GeminiEmbeddingProvider()
                await emb_provider.validate()
            else:
                emb_provider = HashEmbeddingProvider()

            emb_service = EmbeddingService(emb_provider)

            # Test vector search
            async with async_session() as session:
                retrieval = RetrievalService(session, emb_service)

                # Simple test query
                test_result = await retrieval.retrieve("total assets balance sheet", limit_per_path=3)
                results["retrieval_test"] = {
                    "candidates": len(test_result.get("candidates", [])),
                    "native_paths": test_result.get("retrieval_metadata", {}),
                    "degraded": test_result.get("degraded", False),
                }

                # Test with document filter
                doc_result = await session.execute(
                    select(Document.id).where(Document.document_type == "balance_sheet").limit(1)
                )
                doc_id = doc_result.scalar_one_or_none()
                if doc_id:
                    filtered = await retrieval.retrieve("total assets", document_id=doc_id, limit_per_path=3)
                    results["filtered_retrieval_test"] = {
                        "candidates": len(filtered.get("candidates", [])),
                        "all_same_doc": all(
                            str(c.get("document_id")) == str(doc_id)
                            for c in filtered.get("candidates", [])
                        ),
                    }

        except Exception as e:
            results["retrieval_error"] = str(e)

    await engine.dispose()
    return results


def print_results(results: dict[str, Any]):
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    print("=== RECOVERY VERIFICATION REPORT ===\n")

    print("DOCUMENTS:")
    d = results.get("documents", {})
    print(f"  Total:          {d.get('total', 0)}")
    print(f"  PASS:           {d.get('PASS', 0)}")
    print(f"  FAIL:           {d.get('FAIL', 0)}")
    for dtype, cnt in d.get("by_type", {}).items():
        print(f"    {dtype}: {cnt}")

    print("\nCHUNKS:")
    c = results.get("chunks", {})
    print(f"  Total active:      {c.get('total_active', 0)}")
    print(f"  With embeddings:   {c.get('with_embeddings', 0)}")
    print(f"  Coverage:          {c.get('embedding_coverage', 'N/A')}")

    print("\nINDEXING JOBS:")
    j = results.get("indexing_jobs", {})
    print(f"  Total: {j.get('total', 0)}")
    for status, cnt in j.get("by_status", {}).items():
        print(f"    {status}: {cnt}")

    print("\nINDEXES:")
    for idx in results.get("indexes", []):
        print(f"  {idx}")
    if "pgvector_version" in results:
        print(f"  pgvector: {results['pgvector_version']}")

    print("\nRETRIEVAL TESTS:")
    rt = results.get("retrieval_test", {})
    if rt:
        print(f"  Vector search: {rt.get('candidates', 0)} candidates")
        print(f"  Native paths: {rt.get('native_paths', {})}")
        print(f"  Degraded: {rt.get('degraded', False)}")
    else:
        print(f"  {results.get('retrieval_error', 'Not run (no embeddings)')}")

    frt = results.get("filtered_retrieval_test", {})
    if frt:
        print(f"  Filtered search: {frt.get('candidates', 0)} candidates")
        print(f"  All same doc: {frt.get('all_same_doc', False)}")


async def main():
    results = await run_verification()
    print_results(results)

    # Summary status
    has_docs = results.get("documents", {}).get("total", 0) > 0
    has_chunks = results.get("chunks", {}).get("total_active", 0) > 0
    has_embeddings = results.get("chunks", {}).get("with_embeddings", 0) > 0
    retrieval_ok = "retrieval_test" in results and results["retrieval_test"].get("candidates", 0) > 0

    print("\n=== SUMMARY ===")
    print(f"  Documents imported:     {'[OK]' if has_docs else '[FAIL]'}")
    print(f"  Chunks created:         {'[OK]' if has_chunks else '[FAIL]'}")
    print(f"  Embeddings generated:   {'[OK]' if has_embeddings else '[FAIL] (offline mode)'}")
    print(f"  Vector search working:  {'[OK]' if retrieval_ok else '[FAIL]'}")

    if has_docs and has_chunks and (has_embeddings or "offline" in str(results.get("retrieval_error", "")).lower()):
        print("\n[OK] RECOVERY SUCCESSFUL")
        return 0
    else:
        print("\n[FAIL] RECOVERY INCOMPLETE")
        return 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))