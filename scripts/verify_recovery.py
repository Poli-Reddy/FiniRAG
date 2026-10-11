#!/usr/bin/env python
"""
Verify the recovered database and application after restore.

Strict by default (audit finding #4): exits non-zero on ANY of —
- manifest count mismatch (imported rows vs recovery_manifest.json)
- wrong or mixed embedding models among active chunks
- embedding dimension mismatch vs RAG_VECTOR_DIM
- missing pgvector extension, missing required indexes, or a failed
  extension/index inspection query
- any FAILED indexing job, or any PENDING/PROCESSING job left over
  (unless --allow-pending)
- degraded retrieval in strict mode (non-native vector/FTS paths,
  empty results, wrong-document filtered results)
- failed API smoke tests (when --api-url is given)

Without --api-url the HTTP smoke checks are reported as SKIPPED, never
as passed.
"""

import os
import sys
import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, func, text, distinct
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import (
    RAG_VECTOR_DIM, normalize_database_url, settings,
)
from backend.app.models.document import Document
from backend.app.models.rag import DocumentChunk, IndexingJob
from backend.app.services.embedding_service import EmbeddingService
from backend.app.providers.embedding_provider import GeminiEmbeddingProvider, HashEmbeddingProvider
from backend.app.services.retrieval_service import RetrievalService

# Indexes the native paths require. Checked by NAME (audit finding #20):
# a generic "some index was used" is not proof the intended ones exist.
REQUIRED_INDEXES = {
    "document_chunks": {"idx_chunks_vec", "idx_chunks_fts"},
}

DEFAULT_MANIFEST = str(
    Path(__file__).resolve().parent.parent / "evaluation" / "recovery_manifest.json")


def _failures() -> list[str]:
    return []


async def run_verification(manifest_path: str | None = None,
                           api_url: str | None = None,
                           allow_pending: bool = False) -> dict[str, Any]:
    db_url = normalize_database_url(
        os.environ.get("DATABASE_URL") or settings.DATABASE_URL)
    if not db_url or "user:pass@host" in db_url:
        return {"error": "DATABASE_URL not configured", "failures": ["no DATABASE_URL"]}

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    results: dict[str, Any] = {"failures": _failures()}
    fail = results["failures"].append

    async with async_session() as session:
        # 1. Document counts
        total_docs = (await session.execute(select(func.count(Document.id)))).scalar() or 0
        pass_docs = (await session.execute(
            select(func.count(Document.id)).where(Document.processing_status == "PASS")
        )).scalar() or 0
        failed_docs = (await session.execute(
            select(func.count(Document.id)).where(Document.processing_status == "FAILED")
        )).scalar() or 0

        by_type = {}
        for dtype in ("invoice", "balance_sheet", "profit_and_loss", "cash_flow_statement"):
            cnt = (await session.execute(
                select(func.count(Document.id)).where(Document.document_type == dtype)
            )).scalar() or 0
            by_type[dtype] = cnt

        results["documents"] = {
            "total": total_docs, "PASS": pass_docs, "FAILED": failed_docs,
            "by_type": by_type,
        }

        # 2. Manifest comparison (audit finding #4): imported rows must
        # match the import manifest — not a hardcoded expected count.
        manifest_file = manifest_path or DEFAULT_MANIFEST
        try:
            manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
            manifest_ids = set()
            for r in manifest.get("records", []):
                if r.get("status") in ("IMPORTED", "UPSERTED",
                                       "SKIPPED_EXISTING"):
                    # SKIPPED_EXISTING rows carry existing_id instead of id.
                    rid = r.get("id") or r.get("existing_id")
                    if rid:
                        manifest_ids.add(rid)
            db_ids = {str(i) for i in (await session.execute(
                select(Document.id))).scalars().all()}
            missing = sorted(manifest_ids - db_ids)
            extra = sorted(db_ids - manifest_ids)
            results["manifest"] = {
                "file": manifest_file,
                "manifest_records": len(manifest_ids),
                "db_rows": len(db_ids),
                "missing_from_db": missing,
                "extra_in_db": extra,
            }
            if missing or extra:
                fail(f"manifest mismatch: {len(missing)} manifest rows missing "
                     f"from DB, {len(extra)} DB rows not in manifest")
        except FileNotFoundError:
            results["manifest"] = {"file": manifest_file, "status": "NOT_FOUND"}
            fail(f"recovery manifest not found: {manifest_file} — "
                 "run restore with manifest writing enabled")
        except Exception as e:
            results["manifest"] = {"file": manifest_file, "error": str(e)}
            fail(f"manifest unreadable: {type(e).__name__}: {e}")

        # 3. Chunk + embedding-model identity (audit findings #3/#4).
        total_chunks = (await session.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.is_active == True)  # noqa
        )).scalar() or 0

        chunks_with_embedding = (await session.execute(
            select(func.count(DocumentChunk.id)).where(
                DocumentChunk.is_active == True,  # noqa
                DocumentChunk.embedding.is_not(None),
            )
        )).scalar() or 0

        model_rows = (await session.execute(
            select(distinct(DocumentChunk.embedding_model)).where(
                DocumentChunk.is_active == True,  # noqa
                DocumentChunk.embedding.is_not(None),
            )
        )).scalars().all()
        models = sorted(m or "" for m in model_rows)

        dim_bad = 0
        if chunks_with_embedding:
            vec_rows = (await session.execute(
                select(DocumentChunk.embedding).where(
                    DocumentChunk.is_active == True,  # noqa
                    DocumentChunk.embedding.is_not(None),
                ).limit(5000))).scalars().all()
            dim_bad = sum(1 for v in vec_rows
                          if not isinstance(v, (list, tuple)) or len(list(v)) != RAG_VECTOR_DIM)

        results["chunks"] = {
            "total_active": total_chunks,
            "with_embeddings": chunks_with_embedding,
            "embedding_models": models,
            "expected_model": settings.EMBEDDING_MODEL,
            "expected_dim": RAG_VECTOR_DIM,
            "dim_mismatches_sampled": dim_bad,
            "embedding_coverage": f"{(chunks_with_embedding / total_chunks * 100):.1f}%"
                                  if total_chunks else "N/A",
        }
        if total_chunks and not chunks_with_embedding:
            fail("active chunks exist but none carry embeddings")
        if models and models != [settings.EMBEDDING_MODEL]:
            fail(f"active vectors use model(s) {models}, expected exactly "
                 f"[{settings.EMBEDDING_MODEL}] — mixed vector spaces")
        if dim_bad:
            fail(f"{dim_bad} sampled vectors do not match VECTOR({RAG_VECTOR_DIM})")

        # 4. Indexing jobs: FAILED always fails; pending fails by default.
        jobs_by_status = {}
        for status in ("PENDING", "PROCESSING", "COMPLETED", "FAILED"):
            cnt = (await session.execute(
                select(func.count(IndexingJob.id)).where(IndexingJob.status == status)
            )).scalar() or 0
            jobs_by_status[status] = cnt
        results["indexing_jobs"] = {
            "total": sum(jobs_by_status.values()), "by_status": jobs_by_status}
        if jobs_by_status["FAILED"]:
            fail(f"{jobs_by_status['FAILED']} indexing job(s) FAILED")
        pending = jobs_by_status["PENDING"] + jobs_by_status["PROCESSING"]
        if pending and not allow_pending:
            fail(f"{pending} indexing job(s) still pending/processing "
                 "(pass --allow-pending to accept)")

        # 5. Extension + required index NAMES. Any inspection failure is a
        # failure — never swallowed (audit finding #4).
        try:
            idx_rows = (await session.execute(text(
                "SELECT indexname, tablename FROM pg_indexes "
                "WHERE tablename IN ('document_chunks', 'documents') "
                "ORDER BY tablename, indexname"))).fetchall()
            indexes = [f"{r[1]}.{r[0]}" for r in idx_rows]
            results["indexes"] = indexes
            by_table: dict[str, set[str]] = {}
            for table, name in [(r[1], r[0]) for r in idx_rows]:
                by_table.setdefault(table, set()).add(name)
            for table, required in REQUIRED_INDEXES.items():
                absent = required - by_table.get(table, set())
                if absent:
                    fail(f"missing required index(es) on {table}: {sorted(absent)}")

            vec_ver = (await session.execute(text(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ))).scalar()
            results["pgvector_version"] = vec_ver
            if not vec_ver:
                fail("pgvector extension 'vector' is not installed")
        except Exception as e:
            results["index_check_error"] = f"{type(e).__name__}: {e}"
            fail(f"extension/index inspection failed: {type(e).__name__}: {e}")

    # 6. Strict retrieval: native pgvector + native FTS, non-degraded,
    # non-empty, filtered results from the right document.
    if results["chunks"]["with_embeddings"] > 0:
        try:
            if settings.GEMINI_API_KEY or settings.GEMINI_API_KEYS:
                emb_provider: Any = GeminiEmbeddingProvider()
                await emb_provider.validate()
            else:
                emb_provider = HashEmbeddingProvider()

            emb_service = EmbeddingService(emb_provider)

            async with async_session() as session:
                retrieval = RetrievalService(session, emb_service)

                test_result = await retrieval.retrieve(
                    "total assets balance sheet", limit_per_path=3, strict=True)
                results["retrieval_test"] = {
                    "candidates": len(test_result.get("candidates", [])),
                    "vector_path": test_result.get("vector_path"),
                    "keyword_path": test_result.get("keyword_path"),
                    "degraded": test_result.get("degraded", True),
                    "native_errors": test_result.get("native_errors", {}),
                }
                rt = results["retrieval_test"]
                if not rt["candidates"]:
                    fail("strict retrieval returned zero candidates")
                if rt["vector_path"] != "pgvector":
                    fail(f"vector path is {rt['vector_path']!r}, not native 'pgvector'")
                if rt["keyword_path"] != "postgres-fts":
                    fail(f"keyword path is {rt['keyword_path']!r}, not native 'postgres-fts'")
                if rt["degraded"]:
                    fail(f"strict retrieval degraded: {rt['native_errors']}")

                doc_id = (await session.execute(
                    select(Document.id).where(Document.document_type == "balance_sheet")
                    .limit(1))).scalar_one_or_none()
                if doc_id:
                    filtered = await retrieval.retrieve(
                        "total assets", document_id=doc_id,
                        limit_per_path=3, strict=True)
                    cands = filtered.get("candidates", [])
                    same = all(str(c.get("document_id")) == str(doc_id) for c in cands)
                    results["filtered_retrieval_test"] = {
                        "candidates": len(cands), "all_same_doc": same}
                    if not cands:
                        fail("strict filtered retrieval returned zero candidates")
                    if not same:
                        fail("strict filtered retrieval leaked other documents' chunks")
        except Exception as e:
            results["retrieval_error"] = f"{type(e).__name__}: {e}"
            fail(f"strict retrieval raised {type(e).__name__}: {e}")
    else:
        fail("no embeddings available — retrieval not exercised")

    # 7. API smoke tests (only when --api-url is given; otherwise SKIPPED,
    # never passed).
    if api_url:
        api_failures = await _api_smoke(api_url.rstrip("/"))
        results["api_smoke"] = api_failures
        for f in api_failures:
            fail(f)
    else:
        results["api_smoke"] = ["SKIPPED (no --api-url given)"]

    await engine.dispose()
    return results


async def _api_smoke(base: str) -> list[str]:
    """Exercise the real HTTP routes. Every failure is returned, none hidden."""
    import httpx

    failures: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"{base}/api/v1/health")
            if r.status_code != 200:
                failures.append(f"GET /api/v1/health -> {r.status_code}")
            else:
                body = r.json()
                if body.get("database") != "connected":
                    failures.append("health reports database != connected")

            r = await client.get(f"{base}/api/v1/documents?limit=5&offset=0")
            if r.status_code not in (200, 401):
                failures.append(f"GET /api/v1/documents -> {r.status_code}")
            elif r.status_code == 401:
                failures.append("documents listing requires API_KEYS the smoke "
                                "client was not given (set API_KEYS header or "
                                "run against an open dev instance)")

            r = await client.post(f"{base}/api/v1/qa/ask", json={
                "question": "What is the total assets value?"})
            if r.status_code not in (200, 401):
                failures.append(f"POST /api/v1/qa/ask -> {r.status_code}")
            elif r.status_code == 200 and "answer" not in r.json():
                failures.append("qa/ask 200 without an answer payload")
    except Exception as e:
        failures.append(f"API smoke client error: {type(e).__name__}: {e}")
    return failures


def print_results(results: dict[str, Any]):
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    print("=== RECOVERY VERIFICATION REPORT ===\n")

    print("DOCUMENTS:")
    d = results.get("documents", {})
    print(f"  Total:          {d.get('total', 0)}")
    print(f"  PASS:           {d.get('PASS', 0)}")
    print(f"  FAILED:         {d.get('FAILED', 0)}")
    for dtype, cnt in d.get("by_type", {}).items():
        print(f"    {dtype}: {cnt}")

    m = results.get("manifest", {})
    print(f"\nMANIFEST: {m.get('file', '?')}")
    if "manifest_records" in m:
        print(f"  manifest rows: {m['manifest_records']}, db rows: {m['db_rows']}")
        if m.get("missing_from_db"):
            print(f"  missing from DB: {m['missing_from_db'][:10]}")
        if m.get("extra_in_db"):
            print(f"  extra in DB: {m['extra_in_db'][:10]}")
    else:
        print(f"  {m.get('status', m.get('error', 'unavailable'))}")

    print("\nCHUNKS:")
    c = results.get("chunks", {})
    print(f"  Total active:      {c.get('total_active', 0)}")
    print(f"  With embeddings:   {c.get('with_embeddings', 0)}")
    print(f"  Models:            {c.get('embedding_models', [])} "
          f"(expected [{c.get('expected_model')}])")
    print(f"  Dim mismatches:    {c.get('dim_mismatches_sampled', '?')}")
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
    if results.get("index_check_error"):
        print(f"  CHECK ERROR: {results['index_check_error']}")

    print("\nRETRIEVAL TESTS (strict, native paths required):")
    rt = results.get("retrieval_test", {})
    if rt:
        print(f"  candidates: {rt.get('candidates', 0)}, "
              f"vector={rt.get('vector_path')}, keyword={rt.get('keyword_path')}, "
              f"degraded={rt.get('degraded', True)}")
    else:
        print(f"  {results.get('retrieval_error', 'not run')}")
    frt = results.get("filtered_retrieval_test", {})
    if frt:
        print(f"  filtered: {frt.get('candidates', 0)} candidates, "
              f"all_same_doc={frt.get('all_same_doc', False)}")

    print("\nAPI SMOKE:")
    for line in results.get("api_smoke", []):
        print(f"  {line}")

    failures = results.get("failures", [])
    print(f"\n=== {'[OK] RECOVERY SUCCESSFUL' if not failures else '[FAIL] RECOVERY INCOMPLETE'} ===")
    for f in failures:
        print(f"  - {f}")


async def main(argv: list[str] | None = None) -> int:
    """Entry point. `argv` is injectable (audit finding #2)."""
    parser = argparse.ArgumentParser(description="Verify a FiniRAG database recovery")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST,
                        help="Recovery manifest to compare against")
    parser.add_argument("--api-url", type=str, default=None,
                        help="Base URL of a running app for HTTP smoke tests "
                             "(else reported SKIPPED, never passed)")
    parser.add_argument("--allow-pending", action="store_true",
                        help="Accept leftover PENDING/PROCESSING jobs (FAILED "
                             "jobs still fail verification)")
    args = parser.parse_args(argv)

    results = await run_verification(
        manifest_path=args.manifest, api_url=args.api_url,
        allow_pending=args.allow_pending)
    print_results(results)
    return 0 if not results.get("failures") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
