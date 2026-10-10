"""
FiniRAG Indexing service — idempotent, transactional, never reruns OCR.
"""

from __future__ import annotations

import logging
from uuid import UUID

from backend.app.core.config import settings
from backend.app.core.metrics import (
    INDEXING_CHUNKS, INDEXING_JOBS, INDEXING_LATENCY, INDEXING_PENDING, metrics,
)
from backend.app.models.document import Document
from backend.app.repositories.rag_repository import ChunkRepository, IndexingJobRepository
from backend.app.services.chunking_service import ChunkingService
from backend.app.services.embedding_service import EmbeddingService

logger = logging.getLogger("finirag")


class IndexingService:
    def __init__(self, db, chunking: ChunkingService | None = None,
                 embeddings: EmbeddingService | None = None):
        self._db = db
        self._chunks = ChunkRepository(db)
        self._jobs = IndexingJobRepository(db)
        self._chunking = chunking or ChunkingService()
        self._embeddings = embeddings or EmbeddingService()

    async def ensure_job(self, document_id) -> object:
        """Idempotent: reuse active job, else create PENDING.

        The read-then-insert above races under concurrency, so creation runs
        in a savepoint: the loser gets IntegrityError from
        uq_active_job_per_document, rolls back ONLY the savepoint (the
        caller's document insert is untouched), and re-reads the winner.
        """
        from sqlalchemy.exc import IntegrityError

        active = await self._jobs.get_active_for_document(document_id)
        if active is not None:
            return active
        try:
            async with self._db.begin_nested():
                return await self._jobs.create_pending(document_id, settings.INDEX_VERSION)
        except IntegrityError:
            # Lost the race — another request inserted first.
            pass
        winner = await self._jobs.get_active_for_document(document_id)
        if winner is None:  # pragma: no cover - defensive; index guarantees a row
            raise RuntimeError("indexing job vanished after unique conflict")
        return winner

    async def process_job(self, job) -> dict:
        """Run one job: chunks -> embeddings -> persist. Never touches OCR.

        Replacement is atomic via savepoint: deactivate+insert either both
        commit or both roll back, so old active chunks are never lost on error.
        """
        import time as _time

        from sqlalchemy import select

        _start = _time.perf_counter()

        MAX_ATTEMPTS = 5
        if (job.attempt_count or 0) >= MAX_ATTEMPTS and job.status == "PENDING":
            await self._jobs.mark_failed(job, "MAX_ATTEMPTS_EXCEEDED", "retry budget spent")
            metrics.inc(INDEXING_JOBS, {"status": "failed", "error": "MAX_ATTEMPTS_EXCEEDED"})
            metrics.observe(INDEXING_LATENCY, _time.perf_counter() - _start)
            return {"status": "FAILED", "error": "MAX_ATTEMPTS_EXCEEDED"}
        stmt = select(Document).where(Document.id == job.document_id)
        doc = (await self._db.execute(stmt)).scalar_one_or_none()
        if doc is None:
            await self._jobs.mark_failed(job, "DOCUMENT_NOT_FOUND", "document row missing")
            metrics.inc(INDEXING_JOBS, {"status": "failed", "error": "DOCUMENT_NOT_FOUND"})
            metrics.observe(INDEXING_LATENCY, _time.perf_counter() - _start)
            return {"status": "FAILED", "error": "DOCUMENT_NOT_FOUND"}
        if doc.processing_status != "PASS" or not doc.extracted_data:
            # Index only genuinely available content; label clearly.
            if not doc.extracted_data:
                await self._jobs.mark_failed(job, "NO_EXTRACTED_CONTENT", "nothing indexable")
                metrics.inc(INDEXING_JOBS, {"status": "failed", "error": "NO_EXTRACTED_CONTENT"})
                metrics.observe(INDEXING_LATENCY, _time.perf_counter() - _start)
                return {"status": "FAILED", "error": "NO_EXTRACTED_CONTENT"}

        # Reuse existing vectors ONLY for same model+dim+hash (spaces incompatible).
        prior = await self._chunks.active_for_document(job.document_id)
        reuse = {f"{c.content_hash}:{c.embedding_model}:{len(list(c.embedding)) if c.embedding else 0}":
                 c.embedding for c in prior if c.embedding}
        # Stamp reuse lookup with current provider identity at embed time
        # (EmbeddingService keys reuse by hash:model:dim).

        drafts = self._chunking.chunk_document(
            document_id=str(doc.id), document_type=doc.document_type,
            extracted_data=doc.extracted_data,
            # Retained OCR/native text when available; older rows and
            # vision-fallback successes carry NULL and index derived
            # content only — honestly, not silently.
            source_text=doc.source_text or "",
            page_count=(doc.file_validation or {}).get("page_count"),
            document_name=doc.document_name)
        if not drafts:
            await self._jobs.mark_failed(job, "NO_INDEXABLE_CONTENT", "no chunks built")
            metrics.inc(INDEXING_JOBS, {"status": "failed", "error": "NO_INDEXABLE_CONTENT"})
            metrics.observe(INDEXING_LATENCY, _time.perf_counter() - _start)
            return {"status": "FAILED", "error": "NO_INDEXABLE_CONTENT"}
        # Model/dim drift forces full re-embed (old vectors incompatible).
        for d in drafts:
            d["embedding_model"] = self._embeddings.model
        try:
            await self._embeddings.embed_chunks(drafts, existing={
                k: v for k, v in reuse.items()
                if k.endswith(f":{self._embeddings.model}:{self._embeddings.dim}")})
        except Exception as e:
            code = getattr(e, "code", "EMBEDDING_FAILED")
            await self._jobs.mark_failed(job, code, type(e).__name__)
            metrics.inc(INDEXING_JOBS, {"status": "failed", "error": code})
            metrics.observe(INDEXING_LATENCY, _time.perf_counter() - _start)
            return {"status": "FAILED", "error": code}

        for d in drafts:
            d["content_tsv"] = d["content"][:4000]
        try:
            async with self._db.begin_nested():
                await self._chunks.deactivate_document(job.document_id)
                inserted = await self._chunks.upsert_chunks(drafts)
                await self._jobs.mark_completed(job)
        except Exception as e:
            # The savepoint above already rolled back ONLY the chunk
            # replacement on context exit. Never call session.rollback() here:
            # it would also undo the caller's uncommitted document insert and
            # job row, contradicting the guarantee that indexing failure never
            # invalidates a processed document. Instead detach the stale
            # in-memory chunk objects (deactivated flags, unflushed inserts)
            # so they cannot leak into a later flush, re-fetch the still-
            # persisted job row, and record the failure on it.
            self._db.expunge_all()
            fresh = await self._jobs.get_by_id(job.id)
            if fresh is not None:
                try:
                    await self._jobs.mark_failed(fresh, "INDEX_WRITE_FAILED", type(e).__name__)
                except Exception:
                    pass
            metrics.inc(INDEXING_JOBS, {"status": "failed", "error": "INDEX_WRITE_FAILED"})
            metrics.observe(INDEXING_LATENCY, _time.perf_counter() - _start)
            return {"status": "FAILED", "error": "INDEX_WRITE_FAILED"}
        metrics.inc(INDEXING_JOBS, {"status": "completed"})
        metrics.inc(INDEXING_CHUNKS, {"document_type": doc.document_type},
                    amount=float(inserted))
        metrics.observe(INDEXING_LATENCY, _time.perf_counter() - _start)
        await self._refresh_pending_gauge()
        logger.info("Indexing completed", extra={
            "document_id": str(job.document_id), "chunks": inserted})
        return {"status": "COMPLETED", "chunks": inserted}

    async def _refresh_pending_gauge(self) -> None:
        """Best-effort backlog gauge; failure must never fail the job."""
        try:
            from sqlalchemy import func, select

            from backend.app.models.rag import IndexingJob

            pending = (await self._db.execute(
                select(func.count(IndexingJob.id)).where(
                    IndexingJob.status.in_(["PENDING", "PROCESSING"]))
            )).scalar_one()
            metrics.set_gauge(INDEXING_PENDING, float(pending))
        except Exception:
            pass

    async def index_document(self, document_id) -> dict:
        job = await self.ensure_job(document_id)
        if getattr(job, "status", "") == "COMPLETED":
            return {"job_id": str(job.id), "status": "COMPLETED"}
        return {"job_id": str(job.id), **await self.process_job(job)}

    async def status_for_document(self, document_id) -> dict:
        from sqlalchemy import select, desc
        from uuid import UUID as _UUID

        from backend.app.models.rag import IndexingJob

        try:
            did = document_id if isinstance(document_id, _UUID) else _UUID(str(document_id))
        except Exception:
            did = document_id
        stmt = select(IndexingJob).where(IndexingJob.document_id == did).order_by(
            desc(IndexingJob.created_at)).limit(1)
        job = (await self._db.execute(stmt)).scalar_one_or_none()
        count = await self._chunks.count_active(document_id)
        if job is None:
            return {"status": "NOT_INDEXED", "chunk_count": count,
                    "index_version": settings.INDEX_VERSION}
        return {"status": job.status, "job_id": str(job.id), "chunk_count": count,
                "index_version": job.index_version, "attempt_count": job.attempt_count,
                "error_code": job.error_code, "error_message": job.error_message,
                "updated_at": job.updated_at.isoformat() if job.updated_at else None}
