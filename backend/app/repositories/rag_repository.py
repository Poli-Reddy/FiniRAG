"""Repositories for RAG tables."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select, func, desc, delete

from backend.app.models.rag import DocumentChunk, IndexingJob

logger = logging.getLogger("finirag")


def _uuid(value):
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except Exception:
        return value


class ChunkRepository:
    def __init__(self, db):
        self._db = db

    async def upsert_chunks(self, rows: list[dict]) -> int:
        """Idempotent insert: skip identical (document, version, hash) actives."""
        if not rows:
            return 0
        doc_id = _uuid(rows[0]["document_id"])
        version = rows[0].get("index_version", "v1")
        stmt = select(DocumentChunk.content_hash).where(
            DocumentChunk.document_id == doc_id,
            DocumentChunk.index_version == version,
            DocumentChunk.is_active == True,  # noqa: E712
        )
        existing = set((await self._db.execute(stmt)).scalars().all())
        fresh = [r for r in rows if r["content_hash"] not in existing]
        for r in fresh:
            rr = dict(r)
            rr["document_id"] = _uuid(rr.get("document_id"))
            # Chunk dicts carry "chunk_metadata" (the ORM attribute). Accept
            # the legacy bare "metadata" key too — it would otherwise be
            # swallowed by DeclarativeBase.metadata and identity lost.
            if "chunk_metadata" not in rr and "metadata" in rr:
                rr["chunk_metadata"] = rr.pop("metadata")
            else:
                rr.pop("metadata", None)
            self._db.add(DocumentChunk(**rr))
        if fresh:
            await self._db.flush()
        return len(fresh)

    async def deactivate_document(self, document_id) -> int:
        document_id = _uuid(document_id)
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_active == True,  # noqa: E712
        )
        res = await self._db.execute(stmt)
        n = 0
        for c in res.scalars().all():
            c.is_active = False
            n += 1
        if n:
            await self._db.flush()
        return n

    async def active_for_document(self, document_id) -> list[DocumentChunk]:
        document_id = _uuid(document_id)
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_active == True,  # noqa: E712
        ).order_by(DocumentChunk.chunk_index)
        return list((await self._db.execute(stmt)).scalars().all())

    async def count_active(self, document_id) -> int:
        document_id = _uuid(document_id)
        stmt = select(func.count(DocumentChunk.id)).where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.is_active == True,  # noqa: E712
        )
        return (await self._db.execute(stmt)).scalar_one()

    async def all_active(self, limit: int = 5000) -> list[DocumentChunk]:
        stmt = select(DocumentChunk).where(
            DocumentChunk.is_active == True  # noqa: E712
        ).order_by(desc(DocumentChunk.created_at)).limit(limit)
        return list((await self._db.execute(stmt)).scalars().all())


class IndexingJobRepository:
    def __init__(self, db):
        self._db = db

    async def get_active_for_document(self, document_id) -> IndexingJob | None:
        document_id = _uuid(document_id)
        stmt = select(IndexingJob).where(
            IndexingJob.document_id == document_id,
            IndexingJob.status.in_(["PENDING", "PROCESSING"]),
        ).order_by(desc(IndexingJob.created_at)).limit(1)
        return (await self._db.execute(stmt)).scalar_one_or_none()

    async def create_pending(self, document_id, index_version: str) -> IndexingJob:
        job = IndexingJob(document_id=_uuid(document_id), status="PENDING",
                          index_version=index_version, attempt_count=0)
        self._db.add(job)
        await self._db.flush()
        await self._db.refresh(job)
        return job

    async def get_by_id(self, job_id) -> IndexingJob | None:
        job_id = _uuid(job_id)
        stmt = select(IndexingJob).where(IndexingJob.id == job_id)
        return (await self._db.execute(stmt)).scalar_one_or_none()

    async def claim_pending(self, limit: int = 5) -> list[IndexingJob]:
        # Concurrency-safe claim: FOR UPDATE SKIP LOCKED on Postgres so two
        # workers never take the same job. SQLite fallback without locking.
        from datetime import datetime, timezone

        stmt = select(IndexingJob).where(IndexingJob.status == "PENDING").order_by(
            IndexingJob.created_at).limit(limit)
        try:
            stmt = stmt.with_for_update(skip_locked=True)
        except Exception:
            pass
        try:
            jobs = list((await self._db.execute(stmt)).scalars().all())
        except Exception:
            # SQLite or dialects without SKIP LOCKED: plain select.
            stmt = select(IndexingJob).where(IndexingJob.status == "PENDING").order_by(
                IndexingJob.created_at).limit(limit)
            jobs = list((await self._db.execute(stmt)).scalars().all())
        now = datetime.now(timezone.utc)
        for j in jobs:
            j.status = "PROCESSING"
            j.attempt_count = (j.attempt_count or 0) + 1
            j.started_at = now
            j.updated_at = now
        if jobs:
            await self._db.flush()
        return jobs

    async def recover_stale(self, stale_minutes: int = 30, max_attempts: int = 5) -> int:
        """Return crashed PROCESSING jobs to PENDING (bounded retries).

        Jobs stuck PROCESSING longer than stale_minutes go back to PENDING
        unless they exhausted max_attempts (then FAILED/TERMINAL).
        Returns number recovered.
        """
        from datetime import datetime, timezone, timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)
        stmt = select(IndexingJob).where(IndexingJob.status == "PROCESSING")
        jobs = list((await self._db.execute(stmt)).scalars().all())
        n = 0
        now = datetime.now(timezone.utc)
        for j in jobs:
            started = j.started_at
            if started is not None and getattr(started, "tzinfo", None) is None:
                started = started.replace(tzinfo=timezone.utc)
            if started is not None and started > cutoff:
                continue
            if (j.attempt_count or 0) >= max_attempts:
                j.status = "FAILED"
                j.error_code = "MAX_ATTEMPTS_EXCEEDED"
                j.error_message = "stale job exhausted retries"
                j.completed_at = now
            else:
                j.status = "PENDING"
                j.error_code = None
                j.error_message = None
            j.updated_at = now
            n += 1
        if n:
            await self._db.flush()
        return n

    async def mark_completed(self, job: IndexingJob) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        job.status = "COMPLETED"
        job.completed_at = now
        job.updated_at = now
        job.error_code = None
        job.error_message = None
        await self._db.flush()

    async def mark_failed(self, job: IndexingJob, code: str, message: str) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        job.status = "FAILED"
        job.completed_at = now
        job.updated_at = now
        job.error_code = code
        job.error_message = (message or "")[:2000]
        await self._db.flush()
