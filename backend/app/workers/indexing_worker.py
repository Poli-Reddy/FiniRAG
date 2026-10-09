"""Bounded indexing worker — concurrency-safe claims, stale recovery.

Deployment: single Render web service. No background loop is started
automatically (free plan cannot run a second process reliably). Recovery:
- Inline indexing at upload time (bounded, isolated from doc processing).
- `recover_stale_jobs()` at app startup (non-blocking, best-effort).
- Manual: `python scripts/run_indexing_worker.py [--limit N]`.
- API: POST /documents/{id}/reindex.
"""

from __future__ import annotations

import logging

from backend.app.core.database import async_session_factory
from backend.app.repositories.rag_repository import IndexingJobRepository
from backend.app.services.indexing_service import IndexingService

logger = logging.getLogger("finirag")


async def recover_stale_jobs(stale_minutes: int = 30) -> int:
    async with async_session_factory() as db:
        try:
            n = await IndexingJobRepository(db).recover_stale(stale_minutes=stale_minutes)
            await db.commit()
            if n:
                logger.info("Indexing stale recovery", extra={"recovered": n})
            return n
        except Exception as e:
            await db.rollback()
            logger.warning("Stale recovery failed", extra={"error_type": type(e).__name__})
            return 0


async def run_pending_jobs(limit: int = 5) -> dict:
    """Claim up to `limit` PENDING jobs (SKIP LOCKED) and process each."""
    await recover_stale_jobs()
    processed = completed = failed = 0
    async with async_session_factory() as db:
        jobs = await IndexingJobRepository(db).claim_pending(limit=limit)
        job_ids = [j.id for j in jobs]
        await db.commit()
    for job_id in job_ids:
        async with async_session_factory() as db:
            try:
                job = await IndexingJobRepository(db).get_by_id(job_id)
                if job is None:
                    continue
                result = await IndexingService(db).process_job(job)
                await db.commit()
                processed += 1
                if result.get("status") == "COMPLETED":
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                try:
                    job = await IndexingJobRepository(db).get_by_id(job_id)
                    if job is not None:
                        await IndexingJobRepository(db).mark_failed(
                            job, "WORKER_ERROR", type(e).__name__)
                    await db.commit()
                except Exception:
                    await db.rollback()
                failed += 1
                logger.error("Indexing worker error", extra={"error_type": type(e).__name__})
    return {"processed": processed, "completed": completed, "failed": failed}
