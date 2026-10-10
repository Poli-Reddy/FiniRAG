"""Prune inactive chunk versions (finding #23).

Reindexing deactivates old chunks and inserts replacements; old rows stay
for audit but grow the table, indexes and backups without bound. This script
deletes inactive chunks older than the retention window. Nothing active is
ever touched, and documents rows are never deleted.

Usage:
    python scripts/prune_chunks.py [--days 30] [--dry-run]

Run it on a schedule (cron/Render cron job) or manually after bulk
reindexing. Default retention comes from INDEX_HISTORY_RETENTION_DAYS.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")


async def prune(older_than_days: int, dry_run: bool = False,
                session_factory=None) -> dict:
    from sqlalchemy import func, select

    from backend.app.models.rag import DocumentChunk

    # session_factory is injectable so tests run against sqlite; production
    # uses the app factory (Render PostgreSQL). Never default to a global
    # that a test could mistake for production data.
    if session_factory is None:
        from backend.app.core.database import async_session_factory

        session_factory = async_session_factory
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    async with session_factory() as db:
        stmt = select(func.count(DocumentChunk.id)).where(
            DocumentChunk.is_active == False,  # noqa: E712
            DocumentChunk.updated_at < cutoff,
        )
        eligible = (await db.execute(stmt)).scalar_one()
        deleted = 0
        if eligible and not dry_run:
            from sqlalchemy import delete

            res = await db.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.is_active == False,  # noqa: E712
                    DocumentChunk.updated_at < cutoff,
                )
            )
            deleted = res.rowcount or 0
            await db.commit()
        return {"eligible": eligible, "deleted": deleted, "dry_run": dry_run,
                "older_than_days": older_than_days}


def main() -> None:
    from backend.app.core.config import settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=settings.INDEX_HISTORY_RETENTION_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(prune(args.days, dry_run=args.dry_run))
    print(result)


if __name__ == "__main__":
    main()
