"""Bounded manual indexing run (no paid services)."""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")


async def main(limit: int = 5):
    from backend.app.workers.indexing_worker import run_pending_jobs

    print(await run_pending_jobs(limit=limit))


if __name__ == "__main__":
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 5
    asyncio.run(main(limit))
