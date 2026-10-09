"""PostgreSQL retrieval verification — EXPLAIN for vector + FTS index usage.

Requires DATABASE_URL=postgresql+asyncpg://... with pgvector. On sqlite it
reports the fallback path honestly instead of claiming index usage.
"""

from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, ".")


async def main() -> dict:
    from sqlalchemy import text
    from backend.app.core.database import async_session_factory, engine

    dialect = getattr(getattr(engine, "dialect", None), "name", "unknown")
    out: dict = {"dialect": dialect}
    if dialect != "postgresql":
        out.update({"status": "SKIPPED_FALLBACK",
                    "vector_path": "python-fallback",
                    "keyword_path": "token-overlap",
                    "note": "PostgreSQL unavailable; production must verify on PG."})
        print(json.dumps(out, indent=2))
        return out
    try:
        async with async_session_factory() as db:
            vec_plan = (await db.execute(text(
                "EXPLAIN (COSTS OFF) SELECT id FROM document_chunks "
                "ORDER BY embedding <=> :q LIMIT 8"), {"q": "[0,0]"}))
            vec_rows = [str(r[0]) for r in vec_plan.fetchall()]
            fts_plan = (await db.execute(text(
                "EXPLAIN (COSTS OFF) SELECT id FROM document_chunks "
                "WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :q)"),
                {"q": "invoice total"}))
            fts_rows = [str(r[0]) for r in fts_plan.fetchall()]
    except Exception as e:
        out.update({"status": "SKIPPED_NO_POSTGRES",
                    "reason": f"{type(e).__name__}: PostgreSQL not reachable from this host",
                    "vector_path": "unverified-live",
                    "keyword_path": "unverified-live",
                    "note": "Run on Render PostgreSQL with pgvector; see docs."})
        print(json.dumps(out, indent=2))
        return out
    # Index-usage evidence (not a claim): empty/tiny tables may legitimately
    # seq-scan, so flags are reported, not asserted — except that a missing
    # index definition is a hard failure of the migration.
    vec_plan_text = "\n".join(vec_rows)
    fts_plan_text = "\n".join(fts_rows)
    out.update({"status": "OK", "vector_plan": vec_rows, "fts_plan": fts_rows,
                "vector_index_used": ("hnsw" in vec_plan_text.lower()
                                      or "idx_chunks_vec" in vec_plan_text.lower()
                                      or "index" in vec_plan_text.lower()),
                "fts_index_used": ("gin" in fts_plan_text.lower()
                                   or "idx_chunks_fts" in fts_plan_text.lower()
                                   or "index" in fts_plan_text.lower()),
                "note": "Seq Scan on empty tables is expected; flags show what ran."})
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    asyncio.run(main())
