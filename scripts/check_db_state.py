import asyncio
import sys

sys.path.insert(0, ".")

from sqlalchemy import text

from backend.app.core.database import async_session_factory


async def main():
    async with async_session_factory() as db:
        exts = (await db.execute(
            text("SELECT extname FROM pg_extension WHERE extname IN ('vector','pgcrypto')")
        )).fetchall()
        print("extensions:", sorted(r[0] for r in exts))
        tables = (await db.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1")
        )).fetchall()
        print("tables:", [r[0] for r in tables])
        for t in ("documents", "document_chunks", "indexing_jobs"):
            n = (await db.execute(text(f"SELECT COUNT(*) FROM {t}"))).scalar()
            print(f"{t} rows:", n)
        idx = (await db.execute(
            text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename IN "
                 "('document_chunks','indexing_jobs','documents') ORDER BY 1")
        )).fetchall()
        for name, definition in idx:
            print(f"index {name}: {definition}")
        ver = (await db.execute(text("SELECT version_num FROM alembic_version"))).fetchall()
        print("alembic:", [r[0] for r in ver])
        emb = (await db.execute(
            text("SELECT atttypid::regtype::text FROM pg_attribute "
                 "JOIN pg_class ON pg_class.oid=attrelid "
                 "WHERE relname='document_chunks' AND attname='embedding'")
        )).fetchall()
        print("embedding coltype:", [r[0] for r in emb])


asyncio.run(main())
