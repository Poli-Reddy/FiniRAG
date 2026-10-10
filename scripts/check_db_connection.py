#!/usr/bin/env python
"""
Check database connection and diagnose issues before migration.

Reads DATABASE_URL from environment. Verifies:
- Variable is set
- Connection works
- PostgreSQL version
- Database identity
- pgvector extension support
- Existing tables/data
"""

import os
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from backend.app.core.config import settings


async def main():
    # 1. Confirm DATABASE_URL is available
    db_url = os.environ.get("DATABASE_URL") or settings.DATABASE_URL
    if not db_url or "user:pass@host" in db_url:
        print("ERROR: DATABASE_URL not configured (still has placeholder)")
        print("  Set DATABASE_URL environment variable to your Render PostgreSQL URL")
        print("  Format: postgresql+asyncpg://user:password@host:5432/dbname")
        return 1

    # Mask for display
    masked = db_url
    if "@" in masked:
        user_pass, host_db = masked.split("@", 1)
        masked = f"{user_pass.split(':')[0]}:***@{host_db}"
    print(f"DATABASE_URL: {masked}")

    # 2. Connect and execute SELECT 1
    engine = create_async_engine(db_url, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        print("[OK] Connection successful (SELECT 1)")
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return 1

    # 3. Verify database identity and PostgreSQL version
    try:
        async with engine.connect() as conn:
            # Current database
            result = await conn.execute(text("SELECT current_database()"))
            db_name = result.scalar()
            print(f"Database: {db_name}")

            # PostgreSQL version
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"PostgreSQL: {version.split(',')[0]}")

            # 4. Check if database is new or has existing tables
            result = await conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]
            if tables:
                print(f"Existing tables ({len(tables)}): {', '.join(tables)}")
            else:
                print("Database is empty (no tables)")

            # 5. Verify pgvector extension support
            result = await conn.execute(text("""
                SELECT * FROM pg_available_extensions WHERE name = 'vector'
            """))
            vec_ext = result.fetchone()
            if vec_ext:
                print(f"[OK] pgvector extension available: {vec_ext[1]} (default: {vec_ext[2]})")
            else:
                print("[FAIL] pgvector extension NOT available")
                return 1

            # Check if already installed
            result = await conn.execute(text("""
                SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'
            """))
            vec_installed = result.fetchone()
            if vec_installed:
                print(f"[OK] pgvector already installed: version {vec_installed[1]}")
            else:
                print("pgvector not yet installed (will be installed by migration)")

    except Exception as e:
        print(f"[FAIL] Diagnostic query failed: {e}")
        return 1
    finally:
        await engine.dispose()

    print("\nAll pre-migration checks passed.")
    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))