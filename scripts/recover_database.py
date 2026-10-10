#!/usr/bin/env python
"""
FiniRAG Database Recovery - Master Orchestration Script

This script runs the complete recovery process:
1. Check database connection
2. Run Alembic migrations
3. Import processed records from JSON files
4. Rebuild RAG index
5. Verify recovery

Usage:
    python scripts/recover_database.py [--dry-run] [--limit N] [--confirm]

Environment:
    DATABASE_URL must be set to your Render PostgreSQL connection string
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import sub-scripts
from scripts.check_db_connection import main as check_db
from scripts.restore_project_data import main as restore_data
from scripts.rebuild_rag_index import main as rebuild_index
from scripts.verify_recovery import main as verify_recovery


async def run_command(cmd_args: list[str], description: str) -> int:
    """Run a sub-command and return exit code."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    # Save original argv
    original_argv = sys.argv[:]
    sys.argv = cmd_args
    
    try:
        if "Database Connection Check" in description:
            return await check_db()
        elif "Import Processed Records" in description:
            return await restore_data()
        elif "Rebuild RAG Index" in description:
            return await rebuild_index()
        elif "Verify Recovery" in description:
            return await verify_recovery()
    except SystemExit as e:
        return e.code
    except Exception as e:
        print(f"[FAIL] {description} failed with exception: {e}")
        return 1
    finally:
        sys.argv = original_argv
    
    return 0


async def main():
    parser = argparse.ArgumentParser(
        description="FiniRAG Complete Database Recovery Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps executed in order:
  1. Check database connection and pgvector support
  2. Run Alembic migrations (alembic upgrade head)
  3. Import processed records from evaluation/results/*.json
  4. Rebuild RAG index (document_chunks + embeddings)
  5. Verify recovery (counts, retrieval tests)

Environment:
  DATABASE_URL must be set to your Render PostgreSQL URL
  (postgresql+asyncpg://user:password@host:5432/dbname)

Examples:
  # Dry-run everything (no writes)
  python scripts/recover_database.py --dry-run

  # Full recovery (requires --confirm for write steps)
  python scripts/recover_database.py --confirm

  # Limit to first 10 documents for testing
  python scripts/recover_database.py --confirm --limit 10
        """
    )
    parser.add_argument("--dry-run", action="store_true", 
                        help="Run all steps in dry-run mode (no writes)")
    parser.add_argument("--confirm", action="store_true",
                        help="Confirm write operations (required for non-dry-run)")
    parser.add_argument("--limit", type=int,
                        help="Limit number of documents to process (for testing)")
    parser.add_argument("--skip-migrations", action="store_true",
                        help="Skip Alembic migrations (use with caution)")
    parser.add_argument("--skip-import", action="store_true",
                        help="Skip JSON import step")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip RAG index rebuild step")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip final verification step")
    
    args = parser.parse_args()

    # Check DATABASE_URL
    db_url = os.environ.get("DATABASE_URL")
    if not db_url or "user:pass@host" in db_url:
        print("[FAIL] DATABASE_URL not configured")
        print("  Set DATABASE_URL environment variable to your Render PostgreSQL URL")
        print("  Format: postgresql+asyncpg://user:password@host:5432/dbname")
        return 1

    masked = db_url
    if "@" in masked:
        user_pass, host_db = masked.split("@", 1)
        masked = f"{user_pass.split(':')[0]}:***@{host_db}"
    print(f"Target database: {masked}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")

    # Build command arguments for each step
    base_args = []
    if args.dry_run:
        base_args.append("--dry-run")
    if args.confirm:
        base_args.append("--confirm")
    if args.limit:
        base_args.extend(["--limit", str(args.limit)])

    results = {}

    # Step 1: Check database connection
    if not args.skip_migrations:
        print("\n[1/5] Checking database connection...")
        cmd = [sys.argv[0], "check_db"]
        code = await run_command(cmd, "1. Database Connection Check")
        results["db_check"] = code
        if code != 0:
            print("[FAIL] Database connection check failed. Aborting.")
            return 1

        # Step 2: Run Alembic migrations
        print("\n[2/5] Running Alembic migrations...")
        alembic_cmd = [sys.executable, "-m", "alembic", "upgrade", "head"]
        print(f"Running: {' '.join(alembic_cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *alembic_cmd,
            cwd=Path(__file__).resolve().parent.parent,
        )
        await proc.wait()
        if proc.returncode != 0:
            print("[FAIL] Alembic migrations failed. Aborting.")
            return 1
        print("[OK] Migrations completed")
        results["migrations"] = 0
    else:
        print("\n[2/5] Skipping Alembic migrations (--skip-migrations)")

    # Step 3: Import processed records
    if not args.skip_import:
        print("\n[3/5] Importing processed records...")
        cmd = [sys.argv[0], "restore"] + base_args
        code = await run_command(cmd, "3. Import Processed Records")
        results["import"] = code
        if code != 0:
            print("[FAIL] Import failed. Aborting.")
            return 1
    else:
        print("\n[3/5] Skipping import (--skip-import)")

    # Step 4: Rebuild RAG index
    if not args.skip_index:
        print("\n[4/5] Rebuilding RAG index...")
        cmd = [sys.argv[0], "rebuild-index"] + base_args
        code = await run_command(cmd, "4. Rebuild RAG Index")
        results["rebuild_index"] = code
        if code != 0:
            print("[FAIL] Index rebuild failed. Aborting.")
            return 1
    else:
        print("\n[4/5] Skipping index rebuild (--skip-index)")

    # Step 5: Verify recovery
    if not args.skip_verify:
        print("\n[5/5] Verifying recovery...")
        cmd = [sys.argv[0], "verify"]
        code = await run_command(cmd, "5. Verify Recovery")
        results["verify"] = code
        if code != 0:
            print("[WARN] Verification reported issues")
        else:
            print("[OK] Verification passed")
    else:
        print("\n[5/5] Skipping verification (--skip-verify)")

    # Summary
    print("\n" + "="*60)
    print("RECOVERY SUMMARY")
    print("="*60)
    for step, code in results.items():
        status = "[OK]" if code == 0 else "[FAIL]"
        print(f"  {step:20} {status}")
    
    all_ok = all(code == 0 for code in results.values())
    print(f"\nOverall: {'[OK] ALL STEPS PASSED' if all_ok else '[FAIL] SOME STEPS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import os
    sys.exit(asyncio.run(main()))