#!/usr/bin/env python
"""
FiniRAG Database Recovery - Master Orchestration Script.

Runs the complete recovery process:
1. Check database connection (+ pgvector support)
2. Run Alembic migrations
3. Import processed records from JSON files
4. Rebuild RAG index
5. Verify recovery

Usage:
    python scripts/recover_database.py [--dry-run] [--limit N] [--confirm]

Environment:
    DATABASE_URL must be set to your Render PostgreSQL connection string.

Design notes (audit finding #2):
- Child steps are called as FUNCTIONS with explicit argument lists — never
  via sys.argv simulation, which broke argparse dispatch ("unrecognized
  arguments") in the previous implementation.
- --dry-run is strictly read-only: migrations are SKIPPED, importers and
  indexers run in dry-run mode, nothing is committed.
- The DB connectivity check is independent of --skip-migrations; skipping
  migrations skips ONLY the alembic step.
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_db_connection import main as check_db_main
from scripts.restore_project_data import main as restore_main
from scripts.rebuild_rag_index import main as rebuild_main
from scripts.verify_recovery import main as verify_main


async def run_alembic_upgrade(project_root: Path) -> int:
    """Run `alembic upgrade head` as a subprocess. Returns exit code."""
    print("Running: python -m alembic upgrade head")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "alembic", "upgrade", "head",
        cwd=str(project_root),
    )
    await proc.wait()
    return proc.returncode or 0


def _step(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"STEP: {title}")
    print(f"{'=' * 60}")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FiniRAG Complete Database Recovery Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps executed in order:
  1. Check database connection and pgvector support (read-only)
  2. Run Alembic migrations (SKIPPED under --dry-run: strictly no writes)
  3. Import processed records from evaluation/results/*.json
  4. Rebuild RAG index (document_chunks + embeddings)
  5. Verify recovery (counts, retrieval tests)

Environment:
  DATABASE_URL must be set to your Render PostgreSQL URL
  (postgresql://... or postgresql+asyncpg://... — both accepted)

Examples:
  # Dry-run everything (zero writes: no migrations, no inserts, no jobs)
  python scripts/recover_database.py --dry-run

  # Full recovery (requires --confirm for write steps)
  python scripts/recover_database.py --confirm

  # Limit indexed documents for testing (import + index both honor it)
  python scripts/recover_database.py --confirm --limit 10
        """,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Strictly read-only: skip migrations; children "
                             "run in dry-run mode; nothing is committed.")
    parser.add_argument("--confirm", action="store_true",
                        help="Confirm write operations (required for non-dry-run)")
    parser.add_argument("--limit", type=int,
                        help="Limit files indexed/imported (for testing)")
    parser.add_argument("--skip-db-check", action="store_true",
                        help="Skip the connectivity check (not recommended)")
    parser.add_argument("--skip-migrations", action="store_true",
                        help="Skip Alembic migrations ONLY (connectivity check "
                             "still runs unless --skip-db-check is also given)")
    parser.add_argument("--skip-import", action="store_true",
                        help="Skip JSON import step")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip RAG index rebuild step")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip final verification step")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Manifest path shared by import and verify steps")
    parser.add_argument("--allow-hash-embeddings", action="store_true",
                        help="Forwarded to the index rebuild (local test DBs only)")

    args = parser.parse_args(argv)

    # Check DATABASE_URL (never printed).
    db_url = os.environ.get("DATABASE_URL")
    if not db_url or "user:pass@host" in db_url:
        print("[FAIL] DATABASE_URL not configured")
        print("  Set DATABASE_URL environment variable to your Render PostgreSQL URL")
        return 1

    masked = db_url
    if "@" in masked:
        user_pass, host_db = masked.split("@", 1)
        masked = f"{user_pass.split(':')[0]}:***@{host_db}"
    print(f"Target database: {masked}")
    print(f"Mode: {'DRY-RUN (strictly read-only)' if args.dry_run else 'LIVE'}")

    if args.dry_run and args.confirm:
        print("[FAIL] --dry-run and --confirm are mutually exclusive")
        return 1

    project_root = Path(__file__).resolve().parent.parent
    manifest = args.manifest or str(project_root / "evaluation" / "recovery_manifest.json")
    limit_args = ["--limit", str(args.limit)] if args.limit else []
    results: dict[str, int] = {}

    # Step 1: connectivity check (read-only; independent of migrations).
    if not args.skip_db_check:
        _step("1. Database Connection Check")
        code = await check_db_main()
        results["db_check"] = code
        if code != 0:
            print("[FAIL] Database connection check failed. Aborting.")
            return 1
    else:
        print("\n[1/5] Skipping connectivity check (--skip-db-check)")

    # Step 2: migrations — NEVER under --dry-run.
    if args.dry_run:
        print("\n[2/5] Skipping Alembic migrations (--dry-run is strictly read-only)")
        results["migrations"] = 0
    elif not args.skip_migrations:
        _step("2. Alembic Migrations")
        code = await run_alembic_upgrade(project_root)
        results["migrations"] = code
        if code != 0:
            print("[FAIL] Alembic migrations failed. Aborting.")
            return 1
        print("[OK] Migrations completed")
    else:
        print("\n[2/5] Skipping Alembic migrations (--skip-migrations)")

    # Step 3: import (direct call with explicit argv).
    if not args.skip_import:
        _step("3. Import Processed Records")
        import_argv = ["--manifest", manifest] + limit_args
        if args.dry_run:
            import_argv.append("--dry-run")
        else:
            import_argv.append("--confirm")
        code = await restore_main(import_argv)
        results["import"] = code
        if code != 0:
            print("[FAIL] Import failed. Aborting.")
            return 1
    else:
        print("\n[3/5] Skipping import (--skip-import)")

    # Step 4: index rebuild (direct call with explicit argv).
    if not args.skip_index:
        _step("4. Rebuild RAG Index")
        index_argv = list(limit_args)
        if args.allow_hash_embeddings:
            index_argv.append("--allow-hash-embeddings")
        if args.dry_run:
            index_argv.append("--dry-run")
        else:
            index_argv.append("--confirm")
        code = await rebuild_main(index_argv)
        results["rebuild_index"] = code
        if code != 0:
            print("[FAIL] Index rebuild failed. Aborting.")
            return 1
    else:
        print("\n[4/5] Skipping index rebuild (--skip-index)")

    # Step 5: verification (direct call).
    if not args.skip_verify:
        _step("5. Verify Recovery")
        code = await verify_main(["--manifest", manifest])
        results["verify"] = code
        if code != 0:
            print("[WARN] Verification reported issues")
        else:
            print("[OK] Verification passed")
    else:
        print("\n[5/5] Skipping verification (--skip-verify)")

    # Summary
    print("\n" + "=" * 60)
    print("RECOVERY SUMMARY")
    print("=" * 60)
    for step, code in results.items():
        status = "[OK]" if code == 0 else "[FAIL]"
        print(f"  {step:20} {status}")

    all_ok = all(code == 0 for code in results.values())
    print(f"\nOverall: {'[OK] ALL STEPS PASSED' if all_ok else '[FAIL] SOME STEPS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
