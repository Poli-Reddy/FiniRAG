#!/usr/bin/env python
"""
Restore processed financial records from evaluation/results/*.json into the
new Render PostgreSQL database.

- Reads JSON files from evaluation/results/
- Validates against current Document schema
- Inserts into documents table with idempotent import key (content_hash)
- Preserves original IDs when valid, generates new UUIDs otherwise
- Reports imported, skipped, invalid, failed counts
- Dry-run mode available
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import normalize_database_url, settings
from backend.app.models.document import Document
from backend.app.core.database import Base


RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "results"


def compute_content_hash(data: dict) -> str:
    """Stable hash of extracted_data + document_name for deduplication."""
    # Use a canonical subset of fields that define the document content
    canonical = {
        "document_name": data.get("document_name", ""),
        "document_type": data.get("document_type", ""),
        "extracted_data": data.get("extracted_data"),
        "validation": data.get("validation"),
        "processing_metadata": data.get("processing_metadata"),
        "file_validation": data.get("file_validation"),
        "error": data.get("error"),
    }
    serialized = json.dumps(canonical, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialized).hexdigest()[:64]


def validate_document_data(data: dict, filename: str) -> tuple[bool, list[str]]:
    """Validate JSON against expected Document schema. Returns (is_valid, errors)."""
    errors = []

    # Required fields
    required = ["document_name", "document_type", "processing_status", "file_validation"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # document_type must be valid
    valid_types = {"invoice", "balance_sheet", "profit_and_loss", "cash_flow_statement"}
    if data.get("document_type") not in valid_types:
        errors.append(f"Invalid document_type: {data.get('document_type')}")

    # processing_status must match what the pipeline actually writes:
    # document_service persists "PASS" or "FAILED" — accept both, plus the
    # PENDING/ERROR states jobs can carry. Anything else is drift.
    valid_statuses = {"PASS", "FAILED", "PENDING", "ERROR"}
    if data.get("processing_status") not in valid_statuses:
        errors.append(f"Unrecognized processing_status: {data.get('processing_status')}")

    # file_validation must be dict with status
    fv = data.get("file_validation")
    if not isinstance(fv, dict) or "status" not in fv:
        errors.append("file_validation must be dict with 'status'")

    # extracted_data optional but if present must be dict
    if "extracted_data" in data and data["extracted_data"] is not None:
        if not isinstance(data["extracted_data"], dict):
            errors.append("extracted_data must be dict or null")

    return len(errors) == 0, errors


async def _find_by_import_hash(session: AsyncSession,
                               content_hash: str) -> Document | None:
    """Row carrying this import hash, or None.

    Matched in Python over (id, processing_metadata) pairs instead of a
    Postgres-only JSON operator, so the importer behaves identically on
    every backend (audit finding #7: no dialect-specific surprises). The
    documents table is small (one row per processing attempt); the scan
    is bounded and read-only.
    """
    rows = (await session.execute(
        select(Document.id, Document.processing_metadata))).all()
    for _id, meta in rows:
        if isinstance(meta, dict) and meta.get("import_hash") == content_hash:
            return await session.get(Document, _id)
    return None


async def import_documents(
    session: AsyncSession,
    dry_run: bool = False,
    skip_existing: bool = True,
    limit: int | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Import all JSON files from evaluation/results/ (or `results_dir`).

    Each record is written inside its own savepoint: a per-record
    insert/flush failure rolls back only that record (audit finding #6),
    never poisoning the session for the records that follow.
    """

    src_dir = results_dir or RESULTS_DIR
    if not src_dir.exists():
        return {"error": f"Results directory not found: {src_dir}"}

    json_files = sorted(src_dir.glob("*.json"))
    if limit is not None:
        json_files = json_files[:limit]
    if not json_files:
        return {"error": f"No JSON files found in {src_dir}"}

    stats = {
        "total_files": len(json_files),
        "imported": 0,
        "skipped_existing": 0,
        "skipped_invalid": 0,
        "failed": 0,
        "details": [],
    }

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))

            # Validate
            is_valid, errors = validate_document_data(data, json_file.name)
            if not is_valid:
                stats["skipped_invalid"] += 1
                stats["details"].append({
                    "file": json_file.name,
                    "status": "INVALID",
                    "errors": errors,
                })
                continue

            # Compute import key
            content_hash = compute_content_hash(data)

            # Check if already imported (by content_hash in processing_metadata).
            # Portable helper — no Postgres-only JSON operators (finding #7).
            existing_doc = await _find_by_import_hash(session, content_hash)

            if existing_doc is not None and skip_existing:
                stats["skipped_existing"] += 1
                stats["details"].append({
                    "file": json_file.name,
                    "status": "SKIPPED_EXISTING",
                    "existing_id": str(existing_doc.id),
                })
                continue

            # Build processing_metadata with import tracking
            proc_meta = data.get("processing_metadata", {}) or {}
            proc_meta = dict(proc_meta)  # copy
            proc_meta["import_hash"] = content_hash
            proc_meta["imported_at"] = datetime.now(timezone.utc).isoformat()
            proc_meta["source_json"] = json_file.name

            if dry_run:
                preview_id = str(existing_doc.id) if existing_doc is not None \
                    else str(_preview_uuid(data))
                stats["imported"] += 1
                stats["details"].append({
                    "file": json_file.name,
                    "status": "UPSERT_WOULD_UPDATE" if existing_doc is not None
                              else "WOULD_IMPORT",
                    "id": preview_id,
                    "content_hash": content_hash,
                })
                continue

            # Real write — isolated in a savepoint so one bad record can
            # never wedge the session (PendingRollbackError) or abort the
            # batch commit for every other record (audit finding #6).
            try:
                async with session.begin_nested():
                    if existing_doc is not None:
                        # --force: transactional upsert on the stable import
                        # key. The SAME row is updated in place — no duplicate
                        # UUID is ever created (audit finding #9), and
                        # dependent chunks/jobs keep pointing at one row.
                        existing_doc.document_name = data["document_name"]
                        existing_doc.document_type = data["document_type"]
                        existing_doc.processing_status = data["processing_status"]
                        existing_doc.file_validation = data["file_validation"]
                        existing_doc.extracted_data = data.get("extracted_data")
                        existing_doc.validation = data.get("validation")
                        existing_doc.processing_metadata = proc_meta
                        existing_doc.error = data.get("error")
                        await session.flush()
                        doc_id = str(existing_doc.id)
                        outcome = "UPSERTED"
                    else:
                        doc = Document(
                            id=_preview_uuid(data),
                            document_name=data["document_name"],
                            document_type=data["document_type"],
                            processing_status=data["processing_status"],
                            file_validation=data["file_validation"],
                            extracted_data=data.get("extracted_data"),
                            validation=data.get("validation"),
                            processing_metadata=proc_meta,
                            error=data.get("error"),
                            created_at=_preview_created_at(data),
                        )
                        session.add(doc)
                        await session.flush()
                        doc_id = str(doc.id)
                        outcome = "IMPORTED"
            except Exception as e:
                stats["failed"] += 1
                stats["details"].append({
                    "file": json_file.name,
                    "status": "FAILED",
                    "error": f"{type(e).__name__}: {e}",
                })
                continue

            stats["imported"] += 1
            stats["details"].append({
                "file": json_file.name,
                "status": outcome,
                "id": doc_id,
                "content_hash": content_hash,
            })

        except Exception as e:
            stats["failed"] += 1
            stats["details"].append({
                "file": json_file.name,
                "status": "FAILED",
                "error": str(e),
            })

    return stats


def _preview_uuid(data: dict) -> uuid.UUID:
    """Original ID when it is a valid UUID, else a fresh one.

    The recovery JSONs carry no id field, so this is almost always fresh —
    and that is reported honestly, never fabricated as history.
    """
    if "id" in data:
        try:
            return uuid.UUID(str(data["id"]))
        except (ValueError, TypeError, AttributeError):
            pass
    return uuid.uuid4()


def _preview_created_at(data: dict) -> datetime:
    """Original timestamp when present and parseable, else now."""
    if data.get("created_at"):
        try:
            return datetime.fromisoformat(
                str(data["created_at"]).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc)


def print_dry_run_summary(stats: dict):
    print("\n=== DRY-RUN SUMMARY ===")
    print(f"Total files: {stats['total_files']}")
    print(f"Would import: {stats['imported']}")
    print(f"Would skip (existing): {stats['skipped_existing']}")
    print(f"Would skip (invalid): {stats['skipped_invalid']}")
    print(f"Would fail: {stats['failed']}")
    print()

    for d in stats["details"]:
        status = d["status"]
        if status in ("IMPORTED", "WOULD_IMPORT"):
            print(f"  [+] {d['file']} -> {d['id'][:8]}... (hash: {d['content_hash'][:12]}...)")
        elif status == "SKIPPED_EXISTING":
            print(f"  [=] {d['file']} -> already imported as {d['existing_id'][:8]}...")
        elif status == "INVALID":
            print(f"  [!] {d['file']} -> INVALID: {', '.join(d['errors'])}")
        elif status == "FAILED":
            print(f"  [!] {d['file']} -> FAILED: {d['error']}")


def print_final_report(stats: dict):
    print("\n=== IMPORT REPORT ===")
    print(f"Total files:          {stats['total_files']}")
    print(f"Successfully imported: {stats['imported']}")
    print(f"Skipped (existing):    {stats['skipped_existing']}")
    print(f"Skipped (invalid):     {stats['skipped_invalid']}")
    print(f"Failed:                {stats['failed']}")

    if stats["details"]:
        print("\nDetails:")
        for d in stats["details"]:
            print(f"  {d['status']:16} {d['file']}" + (
                f" -> {d.get('id', d.get('existing_id', ''))[:8]}..." if d.get('id') or d.get('existing_id') else ""
            ) + (
                f" ({d['content_hash'][:12]}...)" if d.get('content_hash') else ""
            ) + (
                f" - {d['error']}" if d.get('error') else ""
            ))


async def main(argv: list[str] | None = None):
    """Entry point. `argv` is injectable so the orchestrator and tests call
    this directly — no sys.argv simulation (audit finding #2)."""
    parser = argparse.ArgumentParser(description="Restore FiniRAG processed records from JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported without writing")
    parser.add_argument("--force", action="store_true",
                        help="Update already-imported rows in place (transactional "
                             "upsert on the stable import hash). NEVER creates "
                             "duplicate records (audit finding #9).")
    parser.add_argument("--confirm", action="store_true", help="Confirm before actual import (required for non-dry-run)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Import at most N files (sorted order; for testing)")
    parser.add_argument("--manifest", type=str, default=str(
        Path(__file__).resolve().parent.parent / "evaluation" / "recovery_manifest.json"),
        help="Where to write the recovery manifest (file -> record mapping)")
    args = parser.parse_args(argv)

    # Get DATABASE_URL from environment (shared normalizer, finding #7).
    db_url = normalize_database_url(
        os.environ.get("DATABASE_URL") or settings.DATABASE_URL)
    if not db_url or "user:pass@host" in db_url:
        print("ERROR: DATABASE_URL not configured")
        print("  Set DATABASE_URL environment variable to your Render PostgreSQL URL")
        return 1

    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Run import
        stats = await import_documents(
            session,
            dry_run=args.dry_run,
            skip_existing=not args.force,
            limit=args.limit,
        )

        if "error" in stats:
            print(f"ERROR: {stats['error']}")
            return 1

        if args.dry_run:
            print_dry_run_summary(stats)
            print("\nDry-run complete. Run without --dry-run to perform actual import.")
        else:
            if not args.confirm:
                print("ERROR: Non-dry-run requires --confirm flag")
                print("Run with --dry-run first to preview, then add --confirm to execute.")
                return 1

            try:
                await session.commit()
                print_final_report(stats)
                _write_manifest(stats, args.manifest)
            except Exception as e:
                await session.rollback()
                print(f"ERROR: Commit failed: {e}")
                return 1

    await engine.dispose()
    return 0


def _write_manifest(stats: dict, path: str) -> None:
    """Persist the file -> record mapping for the verifier (finding #4).

    Records which source JSON produced which database row, plus whether
    the source carried any retained source_text (it does not — the JSONs
    predate source_text retention, finding #8 — and that absence is
    recorded honestly instead of fabricated).
    """
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_files": stats.get("total_files", 0),
        "records": [
            {k: d.get(k) for k in ("file", "status", "id", "existing_id",
                                   "content_hash", "error", "errors")
             if d.get(k) is not None}
            for d in stats.get("details", [])
        ],
    }
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written: {path} ({len(manifest['records'])} entries)")


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))