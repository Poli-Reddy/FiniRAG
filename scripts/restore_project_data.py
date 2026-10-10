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
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings
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

    # processing_status should be recognizable
    valid_statuses = {"PASS", "FAIL", "PENDING", "ERROR"}
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


async def import_documents(
    session: AsyncSession,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Import all JSON files from evaluation/results/."""

    if not RESULTS_DIR.exists():
        return {"error": f"Results directory not found: {RESULTS_DIR}"}

    json_files = sorted(RESULTS_DIR.glob("*.json"))
    if not json_files:
        return {"error": f"No JSON files found in {RESULTS_DIR}"}

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

            # Check if already imported (by content_hash in processing_metadata)
            # We store the import hash in processing_metadata.import_hash
            existing = await session.execute(
                select(Document.id).where(
                    func.jsonb_extract_path_text(
                        Document.processing_metadata, "import_hash"
                    ) == content_hash
                ).limit(1)
            )
            existing_id = existing.scalar_one_or_none()

            if existing_id and skip_existing:
                stats["skipped_existing"] += 1
                stats["details"].append({
                    "file": json_file.name,
                    "status": "SKIPPED_EXISTING",
                    "existing_id": str(existing_id),
                })
                continue

            # Prepare document for insert
            # Preserve original ID if it looks like a valid UUID
            doc_id = None
            if "id" in data:
                try:
                    doc_id = uuid.UUID(data["id"])
                except (ValueError, TypeError):
                    pass

            # Parse created_at if present
            created_at = None
            if "created_at" in data and data["created_at"]:
                try:
                    created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            # Build processing_metadata with import tracking
            proc_meta = data.get("processing_metadata", {}) or {}
            proc_meta = dict(proc_meta)  # copy
            proc_meta["import_hash"] = content_hash
            proc_meta["imported_at"] = datetime.now(timezone.utc).isoformat()
            proc_meta["source_json"] = json_file.name

            doc = Document(
                id=doc_id or uuid.uuid4(),
                document_name=data["document_name"],
                document_type=data["document_type"],
                processing_status=data["processing_status"],
                file_validation=data["file_validation"],
                extracted_data=data.get("extracted_data"),
                validation=data.get("validation"),
                processing_metadata=proc_meta,
                error=data.get("error"),
                created_at=created_at or datetime.now(timezone.utc),
            )

            if not dry_run:
                session.add(doc)
                await session.flush()  # Get the ID

            stats["imported"] += 1
            stats["details"].append({
                "file": json_file.name,
                "status": "IMPORTED" if not dry_run else "WOULD_IMPORT",
                "id": str(doc.id),
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


async def main():
    parser = argparse.ArgumentParser(description="Restore FiniRAG processed records from JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported without writing")
    parser.add_argument("--force", action="store_true", help="Re-import even if content hash exists")
    parser.add_argument("--confirm", action="store_true", help="Confirm before actual import (required for non-dry-run)")
    args = parser.parse_args()

    # Get DATABASE_URL from environment
    db_url = os.environ.get("DATABASE_URL") or settings.DATABASE_URL
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
            except Exception as e:
                await session.rollback()
                print(f"ERROR: Commit failed: {e}")
                return 1

    await engine.dispose()
    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))