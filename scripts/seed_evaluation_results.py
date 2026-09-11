"""Seed the measured dataset results into the dashboard PostgreSQL store.

The evaluation harness intentionally bypasses persistence. This script bridges
that deliberate separation after evaluation: it clears the dashboard's old
manual attempts, then inserts only the 50 real Dataset documents from the
already-measured JSON files. It does not call OCR or an LLM.

Usage:
    venv/Scripts/python.exe scripts/seed_evaluation_results.py

The script is intentionally replace-all. Running it again produces one
dashboard row per original dataset document, not duplicate seed rows.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import Json


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"


def _database_url() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("DATABASE_URL is missing from .env")


def _processed_at(metadata: dict) -> datetime:
    value = metadata.get("processed_at")
    if isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def main() -> None:
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    rows = []
    for path in result_files:
        result = json.loads(path.read_text(encoding="utf-8"))
        # The harness marks real files as source=dataset. This excludes all
        # four synthetic file-gate cases without relying on their filenames.
        if result.get("_harness", {}).get("source") != "dataset":
            continue
        metadata = result.get("processing_metadata") or {}
        rows.append(
            (
                str(uuid.uuid4()),
                result["document_name"],
                result["document_type"],
                result["processing_status"],
                Json(result.get("file_validation") or {}),
                Json(result.get("extracted_data")),
                Json(result.get("validation")),
                Json(metadata),
                Json(result.get("error")) if result.get("error") else None,
                _processed_at(metadata),
            )
        )

    if len(rows) != 50:
        raise RuntimeError(
            f"Expected exactly 50 dataset results, found {len(rows)} in {RESULTS_DIR}"
        )

    connection = psycopg2.connect(_database_url(), connect_timeout=20)
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE documents")
                cursor.executemany(
                    """
                    INSERT INTO documents (
                        id, document_name, document_type, processing_status,
                        file_validation, extracted_data, validation,
                        processing_metadata, error, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
                cursor.execute("SELECT COUNT(*) FROM documents")
                count = cursor.fetchone()[0]
        print(f"seeded {count} dataset results into PostgreSQL")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
