"""Baseline documents table — adoption migration for pre-RAG deployments.

Revision ID: 0000_documents
Revises: None

Creates `documents` if missing (fresh DB) and adopts it if present
(existing prod data preserved — no drop/recreate).
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0000_documents"
down_revision = None
branch_labels = None
depends_on = None


def _table_exists(name: str) -> bool:
    """True when the table already exists. Unavailable in offline --sql
    mode (MockConnection); there we return False so DDL is emitted."""
    try:
        return name in sa.inspect(op.get_bind()).get_table_names()
    except Exception:
        return False


# Columns the ORM requires on documents. An adopted table missing any of
# these would pass silently and fail later (FK creation, ORM loads), so
# adoption validates instead of assuming compatibility.
REQUIRED_DOCUMENT_COLUMNS = {
    "id", "document_name", "document_type", "processing_status",
    "file_validation", "extracted_data", "validation",
    "processing_metadata", "error", "created_at",
}

# Expected definition for each critical column: (type family, nullable).
# Audit finding #10: names alone are not compatibility. Type families are
# deliberately coarse (UUID, VARCHAR, JSON/JSONB, TIMESTAMP) so minor
# dialect spellings pass while wrong types fail loudly.
EXPECTED_DOCUMENT_SCHEMA = {
    "id": ("UUID", False),
    "document_name": ("VARCHAR", False),
    "document_type": ("VARCHAR", False),
    "processing_status": ("VARCHAR", False),
    "file_validation": ("JSON", False),
    "extracted_data": ("JSON", True),
    "validation": ("JSON", True),
    "processing_metadata": ("JSON", False),
    "error": ("JSON", True),
    "created_at": ("TIMESTAMP", False),
}

# Indexes the baseline owns. An adopted table missing them gets them
# created (IF NOT EXISTS) instead of failing later on slow queries.
BASELINE_INDEXES = ("idx_documents_name_created", "idx_documents_created")


def _type_family(col_type, dialect_name: str = "") -> str:
    """Coarse type family of an inspector column type."""
    name = type(col_type).__name__.upper()
    compiled = ""
    try:
        compiled = str(col_type.compile(dialect=None)).upper()  # type: ignore[attr-defined]
    except Exception:
        pass
    blob = f"{name} {compiled} {dialect_name}".upper()
    if "UUID" in blob:
        return "UUID"
    if any(k in blob for k in ("JSONB", "JSON")):
        return "JSON"
    if "TIMESTAMP" in blob or "DATETIME" in blob:
        return "TIMESTAMP"
    if any(k in blob for k in ("VARCHAR", "CHAR", "TEXT", "STRING")):
        return "VARCHAR"
    if any(k in blob for k in ("INT", "SERIAL")):
        return "INTEGER"
    if "BOOL" in blob:
        return "BOOLEAN"
    return name


def validate_existing_schema(columns: list[dict],
                             dialect_name: str = "") -> list[str]:
    """Diff an inspected column list against the baseline. Pure function —
    unit-testable without a database. Returns human-readable problems
    (empty = compatible)."""
    problems: list[str] = []
    by_name = {c["name"]: c for c in columns}
    missing = REQUIRED_DOCUMENT_COLUMNS - set(by_name)
    if missing:
        problems.append(f"missing columns: {sorted(missing)}")
    for name, (family, nullable) in EXPECTED_DOCUMENT_SCHEMA.items():
        col = by_name.get(name)
        if col is None:
            continue
        actual_family = _type_family(col.get("type"), dialect_name)
        if actual_family != family:
            problems.append(
                f"column {name!r}: expected {family}, found {actual_family}")
        # Nullability is enforced only for NOT NULL columns: an adopted
        # table that is MORE permissive (nullable) still loads fine, but
        # one that is stricter than the ORM writes would break inserts —
        # both directions are reported, strictness as the louder error.
        actual_nullable = col.get("nullable", True)
        if nullable is False and actual_nullable:
            problems.append(
                f"column {name!r}: must be NOT NULL (ORM writes assume it)")
        elif nullable is True and not actual_nullable:
            problems.append(
                f"column {name!r}: unexpectedly NOT NULL (ORM writes NULLs)")
    return problems


def _existing_columns(name: str) -> list[dict] | None:
    """Inspected column dicts of an existing table, or None when
    uninspectable (offline --sql mode)."""
    try:
        return sa.inspect(op.get_bind()).get_columns(name)
    except Exception:
        return None


def _existing_indexes(name: str) -> set[str]:
    try:
        return {i["name"] for i in sa.inspect(op.get_bind()).get_indexes(name)}
    except Exception:
        return set(BASELINE_INDEXES)  # offline: assume present, DDL emitted anyway


def upgrade() -> None:
    bind = op.get_bind()
    if _table_exists("documents"):
        # Adopt — but only a compatible table. Names, types and nullability
        # are all diffed; a drifted schema fails loudly with the exact
        # diff instead of breaking at FK/ORM time (audit finding #10).
        existing = _existing_columns("documents")
        if existing is not None:
            try:
                dialect = bind.dialect.name if bind is not None else ""
            except Exception:
                dialect = ""
            problems = validate_existing_schema(existing, dialect)
            if problems:
                raise RuntimeError(
                    "Existing documents table is incompatible with the "
                    "baseline schema — refusing to adopt a drifted schema. "
                    "Migrate the deployed schema explicitly instead. "
                    f"Problems: {problems}")
        # Backfill baseline indexes the adopted table may lack.
        have = _existing_indexes("documents")
        if "idx_documents_name_created" not in have:
            op.create_index("idx_documents_name_created", "documents",
                            ["document_name", sa.text("created_at DESC")])
        if "idx_documents_created" not in have:
            op.create_index("idx_documents_created", "documents",
                            [sa.text("created_at DESC")])
        return  # adopt existing table, preserve data
    # Migrations target PostgreSQL only: JSONB matches the ORM schema exactly
    # (models/document.py uses JSONB on PG; its JSON variant is for sqlite
    # tests, which never run migrations).
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("document_name", sa.String(255), nullable=False),
        sa.Column("document_type", sa.String(50), nullable=False),
        sa.Column("processing_status", sa.String(20), nullable=False),
        sa.Column("file_validation", JSONB(), nullable=False),
        sa.Column("extracted_data", JSONB(), nullable=True),
        sa.Column("validation", JSONB(), nullable=True),
        sa.Column("processing_metadata", JSONB(), nullable=False),
        sa.Column("error", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_documents_name_created", "documents",
                    ["document_name", sa.text("created_at DESC")])
    op.create_index("idx_documents_created", "documents",
                    [sa.text("created_at DESC")])


def downgrade() -> None:
    # Non-destructive policy: baseline downgrade does NOT drop documents.
    pass
