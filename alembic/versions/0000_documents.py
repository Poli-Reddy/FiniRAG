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


def _existing_columns(name: str) -> set[str] | None:
    """Column names of an existing table, or None when uninspectable
    (offline --sql mode)."""
    try:
        return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(name)}
    except Exception:
        return None


def upgrade() -> None:
    bind = op.get_bind()
    if _table_exists("documents"):
        # Adopt — but only a compatible table. A drifted schema that passes
        # silently would break FK creation and ORM loads downstream.
        existing = _existing_columns("documents")
        if existing is not None:
            missing = REQUIRED_DOCUMENT_COLUMNS - existing
            if missing:
                raise RuntimeError(
                    "Existing documents table is missing columns "
                    f"{sorted(missing)} — refusing to adopt a drifted schema. "
                    "Migrate the deployed schema explicitly instead."
                )
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
