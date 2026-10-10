"""RAG correctness follow-ups — retained source text + atomic job claims.

Revision ID: 0003_rag_correctness
Revises: 0002_rag_hardening

1. `documents.source_text` (nullable TEXT): the bounded OCR/native text the
   extraction was built from. NULL for vision-fallback and failed
   extractions. Enables verification of evidence snippets against the real
   source instead of trusting model-transcribed text, and lets indexing
   build narrative chunks from the original rather than derived summaries.
   Lives and dies with its row — no separate retention policy needed.
2. Partial unique index `uq_active_job_per_document` on
   `indexing_jobs(document_id)` WHERE status IN ('PENDING','PROCESSING'):
   the database, not application read-then-insert logic, guarantees one
   active job per document. Concurrent `ensure_job` calls that lose the
   race get IntegrityError and re-read the winner.

Downgrade drops only these two artefacts; document rows are never removed.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_rag_correctness"
down_revision = "0002_rag_hardening"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    try:
        return op.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


def upgrade() -> None:
    op.add_column("documents", sa.Column("source_text", sa.Text(), nullable=True))
    op.add_column("document_chunks",
                  sa.Column("company_norm", sa.String(255), nullable=True))
    op.create_index("idx_chunks_company", "document_chunks", ["company_norm"])
    # Partial unique index: one active job per document, enforced by the
    # database. SQLite supports partial indexes too, so tests exercise it.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_active_job_per_document "
        "ON indexing_jobs (document_id) "
        "WHERE status IN ('PENDING', 'PROCESSING')"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_active_job_per_document")
    op.drop_index("idx_chunks_company", table_name="document_chunks")
    # batch_alter_table keeps the downgrade portable to SQLite.
    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_column("source_text")
    with op.batch_alter_table("document_chunks") as batch_op:
        batch_op.drop_column("company_norm")
