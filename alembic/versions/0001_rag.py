"""RAG tables — document_chunks + indexing_jobs (dim 768, gemini-embedding-2).

Revision ID: 0001_rag
Revises: 0000_documents

- Enables the pgvector extension with its CORRECT name: `vector`
  (the extension is `vector`, the package is `pgvector`).
- Fails loudly if the extension is unavailable — never suppress and continue.
- Requires the `documents` table (baseline migration) for FK integrity.
- Vector dim 768 must match EMBEDDING_DIM; a model/dim change needs re-embed.
- Downgrade drops only RAG tables, never documents.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001_rag"
down_revision = "0000_documents"
branch_labels = None
depends_on = None

VECTOR_DIM = 768


def upgrade() -> None:
    # 1. pgvector extension — correct name is `vector`. Raise if unavailable.
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
    if bind_is_postgres() and not _extension_present("vector"):
        raise RuntimeError(
            "pgvector extension 'vector' unavailable — "
            "install pgvector on PostgreSQL before migrating")

    # 2. documents table must exist for FK references.
    if not _table_exists("documents"):
        raise RuntimeError(
            "documents table missing — run baseline 0000_documents first")

    try:
        from pgvector.sqlalchemy import Vector  # noqa

        vec = Vector(VECTOR_DIM)
    except Exception as e:
        raise RuntimeError(
            f"pgvector SQLAlchemy type unavailable: {e}. "
            "pip install pgvector==0.3.6") from e

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("document_id", sa.Uuid(),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("chunk_type", sa.String(40), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_tsv", sa.Text, nullable=True),
        sa.Column("page_number", sa.Integer, nullable=True),
        sa.Column("section_name", sa.String(255), nullable=True),
        sa.Column("document_type", sa.String(50), nullable=False),
        sa.Column("reporting_period", sa.String(50), nullable=True),
        sa.Column("source_evidence", JSONB(), nullable=False, server_default="{}"),
        sa.Column("metadata", JSONB(), nullable=False, server_default="{}"),
        sa.Column("embedding", vec, nullable=True),
        sa.Column("embedding_model", sa.String(128), nullable=False, server_default=""),
        sa.Column("index_version", sa.String(32), nullable=False, server_default="v1"),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_chunks_document", "document_chunks", ["document_id"])
    op.create_index("idx_chunks_doc_type", "document_chunks", ["document_type"])
    op.create_index("idx_chunks_period", "document_chunks", ["reporting_period"])
    op.create_index("idx_chunks_type", "document_chunks", ["chunk_type"])
    op.create_index("idx_chunks_active_doc", "document_chunks", ["document_id", "is_active"])
    # FTS GIN over an expression index (content_tsv column is a cache, not the index).
    op.execute("CREATE INDEX idx_chunks_fts ON document_chunks "
               "USING gin (to_tsvector('english', content))")
    # HNSW cosine index — only after extension + vector column exist.
    op.execute("CREATE INDEX idx_chunks_vec ON document_chunks "
               "USING hnsw (embedding vector_cosine_ops)")
    op.create_table(
        "indexing_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("document_id", sa.Uuid(),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("attempt_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("index_version", sa.String(32), nullable=False, server_default="v1"),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_jobs_document", "indexing_jobs", ["document_id"])
    op.create_index("idx_jobs_status", "indexing_jobs", ["status"])


def bind_is_postgres() -> bool:
    try:
        return op.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


def _table_exists(name: str) -> bool:
    """Inspection is unavailable in offline --sql mode; there we assume the
    table exists for guard checks (DDL for RAG tables is still emitted)."""
    try:
        return name in sa.inspect(op.get_bind()).get_table_names()
    except Exception:
        return True


def _extension_present(name: str) -> bool:
    """Check pg_extension. Unverifiable offline — assume present there and
    let the online run enforce it loudly."""
    try:
        rows = op.get_bind().exec_driver_sql(
            "SELECT extname FROM pg_extension").fetchall()
        return any(r[0] == name for r in rows)
    except Exception:
        return True


def downgrade() -> None:
    # RAG-only teardown; documents preserved.
    op.drop_index("idx_jobs_status", table_name="indexing_jobs")
    op.drop_index("idx_jobs_document", table_name="indexing_jobs")
    op.drop_table("indexing_jobs")
    op.drop_index("idx_chunks_vec", table_name="document_chunks")
    op.drop_index("idx_chunks_fts", table_name="document_chunks")
    op.drop_index("idx_chunks_active_doc", table_name="document_chunks")
    op.drop_index("idx_chunks_type", table_name="document_chunks")
    op.drop_index("idx_chunks_period", table_name="document_chunks")
    op.drop_index("idx_chunks_doc_type", table_name="document_chunks")
    op.drop_index("idx_chunks_document", table_name="document_chunks")
    op.drop_table("document_chunks")
