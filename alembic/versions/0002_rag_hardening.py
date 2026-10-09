"""RAG hardening — explicit HNSW params + active-chunk uniqueness.

Revision ID: 0002_rag_hardening
Revises: 0001_rag

1. Rebuild the HNSW vector index with explicit build parameters
   (m = 16, ef_construction = 64 — pgvector's documented defaults, stated
   explicitly so the build is reproducible rather than relying on server
   defaults that may change). Query-time recall is tuned separately via
   hnsw.ef_search (see HNSW_EF_SEARCH).
2. Add a partial unique index so the database, not just application logic,
   guarantees at most one ACTIVE chunk per (document, index_version,
   content_hash). This closes the race where two concurrent indexing
   attempts could insert duplicate active chunks.

Downgrade drops only these two artefacts; documents and chunk data are
never removed.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_rag_hardening"
down_revision = "0001_rag"
branch_labels = None
depends_on = None

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


def _is_postgres() -> bool:
    try:
        return op.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


def upgrade() -> None:
    if not _is_postgres():
        return  # RAG index tuning is Postgres-specific

    # 1. Rebuild HNSW with explicit, documented build parameters.
    op.execute("DROP INDEX IF EXISTS idx_chunks_vec")
    op.execute(
        "CREATE INDEX idx_chunks_vec ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
    )

    # 2. Enforce unique active chunks at the database level.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_active_hash "
        "ON document_chunks (document_id, index_version, content_hash) "
        "WHERE is_active"
    )


def downgrade() -> None:
    if not _is_postgres():
        return
    op.execute("DROP INDEX IF EXISTS uq_chunks_active_hash")
    op.execute("DROP INDEX IF EXISTS idx_chunks_vec")
    op.execute(
        "CREATE INDEX idx_chunks_vec ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
