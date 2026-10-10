"""
FiniRAG RAG models — document_chunks + indexing_jobs.

document_chunks references the exact processing-attempt row (documents.id).
Old attempts keep their chunks; reprocessing creates new chunks, never
moves old ones. Deduplication is enforced per (document_id, index_version,
content_hash) for active chunks.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Index, Integer, ForeignKey, Text, text, JSON
from sqlalchemy import Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import JSON as GenericJSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from backend.app.core.database import Base

JSON_VARIANT = JSONB().with_variant(GenericJSON(), "sqlite")


class EmbeddingVector(TypeDecorator):
    """pgvector on PostgreSQL, JSON fallback elsewhere (sqlite/tests).

    Stores a list[float] of EMBEDDING_DIM. Rejects non-finite values.
    Similarity search uses pgvector operators on Postgres; the retrieval
    service falls back to Python cosine on other backends.
    """

    impl = GenericJSON
    cache_ok = True

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector

                return Vector(self.dim)
            except Exception:
                pass
        return GenericJSON()

    def process_bind_param(self, value, dialect):
        import math

        if value is None:
            return None
        vals = list(value)
        if len(vals) != self.dim:
            raise ValueError(f"embedding dim mismatch: {len(vals)} != {self.dim}")
        for v in vals:
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError("non-finite embedding value rejected")
        if dialect.name == "postgresql":
            return vals
        return vals

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return list(value)


def embedding_cosine_distance(column, vector):
    """pgvector `<=>` ordering expression, built without relying on the
    column type's comparator (EmbeddingVector must also work on sqlite,
    where pgvector operators do not exist). Only ever EXECUTED on
    PostgreSQL — retrieval_service guards by dialect."""
    from sqlalchemy.sql import operators

    return column.operate(operators.custom_op("<=>"), vector)


def _utcnow():
    return datetime.now(timezone.utc)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(40), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_tsv: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    reporting_period: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Canonical company for exact-match filtering (finding #19): normalized
    # company_name (statements) or vendor_name (invoices), else None. Filters
    # match this column exactly; substring matching is only a legacy fallback
    # for chunks indexed before it existed.
    company_norm: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_evidence: Mapped[dict] = mapped_column(JSON_VARIANT, nullable=False, default=dict)
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSON_VARIANT, nullable=False, default=dict)
    embedding: Mapped[list | None] = mapped_column(EmbeddingVector(768), nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    index_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        Index("idx_chunks_document", "document_id"),
        Index("idx_chunks_doc_type", "document_type"),
        Index("idx_chunks_period", "reporting_period"),
        Index("idx_chunks_type", "chunk_type"),
        Index("idx_chunks_active_doc", "document_id", "is_active"),
        Index("idx_chunks_company", "company_norm"),
    )


class IndexingJob(Base):
    __tablename__ = "indexing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    index_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        Index("idx_jobs_document", "document_id"),
        Index("idx_jobs_status", "status"),
        # One active job per document, enforced by the database (finding #8).
        # Declared here — not only in migration 0003 — so create_all-based
        # environments (tests, dev) carry the same guarantee as migrated
        # production. Partial indexes exist on both PostgreSQL and SQLite.
        Index("uq_active_job_per_document", "document_id", unique=True,
              sqlite_where=text("status IN ('PENDING', 'PROCESSING')"),
              postgresql_where=text("status IN ('PENDING', 'PROCESSING')")),
    )
