"""
FiniRAG SQLAlchemy ORM Model.

Every processing attempt creates a new row. No upserts, no overwrites.
GET-by-name returns the latest row via ORDER BY created_at DESC LIMIT 1.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Index, text, JSON
from sqlalchemy import Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


# Portable JSON: JSONB on PostgreSQL, generic JSON on sqlite/tests.
JSON_VARIANT = JSONB().with_variant(JSON(), "sqlite")


class Document(Base):
    """Represents a single processing attempt of a document."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    document_name: Mapped[str] = mapped_column(String(255), nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    processing_status: Mapped[str] = mapped_column(String(20), nullable=False)

    # JSONB columns for flexible/dynamic data
    file_validation: Mapped[dict] = mapped_column(JSON_VARIANT, nullable=False)
    extracted_data: Mapped[dict | None] = mapped_column(JSON_VARIANT, nullable=True)
    validation: Mapped[dict | None] = mapped_column(JSON_VARIANT, nullable=True)
    processing_metadata: Mapped[dict] = mapped_column(JSON_VARIANT, nullable=False)
    error: Mapped[dict | None] = mapped_column(JSON_VARIANT, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Indexes for efficient queries
    __table_args__ = (
        # GET-by-name: latest processing result for a document
        Index("idx_documents_name_created", "document_name", created_at.desc()),
        # Dashboard: ordered by creation time
        Index("idx_documents_created", created_at.desc()),
    )

    def __repr__(self) -> str:
        return (
            f"<Document(id={self.id}, name='{self.document_name}', "
            f"type='{self.document_type}', status='{self.processing_status}')>"
        )
