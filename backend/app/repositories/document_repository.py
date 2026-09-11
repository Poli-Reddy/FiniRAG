"""
FiniRAG Document Repository.

PostgreSQL persistence layer.
Every processing attempt creates a new row — no upserts, no overwrites.
GET-by-name returns the latest row (ORDER BY created_at DESC, id DESC LIMIT 1).
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.document import Document

logger = logging.getLogger("finirag")


class DocumentRepository:
    """Database operations for documents."""

    def __init__(self, db: AsyncSession):
        self._db = db

    async def create(self, document: Document) -> Document:
        """Insert a new processing attempt."""
        self._db.add(document)
        await self._db.flush()
        await self._db.refresh(document)

        logger.info("Database persist", extra={
            "document_id": str(document.id),
            "document_name": document.document_name,
        })
        return document

    async def get_latest_by_name(self, document_name: str) -> Document | None:
        """
        Get the latest processing result for a document.
        Returns the most recent row, NOT the only row.
        """
        stmt = (
            select(Document)
            .where(Document.document_name == document_name)
            .order_by(desc(Document.created_at), desc(Document.id))
            .limit(1)
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id(self, document_id: UUID) -> Document | None:
        """Get a specific processing attempt by ID."""
        stmt = select(Document).where(Document.id == document_id)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        """
        List all processing attempts, ordered by creation time (newest first).
        Returns (documents, total_count).
        """
        count_stmt = select(func.count(Document.id))
        count_result = await self._db.execute(count_stmt)
        total = count_result.scalar_one()

        stmt = (
            select(Document)
            .order_by(desc(Document.created_at), desc(Document.id))
            .limit(limit)
            .offset(offset)
        )
        result = await self._db.execute(stmt)
        documents = list(result.scalars().all())

        return documents, total
