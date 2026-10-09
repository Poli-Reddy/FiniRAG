"""
FiniRAG API Routes — Document Processing Endpoints.

Endpoints:
  POST /api/v1/documents/process  — Upload and process a document
  GET  /api/v1/documents          — List all processing attempts
  GET  /api/v1/documents/{name}   — Get latest result by filename
  GET  /api/v1/health             — Health check
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.rate_limit import rate_limit
from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.schemas.document import (
    DocumentProcessResponse,
    DocumentListResponse,
    DocumentListItem,
    HealthResponse,
    VALID_DOCUMENT_TYPES,
)
from backend.app.services.document_service import DocumentService

logger = logging.getLogger("finirag")

router = APIRouter(prefix="/api/v1", tags=["documents"])


# ─────────────────────────────────────────────────────────────────────
# POST /api/v1/documents/process
# ─────────────────────────────────────────────────────────────────────

@router.post("/documents/process", response_model=DocumentProcessResponse,
             dependencies=[Depends(rate_limit)])
async def process_document(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload and process a document.

    - document_type: Must be one of: invoice, balance_sheet,
      profit_and_loss, cash_flow_statement
    - file: PDF, JPG, or PNG file (max 10MB, max 3 pages)
    """
    # Validate document type (user selects — no auto-classification)
    if document_type not in VALID_DOCUMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_DOCUMENT_TYPE",
                "message": f"Must be one of: {', '.join(sorted(VALID_DOCUMENT_TYPES))}",
            },
        )

    # Read file bytes
    file_bytes = await file.read()

    # Sanitize filename (path traversal prevention)
    filename = _sanitize_filename(file.filename or "unnamed")

    # Process through the full pipeline
    service = DocumentService(db)
    result = await service.process_document(
        file_bytes=file_bytes,
        filename=filename,
        document_type=document_type,
    )

    # Return 200 even for FAILED processing (it's a valid processing result)
    return result


# ─────────────────────────────────────────────────────────────────────
# GET /api/v1/documents
# ─────────────────────────────────────────────────────────────────────

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List all processing attempts, ordered by newest first."""
    service = DocumentService(db)
    documents, total = await service.list_documents(limit=limit, offset=offset)

    items = []
    for doc in documents:
        validation = doc.validation or {}
        items.append(DocumentListItem(
            id=str(doc.id),
            document_name=doc.document_name,
            document_type=doc.document_type,
            processing_status=doc.processing_status,
            validation_status=validation.get("overall_status") or "NOT_RUN",
            processing_metadata=doc.processing_metadata,
            created_at=doc.created_at.isoformat() if doc.created_at else "",
        ))

    return DocumentListResponse(documents=items, total=total)


# ─────────────────────────────────────────────────────────────────────
# GET /api/v1/documents/{document_name}
# ─────────────────────────────────────────────────────────────────────

@router.get("/documents/{document_name}")
async def get_document_by_name(
    document_name: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get the latest processing result for a document by filename.

    Returns the most recent processing attempt, not the only one.
    """
    service = DocumentService(db)
    document = await service.get_document_by_name(document_name)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "DOCUMENT_NOT_FOUND",
                "message": f"No processing results found for '{document_name}'",
            },
        )

    return {
        "id": str(document.id),
        "document_name": document.document_name,
        "document_type": document.document_type,
        "processing_status": document.processing_status,
        "file_validation": document.file_validation,
        "extracted_data": document.extracted_data,
        "validation": document.validation,
        "processing_metadata": document.processing_metadata,
        "error": document.error,
        "created_at": document.created_at.isoformat() if document.created_at else "",
    }


# ─────────────────────────────────────────────────────────────────────
# GET /api/v1/health
# ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health/readiness: app, DB, embedding config, index counts, provider
    breaker states. Deliberately cheap (one SELECT, one COUNT) and contains
    no secrets — safe to poll from a platform health check."""
    db_status = "disconnected"
    indexed_documents = 0
    try:
        from sqlalchemy import func, select, text

        from backend.app.models.rag import DocumentChunk

        await db.execute(text("SELECT 1"))
        db_status = "connected"
        total_chunks = int((await db.execute(
            select(func.count(DocumentChunk.id)).where(
                DocumentChunk.is_active == True))).scalar() or 0)  # noqa: E712
        indexed_documents = int((await db.execute(
            select(func.count(func.distinct(DocumentChunk.document_id))).where(
                DocumentChunk.is_active == True))).scalar() or 0)  # noqa: E712
    except Exception:
        db_status = "disconnected"
        total_chunks = 0

    emb_configured = bool(settings.GEMINI_API_KEY or settings.GEMINI_API_KEYS)
    try:
        from backend.app.core.resilience import breakers_snapshot

        breakers = breakers_snapshot()
    except Exception:
        breakers = []
    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        database=db_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        embedding_model=settings.EMBEDDING_MODEL,
        embedding_configured=emb_configured,
        retrieval_mode="hybrid" if emb_configured else "keyword-only",
        index_version=settings.INDEX_VERSION,
        indexed_chunks=total_chunks,
        indexed_documents=indexed_documents,
        provider_breakers=breakers,
    )


# ─────────────────────────────────────────────────────────────────────
# GET /api/v1/documents/{document_id}/index-status  + reindex + jobs
# ─────────────────────────────────────────────────────────────────────

@router.get("/documents/{document_id}/index-status")
async def document_index_status(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Indexing status for one processing-attempt record. Never rewrites processing_status."""
    import uuid as _uuid

    try:
        did = _uuid.UUID(str(document_id))
    except Exception:
        raise HTTPException(status_code=400, detail={"code": "INVALID_DOCUMENT_ID",
                                                     "message": "document_id must be UUID."})
    from backend.app.services.indexing_service import IndexingService

    status = await IndexingService(db).status_for_document(did)
    return {"document_id": str(did), **status}


@router.post("/documents/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Idempotent reindex of one processing attempt. Does not rerun extraction."""
    import uuid as _uuid

    try:
        did = _uuid.UUID(str(document_id))
    except Exception:
        raise HTTPException(status_code=400, detail={"code": "INVALID_DOCUMENT_ID",
                                                     "message": "document_id must be UUID."})
    from sqlalchemy import select as _select

    from backend.app.models.document import Document as _Document
    from backend.app.services.indexing_service import IndexingService

    doc = (await db.execute(_select(_Document).where(_Document.id == did))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail={"code": "DOCUMENT_NOT_FOUND",
                                                     "message": "No such processing attempt."})
    svc = IndexingService(db)
    job = await svc.ensure_job(did)
    return {"document_id": str(did), "job_id": str(job.id), "status": job.status}


@router.get("/indexing/jobs/{job_id}")
async def indexing_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    import uuid as _uuid

    try:
        jid = _uuid.UUID(str(job_id))
    except Exception:
        raise HTTPException(status_code=400, detail={"code": "INVALID_JOB_ID",
                                                     "message": "job_id must be UUID."})
    from backend.app.repositories.rag_repository import IndexingJobRepository

    job = await IndexingJobRepository(db).get_by_id(jid)
    if job is None:
        raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND",
                                                     "message": "No such indexing job."})
    return {
        "job_id": str(job.id), "document_id": str(job.document_id),
        "status": job.status, "attempt_count": job.attempt_count,
        "index_version": job.index_version, "error_code": job.error_code,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal.
    Only keep the basename, strip directory components.
    """
    import os
    # Remove any directory components
    basename = os.path.basename(filename)
    # Remove null bytes and other problematic chars
    basename = basename.replace("\x00", "").strip()
    if not basename:
        basename = "unnamed"
    return basename
