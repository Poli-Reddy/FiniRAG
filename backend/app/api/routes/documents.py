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

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.rate_limit import rate_limit
from backend.app.api.auth import require_api_key
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

router = APIRouter(prefix="/api/v1", tags=["documents"],
                     dependencies=[Depends(require_api_key)])

# Multipart framing overhead slack for the Content-Length pre-check: the raw
# body is always larger than the file itself, so the header gate allows a
# small margin and the bounded read below enforces the exact file limit.
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024

# Health stays OUTSIDE the auth gate: platform probes (Render
# healthCheckPath) and uptime monitors must reach it without credentials.
# It carries no document data — only status counts and config labels.
public_router = APIRouter(prefix="/api/v1", tags=["health"])


# ─────────────────────────────────────────────────────────────────────
# POST /api/v1/documents/process
# ─────────────────────────────────────────────────────────────────────

@router.post("/documents/process", response_model=DocumentProcessResponse,
             dependencies=[Depends(rate_limit)])
async def process_document(
    request: Request,
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

    # Bounded read: the size gate must trigger BEFORE the whole body is
    # buffered, or a huge upload is a memory-exhaustion vector regardless of
    # the configured limit. Reject early on Content-Length, then read at most
    # limit+1 bytes so an oversized body is cut off mid-stream.
    max_bytes = settings.max_file_size_bytes
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes + _MULTIPART_OVERHEAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={"code": "FILE_TOO_LARGE",
                            "message": f"Uploads are limited to {settings.MAX_FILE_SIZE_MB} MB."},
                )
        except ValueError:
            pass  # malformed header: fall through to the bounded read
    file_bytes = await file.read(size=max_bytes + 1)
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={"code": "FILE_TOO_LARGE",
                    "message": f"Uploads are limited to {settings.MAX_FILE_SIZE_MB} MB."},
        )

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

@router.get("/documents", response_model=DocumentListResponse,
             dependencies=[Depends(rate_limit)])
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

@router.get("/documents/{document_name}", dependencies=[Depends(rate_limit)])
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

@public_router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Liveness + cheap readiness. Deliberately cheap (indexed metadata
    queries only) and secret-free — safe to poll from a platform check.

    Readiness is reported per subsystem, never as one blended verdict:
      database:            SELECT 1 probe (datastore liveness)
      vector_ready:        pgvector extension present AND HNSW index exists
      fts_ready:           GIN full-text index exists
      native_paths_ready:  both of the above (what strict retrieval needs)
      retrieval_mode:      CONFIGURED semantic search ("hybrid" = key present,
                           NOT proof of live quota or semantics)
    No embedding or LLM calls are made here; use scripts/verify_retrieval.py
    and scripts/live_embedding_smoke.py for live provider proof.
    """
    from backend.app.core import database as _database

    async def _scalar_one(query, default=0):
        try:
            return int((await db.execute(query)).scalar() or default)
        except Exception:
            return default

    db_status = "disconnected"
    indexed_documents = 0
    total_chunks = 0
    vector_ready = False
    fts_ready = False
    try:
        from sqlalchemy import func, select, text

        from backend.app.models.rag import DocumentChunk

        await db.execute(text("SELECT 1"))
        db_status = "connected"
        total_chunks = await _scalar_one(
            select(func.count(DocumentChunk.id)).where(
                DocumentChunk.is_active == True))  # noqa: E712
        indexed_documents = await _scalar_one(
            select(func.count(func.distinct(DocumentChunk.document_id))).where(
                DocumentChunk.is_active == True))  # noqa: E712
    except Exception:
        db_status = "disconnected"
    # Index-existence probe is independent: a missing pg_indexes view (or a
    # stubbed session in tests) must not flip the datastore verdict above.
    try:
        if db_status == "connected" and _database.engine.dialect.name == "postgresql":
            from sqlalchemy import text as _text

            idx = (await db.execute(_text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'document_chunks'"))).fetchall()
            names = {r[0] for r in idx}
            vector_ready = "idx_chunks_vec" in names
            fts_ready = "idx_chunks_fts" in names
    except Exception:
        vector_ready = False
        fts_ready = False

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
        native_paths_ready=vector_ready and fts_ready,
        vector_ready=vector_ready,
        fts_ready=fts_ready,
        index_version=settings.INDEX_VERSION,
        indexed_chunks=total_chunks,
        indexed_documents=indexed_documents,
        provider_breakers=breakers,
    )


# ─────────────────────────────────────────────────────────────────────
# GET /api/v1/documents/{document_id}/index-status  + reindex + jobs
# ─────────────────────────────────────────────────────────────────────

@router.get("/documents/{document_id}/index-status",
             dependencies=[Depends(rate_limit)])
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


@router.post("/documents/{document_id}/reindex",
              dependencies=[Depends(rate_limit)])
async def reindex_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Idempotent reindex of one processing attempt. Does not rerun extraction.

    Queue semantics: this endpoint only *queues* work (creates or reuses a
    PENDING job) and returns immediately. A worker drains the queue —
    inline at upload time, or via `python scripts/run_indexing_worker.py`.
    Poll `GET /indexing/jobs/{job_id}` until COMPLETED/FAILED. `queued` is
    True when work remains; False when the active job is already COMPLETED.
    """
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
    queued = job.status in ("PENDING", "PROCESSING")
    return {"document_id": str(did), "job_id": str(job.id), "status": job.status,
            "queued": queued,
            "message": ("Job queued — drain with scripts/run_indexing_worker.py "
                        "and poll GET /indexing/jobs/{job_id}.")
                       if queued else "Already indexed — no work queued."}


@router.get("/indexing/jobs/{job_id}", dependencies=[Depends(rate_limit)])
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
