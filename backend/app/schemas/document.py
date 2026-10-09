"""
FiniRAG API Request/Response Schemas.

Pydantic models for the REST API contract (Section 21 of Gate 2.1).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# File Validation
# ─────────────────────────────────────────────────────────────────────

class FileValidationResult(BaseModel):
    """Result of pre-extraction file validation."""
    file_type: str = ""              # "image/jpeg", "application/pdf"
    is_supported: bool = False
    is_readable: bool = False
    page_count: int = 0
    status: str = "FAIL"             # "PASS" | "FAIL"
    error: str | None = None


# ─────────────────────────────────────────────────────────────────────
# Processing Metadata
# ─────────────────────────────────────────────────────────────────────

class ProcessingMetadata(BaseModel):
    """Metadata about how the document was processed."""
    text_extraction_method: str = ""  # "native_pdf" | "ocr"
    ocr_provider: str | None = None   # "google_vision" | "<llm>_vision" | "none"
    llm_provider: str = ""            # "gemini"
    processed_at: str = ""            # ISO timestamp
    processing_time_ms: int = 0


# ─────────────────────────────────────────────────────────────────────
# API Response Models
# ─────────────────────────────────────────────────────────────────────

class DocumentProcessResponse(BaseModel):
    """Full response for POST /api/v1/documents/process."""
    document_name: str
    document_type: str
    processing_status: str           # "PASS" | "FAILED"
    file_validation: FileValidationResult
    extracted_data: dict | None = None
    validation: dict | None = None
    processing_metadata: ProcessingMetadata
    error: dict | None = None

    # No Decimal encoder here on purpose. extracted_data and validation
    # arrive already JSON-dumped by their own models, where each Decimal is
    # serialized to an exact string. A float encoder at this level would
    # only reintroduce the binary rounding those serializers exist to avoid.


class DocumentListItem(BaseModel):
    """Summary item for GET /api/v1/documents."""
    id: str
    document_name: str
    document_type: str
    processing_status: str
    # Kept distinct from processing_status: a readable document whose
    # arithmetic does not reconcile is processing_status=PASS with
    # validation_status=FAIL, and the dashboard has to show both or the
    # two failure modes look identical (Trap D31). "NOT_RUN" when the
    # document never reached validation.
    validation_status: str = "NOT_RUN"
    processing_metadata: dict
    created_at: str


class DocumentListResponse(BaseModel):
    """Response for GET /api/v1/documents."""
    documents: list[DocumentListItem] = []
    total: int = 0


class HealthResponse(BaseModel):
    """Response for GET /api/v1/health."""
    status: str = "healthy"
    database: str = "connected"
    timestamp: str = ""
    # RAG readiness (no secrets): configured vs missing key, index version.
    embedding_model: str = ""
    embedding_configured: bool = False
    retrieval_mode: str = "keyword-only"
    index_version: str = "v1"


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: ErrorDetail


class ErrorDetail(BaseModel):
    """Error detail within ErrorResponse."""
    code: str                        # "UNSUPPORTED_FILE_TYPE", etc.
    message: str


# ─────────────────────────────────────────────────────────────────────
# Valid Document Types (Trap P10 — no classification, user selects)
# ─────────────────────────────────────────────────────────────────────

VALID_DOCUMENT_TYPES = {
    "invoice",
    "balance_sheet",
    "profit_and_loss",
    "cash_flow_statement",
}
