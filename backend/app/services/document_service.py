"""
FiniRAG Document Service — Pipeline Orchestrator.

The master pipeline:
  Upload → File Validation → Text Extraction Router
    → Native PDF or Render → OCR → Multi-page Aggregation
    → LLM Extraction → Normalization → Financial Validation
    → PostgreSQL Persist → API Response

This module ties all services together into a single flow.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.document import Document
from backend.app.providers.errors import LLMOutputTruncatedError, LLMRateLimitError
from backend.app.repositories import DocumentRepository
from backend.app.schemas.document import (
    DocumentProcessResponse,
    FileValidationResult,
    ProcessingMetadata,
)
from backend.app.services.document_validation_service import DocumentValidationService
from backend.app.services.text_extraction_router import TextExtractionRouter
from backend.app.services.image_preprocessor import ImagePreprocessor
from backend.app.services.ocr_service import OCRService
from backend.app.services.extraction_service import ExtractionService
from backend.app.services.financial_validation_service import FinancialValidationService

logger = logging.getLogger("finirag")

PAGE_SEPARATOR = "\n\n--- PAGE BREAK ---\n\n"


def _retain_source_text(text: str | None) -> str | None:
    """
    Bounded source text for the document row (finding #9).

    Returns None when there is nothing worth retaining (vision-fallback and
    failed extractions have no text) so NULL honestly means "no source".
    Otherwise caps at MAX_SOURCE_TEXT_CHARS — one pathological document must
    not bloat its row; page breaks survive because truncation cuts the tail.
    """
    from backend.app.core.config import settings

    if not text or not text.strip():
        return None
    cap = settings.MAX_SOURCE_TEXT_CHARS
    return text if len(text) <= cap else text[:cap]


def _join_pages(page_texts: list[str]) -> str:
    """
    Join per-page text, or return "" when no page produced any.

    Joining a list of empty strings still leaves the separators behind,
    and "--- PAGE BREAK ---" is not whitespace: a `.strip()` emptiness
    check reads it as content. A document whose OCR failed on every page
    would then be sent to the LLM as nothing but markers instead of
    going to the vision fallback — and only ever on multi-page files,
    since a single empty page joins to "".

    Pages that did produce text keep their position. Dropping a blank
    page would renumber every page after it, and evidence page numbers
    are quoted back to the user.
    """
    if not any(t.strip() for t in page_texts):
        return ""
    return PAGE_SEPARATOR.join(page_texts)


class DocumentService:
    """Orchestrates the full document processing pipeline."""

    def __init__(self, db: AsyncSession):
        self._db = db
        self._repo = DocumentRepository(db)
        self._file_validator = DocumentValidationService()
        self._text_router = TextExtractionRouter()
        self._preprocessor = ImagePreprocessor()
        self._ocr_service = OCRService()
        self._extraction_service = ExtractionService()
        self._financial_validator = FinancialValidationService()

    async def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        document_type: str,
    ) -> DocumentProcessResponse:
        """
        Process a single document through the full pipeline.

        Returns a complete DocumentProcessResponse.
        """
        start_time = time.time()

        logger.info("Request received", extra={
            "document_name": filename,
            "document_type": document_type,
        })

        # ── Step 1: File Validation ──
        file_result = self._file_validator.validate(file_bytes, filename)

        if file_result.status != "PASS":
            return await self._persist_failure(
                filename=filename,
                document_type=document_type,
                file_result=file_result,
                error_code=file_result.error or "VALIDATION_FAILED",
                start_time=start_time,
            )

        # ── Step 2: Text Extraction (Router decides native vs OCR) ──
        try:
            text, extraction_method, ocr_provider = await self._extract_text(
                file_bytes, file_result.file_type
            )
        except Exception as e:
            logger.error("Text extraction failed", extra={
                "document_name": filename,
                "error_type": type(e).__name__,
            })
            return await self._persist_failure(
                filename=filename,
                document_type=document_type,
                file_result=file_result,
                error_code="TEXT_EXTRACTION_FAILED",
                start_time=start_time,
            )

        if not text.strip():
            # ── Vision Fallback: OCR returned empty text ──
            # Send rendered images to the LLM for extraction.
            #
            # Rendering and extracting are reported separately on purpose.
            # Collapsing both into EMPTY_TEXT — as this did until the dataset
            # run of 2026-09-10 — tells a user whose upload was merely
            # throttled that their document had no readable text, and sends
            # them off to rescan a file that was never the problem.
            logger.info("OCR text empty, using vision fallback", extra={
                "document_name": filename,
            })
            try:
                images = self._preprocessor.render_pdf_pages(file_bytes) \
                    if file_result.file_type == "application/pdf" \
                    else self._preprocessor.preprocess_images(file_bytes, file_result.file_type)
            except Exception as e:
                logger.error("Vision fallback rendering failed", extra={
                    "document_name": filename,
                    "error_type": type(e).__name__,
                })
                return await self._persist_failure(
                    filename=filename,
                    document_type=document_type,
                    file_result=file_result,
                    error_code="TEXT_EXTRACTION_FAILED",
                    start_time=start_time,
                )

            if not images:
                # OCR found no text and there is no page to look at either.
                # This is the only branch EMPTY_TEXT genuinely describes.
                logger.error("Vision fallback has no pages to read", extra={
                    "document_name": filename,
                })
                return await self._persist_failure(
                    filename=filename,
                    document_type=document_type,
                    file_result=file_result,
                    error_code="EMPTY_TEXT",
                    start_time=start_time,
                )

            try:
                extracted_data = await self._extraction_service.extract_from_images(
                    images=images,
                    document_type=document_type,
                )
                # Named after the backend that actually read the pages, so a
                # document extracted by the fallback does not report itself as
                # having been read by the primary.
                answered_by = getattr(
                    self._extraction_service, "last_llm_provider", "gemini"
                )
                extraction_method = f"{answered_by}_vision"
                ocr_provider = f"{answered_by}_vision"
            except LLMRateLimitError as e:
                logger.error("Vision fallback throttled", extra={
                    "document_name": filename,
                    "retry_after_seconds": e.retry_after_seconds,
                })
                return await self._persist_failure(
                    filename=filename,
                    document_type=document_type,
                    file_result=file_result,
                    error_code="LLM_RATE_LIMITED",
                    start_time=start_time,
                    extraction_method=extraction_method,
                    ocr_provider=ocr_provider,
                )
            except LLMOutputTruncatedError as e:
                logger.error("Vision fallback output truncated", extra={
                    "document_name": filename,
                    "completion_tokens": e.completion_tokens,
                })
                return await self._persist_failure(
                    filename=filename,
                    document_type=document_type,
                    file_result=file_result,
                    error_code="LLM_OUTPUT_TRUNCATED",
                    start_time=start_time,
                    extraction_method=extraction_method,
                    ocr_provider=ocr_provider,
                )
            except Exception as e:
                logger.error("Vision fallback failed", extra={
                    "document_name": filename,
                    "error_type": type(e).__name__,
                })
                return await self._persist_failure(
                    filename=filename,
                    document_type=document_type,
                    file_result=file_result,
                    error_code="LLM_EXTRACTION_FAILED",
                    start_time=start_time,
                    extraction_method=extraction_method,
                    ocr_provider=ocr_provider,
                )
        else:
            # ── Step 3: LLM Extraction (text-based) ──
            try:
                extracted_data = await self._extraction_service.extract(
                    text=text,
                    document_type=document_type,
                )
            except LLMRateLimitError as e:
                logger.error("LLM extraction throttled", extra={
                    "document_name": filename,
                    "retry_after_seconds": e.retry_after_seconds,
                })
                return await self._persist_failure(
                    filename=filename,
                    document_type=document_type,
                    file_result=file_result,
                    error_code="LLM_RATE_LIMITED",
                    start_time=start_time,
                    extraction_method=extraction_method,
                    ocr_provider=ocr_provider,
                )
            except LLMOutputTruncatedError as e:
                logger.error("LLM output truncated", extra={
                    "document_name": filename,
                    "completion_tokens": e.completion_tokens,
                })
                return await self._persist_failure(
                    filename=filename,
                    document_type=document_type,
                    file_result=file_result,
                    error_code="LLM_OUTPUT_TRUNCATED",
                    start_time=start_time,
                    extraction_method=extraction_method,
                    ocr_provider=ocr_provider,
                )
            except Exception as e:
                logger.error("LLM extraction failed", extra={
                    "document_name": filename,
                    "error_type": type(e).__name__,
                })
                return await self._persist_failure(
                    filename=filename,
                    document_type=document_type,
                    file_result=file_result,
                    error_code="LLM_EXTRACTION_FAILED",
                    start_time=start_time,
                    extraction_method=extraction_method,
                    ocr_provider=ocr_provider,
                )

        # ── Step 4: Financial Validation ──
        try:
            validation_result = self._financial_validator.validate(
                extracted_data=extracted_data,
                document_type=document_type,
            )
            validation_dict = validation_result.model_dump(mode="json")
        except Exception as e:
            logger.error("Validation failed", extra={
                "document_name": filename,
                "error_type": type(e).__name__,
            })
            # Validation failure is NOT a processing failure
            validation_dict = {"checks": [], "overall_status": "NOT_APPLICABLE", "summary": {}}

        # ── Step 5: Persist ──
        elapsed_ms = int((time.time() - start_time) * 1000)
        metadata = ProcessingMetadata(
            text_extraction_method=extraction_method,
            ocr_provider=ocr_provider,
            # Which backend actually answered, not which one was tried first.
            # With a fallback chain in place a hardcoded "gemini" would credit
            # Gemini for numbers Groq read.
            llm_provider=getattr(
                self._extraction_service, "last_llm_provider", "gemini"
            ),
            processed_at=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=elapsed_ms,
        )

        document = Document(
            document_name=filename,
            document_type=document_type,
            processing_status="PASS",
            file_validation=file_result.model_dump(),
            extracted_data=extracted_data,
            validation=validation_dict,
            processing_metadata=metadata.model_dump(),
            # Retain the source the extraction was built from (bounded).
            # Vision-fallback successes have no text — NULL, honestly.
            source_text=_retain_source_text(text),
        )

        document = await self._repo.create(document)

        # ── Step 6: Indexing (separate pipeline, same request) ──
        # A FAILED index must never fail the document. The job row survives
        # restarts; a bounded worker or reindex endpoint can resume it.
        try:
            from backend.app.services.indexing_service import IndexingService

            indexer = IndexingService(self._db)
            job = await indexer.ensure_job(document.id)
            try:
                await indexer.process_job(job)
            except Exception as e:
                logger.error("Inline indexing failed, job retained", extra={
                    "document_name": filename, "error_type": type(e).__name__})
        except Exception as e:
            logger.error("Indexing job setup failed", extra={
                "document_name": filename, "error_type": type(e).__name__})

        from backend.app.core.metrics import (
            DOCUMENT_LATENCY, DOCUMENTS_PROCESSED, metrics,
        )

        metrics.inc(DOCUMENTS_PROCESSED, {"document_type": document_type,
                                          "processing_status": "PASS",
                                          "validation_status": validation_dict.get(
                                              "overall_status", "NOT_APPLICABLE")})
        metrics.observe(DOCUMENT_LATENCY, elapsed_ms / 1000.0,
                        {"document_type": document_type})

        return DocumentProcessResponse(
            document_name=filename,
            document_type=document_type,
            processing_status="PASS",
            file_validation=file_result,
            extracted_data=extracted_data,
            validation=validation_dict,
            processing_metadata=metadata,
        )

    async def _extract_text(
        self,
        file_bytes: bytes,
        file_type: str,
    ) -> tuple[str, str, str | None]:
        """
        Extract text using the routing strategy:
          PDF → try native text → fallback to render+OCR
          Image → preprocess → OCR

        Returns: (aggregated_text, method, ocr_provider_or_none)
        """
        # Try native PDF text first (Correction #1)
        native_result = self._text_router.should_use_native_text(
            file_bytes, file_type
        )

        if native_result is not None:
            # Native text is usable — aggregate all pages
            return _join_pages(native_result.text_by_page), "native_pdf", None

        # Fall back to render + OCR
        if file_type == "application/pdf":
            images = self._preprocessor.render_pdf_pages(file_bytes)
        else:
            images = self._preprocessor.preprocess_images(file_bytes, file_type)

        # OCR all pages
        ocr_results = await self._ocr_service.extract_text_from_images(images)

        # Aggregate text from all pages (multi-page support for CF statements)
        page_texts = [result.text for result in ocr_results]

        # Record every provider that contributed, in the order they first
        # did. Pages that no provider could read report "none" and are left
        # out, so a document that was only partly readable does not claim
        # clean OCR coverage — that partial page is the lower-confidence
        # part of the document a reviewer most needs to know about.
        used = [r.provider for r in ocr_results if r.provider != "none"]
        ocr_provider = "+".join(dict.fromkeys(used)) or "none"

        return _join_pages(page_texts), "ocr", ocr_provider

    async def _persist_failure(
        self,
        filename: str,
        document_type: str,
        file_result: FileValidationResult,
        error_code: str,
        start_time: float,
        extraction_method: str = "",
        ocr_provider: str | None = None,
    ) -> DocumentProcessResponse:
        """Persist a failed processing attempt."""
        elapsed_ms = int((time.time() - start_time) * 1000)

        metadata = ProcessingMetadata(
            text_extraction_method=extraction_method,
            ocr_provider=ocr_provider,
            processed_at=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=elapsed_ms,
        )

        error_dict = {"code": error_code, "message": error_code}

        document = Document(
            document_name=filename,
            document_type=document_type,
            processing_status="FAILED",
            file_validation=file_result.model_dump(),
            extracted_data=None,
            validation=None,
            processing_metadata=metadata.model_dump(),
            error=error_dict,
        )

        await self._repo.create(document)

        from backend.app.core.metrics import (
            DOCUMENT_LATENCY, DOCUMENTS_PROCESSED, metrics,
        )

        metrics.inc(DOCUMENTS_PROCESSED, {"document_type": document_type,
                                          "processing_status": "FAILED",
                                          "error_code": error_code})
        metrics.observe(DOCUMENT_LATENCY, elapsed_ms / 1000.0,
                        {"document_type": document_type})

        return DocumentProcessResponse(
            document_name=filename,
            document_type=document_type,
            processing_status="FAILED",
            file_validation=file_result,
            processing_metadata=metadata,
            error=error_dict,
        )

    async def get_document_by_name(
        self, document_name: str
    ) -> Document | None:
        """Get the latest processing result for a document."""
        return await self._repo.get_latest_by_name(document_name)

    async def list_documents(
        self, limit: int = 100, offset: int = 0
    ) -> tuple[list[Document], int]:
        """List all processing attempts."""
        return await self._repo.list_all(limit=limit, offset=offset)
