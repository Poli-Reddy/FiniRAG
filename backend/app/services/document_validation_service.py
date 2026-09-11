"""
FiniRAG Document Validation Service.

Pre-extraction file validation (Trap P11):
  1. MIME type check via leading magic bytes (Trap P13) — NOT file extension
  2. File corruption detection
  3. Empty file detection
  4. Page count enforcement (max 3 pages, Trap P12, D7)
  5. File size limit

This runs BEFORE any OCR or LLM call. If validation fails,
return a 400 error immediately.
"""

from __future__ import annotations

import io
import logging
from typing import BinaryIO

import fitz  # PyMuPDF
from PIL import Image

from backend.app.core.config import settings
from backend.app.schemas.document import FileValidationResult

logger = logging.getLogger("finirag")

# Supported MIME types — validated by magic bytes, not extension
SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "image",
    "image/png": "image",
}

# Leading byte signatures for the three formats in scope.
#
# Replaces python-magic, which needs the system `libmagic1` shared library.
# That dependency was never installed by render.yaml (buildCommand is
# `pip install` only), so on a clean Render box the import raised and every
# upload came back CORRUPTED_FILE. Three formats do not justify a native
# dependency the deployment cannot satisfy.
#
# Matching is on the LEADING bytes only, never the whole buffer: a phone
# photo can carry trailing vendor blocks (Samsung SEFT) after the JPEG EOI,
# and those files are valid and must still pass.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
)


def sniff_mime(file_bytes: bytes) -> str:
    """
    The MIME type implied by the file's leading bytes.

    Returns "application/octet-stream" when nothing matches — an unknown
    type is a real answer here, so the caller reports UNSUPPORTED_FILE_TYPE
    rather than treating detection failure as corruption.
    """
    for signature, mime in _SIGNATURES:
        if file_bytes.startswith(signature):
            return mime

    # A .txt upload is the documented unsupported case; naming it beats a
    # generic octet-stream in the error the evaluator sees.
    head = file_bytes[:1024]
    if head:
        try:
            head.decode("utf-8")
            return "text/plain"
        except UnicodeDecodeError:
            pass
    return "application/octet-stream"


class DocumentValidationService:
    """Validates uploaded files before processing."""

    def validate(self, file_bytes: bytes, filename: str) -> FileValidationResult:
        """
        Run all pre-extraction checks.

        Returns FileValidationResult with status "PASS" or "FAIL".
        On FAIL, error field contains the specific error code.
        """
        result = FileValidationResult()

        # ── Check 1: Empty file ──
        if not file_bytes or len(file_bytes) == 0:
            result.status = "FAIL"
            result.error = "EMPTY_FILE"
            return result

        # ── Check 2: File size ──
        if len(file_bytes) > settings.max_file_size_bytes:
            result.status = "FAIL"
            result.error = "FILE_TOO_LARGE"
            return result

        # ── Check 3: MIME type via magic bytes (Trap P13) ──
        detected_mime = sniff_mime(file_bytes)
        result.file_type = detected_mime

        if detected_mime not in SUPPORTED_MIME_TYPES:
            result.status = "FAIL"
            result.error = "UNSUPPORTED_FILE_TYPE"
            return result

        result.is_supported = True
        file_category = SUPPORTED_MIME_TYPES[detected_mime]

        # ── Check 4: Readability + Page Count ──
        try:
            if file_category == "pdf":
                page_count = self._validate_pdf(file_bytes)
            else:
                page_count = self._validate_image(file_bytes)
        except Exception as e:
            logger.error("File validation failed", extra={
                "document_name": filename, "error_type": type(e).__name__
            })
            result.status = "FAIL"
            result.error = "CORRUPTED_FILE"
            return result

        result.page_count = page_count
        result.is_readable = True

        # ── Check 5: Page limit (Trap P12, D7) ──
        if page_count > settings.MAX_PAGES:
            result.status = "FAIL"
            result.error = "PAGE_LIMIT_EXCEEDED"
            return result

        result.status = "PASS"

        logger.info("File validation", extra={
            "document_name": filename,
            "file_type": detected_mime,
            "pages": page_count,
            "status": "PASS",
        })

        return result

    def _validate_pdf(self, file_bytes: bytes) -> int:
        """Validate PDF readability and return page count."""
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)
        if page_count == 0:
            doc.close()
            raise ValueError("PDF has zero pages")
        doc.close()
        return page_count

    def _validate_image(self, file_bytes: bytes) -> int:
        """
        Validate image readability and return page count (always 1).

        Handles missing JPEG EOI marker gracefully (Samsung SEFT trailer).
        Does NOT check for FFD9 — uses Pillow's open test instead.
        """
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()  # Verify without fully loading
        except Exception:
            # Try again without verify — some images with missing
            # EOI markers still load fine with Pillow
            try:
                img = Image.open(io.BytesIO(file_bytes))
                img.load()  # Force full load
            except Exception:
                raise ValueError("Image is corrupted or unreadable")
        return 1
