"""
FiniRAG Text Extraction Router (Correction #1).

Decides whether to use native PDF text extraction (PyMuPDF) or
render-and-OCR based on the document content.

Flow:
  PDF → probe for native text → usable? → native extraction
                                     └──→ render 300 DPI → OCR
  JPG/PNG → always render → OCR

A PDF has "usable" native text when:
  1. Text length >= 50 characters
  2. Contains at least one financial keyword (Amount, Total, etc.)
  3. Text is NOT just whitespace or noise

Only one PDF in the dataset (CF Statement 2022) has usable native text.
All others are vector-outlined (no text layer, no images — just paths).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import fitz  # PyMuPDF

logger = logging.getLogger("finirag")

# Keywords that indicate genuinely useful text content
FINANCIAL_KEYWORDS = [
    "total", "amount", "balance", "income", "expenditure",
    "profit", "loss", "cash", "net", "assets", "liabilities",
    "interest", "tax", "date", "period", "schedule",
    "operating", "investing", "financing",
    "invoice", "receipt", "qty", "quantity", "price",
    "subtotal", "discount", "payment", "due",
]


@dataclass
class TextExtractionResult:
    """Result of text extraction routing."""
    method: str                    # "native_pdf" or "ocr"
    text_by_page: list[str]        # Text per page
    total_chars: int = 0


class TextExtractionRouter:
    """Routes between native PDF text extraction and OCR."""

    MIN_TEXT_LENGTH = 50  # Minimum chars to consider native text usable
    MIN_KEYWORD_MATCHES = 1  # Minimum financial keywords required

    def probe_native_text(self, file_bytes: bytes) -> TextExtractionResult | None:
        """
        Attempt native text extraction from PDF using PyMuPDF.

        Returns TextExtractionResult if native text is usable, None otherwise.
        """
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception:
            return None

        text_by_page = []
        total_chars = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            text_by_page.append(text)
            total_chars += len(text)

        doc.close()

        # ── Usability check ──
        if total_chars < self.MIN_TEXT_LENGTH:
            logger.info("Text extraction routing", extra={
                "method": "ocr",
                "reason": f"native text too short ({total_chars} chars)",
            })
            return None

        # Check for financial keywords
        all_text_lower = " ".join(text_by_page).lower()
        keyword_count = sum(
            1 for kw in FINANCIAL_KEYWORDS if kw in all_text_lower
        )

        if keyword_count < self.MIN_KEYWORD_MATCHES:
            logger.info("Text extraction routing", extra={
                "method": "ocr",
                "reason": f"insufficient keywords ({keyword_count})",
            })
            return None

        logger.info("Text extraction routing", extra={
            "method": "native_pdf",
            "chars": total_chars,
            "keywords_matched": keyword_count,
        })

        return TextExtractionResult(
            method="native_pdf",
            text_by_page=text_by_page,
            total_chars=total_chars,
        )

    def should_use_native_text(self, file_bytes: bytes, file_type: str) -> TextExtractionResult | None:
        """
        Main routing decision.

        Only PDFs can have native text. Images always go through OCR.
        """
        if file_type != "application/pdf":
            return None

        return self.probe_native_text(file_bytes)
