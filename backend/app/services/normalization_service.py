"""
FiniRAG Normalization Service.

Converts raw LLM-extracted values into canonical ExtractedValue objects.
This is where the dataset's tricky edge cases are handled:

  - Parenthesized negatives: (2,500.00) → -2500.0  (Trap D26)
  - European decimal comma: 3,49 → 3.49  (Trap D9)
  - Standalone dash → EXPLICIT_ZERO with value=0.0  (Correction #3)
  - Currency symbol removal
  - Unit detection (₹ in '000, ₹ in crore)
  - EPS exception: always in ₹ per share, NOT document unit (Trap D8)
  - Date handling (preserve ambiguous, normalize unambiguous)

All monetary values are normalized to Decimal (Correction #8).
"""

from __future__ import annotations

import re
import logging
from decimal import Decimal, InvalidOperation

from backend.app.schemas.extraction import (
    ExtractedValue, Evidence, FieldStatus,
    InvoiceExtraction, BalanceSheetExtraction,
    ProfitLossExtraction, CashFlowExtraction,
)

logger = logging.getLogger("finirag")

# ─────────────────────────────────────────────────────────────────────
# Currency / Unit Patterns
# ─────────────────────────────────────────────────────────────────────

CURRENCY_SYMBOLS = re.compile(r"[₹$€£¥]", re.UNICODE)

# Alphabetic currency codes / prefixes (INR, RM, MYR, Rs., USD …) and
# trailing accounting markers (CR, DR). Stripped positionally rather than
# by character class so that "MYR 12.50" does not lose its Y.
LEADING_ALPHA = re.compile(r"^[A-Za-z]{1,4}\.?\s*")
TRAILING_ALPHA = re.compile(r"\s*[A-Za-z]{1,3}\.?$")

# Space-like characters used as digit-group separators.
SPACE_CHARS = re.compile(r"[\s  ]")

# What a parseable number may consist of once cleaned.
NUMERIC_ONLY = re.compile(r"^[0-9.,]+$")

NIL_TOKENS = ("-", "−", "–", "—", "Nil", "nil", "NIL", "N.A.", "NA")

UNIT_KEYWORDS = {
    "in thousands": "INR_THOUSANDS",
    "'000": "INR_THOUSANDS",
    "in '000": "INR_THOUSANDS",
    "in crore": "INR_CRORE",
    "in crores": "INR_CRORE",
    "₹ in crore": "INR_CRORE",
    "c in crore": "INR_CRORE",  # OCR artifact: ₹ → C (Trap A9)
}


class NormalizationService:
    """Normalizes raw extracted values into canonical form."""

    def normalize_number(self, raw_value: str) -> tuple[Decimal | None, FieldStatus]:
        """
        Normalize a raw string to a Decimal value and determine status.

        Returns:
            (value, status) tuple
        """
        if raw_value is None:
            return None, FieldStatus.MISSING

        # Convert to string and strip
        text = str(raw_value).strip()

        if not text:
            return None, FieldStatus.MISSING

        # ── Standalone dash → EXPLICIT_ZERO (Correction #3) ──
        if text in NIL_TOKENS:
            return Decimal("0"), FieldStatus.EXPLICIT_ZERO

        # ── Remove currency symbols, percent signs, dot-leader noise ──
        cleaned = CURRENCY_SYMBOLS.sub("", text)
        cleaned = cleaned.replace("%", "")
        cleaned = re.sub(r"\.{2,}", "", cleaned).strip()

        # ── Parenthesized negatives: (2,500.00) → -2500.00 (Trap D26) ──
        is_negative = False
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1].strip()
            is_negative = True
        elif cleaned[:1] in ("-", "−", "–", "—"):
            cleaned = cleaned[1:].strip()
            is_negative = True
        elif cleaned[-1:] in ("-", "−"):
            # Some OCR output places the sign after the digits
            cleaned = cleaned[:-1].strip()
            is_negative = True

        # ── Strip alphabetic currency codes and accounting markers ──
        cleaned = LEADING_ALPHA.sub("", cleaned)
        cleaned = TRAILING_ALPHA.sub("", cleaned)

        # ── Remove digit-group spaces ──
        cleaned = SPACE_CHARS.sub("", cleaned).strip()

        # Anything left over that is not digits and separators is unreadable
        if not cleaned or not NUMERIC_ONLY.match(cleaned):
            return None, FieldStatus.UNREADABLE

        # ── Detect decimal convention ──
        value = self._parse_with_decimal_detection(cleaned)
        if value is None:
            return None, FieldStatus.UNREADABLE

        if is_negative:
            value = -value

        # Determine status
        if value == 0:
            return Decimal("0"), FieldStatus.EXPLICIT_ZERO
        else:
            return value, FieldStatus.EXPLICIT

    def _parse_with_decimal_detection(self, text: str) -> Decimal | None:
        """
        Parse a number string with automatic decimal convention detection.

        Handles:
          - Standard: 1,234.56 → 1234.56
          - European: 3,49 → 3.49 (single comma, ≤2 digits after)
          - No decimal: 17,995,066 → 17995066
          - Already decimal: 83.33 → 83.33
        """
        if not text:
            return None

        has_comma = "," in text
        has_period = "." in text

        try:
            if has_comma and has_period:
                # Standard: comma is thousands, period is decimal
                cleaned = text.replace(",", "")
                return Decimal(cleaned)

            elif has_comma and not has_period:
                # Could be European decimal OR thousands separator
                parts = text.split(",")
                last_part = parts[-1]

                if len(parts) == 2 and len(last_part) <= 2:
                    # European decimal comma: 3,49 → 3.49
                    cleaned = text.replace(",", ".")
                    return Decimal(cleaned)
                else:
                    # Thousands separator: 17,995,066
                    cleaned = text.replace(",", "")
                    return Decimal(cleaned)

            elif has_period and not has_comma:
                # Standard decimal point
                return Decimal(text)

            else:
                # Plain integer
                return Decimal(text)

        except (InvalidOperation, ValueError):
            return None

    def detect_unit_code(self, ocr_text: str) -> str:
        """
        Detect the unit convention from document header text.

        Returns unit code string or empty string.
        """
        text_lower = ocr_text.lower()
        for keyword, code in UNIT_KEYWORDS.items():
            if keyword in text_lower:
                return code
        return ""

    def build_extracted_value(
        self,
        raw_value: str | None,
        source_text: str = "",
        page_number: int = 1,
        bounding_box: list[float] | None = None,
        unit: str | None = None,
        source_verified: bool = False,
    ) -> ExtractedValue:
        """
        Build a complete ExtractedValue from a raw string.

        Preserves both the original value and the normalized value.
        Evidence carries a transcribed snippet, NOT a verified quotation:
        an absent snippet stays empty (it is never echoed from the raw
        value — the raw value is itself model-transcribed, so echoing it
        would manufacture verbatim provenance). Callers that checked the
        snippet against retained source text pass source_verified=True.
        `unit` records the scale the figure is stated in (INR_THOUSANDS,
        INR_CRORE) so a value can never be read without its scale.
        """
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            return ExtractedValue(
                value=None,
                original_value=str(raw_value) if raw_value else "",
                status=FieldStatus.MISSING,
                evidence=Evidence(
                    source_text=source_text,
                    page_number=page_number,
                    bounding_box=bounding_box,
                ),
                unit=unit,
            )

        value, status = self.normalize_number(str(raw_value))

        return ExtractedValue(
            value=value,
            original_value=str(raw_value),
            status=status,
            evidence=Evidence(
                source_text=source_text,
                page_number=page_number,
                bounding_box=bounding_box,
                verified=source_verified and bool(source_text.strip()),
            ),
            unit=unit,
        )

    def build_text_value(
        self,
        raw_value: str | None,
        source_text: str = "",
        page_number: int = 1,
        unit: str | None = None,
        source_verified: bool = False,
    ) -> ExtractedValue:
        """Build an ExtractedValue for non-numeric text fields."""
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            return ExtractedValue(
                value=None,
                original_value="",
                status=FieldStatus.MISSING,
                evidence=Evidence(source_text=source_text, page_number=page_number),
                unit=unit,
            )

        return ExtractedValue(
            value=str(raw_value).strip(),
            original_value=str(raw_value),
            status=FieldStatus.EXPLICIT,
            evidence=Evidence(
                source_text=source_text,
                page_number=page_number,
                verified=source_verified and bool(source_text.strip()),
            ),
            unit=unit,
        )
