"""
Unit tests for NormalizationService.

This is where the dataset's number-formatting traps are pinned down:
parenthesised negatives, nil dashes, thousands separators, European
decimal commas, currency prefixes, and unit descriptors.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from backend.app.schemas.extraction import FieldStatus
from backend.app.services.normalization_service import NormalizationService


@pytest.fixture()
def n() -> NormalizationService:
    return NormalizationService()


# ─────────────────────────────────────────────────────────────────────
# Missing vs zero vs unreadable (Trap P7)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [None, "", "   "])
def test_absent_values_are_missing_not_zero(n, raw):
    value, status = n.normalize_number(raw)
    assert status is FieldStatus.MISSING
    assert value is None


@pytest.mark.parametrize("raw", ["-", "−", "–", "—", "Nil", "nil", "NIL"])
def test_nil_dashes_are_explicit_zero(n, raw):
    value, status = n.normalize_number(raw)
    assert status is FieldStatus.EXPLICIT_ZERO
    assert value == Decimal("0")


@pytest.mark.parametrize("raw", ["0", "0.00", "0,00"])
def test_literal_zero_is_explicit_zero(n, raw):
    value, status = n.normalize_number(raw)
    assert status is FieldStatus.EXPLICIT_ZERO
    assert value == Decimal("0")


@pytest.mark.parametrize("raw", ["####", "abc", "N/A", "??"])
def test_illegible_values_are_unreadable_not_zero(n, raw):
    value, status = n.normalize_number(raw)
    assert status is FieldStatus.UNREADABLE
    assert value is None


# ─────────────────────────────────────────────────────────────────────
# Sign handling (Trap D26)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("(2,500.00)", Decimal("-2500.00")),
    ("(1,234)", Decimal("-1234")),
    ("(0.01)", Decimal("-0.01")),
    ("-2,500.00", Decimal("-2500.00")),
    ("−3,988.06", Decimal("-3988.06")),   # U+2212 minus
])
def test_parentheses_and_minus_mean_negative(n, raw, expected):
    value, status = n.normalize_number(raw)
    assert status is FieldStatus.EXPLICIT
    assert value == expected


def test_positive_values_stay_positive(n):
    value, status = n.normalize_number("19,069.34")
    assert status is FieldStatus.EXPLICIT
    assert value == Decimal("19069.34")


# ─────────────────────────────────────────────────────────────────────
# Separator conventions
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("1,234.56", Decimal("1234.56")),
    ("17,995,066,442", Decimal("17995066442")),
    ("1,799,506.63", Decimal("1799506.63")),
    ("228834.51", Decimal("228834.51")),
    ("83.33", Decimal("83.33")),
    ("5", Decimal("5")),
])
def test_standard_separators(n, raw, expected):
    value, _ = n.normalize_number(raw)
    assert value == expected


def test_european_decimal_comma(n):
    """3,49 is 3.49 — a single comma with <=2 trailing digits (Trap D9)."""
    value, status = n.normalize_number("3,49")
    assert status is FieldStatus.EXPLICIT
    assert value == Decimal("3.49")


def test_three_trailing_digits_is_thousands_not_decimal(n):
    """1,234 must be one thousand two hundred thirty four — not 1.234."""
    value, _ = n.normalize_number("1,234")
    assert value == Decimal("1234")


def test_decimal_precision_is_exact_not_float(n):
    """Money must never round-trip through binary float."""
    value, _ = n.normalize_number("0.1")
    assert value == Decimal("0.1")
    assert value + Decimal("0.2") == Decimal("0.3")


def test_dot_leaders_are_stripped(n):
    value, _ = n.normalize_number("1,234.56")
    assert value == Decimal("1234.56")
    value, _ = n.normalize_number("500......")
    assert value == Decimal("500")


# ─────────────────────────────────────────────────────────────────────
# Currency prefixes
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("₹ 1,234.56", Decimal("1234.56")),
    ("$1,234.56", Decimal("1234.56")),
    ("€1.234", Decimal("1.234")),
    ("£99.99", Decimal("99.99")),
    ("RM 12.50", Decimal("12.50")),      # Malaysian receipts in the dataset
    ("MYR 12.50", Decimal("12.50")),     # ISO code form of the same currency
    ("INR 500", Decimal("500")),
])
def test_currency_prefixes_are_stripped(n, raw, expected):
    value, status = n.normalize_number(raw)
    assert status is FieldStatus.EXPLICIT, f"{raw!r} -> {status}"
    assert value == expected


def test_percent_suffix_on_tax_rate_is_parsed(n):
    """A tax rate printed as '10%' must parse to 10, not become UNREADABLE."""
    value, status = n.normalize_number("10%")
    assert status is FieldStatus.EXPLICIT
    assert value == Decimal("10")


def test_trailing_credit_debit_markers(n):
    """'1,234.56 CR' should still yield a number."""
    value, status = n.normalize_number("1,234.56 CR")
    assert status is FieldStatus.EXPLICIT
    assert value == Decimal("1234.56")


# ─────────────────────────────────────────────────────────────────────
# Unit detection (Trap D5 — unit switches mid-decade)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("(₹ in '000)", "INR_THOUSANDS"),
    ("Rupees in thousands", "INR_THOUSANDS"),
    ("₹ in crore", "INR_CRORE"),
    ("(₹ in crores)", "INR_CRORE"),
    ("C in crore", "INR_CRORE"),          # OCR artifact: ₹ misread as C
    ("no unit descriptor here", ""),
])
def test_detect_unit_code(n, text, expected):
    assert n.detect_unit_code(text) == expected


# ─────────────────────────────────────────────────────────────────────
# build_extracted_value contract
# ─────────────────────────────────────────────────────────────────────

def test_build_extracted_value_preserves_original_text(n):
    out = n.build_extracted_value("(2,500.00)")
    assert out.original_value == "(2,500.00)"
    assert out.value == Decimal("-2500.00")
    assert out.status is FieldStatus.EXPLICIT


def test_build_extracted_value_missing_has_none_value(n):
    out = n.build_extracted_value(None)
    assert out.status is FieldStatus.MISSING
    assert out.value is None


def test_build_extracted_value_carries_unit(n):
    """Each value must carry its unit so crore and '000 are never confused."""
    out = n.build_extracted_value("1,799,506.63", unit="INR_CRORE")
    assert out.unit == "INR_CRORE"


def test_build_text_value_is_not_numeric(n):
    out = n.build_text_value("HDFC Bank Limited")
    assert out.value == "HDFC Bank Limited"
    assert out.status is FieldStatus.EXPLICIT


def test_evidence_holds_raw_text_not_normalized(n):
    """A supplied snippet is kept verbatim-flagged-off; absence stays empty."""
    out = n.build_extracted_value("(2,500.00)", source_text="(2,500.00)")
    assert out.evidence.source_text == "(2,500.00)"
    assert out.evidence.verified is False


def test_evidence_absent_snippet_is_empty_not_echo(n):
    """Without a snippet there is no quotation — never echo the raw value."""
    out = n.build_extracted_value("(2,500.00)")
    assert out.evidence.source_text == ""
    assert out.evidence.verified is False
