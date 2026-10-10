"""
FiniRAG Extraction Schemas — Canonical Internal Document Model.

This module defines the atomic data structures that flow through:
  normalization → evidence → validation → persistence

Key invariants (Gate 2.1 Corrections #3, #7):
  - EXPLICIT_ZERO always has value = Decimal("0") (NEVER None)
  - MISSING always has value = None
  - UNREADABLE always has value = None
  - Monetary values use Decimal, not float (Correction #8)

All 4 document types have separate extraction schemas (Trap D34).
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_serializer


# ─────────────────────────────────────────────────────────────────────
# Core Atomic Types
# ─────────────────────────────────────────────────────────────────────

class FieldStatus(str, Enum):
    """Status of an extracted field. Missing ≠ zero ≠ unreadable (Trap P7)."""
    EXPLICIT = "EXPLICIT"             # Value exists and was read (non-zero)
    EXPLICIT_ZERO = "EXPLICIT_ZERO"   # Document shows 0, 0.00, or dash-nil
    MISSING = "MISSING"               # Field not present in document
    UNREADABLE = "UNREADABLE"         # Field present but OCR cannot read it


class Evidence(BaseModel):
    """Source evidence for an extracted value (Trap P9, D23)."""
    source_text: str = ""             # Exact OCR text, NOT corrected/normalized
    page_number: int = 1              # 1-indexed
    bounding_box: list[float] | None = None  # [x1, y1, x2, y2] — never fabricated
    # True ONLY when the snippet was checked against retained source text
    # (chunking verification). Model-transcribed snippets — including the
    # LLM evidence_map and any echo of the raw value — are NEVER verified.
    verified: bool = False


class ExtractedValue(BaseModel):
    """
    The atomic unit of every extracted field.

    Invariants:
      - EXPLICIT:      value is not None (numeric, string, etc.)
      - EXPLICIT_ZERO: value == Decimal("0") or 0.0 (NEVER None)
      - MISSING:       value is None
      - UNREADABLE:    value is None
    """
    value: Any = None                       # Normalized value (Decimal, str, None)
    original_value: str = ""                # Exact text from document
    status: FieldStatus = FieldStatus.MISSING
    evidence: Evidence = Field(default_factory=Evidence)
    unit: str | None = None                 # "INR_THOUSANDS", "INR_CRORE", etc.

    @field_serializer("value", when_used="json")
    def _serialize_value(self, v: Any) -> Any:
        """Money is serialized as an exact string, never a binary float."""
        if isinstance(v, Decimal):
            return str(v)
        return v


class AdditionalField(BaseModel):
    """
    A field extracted from the document that doesn't map to a core schema field.
    Preserves original labels (Trap P1, D17, D20).
    """
    label: str                              # Original label: "Purchase Order No."
    value: ExtractedValue


class AdditionalTable(BaseModel):
    """Supplementary table (GST summary, VAT summary, etc.)."""
    table_name: str                         # "GST Summary"
    headers: list[str] = []
    rows: list[dict] = []
    page_number: int = 1


# ─────────────────────────────────────────────────────────────────────
# Helpers — used by validation services
# ─────────────────────────────────────────────────────────────────────

def is_usable(ev: ExtractedValue) -> bool:
    """Returns True if the value can participate in arithmetic validation."""
    return ev.status in (FieldStatus.EXPLICIT, FieldStatus.EXPLICIT_ZERO)


def safe_value(ev: ExtractedValue) -> Decimal:
    """
    Returns the numeric value as Decimal.
    Only call when is_usable() is True.

    Raises ValueError rather than asserting: a bare assert disappears
    under `python -O`, which would silently turn a MISSING field into
    zero — precisely the confusion the status model exists to prevent.
    """
    if not is_usable(ev):
        raise ValueError(f"Cannot use value with status {ev.status}")
    if ev.value is None:
        # EXPLICIT_ZERO with a null value is possible (dash-nil normalizes to
        # Decimal 0 upstream, but a hand-built or drifted record may carry
        # None). Only that status may read as zero — an EXPLICIT null is
        # corrupt data, and treating it as zero is exactly the confusion the
        # status model exists to prevent.
        if ev.status is FieldStatus.EXPLICIT_ZERO:
            return Decimal("0")
        raise ValueError(f"EXPLICIT value is None (status {ev.status})")
    if isinstance(ev.value, Decimal):
        return ev.value
    return Decimal(str(ev.value))


# ─────────────────────────────────────────────────────────────────────
# Invoice Extraction Schema (Trap D9, D10, D11, D17, D18, D20)
# ─────────────────────────────────────────────────────────────────────

class InvoiceLineItem(BaseModel):
    """A single line item from an invoice."""
    description: ExtractedValue = Field(default_factory=ExtractedValue)
    quantity: ExtractedValue = Field(default_factory=ExtractedValue)
    unit_of_measure: ExtractedValue = Field(default_factory=ExtractedValue)
    unit_price: ExtractedValue = Field(default_factory=ExtractedValue)
    amount: ExtractedValue = Field(default_factory=ExtractedValue)
    tax_code: ExtractedValue = Field(default_factory=ExtractedValue)
    tax_rate: ExtractedValue = Field(default_factory=ExtractedValue)
    discount_amount: ExtractedValue = Field(default_factory=ExtractedValue)
    discount_rate: ExtractedValue = Field(default_factory=ExtractedValue)


class InvoiceExtraction(BaseModel):
    """Complete extraction result for an invoice document."""

    # Core fields
    invoice_number: ExtractedValue = Field(default_factory=ExtractedValue)
    invoice_date: ExtractedValue = Field(default_factory=ExtractedValue)
    due_date: ExtractedValue = Field(default_factory=ExtractedValue)
    vendor_name: ExtractedValue = Field(default_factory=ExtractedValue)
    vendor_address: ExtractedValue = Field(default_factory=ExtractedValue)
    vendor_tax_id: ExtractedValue = Field(default_factory=ExtractedValue)
    customer_name: ExtractedValue = Field(default_factory=ExtractedValue)
    customer_address: ExtractedValue = Field(default_factory=ExtractedValue)
    customer_tax_id: ExtractedValue = Field(default_factory=ExtractedValue)
    currency: ExtractedValue = Field(default_factory=ExtractedValue)
    subtotal: ExtractedValue = Field(default_factory=ExtractedValue)
    tax_amount: ExtractedValue = Field(default_factory=ExtractedValue)
    tax_rate: ExtractedValue = Field(default_factory=ExtractedValue)
    discount: ExtractedValue = Field(default_factory=ExtractedValue)
    total_amount: ExtractedValue = Field(default_factory=ExtractedValue)
    amount_paid: ExtractedValue = Field(default_factory=ExtractedValue)
    amount_due: ExtractedValue = Field(default_factory=ExtractedValue)
    cash_paid: ExtractedValue = Field(default_factory=ExtractedValue)
    change_given: ExtractedValue = Field(default_factory=ExtractedValue)
    rounding_adjustment: ExtractedValue = Field(default_factory=ExtractedValue)
    payment_method: ExtractedValue = Field(default_factory=ExtractedValue)

    # Line items
    line_items: list[InvoiceLineItem] = []

    # Additional tables (GST summary, VAT summary)
    additional_tables: list[AdditionalTable] = []

    # Dynamic additional fields preserving original labels
    additional_fields: list[AdditionalField] = []

    # Tax-inclusive flag detected from document text
    tax_inclusive: bool | None = None

    # Whether subtotal is confirmed as pre-tax base (Correction #7)
    subtotal_confirmed_pretax: bool = False


# ─────────────────────────────────────────────────────────────────────
# Balance Sheet Extraction Schema (Trap D3, D4, D5, D24, D25)
# ─────────────────────────────────────────────────────────────────────

class BalanceSheetPeriod(BaseModel):
    """Data for a single reporting period in a Balance Sheet."""
    year: str = ""

    # Dynamic dicts — keys are ACTUAL labels from the document
    capital_and_liabilities: dict[str, ExtractedValue] = {}
    total_capital_and_liabilities: ExtractedValue = Field(default_factory=ExtractedValue)

    assets: dict[str, ExtractedValue] = {}
    total_assets: ExtractedValue = Field(default_factory=ExtractedValue)

    # Off-balance items
    contingent_liabilities: ExtractedValue = Field(default_factory=ExtractedValue)
    bills_for_collection: ExtractedValue = Field(default_factory=ExtractedValue)

    # Schedule references
    schedule_refs: dict[str, str] = {}

    additional_fields: list[AdditionalField] = []


class BalanceSheetExtraction(BaseModel):
    """Complete extraction result for a Balance Sheet."""
    company_name: ExtractedValue = Field(default_factory=ExtractedValue)
    statement_type: ExtractedValue = Field(default_factory=ExtractedValue)
    unit_descriptor: ExtractedValue = Field(default_factory=ExtractedValue)
    unit_code: str = ""  # "INR_THOUSANDS" or "INR_CRORE"

    # Keyed by year string: {"2024": ..., "2023": ...}
    periods: dict[str, BalanceSheetPeriod] = {}


# ─────────────────────────────────────────────────────────────────────
# P&L Extraction Schema (Trap D5, D8, D29)
# ─────────────────────────────────────────────────────────────────────

class ProfitLossPeriod(BaseModel):
    """Data for a single reporting period in a P&L Statement."""
    year: str = ""

    # Income
    interest_earned: ExtractedValue = Field(default_factory=ExtractedValue)
    other_income: ExtractedValue = Field(default_factory=ExtractedValue)
    total_income: ExtractedValue = Field(default_factory=ExtractedValue)

    # Expenditure
    interest_expended: ExtractedValue = Field(default_factory=ExtractedValue)
    operating_expenses: ExtractedValue = Field(default_factory=ExtractedValue)
    provisions_and_contingencies: ExtractedValue = Field(default_factory=ExtractedValue)
    total_expenditure: ExtractedValue = Field(default_factory=ExtractedValue)

    # Profit chain
    net_profit_before_minority: ExtractedValue = Field(default_factory=ExtractedValue)
    minority_interest: ExtractedValue = Field(default_factory=ExtractedValue)
    share_in_profits_of_associates: ExtractedValue = Field(default_factory=ExtractedValue)
    consolidated_net_profit: ExtractedValue = Field(default_factory=ExtractedValue)

    # Appropriations
    addition_on_amalgamation: ExtractedValue = Field(default_factory=ExtractedValue)
    brought_forward_profit: ExtractedValue = Field(default_factory=ExtractedValue)
    total_available_for_appropriation: ExtractedValue = Field(default_factory=ExtractedValue)
    appropriations: dict[str, ExtractedValue] = {}
    balance_carried_forward: ExtractedValue = Field(default_factory=ExtractedValue)

    # Per share (ALWAYS in ₹, NOT document unit) — Trap D8
    basic_eps: ExtractedValue = Field(default_factory=ExtractedValue)
    diluted_eps: ExtractedValue = Field(default_factory=ExtractedValue)
    face_value_per_share: ExtractedValue = Field(default_factory=ExtractedValue)

    additional_fields: list[AdditionalField] = []
    schedule_refs: dict[str, str] = {}


class ProfitLossExtraction(BaseModel):
    """Complete extraction result for a P&L Statement."""
    company_name: ExtractedValue = Field(default_factory=ExtractedValue)
    statement_type: ExtractedValue = Field(default_factory=ExtractedValue)
    unit_descriptor: ExtractedValue = Field(default_factory=ExtractedValue)
    unit_code: str = ""

    periods: dict[str, ProfitLossPeriod] = {}


# ─────────────────────────────────────────────────────────────────────
# Cash Flow Extraction Schema (Trap D6, D26, D27, D28)
# ─────────────────────────────────────────────────────────────────────

class CashFlowPeriod(BaseModel):
    """Data for a single reporting period in a Cash Flow Statement."""
    year: str = ""

    # Dynamic line items within each activity section
    operating_items: dict[str, ExtractedValue] = {}
    net_cash_from_operating: ExtractedValue = Field(default_factory=ExtractedValue)

    investing_items: dict[str, ExtractedValue] = {}
    net_cash_from_investing: ExtractedValue = Field(default_factory=ExtractedValue)

    financing_items: dict[str, ExtractedValue] = {}
    net_cash_from_financing: ExtractedValue = Field(default_factory=ExtractedValue)

    # Reconciliation
    fx_translation_adjustment: ExtractedValue = Field(default_factory=ExtractedValue)
    net_increase_in_cash: ExtractedValue = Field(default_factory=ExtractedValue)
    cash_acquired_on_amalgamation: ExtractedValue = Field(default_factory=ExtractedValue)

    # Structural position detection (Correction #4) — NOT year-hardcoded
    # "before_net_increase" | "after_net_increase" | None (absent)
    amalgamation_position: str | None = None

    opening_cash: ExtractedValue = Field(default_factory=ExtractedValue)
    closing_cash: ExtractedValue = Field(default_factory=ExtractedValue)

    additional_fields: list[AdditionalField] = []


class CashFlowExtraction(BaseModel):
    """Complete extraction result for a Cash Flow Statement."""
    company_name: ExtractedValue = Field(default_factory=ExtractedValue)
    statement_type: ExtractedValue = Field(default_factory=ExtractedValue)
    unit_descriptor: ExtractedValue = Field(default_factory=ExtractedValue)
    unit_code: str = ""

    periods: dict[str, CashFlowPeriod] = {}
