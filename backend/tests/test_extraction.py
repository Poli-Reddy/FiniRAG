"""
Unit tests for ExtractionService's parse-and-normalize layer.

The LLM call itself is stubbed out: what is under test here is the
deterministic half — how a raw JSON blob from the model is turned into
typed ExtractedValue objects, and whether the scale (crore vs '000 vs
₹ per share) survives that journey.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from backend.app.schemas.extraction import FieldStatus
from backend.app.services.extraction_service import PROMPT_MAP, ExtractionService


@pytest.fixture()
def svc(monkeypatch) -> ExtractionService:
    """An ExtractionService with the Gemini client never constructed."""
    monkeypatch.setattr(
        "backend.app.services.extraction_service.GeminiProvider",
        lambda *args, **kwargs: object(),
    )
    return ExtractionService()


# ─────────────────────────────────────────────────────────────────────
# Prompt routing (Trap D34 — four separate schemas, no auto-classify)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("doc_type", [
    "invoice", "balance_sheet", "profit_and_loss", "cash_flow_statement",
])
def test_every_document_type_has_its_own_prompt(doc_type):
    assert doc_type in PROMPT_MAP
    assert PROMPT_MAP[doc_type].strip()


def test_prompts_are_distinct_per_document_type():
    """A shared prompt would mean the schemas were never really separate."""
    assert len(set(PROMPT_MAP.values())) == len(PROMPT_MAP)


def test_unknown_document_type_is_rejected(svc):
    with pytest.raises(ValueError):
        svc._parse_and_normalize({}, "tax_return", "")


# ─────────────────────────────────────────────────────────────────────
# Invoice parsing
# ─────────────────────────────────────────────────────────────────────

def test_invoice_parse_normalizes_amounts(svc):
    raw = {
        "invoice_number": "INV-001",
        "currency": "MYR",
        "subtotal": "23.50",
        "tax_amount": "1.41",
        "total_amount": "24.91",
        "line_items": [
            {"description": "Nasi Lemak", "quantity": "2",
             "unit_price": "11.75", "amount": "23.50"},
        ],
    }
    out = svc._parse_invoice(raw, "")
    assert out.invoice_number.value == "INV-001"
    assert out.subtotal.value == Decimal("23.50")
    assert out.line_items[0].amount.value == Decimal("23.50")
    assert out.line_items[0].quantity.value == Decimal("2")


def test_invoice_monetary_fields_carry_the_currency(svc):
    """An RM amount must not be readable as an INR one downstream."""
    raw = {"currency": "MYR", "total_amount": "24.91",
           "line_items": [{"unit_price": "11.75", "amount": "23.50"}]}
    out = svc._parse_invoice(raw, "")
    assert out.total_amount.unit == "MYR"
    assert out.line_items[0].amount.unit == "MYR"


def test_invoice_tax_rate_is_marked_as_a_percentage_not_money(svc):
    raw = {"currency": "MYR", "tax_rate": "6"}
    out = svc._parse_invoice(raw, "")
    assert out.tax_rate.unit == "PERCENT"


def test_invoice_null_fields_become_missing_not_zero(svc):
    out = svc._parse_invoice({"subtotal": None, "tax_amount": None}, "")
    assert out.subtotal.status is FieldStatus.MISSING
    assert out.subtotal.value is None


def test_invoice_dash_becomes_explicit_zero(svc):
    out = svc._parse_invoice({"discount": "-"}, "")
    assert out.discount.status is FieldStatus.EXPLICIT_ZERO
    assert out.discount.value == Decimal("0")


def test_invoice_preserves_unmapped_fields_with_original_labels(svc):
    """Trap P1/D17: a field we have no column for must not be dropped."""
    raw = {"additional_fields": [
        {"label": "Purchase Order No.", "value": "PO-99812"},
        {"label": "GSTIN", "value": "27AAACH1234K1Z5"},
    ]}
    out = svc._parse_invoice(raw, "")
    labels = [f.label for f in out.additional_fields]
    assert labels == ["Purchase Order No.", "GSTIN"]
    assert out.additional_fields[0].value.value == "PO-99812"


def test_invoice_preserves_supplementary_tables(svc):
    raw = {"additional_tables": [{
        "table_name": "GST Summary",
        "headers": ["Code", "Amount", "Tax"],
        "rows": [{"Code": "SR", "Amount": "23.50", "Tax": "1.41"}],
    }]}
    out = svc._parse_invoice(raw, "")
    assert out.additional_tables[0].table_name == "GST Summary"
    assert out.additional_tables[0].rows[0]["Code"] == "SR"


def test_invoice_tax_inclusive_flag_is_tri_state(svc):
    """None means "the document did not say" — not False."""
    assert svc._parse_invoice({}, "").tax_inclusive is None
    assert svc._parse_invoice({"tax_inclusive": True}, "").tax_inclusive is True
    assert svc._parse_invoice({"tax_inclusive": False}, "").tax_inclusive is False


def test_invoice_subtotal_pretax_defaults_to_unconfirmed(svc):
    """Correction #7: absence of evidence is not confirmation."""
    assert svc._parse_invoice({}, "").subtotal_confirmed_pretax is False


# ─────────────────────────────────────────────────────────────────────
# Balance sheet parsing (Trap D3, D4, D5)
# ─────────────────────────────────────────────────────────────────────

def _bs_raw() -> dict:
    return {
        "company_name": "HDFC Bank Limited",
        "unit_code": "INR_THOUSANDS",
        "periods": {
            "2021": {
                "year": "2021",
                "capital_and_liabilities": {"Capital": "5,512,776"},
                "total_capital_and_liabilities": "17,995,066,442",
                "assets": {"Investments": "4,388,231,117"},
                "total_assets": "17,995,066,442",
                "schedule_refs": {"Capital": "1"},
            },
            "2020": {
                "year": "2020",
                "capital_and_liabilities": {"Capital": "5,480,671"},
                "total_capital_and_liabilities": "15,807,732,376",
                "assets": {"Investments": "3,919,106,594"},
                "total_assets": "15,807,732,376",
            },
        },
    }


def test_balance_sheet_parses_every_period(svc):
    """Comparative columns are separate periods, not one merged blob."""
    out = svc._parse_balance_sheet(_bs_raw(), "")
    assert set(out.periods) == {"2021", "2020"}
    assert out.periods["2021"].total_assets.value == Decimal("17995066442")
    assert out.periods["2020"].total_assets.value == Decimal("15807732376")


def test_balance_sheet_keeps_original_row_labels(svc):
    """Row labels are the document's, not a canonical vocabulary of ours."""
    out = svc._parse_balance_sheet(_bs_raw(), "")
    assert "Investments" in out.periods["2021"].assets
    assert "Capital" in out.periods["2021"].capital_and_liabilities


def test_balance_sheet_stamps_unit_on_every_value(svc):
    """A figure without its scale is unusable — thousands or crore?"""
    out = svc._parse_balance_sheet(_bs_raw(), "")
    p = out.periods["2021"]
    assert p.total_assets.unit == "INR_THOUSANDS"
    assert p.assets["Investments"].unit == "INR_THOUSANDS"
    assert p.capital_and_liabilities["Capital"].unit == "INR_THOUSANDS"


def test_balance_sheet_falls_back_to_header_text_for_unit(svc):
    """When the model omits unit_code, read it off the document header."""
    raw = _bs_raw()
    raw["unit_code"] = ""
    out = svc._parse_balance_sheet(raw, "Consolidated Balance Sheet (₹ in crore)")
    assert out.unit_code == "INR_CRORE"
    assert out.periods["2021"].total_assets.unit == "INR_CRORE"


def test_balance_sheet_schedule_refs_survive(svc):
    out = svc._parse_balance_sheet(_bs_raw(), "")
    assert out.periods["2021"].schedule_refs == {"Capital": "1"}


# ─────────────────────────────────────────────────────────────────────
# P&L parsing (Trap D8 — EPS is not scaled)
# ─────────────────────────────────────────────────────────────────────

def test_profit_loss_eps_never_carries_the_document_scale(svc):
    """
    A statement in crore does not make EPS ₹79.25 crore. Stamping the
    document unit onto EPS would inflate it by seven orders of magnitude.
    """
    raw = {
        "unit_code": "INR_CRORE",
        "periods": {"2024": {
            "year": "2024",
            "total_income": "4,07,994.77",
            "basic_eps": "79.25",
            "diluted_eps": "78.90",
            "face_value_per_share": "1",
        }},
    }
    out = svc._parse_profit_loss(raw, "")
    period = out.periods["2024"]
    assert period.total_income.unit == "INR_CRORE"
    assert period.basic_eps.unit == "INR_PER_SHARE"
    assert period.diluted_eps.unit == "INR_PER_SHARE"
    assert period.basic_eps.value == Decimal("79.25")


def test_profit_loss_appropriations_keep_document_labels(svc):
    raw = {"periods": {"2024": {
        "appropriations": {
            "Transfer to Statutory Reserve": "1,000.00",
            "Proposed Dividend": "500.00",
        },
    }}}
    out = svc._parse_profit_loss(raw, "")
    approps = out.periods["2024"].appropriations
    assert set(approps) == {"Transfer to Statutory Reserve", "Proposed Dividend"}
    assert approps["Proposed Dividend"].value == Decimal("500.00")


def test_profit_loss_absent_minority_interest_is_missing(svc):
    """Not every group has a minority interest — that is MISSING, not 0."""
    out = svc._parse_profit_loss({"periods": {"2024": {}}}, "")
    assert out.periods["2024"].minority_interest.status is FieldStatus.MISSING


# ─────────────────────────────────────────────────────────────────────
# Cash flow parsing (Trap D26, D27, D28)
# ─────────────────────────────────────────────────────────────────────

def test_cash_flow_parenthesised_outflows_become_negative(svc):
    raw = {"periods": {"2024": {
        "financing_items": {"Dividends paid": "(10,849.22)"},
        "net_cash_from_financing": "(3,988.06)",
    }}}
    out = svc._parse_cash_flow(raw, "")
    period = out.periods["2024"]
    assert period.net_cash_from_financing.value == Decimal("-3988.06")
    assert period.financing_items["Dividends paid"].value == Decimal("-10849.22")


def test_cash_flow_amalgamation_position_is_read_not_inferred(svc):
    """Correction #4: the position comes from the document's layout."""
    raw = {"periods": {"2024": {
        "cash_acquired_on_amalgamation": "11,781.71",
        "amalgamation_position": "after_net_increase",
    }}}
    out = svc._parse_cash_flow(raw, "")
    assert out.periods["2024"].amalgamation_position == "after_net_increase"


def test_cash_flow_absent_amalgamation_position_stays_none(svc):
    """No structural evidence means no assumption."""
    out = svc._parse_cash_flow({"periods": {"2024": {}}}, "")
    assert out.periods["2024"].amalgamation_position is None


def test_cash_flow_section_items_keep_document_labels(svc):
    raw = {"periods": {"2024": {
        "operating_items": {"Net Profit before tax": "800", "Depreciation": "200"},
    }}}
    out = svc._parse_cash_flow(raw, "")
    assert set(out.periods["2024"].operating_items) == {
        "Net Profit before tax", "Depreciation",
    }


# ─────────────────────────────────────────────────────────────────────
# Serialization contract — this dict is what gets persisted as JSONB
# ─────────────────────────────────────────────────────────────────────

def test_parsed_output_is_json_serializable(svc):
    """model_dump(mode="json") must not leave Decimals in the payload."""
    import json

    out = svc._parse_and_normalize(_bs_raw(), "balance_sheet", "")
    json.dumps(out)  # raises TypeError if a Decimal survived


def test_serialized_decimals_are_exact_strings_not_floats(svc):
    """0.1 + 0.2 must not become 0.30000000000000004 in the database."""
    out = svc._parse_and_normalize(
        {"periods": {"2021": {"total_assets": "17,995,066,442.55"}}},
        "balance_sheet", "",
    )
    serialized = out["periods"]["2021"]["total_assets"]["value"]
    assert serialized == "17995066442.55"
    assert Decimal(serialized) == Decimal("17995066442.55")


def test_evidence_quotes_the_document_not_our_normalized_number(svc):
    """Trap P9: evidence must be auditable against the source."""
    out = svc._parse_invoice({"total_amount": "(2,500.00)"}, "")
    assert out.total_amount.original_value == "(2,500.00)"
    assert out.total_amount.evidence.source_text == "(2,500.00)"
    assert out.total_amount.value == Decimal("-2500.00")


@pytest.mark.asyncio
async def test_truncation_falls_back_to_split_by_period(svc):
    """A document too big for one output budget is extracted year by year."""
    from backend.app.providers.errors import LLMOutputTruncatedError
    from backend.app.providers.llm_provider import LLMProvider

    class SplitScript(LLMProvider):
        def __init__(self):
            self.calls = 0

        async def extract_structured(self, text, document_type, system_prompt):
            self.calls += 1
            if self.calls == 1:
                raise LLMOutputTruncatedError(999)
            if system_prompt.startswith("List every comparative year/period"):
                return {"periods": ["2024", "2023"]}
            year = "2024" if "Extract ONLY the 2024" in system_prompt else "2023"
            return {"periods": {year: {"total_assets": "100"}}}

        async def extract_from_images(self, images, document_type, system_prompt):
            raise AssertionError("text path under test")

    svc._llm = SplitScript()
    out = await svc.extract(text="t", document_type="balance_sheet")
    assert set(out["periods"]) == {"2024", "2023"}
    assert svc._llm.calls == 4  # full + years + 2 splits


@pytest.mark.asyncio
async def test_split_gives_up_honestly_when_periods_unclear(svc):
    """If even the year list fails, the original truncation stands."""
    from backend.app.providers.errors import LLMOutputTruncatedError
    from backend.app.providers.llm_provider import LLMProvider

    class AlwaysTruncated(LLMProvider):
        async def extract_structured(self, text, document_type, system_prompt):
            raise LLMOutputTruncatedError(999)

        async def extract_from_images(self, images, document_type, system_prompt):
            raise LLMOutputTruncatedError(999)

    svc._llm = AlwaysTruncated()
    with pytest.raises(LLMOutputTruncatedError):
        await svc.extract(text="t", document_type="balance_sheet")
