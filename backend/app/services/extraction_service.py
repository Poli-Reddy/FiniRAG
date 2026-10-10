"""
FiniRAG Extraction Service.

Coordinates the LLM extraction step:
  1. Build document-type-specific prompts
  2. Send aggregated text to LLM
  3. Parse LLM response into typed extraction schemas
  4. Normalize all values via NormalizationService

LLM extracts → deterministic Python validates (Gate 2.1 principle).
"""

from __future__ import annotations

import logging
from typing import Any

from backend.app.core.config import settings
from backend.app.providers.errors import LLMOutputTruncatedError
from backend.app.providers.fallback_provider import FallbackLLMProvider
from backend.app.providers.llm_provider import LLMProvider
from backend.app.providers.gemini_provider import GeminiProvider
from backend.app.schemas.extraction import (
    ExtractedValue, Evidence, FieldStatus,
    AdditionalField, AdditionalTable,
    InvoiceExtraction, InvoiceLineItem,
    BalanceSheetExtraction, BalanceSheetPeriod,
    ProfitLossExtraction, ProfitLossPeriod,
    CashFlowExtraction, CashFlowPeriod,
)
from backend.app.services.normalization_service import NormalizationService

logger = logging.getLogger("finirag")


# ─────────────────────────────────────────────────────────────────────
# Extraction Prompts
# ─────────────────────────────────────────────────────────────────────

INVOICE_PROMPT = """You are a document data extraction system. Extract all information from this invoice/receipt.

Return a JSON object with these fields (use null for missing fields):
{
  "invoice_number": "value or null",
  "invoice_date": "value or null",
  "due_date": "value or null",
  "vendor_name": "value or null",
  "vendor_address": "value or null",
  "vendor_tax_id": "value or null",
  "customer_name": "value or null",
  "customer_address": "value or null",
  "customer_tax_id": "value or null",
  "currency": "value or null",
  "subtotal": "value or null",
  "tax_amount": "value or null",
  "tax_rate": "value or null (percentage number, e.g. 6 for 6%)",
  "discount": "value or null",
  "total_amount": "value or null",
  "amount_paid": "value or null",
  "amount_due": "value or null",
  "cash_paid": "value or null",
  "change_given": "value or null",
  "rounding_adjustment": "value or null",
  "payment_method": "value or null",
  "tax_inclusive": true/false/null,
  "subtotal_confirmed_pretax": true/false,
  "line_items": [
    {
      "description": "text",
      "quantity": "value or null",
      "unit_of_measure": "value or null",
      "unit_price": "value or null",
      "amount": "value or null",
      "tax_code": "value or null",
      "tax_rate": "value or null",
      "discount_amount": "value or null",
      "discount_rate": "value or null"
    }
  ],
  "additional_tables": [
    {
      "table_name": "e.g. GST Summary",
      "headers": ["col1", "col2"],
      "rows": [{"col1": "val1", "col2": "val2"}]
    }
  ],
  "additional_fields": [
    {"label": "Original Label", "value": "value"}
  ]
}

IMPORTANT RULES:
- Extract every visible field, even if it doesn't map to the core schema.
- Use additional_fields for any field not in the core schema (PO numbers, GSTIN, HSN codes, bank details, table numbers, cashier names, etc.).
- For tax_inclusive: set true if the document shows "inclusive" or "incl." keywords, false if "exclusive" or "excl.", null if ambiguous.
- For subtotal_confirmed_pretax: set true ONLY if the document explicitly labels the subtotal as a pre-tax amount (e.g., "Subtotal (Excl. GST)", "Taxable Amount"). Set false if ambiguous.
- Keep original values exactly as shown in the document (don't normalize numbers).
- Transcribe, never correct: if OCR/text is garbled, return it as-is or null — never guess what it "should" be.
- Use null for missing fields, NOT empty strings or zeros.
- For standalone dashes (-, −) that represent nil/zero, return "-".
"""

BALANCE_SHEET_PROMPT = """You are a document data extraction system. Extract all information from this Balance Sheet.

Return a JSON object with:
{
  "company_name": "value or null",
  "statement_type": "Balance Sheet",
  "unit_descriptor": "e.g. '₹ in \\'000' or '₹ in crore'",
  "unit_code": "INR_THOUSANDS or INR_CRORE",
  "periods": {
    "2024": {
      "year": "2024",
      "capital_and_liabilities": {
        "Share Capital": "value",
        "Reserves and Surplus": "value",
        ...every row label and value
      },
      "total_capital_and_liabilities": "value",
      "assets": {
        "Cash and Balances with RBI": "value",
        "Fixed Assets": "value",
        "Goodwill on Consolidation": "value",
        ...every row label and value
      },
      "total_assets": "value",
      "contingent_liabilities": "value or null",
      "bills_for_collection": "value or null",
      "schedule_refs": {"Share Capital": "1", ...}
    },
    "2023": { ...same structure }
  }
}

IMPORTANT RULES:
- Extract ALL periods shown (typically current year and previous year).
- Use the ACTUAL years printed in the document as periods keys — the 2024/2023 above are examples only.
- Use exact row labels from the document as dictionary keys.
- Include Goodwill on Consolidation if present (even without schedule number).
- Keep original values as shown in the document.
- Standalone dashes (-) represent nil/zero — return them as "-".
- The unit descriptor describes the unit convention used (₹ in '000 vs ₹ in crore).
"""

PROFIT_LOSS_PROMPT = """You are a document data extraction system. Extract all information from this Profit & Loss statement.

Return a JSON object with:
{
  "company_name": "value or null",
  "statement_type": "Profit and Loss Account",
  "unit_descriptor": "e.g. '₹ in \\'000' or '₹ in crore'",
  "unit_code": "INR_THOUSANDS or INR_CRORE",
  "periods": {
    "2024": {
      "year": "2024",
      "interest_earned": "value",
      "other_income": "value",
      "total_income": "value",
      "interest_expended": "value",
      "operating_expenses": "value",
      "provisions_and_contingencies": "value",
      "total_expenditure": "value",
      "net_profit_before_minority": "value or null",
      "minority_interest": "value or null",
      "share_in_profits_of_associates": "value or null",
      "consolidated_net_profit": "value",
      "addition_on_amalgamation": "value or null",
      "brought_forward_profit": "value",
      "total_available_for_appropriation": "value",
      "appropriations": {
        "Transfer to Statutory Reserve": "value",
        ...every row label and value
      },
      "balance_carried_forward": "value",
      "basic_eps": "value (always in ₹ per share, NOT in document unit)",
      "diluted_eps": "value (always in ₹ per share, NOT in document unit)",
      "face_value_per_share": "value or null",
      "schedule_refs": {"Interest Earned": "13", ...}
    },
    "2023": { ...same structure }
  }
}

IMPORTANT RULES:
- EPS values are ALWAYS in ₹ per share, regardless of the document's unit convention.
- Extract ALL periods shown. Use the ACTUAL years printed in the document — 2024/2023 above are examples only.
- Use exact row labels from the document.
- Standalone dashes represent nil/zero — return them as "-".
"""

CASH_FLOW_PROMPT = """You are a document data extraction system. Extract all information from this Cash Flow Statement.

Return a JSON object with:
{
  "company_name": "value or null",
  "statement_type": "Cash Flow Statement",
  "unit_descriptor": "e.g. '₹ in \\'000' or '₹ in crore'",
  "unit_code": "INR_THOUSANDS or INR_CRORE",
  "periods": {
    "2024": {
      "year": "2024",
      "operating_items": {
        "Net Profit before tax": "value",
        ...every operating activity line
      },
      "net_cash_from_operating": "value",
      "investing_items": {
        "Purchase of Fixed Assets": "value",
        ...every investing activity line
      },
      "net_cash_from_investing": "value",
      "financing_items": {
        "Dividends Paid": "value",
        ...every financing activity line
      },
      "net_cash_from_financing": "value",
      "fx_translation_adjustment": "value or null",
      "net_increase_in_cash": "value",
      "cash_acquired_on_amalgamation": "value or null",
      "amalgamation_position": "before_net_increase or after_net_increase or null",
      "opening_cash": "value",
      "closing_cash": "value"
    },
    "2023": { ...same structure }
  }
}

IMPORTANT RULES:
- amalgamation_position: Report the structural position of "Cash Acquired on Amalgamation":
  - "before_net_increase": if the row appears ABOVE the "Net Increase in Cash" line
  - "after_net_increase": if the row appears BELOW the "Net Increase in Cash" line
  - null: if the row does not exist in this period
- This is critical for correct cash flow reconciliation.
- ROW ALIGNMENT (critical): these statements contain UNLABELLED SUBTOTAL ROWS —
  a figure printed on its own line with no description, above "Direct taxes paid"
  or before a section's net line. Do NOT emit an unlabelled subtotal as a line
  item, and do NOT let it shift the alignment: every value you return must be
  the figure printed on the SAME ROW as its label. Read each row as a unit
  (label + that row's own number), never by position in a list of numbers.
- net_cash_from_operating / net_cash_from_investing / net_cash_from_financing
  must come from the row explicitly labelled "Net cash from/used in ... activities"
  — NOT from an unlabelled subtotal that appears just above it.
- Only include rows that carry their own descriptive label in operating_items,
  investing_items and financing_items. Section subtotals and section net lines
  belong in the net_cash_from_* fields, not in the item dictionaries.
- Parenthesized amounts like (10,849.22) represent negative values — keep the parentheses.
- Extract ALL periods shown (this may be multi-page — extract everything across all pages). Use the ACTUAL years printed in the document — 2024/2023 above are examples only.
- Standalone dashes represent nil/zero — return them as "-".
"""

# ─────────────────────────────────────────────────────────────────────
# Evidence grounding (Traps P9, D22, D23)
# ─────────────────────────────────────────────────────────────────────
# Confidence is OPTIONAL per spec and therefore omitted. Evidence is NOT:
# every monetary/total/key value should carry the exact printed snippet
# plus the page it appeared on, so a reviewer can trace numbers back to
# the source. The parser treats evidence_map as advisory — an omitted or
# malformed entry falls back to echoing the transcribed value, never to a
# failure — but the model must never invent snippets for values it did
# not actually see (Trap D22: no silent correction, no guessing).

EVIDENCE_INSTRUCTIONS = """
EVIDENCE (required for monetary/total/key values):
- Alongside the fields above, return an "evidence_map" object:
  {
    "evidence_map": {
      "<field key or exact row label>": {"source_text": "<exact text as printed>", "page": 1},
      "line::<line-item description>": {"source_text": "<exact row text as printed>", "page": 1}
    }
  }
- "source_text" must be transcribed EXACTLY as printed (including
  parentheses, commas, currency symbols). Never clean it up, never
  correct OCR artefacts, never write a snippet for a value you did not
  see — omit the entry instead.
- "page" is the 1-based page number where the value appears. For
  multi-page documents report the actual page per value.
- If a value is genuinely missing from the document, use null for the
  value and omit its evidence_map entry. Missing is not zero.
"""

INVOICE_PROMPT = INVOICE_PROMPT + EVIDENCE_INSTRUCTIONS
BALANCE_SHEET_PROMPT = BALANCE_SHEET_PROMPT + EVIDENCE_INSTRUCTIONS
PROFIT_LOSS_PROMPT = PROFIT_LOSS_PROMPT + EVIDENCE_INSTRUCTIONS
CASH_FLOW_PROMPT = CASH_FLOW_PROMPT + EVIDENCE_INSTRUCTIONS

PROMPT_MAP = {
    "invoice": INVOICE_PROMPT,
    "balance_sheet": BALANCE_SHEET_PROMPT,
    "profit_and_loss": PROFIT_LOSS_PROMPT,
    "cash_flow_statement": CASH_FLOW_PROMPT,
}


PERIOD_LIST_PROMPT = """List every comparative year/period shown in this financial document.
Return ONLY this JSON object and nothing else:
{"periods": ["2024", "2023"]}
Use the exact year/period labels printed in the document. At most 4 entries."""

PERIOD_SPLIT_PROMPT_TEMPLATE = """Extract ONLY the {year} period/column from this financial document. Ignore all other periods.
Return the FULL schema from the system instructions with "periods" containing ONLY "{year}".
Transcribe values exactly as printed; missing values are null; include evidence_map entries for the {year} values you extract."""

PERIOD_SPLIT_COMPACT_SUFFIX = """
Output budget is tight: omit evidence_map entirely and keep every value on one line. Exact numbers matter more than snippets here."""


def _evidence_lookup(raw: dict) -> dict[str, tuple[str, int]]:
    """
    Normalize the LLM's evidence_map into {lowercased key: (source_text, page)}.

    Defensive by design: the map is advisory, so anything malformed is
    skipped rather than raised. Absent entries leave the evidence snippet
    empty — the parser never echoes the transcribed value as a quotation,
    and only the chunking verification pass against retained source text
    may promote a snippet to verbatim.
    """
    lookup: dict[str, tuple[str, int]] = {}
    ev = raw.get("evidence_map")
    if not isinstance(ev, dict):
        return lookup
    for key, entry in ev.items():
        if not isinstance(entry, dict):
            continue
        snippet = entry.get("source_text")
        if not isinstance(snippet, str) or not snippet.strip():
            continue
        try:
            page = int(entry.get("page", 1))
        except (TypeError, ValueError):
            page = 1
        lookup[str(key).strip().lower()] = (snippet, max(1, page))
    return lookup


# ─────────────────────────────────────────────────────────────────────
# Provider assembly
# ─────────────────────────────────────────────────────────────────────

def build_llm_provider() -> LLMProvider:
    """
    Gemini, with Groq behind it when a key is configured.

    Returns a bare GeminiProvider when no Groq key is set, so a deployment
    without one behaves exactly as it did before the fallback existed rather
    than paying for a chain of length one.

    Constructing the fallback is allowed to fail without taking the primary
    down. A missing `groq` package or a malformed key should degrade the
    service to single-provider, not stop it from starting — the fallback's
    entire job is to make failures less costly, and it would be a poor trade
    if its own absence became one.
    """
    from backend.app.core.config import GEMINI_KEYS, GROQ_KEYS

    # Shared pools so a multi-document run spreads across keys: each call
    # borrows the lowest live key and a refused key retires without retry.
    # Empty pools mean "not configured" — the provider then runs single-key.
    gemini_pool = GEMINI_KEYS if len(GEMINI_KEYS) else None
    primary = GeminiProvider(pool=gemini_pool)

    has_groq_key = bool(settings.GROQ_API_KEY or settings.GROQ_API_KEYS or len(GROQ_KEYS))
    if not has_groq_key:
        return primary

    try:
        from backend.app.providers.groq_provider import GroqProvider

        groq_pool = GROQ_KEYS if len(GROQ_KEYS) else None
        chain = FallbackLLMProvider([("gemini", primary), ("groq", GroqProvider(pool=groq_pool))])
        logger.info("LLM fallback chain enabled", extra={"providers": chain.names})
        return chain
    except Exception as e:
        logger.error("Fallback provider unavailable, continuing without it", extra={
            "error_type": type(e).__name__,
        })
        return primary


class ExtractionService:
    """Coordinates LLM extraction and result parsing."""

    def __init__(self):
        self._llm: LLMProvider = build_llm_provider()
        self._normalizer = NormalizationService()

    @property
    def last_llm_provider(self) -> str:
        """
        Which backend produced the most recent extraction.

        Read by the pipeline for the response metadata. A fallback that went
        unrecorded would attribute a document's numbers to a model that never
        saw it. Safe to keep on the instance because a DocumentService — and
        so an ExtractionService — is built per request.
        """
        return getattr(self._llm, "last_provider", None) or settings.LLM_PROVIDER

    async def extract(
        self,
        text: str,
        document_type: str,
    ) -> dict:
        """
        Extract structured data from document text.

        1. Select prompt for document type
        2. Send to LLM
        3. Parse response into typed schema
        4. Normalize all values

        Returns the extraction as a serializable dict.
        """
        prompt = PROMPT_MAP.get(document_type)
        if not prompt:
            raise ValueError(f"Unknown document type: {document_type}")

        # Call LLM
        try:
            raw_result = await self._llm.extract_structured(
                text=text,
                document_type=document_type,
                system_prompt=prompt,
            )
        except LLMOutputTruncatedError as first_error:
            # The whole document does not fit the backend's output budget.
            # Fall back to one extraction per comparative period and merge.
            try:
                raw_result = await self._extract_split_by_period(
                    document_type=document_type,
                    prompt=prompt,
                    text=text,
                )
            except Exception:
                raise first_error

        return self._parse_and_normalize(raw_result, document_type, text)

    async def extract_from_images(
        self,
        images: list,
        document_type: str,
    ) -> dict:
        """
        Extract structured data by sending page images to the LLM.

        Used when OCR returns empty text (e.g. the Vision API disabled or its
        credentials invalid, or vector-outlined PDFs). On this dataset that is
        every document, so this is the primary path in practice rather than a
        rare fallback.
        """
        prompt = PROMPT_MAP.get(document_type)
        if not prompt:
            raise ValueError(f"Unknown document type: {document_type}")

        # Vision-capable backend; may be the fallback if the primary refuses
        try:
            raw_result = await self._llm.extract_from_images(
                images=images,
                document_type=document_type,
                system_prompt=prompt,
            )
        except LLMOutputTruncatedError as first_error:
            try:
                raw_result = await self._extract_split_by_period(
                    document_type=document_type,
                    prompt=prompt,
                    images=images,
                )
            except Exception:
                raise first_error

        return self._parse_and_normalize(raw_result, document_type, "")

    async def _extract_split_by_period(
        self,
        document_type: str,
        prompt: str,
        text: str = "",
        images: list | None = None,
    ) -> dict:
        """
        One LLM extraction per comparative period, merged afterwards.

        Only meaningful for statement types (invoices are single-period and
        re-raise via the caller). Year labels come from the document itself
        via a cheap list call — never assumed. A shared row snippet carries
        all periods' figures, so evidence_map entries merged first-wins
        stay valid for every year.
        """
        if document_type == "invoice":
            raise ValueError("single-period document cannot be split")

        if images is not None:
            years_raw = await self._llm.extract_from_images(
                images=images,
                document_type=document_type,
                system_prompt=PERIOD_LIST_PROMPT,
            )
        else:
            years_raw = await self._llm.extract_structured(
                text=text,
                document_type=document_type,
                system_prompt=PERIOD_LIST_PROMPT,
            )

        years = years_raw.get("periods")
        if (
            not isinstance(years, list)
            or not 1 <= len(years) <= 4
            or not all(isinstance(y, str) and y.strip() for y in years)
        ):
            raise ValueError("could not list document periods for split extraction")

        from backend.app.providers.errors import LLMOutputTruncatedError as _Truncated

        merged: dict = {}
        merged_periods: dict = {}
        merged_evidence: dict = {}
        for year in years:
            split_prompt = (
                f"{prompt}\n\n"
                + PERIOD_SPLIT_PROMPT_TEMPLATE.format(year=year.strip())
            )
            try:
                if images is not None:
                    part = await self._llm.extract_from_images(
                        images=images,
                        document_type=document_type,
                        system_prompt=split_prompt,
                    )
                else:
                    part = await self._llm.extract_structured(
                        text=text,
                        document_type=document_type,
                        system_prompt=split_prompt,
                    )
            except _Truncated:
                # One year still exceeds the output budget (evidence_map is
                # the bulk). Retry compact: exact numbers, no snippets —
                # the parser's echo fallback keeps evidence present.
                split_prompt += PERIOD_SPLIT_COMPACT_SUFFIX
                if images is not None:
                    part = await self._llm.extract_from_images(
                        images=images,
                        document_type=document_type,
                        system_prompt=split_prompt,
                    )
                else:
                    part = await self._llm.extract_structured(
                        text=text,
                        document_type=document_type,
                        system_prompt=split_prompt,
                    )
            if not merged:
                merged = {k: v for k, v in part.items() if k != "periods"}
            for pkey, pval in (part.get("periods") or {}).items():
                merged_periods.setdefault(pkey, pval)
            for ekey, eval_ in (part.get("evidence_map") or {}).items():
                merged_evidence.setdefault(ekey, eval_)
        merged["periods"] = merged_periods
        if merged_evidence:
            merged["evidence_map"] = merged_evidence
        if not merged_periods:
            raise ValueError("split extraction returned no periods")
        return merged

    def _parse_and_normalize(
        self,
        raw_result: dict,
        document_type: str,
        ocr_text: str,
    ) -> dict:
        """Parse and normalize LLM output into typed extraction schema."""
        if document_type == "invoice":
            extraction = self._parse_invoice(raw_result, ocr_text)
        elif document_type == "balance_sheet":
            extraction = self._parse_balance_sheet(raw_result, ocr_text)
        elif document_type == "profit_and_loss":
            extraction = self._parse_profit_loss(raw_result, ocr_text)
        elif document_type == "cash_flow_statement":
            extraction = self._parse_cash_flow(raw_result, ocr_text)
        else:
            raise ValueError(f"Unknown document type: {document_type}")

        return extraction.model_dump(mode="json")

    # ─────────────────────────────────────────────────────────────
    # Invoice Parsing
    # ─────────────────────────────────────────────────────────────

    def _parse_invoice(self, raw: dict, ocr_text: str) -> InvoiceExtraction:
        """Parse LLM response into InvoiceExtraction."""
        n = self._normalizer
        ev = _evidence_lookup(raw)

        def _num(key: str, value: Any, unit: str | None = None):
            src, pg = ev.get(key.lower(), ("", 1))
            return n.build_extracted_value(value, unit=unit, source_text=src, page_number=pg)

        def _txt(key: str, value: Any):
            src, pg = ev.get(key.lower(), ("", 1))
            return n.build_text_value(value, source_text=src, page_number=pg)

        # An invoice's "unit" is its currency. Stamping it onto every monetary
        # field keeps an RM amount from being read as an INR one downstream.
        currency = raw.get("currency")
        cur = str(currency).strip() if currency else None

        extraction = InvoiceExtraction(
            invoice_number=_txt("invoice_number", raw.get("invoice_number")),
            invoice_date=_txt("invoice_date", raw.get("invoice_date")),
            due_date=_txt("due_date", raw.get("due_date")),
            vendor_name=_txt("vendor_name", raw.get("vendor_name")),
            vendor_address=_txt("vendor_address", raw.get("vendor_address")),
            vendor_tax_id=_txt("vendor_tax_id", raw.get("vendor_tax_id")),
            customer_name=_txt("customer_name", raw.get("customer_name")),
            customer_address=_txt("customer_address", raw.get("customer_address")),
            customer_tax_id=_txt("customer_tax_id", raw.get("customer_tax_id")),
            currency=_txt("currency", raw.get("currency")),
            subtotal=_num("subtotal", raw.get("subtotal"), cur),
            tax_amount=_num("tax_amount", raw.get("tax_amount"), cur),
            tax_rate=_num("tax_rate", raw.get("tax_rate"), "PERCENT"),
            discount=_num("discount", raw.get("discount"), cur),
            total_amount=_num("total_amount", raw.get("total_amount"), cur),
            amount_paid=_num("amount_paid", raw.get("amount_paid"), cur),
            amount_due=_num("amount_due", raw.get("amount_due"), cur),
            cash_paid=_num("cash_paid", raw.get("cash_paid"), cur),
            change_given=_num("change_given", raw.get("change_given"), cur),
            rounding_adjustment=_num("rounding_adjustment", raw.get("rounding_adjustment"), cur),
            payment_method=_txt("payment_method", raw.get("payment_method")),
            tax_inclusive=raw.get("tax_inclusive"),
            subtotal_confirmed_pretax=raw.get("subtotal_confirmed_pretax", False),
        )

        # Parse line items (evidence keyed "line::<description>" per prompt)
        for item_raw in raw.get("line_items", []):
            desc = item_raw.get("description")
            line_src, line_pg = ev.get(f"line::{str(desc or '').strip().lower()}", ("", 1))
            if (not line_src) and isinstance(desc, str):
                line_src, line_pg = ev.get(desc.strip().lower(), ("", 1))

            def _lnum(key: str, value: Any, unit: str | None = None):
                src, pg = ev.get(f"line::{str(desc or '').strip().lower()}::{key}", (line_src, line_pg))
                return n.build_extracted_value(value, unit=unit, source_text=src, page_number=pg)

            line = InvoiceLineItem(
                description=n.build_text_value(desc, source_text=line_src, page_number=line_pg),
                quantity=_lnum("quantity", item_raw.get("quantity"),
                               item_raw.get("unit_of_measure") or None),
                unit_of_measure=n.build_text_value(item_raw.get("unit_of_measure")),
                unit_price=_lnum("unit_price", item_raw.get("unit_price"), cur),
                amount=_lnum("amount", item_raw.get("amount"), cur),
                tax_code=n.build_text_value(item_raw.get("tax_code")),
                tax_rate=_lnum("tax_rate", item_raw.get("tax_rate"), "PERCENT"),
                discount_amount=_lnum("discount_amount", item_raw.get("discount_amount"), cur),
                discount_rate=_lnum("discount_rate", item_raw.get("discount_rate"), "PERCENT"),
            )
            extraction.line_items.append(line)

        # Parse additional tables
        for table_raw in raw.get("additional_tables", []):
            table = AdditionalTable(
                table_name=table_raw.get("table_name", ""),
                headers=table_raw.get("headers", []),
                rows=table_raw.get("rows", []),
            )
            extraction.additional_tables.append(table)

        # Parse additional fields
        for field_raw in raw.get("additional_fields", []):
            af = AdditionalField(
                label=field_raw.get("label", ""),
                value=n.build_text_value(field_raw.get("value")),
            )
            extraction.additional_fields.append(af)

        return extraction

    # ─────────────────────────────────────────────────────────────
    # Balance Sheet Parsing
    # ─────────────────────────────────────────────────────────────

    def _parse_balance_sheet(self, raw: dict, ocr_text: str) -> BalanceSheetExtraction:
        """Parse LLM response into BalanceSheetExtraction."""
        n = self._normalizer
        ev = _evidence_lookup(raw)

        def _num(key: str, value: Any, unit: str | None = None):
            src, pg = ev.get(str(key).strip().lower(), ("", 1))
            return n.build_extracted_value(value, unit=unit, source_text=src, page_number=pg)

        def _txt(key: str, value: Any):
            src, pg = ev.get(str(key).strip().lower(), ("", 1))
            return n.build_text_value(value, source_text=src, page_number=pg)

        # Resolve the scale once and stamp it onto every monetary value, so a
        # figure can never be read downstream without knowing whether it is
        # thousands or crore (Trap D5 — the unit switches between documents).
        unit = raw.get("unit_code", "") or n.detect_unit_code(ocr_text)
        unit = unit or None

        extraction = BalanceSheetExtraction(
            company_name=_txt("company_name", raw.get("company_name")),
            statement_type=_txt("statement_type", raw.get("statement_type")),
            unit_descriptor=_txt("unit_descriptor", raw.get("unit_descriptor")),
            unit_code=unit or "",
        )

        for period_key, period_raw in raw.get("periods", {}).items():
            period = BalanceSheetPeriod(year=period_raw.get("year", period_key))

            # Capital & Liabilities (dynamic dict — exact doc labels double
            # as evidence keys, so schema evolution needs no code change)
            for label, val in period_raw.get("capital_and_liabilities", {}).items():
                period.capital_and_liabilities[label] = _num(label, val, unit)

            period.total_capital_and_liabilities = _num(
                "total_capital_and_liabilities",
                period_raw.get("total_capital_and_liabilities"), unit,
            )

            # Assets (dynamic dict)
            for label, val in period_raw.get("assets", {}).items():
                period.assets[label] = _num(label, val, unit)

            period.total_assets = _num(
                "total_assets", period_raw.get("total_assets"), unit,
            )

            # Off-balance items
            period.contingent_liabilities = _num(
                "contingent_liabilities", period_raw.get("contingent_liabilities"), unit,
            )
            period.bills_for_collection = _num(
                "bills_for_collection", period_raw.get("bills_for_collection"), unit,
            )

            period.schedule_refs = period_raw.get("schedule_refs", {})
            extraction.periods[period_key] = period

        return extraction

    # ─────────────────────────────────────────────────────────────
    # P&L Parsing
    # ─────────────────────────────────────────────────────────────

    def _parse_profit_loss(self, raw: dict, ocr_text: str) -> ProfitLossExtraction:
        """Parse LLM response into ProfitLossExtraction."""
        n = self._normalizer
        ev = _evidence_lookup(raw)

        def _num(key: str, value: Any, unit: str | None = None):
            src, pg = ev.get(str(key).strip().lower(), ("", 1))
            return n.build_extracted_value(value, unit=unit, source_text=src, page_number=pg)

        def _txt(key: str, value: Any):
            src, pg = ev.get(str(key).strip().lower(), ("", 1))
            return n.build_text_value(value, source_text=src, page_number=pg)

        unit = raw.get("unit_code", "") or n.detect_unit_code(ocr_text)
        unit = unit or None

        extraction = ProfitLossExtraction(
            company_name=_txt("company_name", raw.get("company_name")),
            statement_type=_txt("statement_type", raw.get("statement_type")),
            unit_descriptor=_txt("unit_descriptor", raw.get("unit_descriptor")),
            unit_code=unit or "",
        )

        for period_key, period_raw in raw.get("periods", {}).items():
            period = ProfitLossPeriod(year=period_raw.get("year", period_key))

            # Income
            period.interest_earned = _num("interest_earned", period_raw.get("interest_earned"), unit)
            period.other_income = _num("other_income", period_raw.get("other_income"), unit)
            period.total_income = _num("total_income", period_raw.get("total_income"), unit)

            # Expenditure
            period.interest_expended = _num("interest_expended", period_raw.get("interest_expended"), unit)
            period.operating_expenses = _num("operating_expenses", period_raw.get("operating_expenses"), unit)
            period.provisions_and_contingencies = _num("provisions_and_contingencies", period_raw.get("provisions_and_contingencies"), unit)
            period.total_expenditure = _num("total_expenditure", period_raw.get("total_expenditure"), unit)

            # Profit chain
            period.net_profit_before_minority = _num("net_profit_before_minority", period_raw.get("net_profit_before_minority"), unit)
            period.minority_interest = _num("minority_interest", period_raw.get("minority_interest"), unit)
            period.share_in_profits_of_associates = _num("share_in_profits_of_associates", period_raw.get("share_in_profits_of_associates"), unit)
            period.consolidated_net_profit = _num("consolidated_net_profit", period_raw.get("consolidated_net_profit"), unit)

            # Appropriations
            period.addition_on_amalgamation = _num("addition_on_amalgamation", period_raw.get("addition_on_amalgamation"), unit)
            period.brought_forward_profit = _num("brought_forward_profit", period_raw.get("brought_forward_profit"), unit)
            period.total_available_for_appropriation = _num("total_available_for_appropriation", period_raw.get("total_available_for_appropriation"), unit)
            period.balance_carried_forward = _num("balance_carried_forward", period_raw.get("balance_carried_forward"), unit)

            # Dynamic appropriations dict
            for label, val in period_raw.get("appropriations", {}).items():
                period.appropriations[label] = _num(label, val, unit)

            # EPS — always ₹ per share, NEVER the document unit (Trap D8).
            # Stamping the statement's crore/'000 scale here would inflate a
            # ₹79.25 EPS by seven orders of magnitude.
            period.basic_eps = _num("basic_eps", period_raw.get("basic_eps"), "INR_PER_SHARE")
            period.diluted_eps = _num("diluted_eps", period_raw.get("diluted_eps"), "INR_PER_SHARE")
            period.face_value_per_share = _num("face_value_per_share", period_raw.get("face_value_per_share"), "INR_PER_SHARE")

            period.schedule_refs = period_raw.get("schedule_refs", {})
            extraction.periods[period_key] = period

        return extraction

    # ─────────────────────────────────────────────────────────────
    # Cash Flow Parsing
    # ─────────────────────────────────────────────────────────────

    def _parse_cash_flow(self, raw: dict, ocr_text: str) -> CashFlowExtraction:
        """Parse LLM response into CashFlowExtraction."""
        n = self._normalizer
        ev = _evidence_lookup(raw)

        def _num(key: str, value: Any, unit: str | None = None):
            src, pg = ev.get(str(key).strip().lower(), ("", 1))
            return n.build_extracted_value(value, unit=unit, source_text=src, page_number=pg)

        def _txt(key: str, value: Any):
            src, pg = ev.get(str(key).strip().lower(), ("", 1))
            return n.build_text_value(value, source_text=src, page_number=pg)

        unit = raw.get("unit_code", "") or n.detect_unit_code(ocr_text)
        unit = unit or None

        extraction = CashFlowExtraction(
            company_name=_txt("company_name", raw.get("company_name")),
            statement_type=_txt("statement_type", raw.get("statement_type")),
            unit_descriptor=_txt("unit_descriptor", raw.get("unit_descriptor")),
            unit_code=unit or "",
        )

        for period_key, period_raw in raw.get("periods", {}).items():
            period = CashFlowPeriod(year=period_raw.get("year", period_key))

            # Operating activities (exact doc labels double as evidence keys)
            for label, val in period_raw.get("operating_items", {}).items():
                period.operating_items[label] = _num(label, val, unit)
            period.net_cash_from_operating = _num("net_cash_from_operating", period_raw.get("net_cash_from_operating"), unit)

            # Investing activities
            for label, val in period_raw.get("investing_items", {}).items():
                period.investing_items[label] = _num(label, val, unit)
            period.net_cash_from_investing = _num("net_cash_from_investing", period_raw.get("net_cash_from_investing"), unit)

            # Financing activities
            for label, val in period_raw.get("financing_items", {}).items():
                period.financing_items[label] = _num(label, val, unit)
            period.net_cash_from_financing = _num("net_cash_from_financing", period_raw.get("net_cash_from_financing"), unit)

            # Reconciliation
            period.fx_translation_adjustment = _num("fx_translation_adjustment", period_raw.get("fx_translation_adjustment"), unit)
            period.net_increase_in_cash = _num("net_increase_in_cash", period_raw.get("net_increase_in_cash"), unit)
            period.cash_acquired_on_amalgamation = _num("cash_acquired_on_amalgamation", period_raw.get("cash_acquired_on_amalgamation"), unit)
            period.amalgamation_position = period_raw.get("amalgamation_position")
            period.opening_cash = _num("opening_cash", period_raw.get("opening_cash"), unit)
            period.closing_cash = _num("closing_cash", period_raw.get("closing_cash"), unit)

            extraction.periods[period_key] = period

        return extraction
