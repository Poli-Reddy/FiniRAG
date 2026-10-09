"""
FiniRAG FinancialQueryService — SQL-backed structured access.

Never executes model-generated SQL. All queries use SQLAlchemy bound
parameters against whitelisted field identifiers. MISSING is never zero.
Decimal arithmetic throughout; incompatible units/currencies are rejected.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select, desc

from backend.app.models.document import Document

logger = logging.getLogger("finirag")

# Whitelisted logical fields per document type -> path inside extracted_data.
INVOICE_FIELDS = {"invoice_number", "invoice_date", "due_date", "vendor_name",
                  "customer_name", "currency", "subtotal", "tax_amount", "tax_rate",
                  "discount", "total_amount", "amount_paid", "amount_due",
                  "cash_paid", "change_given", "rounding_adjustment", "payment_method"}
STATEMENT_SCALARS = {"company_name", "statement_type", "unit_descriptor", "unit_code"}
PL_FIELDS = {"interest_earned", "other_income", "total_income", "interest_expended",
             "operating_expenses", "provisions_and_contingencies", "total_expenditure",
             "net_profit_before_minority", "minority_interest",
             "share_in_profits_of_associates", "consolidated_net_profit",
             "addition_on_amalgamation", "brought_forward_profit",
             "total_available_for_appropriation", "balance_carried_forward",
             "basic_eps", "diluted_eps", "face_value_per_share"}
CF_FIELDS = {"net_cash_from_operating", "net_cash_from_investing",
             "net_cash_from_financing", "fx_translation_adjustment",
             "net_increase_in_cash", "cash_acquired_on_amalgamation",
             "opening_cash", "closing_cash"}


def _field_blob(data: dict, *path) -> dict | None:
    cur = data
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur if isinstance(cur, dict) and "status" in cur else None


def _dec(value) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


class FinancialQueryService:
    def __init__(self, db):
        self._db = db

    async def _load_doc(self, document_id=None, document_name: str | None = None,
                        document_type: str | None = None):
        if document_id is not None:
            try:
                did = document_id if isinstance(document_id, UUID) else UUID(str(document_id))
            except Exception:
                return None, {"error": "INVALID_DOCUMENT_ID"}
            stmt = select(Document).where(Document.id == did)
            doc = (await self._db.execute(stmt)).scalar_one_or_none()
            if doc is None:
                return None, {"error": "DOCUMENT_NOT_FOUND"}
            return doc, None
        if document_name:
            stmt = select(Document).where(Document.document_name == document_name)
            if document_type:
                stmt = stmt.where(Document.document_type == document_type)
            stmt = stmt.order_by(desc(Document.created_at), desc(Document.id)).limit(1)
            doc = (await self._db.execute(stmt)).scalar_one_or_none()
            if doc is None:
                return None, {"error": "DOCUMENT_NOT_FOUND"}
            return doc, None
        return None, {"error": "NO_DOCUMENT_FILTER"}

    def _resolve_invoice(self, data: dict, field: str):
        if field not in INVOICE_FIELDS:
            return None, {"error": "UNSUPPORTED_FIELD"}
        blob = _field_blob(data, field)
        if blob is None:
            return None, {"error": "FIELD_NOT_FOUND", "insufficient": True}
        return self._pack(blob, field, None), None

    def _resolve_statement(self, doc_type: str, data: dict, field: str, period: str | None):
        periods = data.get("periods") or {}
        if period is None:
            if len(periods) == 1:
                period = next(iter(periods))
            else:
                return None, {"error": "AMBIGUOUS_PERIOD",
                              "insufficient": True,
                              "detail": f"period required, available: {sorted(periods)}"}
        pdata = periods.get(period)
        if not isinstance(pdata, dict):
            return None, {"error": "PERIOD_NOT_FOUND", "insufficient": True}
        # Scalar top-level (company_name etc.) has no period.
        if field in STATEMENT_SCALARS:
            blob = _field_blob(data, field)
            if blob is None:
                return None, {"error": "FIELD_NOT_FOUND", "insufficient": True}
            return self._pack(blob, field, period), None
        # Direct scalar inside period.
        blob = _field_blob(pdata, field)
        if blob is not None:
            return self._pack(blob, field, period), None
        # Dynamic dict sections: assets, capital_and_liabilities, appropriations,
        # operating/investing/financing_items.
        for section in ("capital_and_liabilities", "assets", "appropriations",
                        "operating_items", "investing_items", "financing_items"):
            sect = pdata.get(section) or {}
            if field in sect:
                return self._pack(sect[field], field, period, section=section), None
        # Totals that live at period level with fixed names.
        return None, {"error": "UNSUPPORTED_FIELD"}

    def _pack(self, blob: dict, field: str, period, section: str | None = None) -> dict:
        return {"field": field, "period": period, "section": section,
                "value": blob.get("value"), "original_value": blob.get("original_value", ""),
                "status": blob.get("status", "MISSING"),
                "unit": blob.get("unit"),
                "evidence": blob.get("evidence") or {}}

    async def get_value(self, *, field: str, document_id=None, document_name: str | None = None,
                        document_type: str | None = None, period: str | None = None) -> dict:
        doc, err = await self._load_doc(document_id, document_name, document_type)
        if err:
            return {"ok": False, **err}
        dtype = document_type or doc.document_type
        data = doc.extracted_data or {}
        if dtype == "invoice":
            packed, ferr = self._resolve_invoice(data, field)
        elif dtype in ("balance_sheet", "profit_and_loss", "cash_flow_statement"):
            packed, ferr = self._resolve_statement(dtype, data, field, period)
        else:
            return {"ok": False, "error": "UNSUPPORTED_DOCUMENT_TYPE"}
        if ferr:
            return {"ok": False, **ferr, "document_id": str(doc.id),
                    "document_name": doc.document_name}
        if packed["status"] in ("MISSING", "UNREADABLE"):
            return {"ok": False, "error": "VALUE_NOT_AVAILABLE",
                    "insufficient": True, "status": packed["status"], **packed,
                    "document_id": str(doc.id), "document_name": doc.document_name}
        return {"ok": True, **packed, "document_id": str(doc.id),
                "document_name": doc.document_name, "document_type": dtype,
                "unit_code": (data.get("unit_code") or "")}

    async def compare(self, *, field: str, period_a: str, period_b: str,
                      document_id=None, document_name: str | None = None,
                      document_type: str | None = None) -> dict:
        ra = await self.get_value(field=field, document_id=document_id,
                                  document_name=document_name,
                                  document_type=document_type, period=period_a)
        rb = await self.get_value(field=field, document_id=document_id,
                                  document_name=document_name,
                                  document_type=document_type, period=period_b)
        if not ra.get("ok") or not rb.get("ok"):
            return {"ok": False, "error": "COMPARISON_UNAVAILABLE", "a": ra, "b": rb}
        # Unit/currency guard: never silently compare incompatible scales.
        ua, ub = ra.get("unit"), rb.get("unit")
        if ua and ub and ua != ub:
            return {"ok": False, "error": "INCOMPATIBLE_UNITS",
                    "detail": f"{ua} vs {ub}", "a": ra, "b": rb}
        da, db = _dec(ra["value"]), _dec(rb["value"])
        if da is None or db is None:
            return {"ok": False, "error": "NON_NUMERIC_VALUE", "a": ra, "b": rb}
        diff = db - da
        return {"ok": True, "field": field, "period_a": period_a, "period_b": period_b,
                "value_a": str(da), "value_b": str(db), "difference": str(diff),
                "unit": ua or ub, "a": ra, "b": rb}

    async def validation_status(self, *, document_id=None, document_name: str | None = None,
                                check_name: str | None = None) -> dict:
        doc, err = await self._load_doc(document_id, document_name)
        if err:
            return {"ok": False, **err}
        validation = doc.validation or {}
        checks = validation.get("checks") or []
        if check_name:
            checks = [c for c in checks if c.get("name") == check_name]
            if not checks:
                return {"ok": False, "error": "CHECK_NOT_FOUND"}
        return {"ok": True, "document_id": str(doc.id), "document_name": doc.document_name,
                "overall_status": validation.get("overall_status", "NOT_APPLICABLE"),
                "summary": validation.get("summary", {}), "checks": checks}
