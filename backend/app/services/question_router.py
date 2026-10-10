"""
FiniRAG Question router — deterministic-first, LLM only when ambiguous.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel

logger = logging.getLogger("finirag")

INTENTS = ("DOCUMENT_LOOKUP", "CONCEPTUAL_QUESTION", "EXACT_FINANCIAL_VALUE",
           "FINANCIAL_COMPARISON", "RECONCILIATION_STATUS", "DISCREPANCY_ANALYSIS",
           "MULTI_DOCUMENT_QUESTION", "INSUFFICIENT_OR_UNSUPPORTED")


class RouteDecision(BaseModel):
    intents: list[str]
    needs_sql: bool = False
    needs_retrieval: bool = True
    needs_validation: bool = False
    filters: dict = {}
    reason: str = ""
    llm_used: bool = False


_PATTERNS = [
    (re.compile(r"\b(reconcil|validation|validat|check.*pass|does.*balance|do.*books.*balance)\b", re.I), "RECONCILIATION_STATUS", True, False, True),
    (re.compile(r"\b(discrepancy|mismatch|why.*fail|failed.*check|error in|inconsisten)\b", re.I), "DISCREPANCY_ANALYSIS", True, True, True),
    (re.compile(r"\b(compare|comparison|difference|change between|vs\.?|versus|growth|increase from|across (periods|years|documents))\b", re.I), "FINANCIAL_COMPARISON", True, True, False),
    (re.compile(r"\b(what is the|how much|total|amount of|value of|figure for|show me the|give me the)\b.*\b(total|subtotal|tax|assets?|liabilit|income|expenditure|profit|cash|equity|revenue|balance)\b", re.I), "EXACT_FINANCIAL_VALUE", True, False, False),
    (re.compile(r"\b(invoice (number|date|id)|find invoice|show invoice|list (invoices|documents))\b", re.I), "DOCUMENT_LOOKUP", False, True, False),
    (re.compile(r"\b(what is|what does|explain|define|meaning of|how does|why is)\b", re.I), "CONCEPTUAL_QUESTION", False, True, False),
]

_MULTI = re.compile(r"\b(across|both|all documents|multiple|compare.*and)\b", re.I)
_UNSUPPORTED = re.compile(r"\b(stock price|forecast|predict|recommend|should i (buy|invest)|tax advice|legal advice)\b", re.I)


class QuestionRouter:
    def classify(self, question: str, filters: dict | None = None) -> RouteDecision:
        q = (question or "").strip()
        filters = filters or {}
        if not q:
            return RouteDecision(intents=["INSUFFICIENT_OR_UNSUPPORTED"], needs_sql=False,
                                 needs_retrieval=False, reason="empty question")
        if _UNSUPPORTED.search(q):
            return RouteDecision(intents=["INSUFFICIENT_OR_UNSUPPORTED"], needs_sql=False,
                                 needs_retrieval=False, reason="unsupported topic")
        intents: list[str] = []
        needs_sql = needs_ret = needs_val = False
        for rx, intent, sql, ret, val in _PATTERNS:
            if rx.search(q):
                intents.append(intent)
                needs_sql = needs_sql or sql
                needs_ret = needs_ret or ret
                needs_val = needs_val or val
        if _MULTI.search(q) or (filters.get("document_id") is None and "across" in q.lower()):
            if "MULTI_DOCUMENT_QUESTION" not in intents:
                intents.append("MULTI_DOCUMENT_QUESTION")
            needs_ret = True
        if not intents:
            # Ambiguous: default to grounded retrieval, not SQL numbers.
            intents = ["CONCEPTUAL_QUESTION"]
            needs_ret = True
        # Exact-value questions that also mention validation need both.
        if "EXACT_FINANCIAL_VALUE" in intents and re.search(r"\bvalidat|reconcil\b", q, re.I):
            intents.append("RECONCILIATION_STATUS")
            needs_val = True
        return RouteDecision(intents=intents, needs_sql=needs_sql,
                             needs_retrieval=needs_ret, needs_validation=needs_val,
                             filters=filters, reason="rules-first")
