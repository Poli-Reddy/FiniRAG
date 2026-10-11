"""
FiniRAG QA routes — POST /api/v1/qa/ask.

Pipeline: validate -> route -> selective SQL/retrieval/validation ->
evidence -> answer -> citation checks. Tools invoked only per intent, and the
whole pipeline is bounded by QA_REQUEST_TIMEOUT_S so one slow provider call
cannot pin a worker indefinitely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.rate_limit import rate_limit
from backend.app.api.auth import require_api_key
from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.schemas.document import VALID_DOCUMENT_TYPES
from backend.app.schemas.qa import QAAskRequest, QAAskResponse
from backend.app.services.answer_service import AnswerService
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.financial_query_service import FinancialQueryService
from backend.app.services.question_router import QuestionRouter
from backend.app.services.retrieval_service import RetrievalService

logger = logging.getLogger("finirag")

router = APIRouter(prefix="/api/v1/qa", tags=["qa"])

router_instance = QuestionRouter()

# Curated sample questions grounded in the real 50-document corpus
# (evaluation/rag_real_questions.json). Served WITHOUT expected answers:
# id/question/category only — ground truth never leaves the server.
# backend/app/api/routes/qa.py is five levels below the project root.
_SAMPLES_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "evaluation" / "rag_real_questions.json"


@router.get("/samples",
            dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def question_samples() -> dict:
    """Clickable sample questions for the dashboard Q&A panel."""
    try:
        bank = json.loads(_SAMPLES_PATH.read_text(encoding="utf-8"))
        return {"samples": [
            {"id": q.get("id"), "question": q.get("question"),
             "category": q.get("category")}
            for q in bank if q.get("id") and q.get("question")],
            "count": len(bank)}
    except Exception as e:
        logger.warning("Sample questions unavailable", extra={
            "error_type": type(e).__name__})
        return {"samples": [], "count": 0}

# Heuristic field vocabulary for exact-value extraction from free text.
# Maps to whitelisted FinancialQueryService field ids only — never to dynamic
# section labels, which vary per document and are resolved at query time.
# Order matters: specific phrases first so "total amount" wins over "amount".
_FIELD_ALIASES = {
    "total_amount": ["total amount", "invoice total", "grand total",
                     "total payable"],
    "amount_due": ["amount due", "balance due"],
    "subtotal": ["subtotal", "sub total"],
    "tax_amount": ["tax amount", "gst", "sales tax"],
    "invoice_number": ["invoice number", "invoice id", "invoice #", "invoice no"],
    "invoice_date": ["invoice date", "date of invoice", "billed on"],
    "due_date": ["due date", "payment due", "pay by"],
    "vendor_name": ["vendor name", "vendor", "seller", "billed by", "supplier"],
    "customer_name": ["customer name", "customer", "buyer", "bill to", "billed to"],
    "amount_paid": ["amount paid", "paid amount", "payment received"],
    "cash_paid": ["cash paid", "cash tendered"],
    "change_given": ["change given", "change due"],
    "discount": ["discount"],
    "payment_method": ["payment method", "payment mode", "paid by", "paid via"],
    "currency": ["currency"],
    "total_assets": ["total assets"],
    "total_capital_and_liabilities": ["total capital and liabilities",
                                      "total liabilities", "total equity",
                                      "balance sheet total"],
    "company_name": ["company name", "company"],
    "total_income": ["total income", "revenue", "gross income"],
    "total_expenditure": ["total expenditure", "total expenses", "expenses"],
    "interest_earned": ["interest earned"],
    "interest_expended": ["interest expended", "interest paid"],
    "other_income": ["other income"],
    "operating_expenses": ["operating expenses"],
    "consolidated_net_profit": ["consolidated net profit", "net profit", "net income",
                                "bottom line"],
    "basic_eps": ["basic eps", "earnings per share", "eps"],
    "net_cash_from_operating": ["net cash from operating", "operating cash flow",
                                "cash from operations"],
    "net_cash_from_investing": ["net cash from investing", "investing cash flow"],
    "net_cash_from_financing": ["net cash from financing", "financing cash flow"],
    "net_increase_in_cash": ["net increase in cash", "net increase in cash and"],
    "closing_cash": ["closing cash", "closing balance", "cash at end"],
    "opening_cash": ["opening cash", "opening balance", "cash at beginning"],
    "minority_interest": ["minority interest"],
    "tax_rate": ["tax rate", "gst rate", "vat rate"],
}


def _guess_field(question: str) -> str | None:
    q = question.lower()
    for field, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            # Short aliases ("eps", "gst") must match whole words — otherwise
            # "steps" would resolve to earnings-per-share. Longer phrases are
            # distinctive enough for substring matching.
            if len(alias) <= 4:
                if re.search(r"\b" + re.escape(alias) + r"\b", q):
                    return field
            elif alias in q:
                return field
    return None


def _guess_periods(question: str) -> list[str]:
    # Non-capturing prefix: return full four-digit years, not "19"/"20".
    return re.findall(r"\b(?:19|20)\d{2}\b", question)


async def _resolve_document_from_question(db, question: str,
                                          document_type: str | None = None) -> str | None:
    """Match a document filename embedded in free text (finding #13).

    Exact-value questions name their document in the question far more often
    than in a separate filter field ("total on invoice X?"). Resolution is a
    bounded distinct-name lookup with longest case-insensitive substring
    match — no LLM, no guessing, and the chosen name is reported so the
    caller can see what was assumed.
    """
    from sqlalchemy import select

    from backend.app.models.document import Document

    q = (question or "").lower()
    stmt = select(Document.document_name).distinct()
    if document_type:
        stmt = stmt.where(Document.document_type == document_type)
    names = (await db.execute(stmt.limit(2000))).scalars().all()
    best: str | None = None
    for name in names:
        if name and name.lower() in q:
            if best is None or len(name) > len(best):
                best = name
    return best


def _insufficient(request_id: str, intents: list[str], meta: dict | None = None,
                  limitations: list[str] | None = None) -> QAAskResponse:
    return QAAskResponse(
        answer="I don't have sufficient evidence to answer this question.",
        status="INSUFFICIENT_EVIDENCE",
        answer_provider="none",
        limitations=list(limitations or []),
        retrieval_metadata=meta or {"request_id": request_id, "intents": intents},
        request_id=request_id, intents=intents,
    )


@router.post("/ask", response_model=QAAskResponse,
             dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def ask_question(body: QAAskRequest, db: AsyncSession = Depends(get_db)):
    request_id = str(uuid.uuid4())
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail={"code": "EMPTY_QUESTION",
                                                     "message": "Question is required."})
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail={"code": "QUESTION_TOO_LONG",
                                                     "message": "Question exceeds 2000 chars."})
    if body.document_type and body.document_type not in VALID_DOCUMENT_TYPES:
        raise HTTPException(status_code=400, detail={"code": "INVALID_DOCUMENT_TYPE",
                                                     "message": "Unsupported document_type."})
    doc_uuid = None
    if body.document_id:
        try:
            doc_uuid = uuid.UUID(str(body.document_id))
        except Exception:
            raise HTTPException(status_code=400, detail={"code": "INVALID_DOCUMENT_ID",
                                                         "message": "document_id must be UUID."})

    try:
        return await asyncio.wait_for(
            _answer_pipeline(body, question, doc_uuid, request_id, db),
            timeout=settings.QA_REQUEST_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.error("QA request timed out", extra={
            "request_id": request_id, "timeout_s": settings.QA_REQUEST_TIMEOUT_S})
        raise HTTPException(
            status_code=504,
            detail={"code": "QA_TIMEOUT",
                    "message": "The answer took too long to produce. Please retry."})


async def _answer_pipeline(body: QAAskRequest, question: str, doc_uuid,
                           request_id: str, db: AsyncSession) -> QAAskResponse:
    decision = router_instance.classify(question, filters={
        "document_id": str(doc_uuid) if doc_uuid else None,
        "document_name": body.document_name,
        "document_type": body.document_type,
        "reporting_period": body.reporting_period,
        "company": body.company})
    # Unsupported topics abstain outright — unless the caller scoped the
    # question to a specific document, in which case a retrieval-grounded
    # answer from that document is legitimate ("what does the forecast
    # section of X say?" is a lookup, not advice). The answer path still
    # carries limitations and never invents values.
    if decision.intents == ["INSUFFICIENT_OR_UNSUPPORTED"]:
        if doc_uuid is not None or body.document_name:
            logger.info("Unsupported topic with document scope; answering grounded", extra={
                "request_id": request_id})
            decision.intents = ["CONCEPTUAL_QUESTION"]
            decision.needs_retrieval = True
            decision.reason = "unsupported-topic-with-scope"
        else:
            return _insufficient(request_id, decision.intents)

    fq = FinancialQueryService(db)
    retrieval_meta: dict = {"request_id": request_id, "intents": decision.intents,
                            "route_reason": decision.reason}
    sql_results: list[dict] = []
    validation: dict | None = None
    candidates: list[dict] = []

    # Selective tool invocation per intent.
    # If the caller named no document but the question embeds a known
    # filename, resolve it — otherwise exact-value questions fail whenever
    # the scope lives only in free text (finding #13).
    resolved_name = body.document_name
    if decision.needs_sql and doc_uuid is None and not body.document_name:
        resolved_name = await _resolve_document_from_question(
            db, question, body.document_type)
        if resolved_name:
            retrieval_meta["document_resolved_from_question"] = resolved_name
    if decision.needs_sql:
        field = _guess_field(question)
        periods = _guess_periods(question)
        if "FINANCIAL_COMPARISON" in decision.intents and field and len(periods) >= 2:
            cmp = await fq.compare(field=field, period_a=periods[0], period_b=periods[1],
                                   document_id=doc_uuid, document_name=resolved_name,
                                   document_type=body.document_type)
            retrieval_meta["comparison"] = cmp
            # Wrap comparison operands as evidence facts, carrying each
            # operand's own source provenance (finding #14): real document
            # ids, names, periods, units, statuses and evidence — never
            # empty placeholders a citation cannot resolve.
            if cmp.get("ok"):
                for side, period in (("a", periods[0]), ("b", periods[1])):
                    operand = cmp[side]
                    sql_results.append({
                        "ok": True, "field": f"{field}@{period}",
                        "value": operand.get("value"),
                        "original_value": operand.get("original_value", ""),
                        "status": operand.get("status", "EXPLICIT"),
                        "unit": operand.get("unit") or cmp.get("unit"),
                        "evidence": operand.get("evidence") or {},
                        "document_id": operand.get("document_id", ""),
                        "document_name": operand.get("document_name", ""),
                        "document_type": operand.get("document_type", ""),
                        "period": period,
                    })
            else:
                sql_results.append({"ok": False, "field": field, "error": cmp.get("error")})
        elif field:
            period = body.reporting_period or (periods[0] if periods else None)
            res = await fq.get_value(field=field, document_id=doc_uuid,
                                     document_name=resolved_name,
                                     document_type=body.document_type, period=period)
            sql_results.append(res)
        else:
            retrieval_meta["sql_skipped"] = "no whitelisted field matched"
        # Retrieval fallback: when SQL produced nothing usable (unknown
        # wording, missing scope, missing value), try passages instead of
        # stopping silently (findings #13/#18). Recorded unconditionally so
        # the reason is visible even when another intent already planned
        # a retrieval pass.
        if field is None or any(not r.get("ok") for r in sql_results):
            retrieval_meta["retrieval_fallback"] = "sql yielded nothing usable"
            decision.needs_retrieval = True

    if decision.needs_validation or "RECONCILIATION_STATUS" in decision.intents \
            or "DISCREPANCY_ANALYSIS" in decision.intents:
        validation = await fq.validation_status(document_id=doc_uuid,
                                                document_name=body.document_name)

    if decision.needs_retrieval:
        out = await RetrievalService(db).retrieve(
            question, document_id=doc_uuid,
            document_type=body.document_type,
            reporting_period=body.reporting_period,
            company=body.company)
        candidates = out.get("candidates", [])
        retrieval_meta.update({k: v for k, v in out.items() if k != "candidates"})

    # Explicit insufficient-evidence when nothing supports an answer.
    if not sql_results and not candidates and not (validation or {}).get("checks"):
        return _insufficient(request_id, decision.intents, meta=retrieval_meta)

    evidence = EvidenceService().build(sql_results=sql_results, validation=validation,
                                       candidates=candidates)
    answer = await AnswerService().answer(question=question, evidence=evidence,
                                          retrieval_metadata=retrieval_meta,
                                          validation=validation)
    calc_meta = retrieval_meta.get("comparison")
    calculations = []
    if isinstance(calc_meta, dict) and calc_meta.get("ok"):
        calculations = [{"field": calc_meta.get("field"), "periods": [calc_meta.get("period_a"),
                         calc_meta.get("period_b")], "difference": calc_meta.get("difference"),
                         "unit": calc_meta.get("unit")}]
    return QAAskResponse(answer=answer.answer, status=answer.status,
                         answer_provider=answer.provider,
                         citations=answer.citations, financial_facts=answer.financial_facts,
                         calculations=calculations,
                         validation_checks=answer.validation_checks,
                         limitations=answer.limitations,
                         retrieval_metadata={**retrieval_meta, **answer.retrieval_metadata},
                         request_id=request_id, intents=decision.intents)
