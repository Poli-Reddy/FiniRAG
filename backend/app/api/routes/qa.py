"""
FiniRAG QA routes — POST /api/v1/qa/ask.

Pipeline: validate -> route -> selective SQL/retrieval/validation ->
evidence -> answer -> citation checks. Tools invoked only per intent, and the
whole pipeline is bounded by QA_REQUEST_TIMEOUT_S so one slow provider call
cannot pin a worker indefinitely.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.rate_limit import rate_limit
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

# Heuristic field vocabulary for exact-value extraction from free text.
_FIELD_ALIASES = {
    "total_amount": ["total amount", "invoice total", "grand total"],
    "subtotal": ["subtotal", "sub total"],
    "tax_amount": ["tax amount", "gst", "tax"],
    "total_assets": ["total assets"],
    "total_capital_and_liabilities": ["total capital", "total liabilities"],
    "consolidated_net_profit": ["net profit", "consolidated profit"],
    "total_income": ["total income"],
    "total_expenditure": ["total expenditure"],
    "net_increase_in_cash": ["net increase in cash", "net increase"],
    "closing_cash": ["closing cash"],
    "opening_cash": ["opening cash"],
    "invoice_number": ["invoice number", "invoice id", "invoice #"],
    "invoice_date": ["invoice date"],
}


def _guess_field(question: str) -> str | None:
    q = question.lower()
    for field, aliases in _FIELD_ALIASES.items():
        if any(a in q for a in aliases):
            return field
    return None


def _guess_periods(question: str) -> list[str]:
    # Non-capturing prefix: return full four-digit years, not "19"/"20".
    return re.findall(r"\b(?:19|20)\d{2}\b", question)


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


@router.post("/ask", response_model=QAAskResponse, dependencies=[Depends(rate_limit)])
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
    # Unsupported -> explicit insufficient-evidence, no tools called.
    if decision.intents == ["INSUFFICIENT_OR_UNSUPPORTED"]:
        return _insufficient(request_id, decision.intents)

    fq = FinancialQueryService(db)
    retrieval_meta: dict = {"request_id": request_id, "intents": decision.intents}
    sql_results: list[dict] = []
    validation: dict | None = None
    candidates: list[dict] = []

    # Selective tool invocation per intent.
    if decision.needs_sql:
        field = _guess_field(question)
        periods = _guess_periods(question)
        if "FINANCIAL_COMPARISON" in decision.intents and field and len(periods) >= 2:
            cmp = await fq.compare(field=field, period_a=periods[0], period_b=periods[1],
                                   document_id=doc_uuid, document_name=body.document_name,
                                   document_type=body.document_type)
            retrieval_meta["comparison"] = cmp
            # Wrap comparison operands as evidence facts.
            if cmp.get("ok"):
                for period, value in ((periods[0], cmp["value_a"]), (periods[1], cmp["value_b"])):
                    sql_results.append({"ok": True, "field": f"{field}@{period}",
                                        "value": value, "original_value": value,
                                        "status": "EXPLICIT", "unit": cmp.get("unit"),
                                        "evidence": {}, "document_id": "",
                                        "document_name": body.document_name or "",
                                        "document_type": body.document_type or ""})
            else:
                sql_results.append({"ok": False, "field": field, "error": cmp.get("error")})
        elif field:
            period = body.reporting_period or (periods[0] if periods else None)
            res = await fq.get_value(field=field, document_id=doc_uuid,
                                     document_name=body.document_name,
                                     document_type=body.document_type, period=period)
            sql_results.append(res)
        else:
            retrieval_meta["sql_skipped"] = "no whitelisted field matched"

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
