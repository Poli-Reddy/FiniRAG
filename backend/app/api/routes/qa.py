"""
FiniRAG QA routes — POST /api/v1/qa/ask.

Pipeline: validate -> route -> selective SQL/retrieval/validation ->
evidence -> answer -> citation checks. Tools invoked only per intent.
"""

from __future__ import annotations

import logging
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

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


@router.post("/ask", response_model=QAAskResponse)
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

    decision = router_instance.classify(question, filters={
        "document_id": str(doc_uuid) if doc_uuid else None,
        "document_name": body.document_name,
        "document_type": body.document_type,
        "reporting_period": body.reporting_period,
        "company": body.company})
    # Unsupported -> explicit insufficient-evidence, no tools called.
    if decision.intents == ["INSUFFICIENT_OR_UNSUPPORTED"]:
        result = await AnswerService().answer(
            question=question, evidence={"items": []},
            retrieval_metadata={"request_id": request_id, "intents": decision.intents})
        return QAAskResponse(answer=result.answer, status="INSUFFICIENT_EVIDENCE",
                             answer_provider=result.provider,
                             limitations=result.limitations,
                             retrieval_metadata={"request_id": request_id,
                                                 "intents": decision.intents},
                             request_id=request_id, intents=decision.intents)

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
                sql_results.append({"ok": True, "field": f"{field}@{periods[0]}",
                                    "value": cmp["value_a"], "original_value": cmp["value_a"],
                                    "status": "EXPLICIT", "unit": cmp.get("unit"),
                                    "evidence": {}, "document_id": "",
                                    "document_name": body.document_name or "",
                                    "document_type": body.document_type or ""})
                sql_results.append({"ok": True, "field": f"{field}@{periods[1]}",
                                    "value": cmp["value_b"], "original_value": cmp["value_b"],
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
        ret = RetrievalService(db)
        out = await ret.retrieve(question, document_id=doc_uuid,
                                 document_type=body.document_type,
                                 reporting_period=body.reporting_period,
                                 company=body.company)
        candidates = out.get("candidates", [])
        retrieval_meta.update({k: v for k, v in out.items() if k != "candidates"})
        if "FINANCIAL_COMPARISON" in decision.intents and not candidates and sql_results:
            pass  # numbers alone suffice; no forced retrieval

    # Explicit insufficient-evidence when nothing supports an answer.
    if not sql_results and not candidates and not (validation or {}).get("checks"):
        result = await AnswerService().answer(question=question, evidence={"items": []},
                                              retrieval_metadata=retrieval_meta)
        return QAAskResponse(answer=result.answer, status="INSUFFICIENT_EVIDENCE",
                             answer_provider=result.provider,
                             retrieval_metadata=retrieval_meta, request_id=request_id,
                             intents=decision.intents)

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
