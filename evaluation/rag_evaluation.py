"""RAG evaluation — OFFLINE smoke test (hash embeddings, sqlite, synthetic data).

NOT production semantic quality. For live-provider quality, run
evaluation/rag_live_evaluation.py with GEMINI_API_KEY + PostgreSQL and report
those numbers separately. Metrics here use actual ranks:
  Recall@k from top-k relevant docs; MRR from first relevant rank;
  citation support at claim level (EvidenceService.supports);
  answer correctness vs expected values/snippets; abstention vs expectation.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUESTIONS = ROOT / "rag_questions.json"
OUT = ROOT / "rag_report.json"

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession  # noqa: E402

from backend.app.core.database import Base  # noqa: E402
import backend.app.models  # noqa: F401
from backend.app.providers.embedding_provider import HashEmbeddingProvider  # noqa: E402
from backend.app.services.chunking_service import ChunkingService  # noqa: E402
from backend.app.services.embedding_service import EmbeddingService  # noqa: E402
from backend.app.services.indexing_service import IndexingService  # noqa: E402
from backend.app.services.retrieval_service import RetrievalService  # noqa: E402
from backend.app.services.question_router import QuestionRouter  # noqa: E402
from backend.app.services.financial_query_service import FinancialQueryService  # noqa: E402
from backend.app.services.evidence_service import EvidenceService  # noqa: E402
from backend.app.services.answer_service import AnswerService  # noqa: E402
from backend.app.models.document import Document  # noqa: E402

K = 5


def _invoice(doc_name: str, total: str = "108.00"):
    def ev(v, st="EXPLICIT"):
        return {"value": v, "original_value": str(v), "status": st,
                "evidence": {"source_text": str(v), "page_number": 1}, "unit": "USD"}
    return {"invoice_number": {**ev(doc_name), "value": doc_name},
            "total_amount": ev(total), "subtotal": ev("100.00"), "tax_amount": ev("8.00"),
            "currency": {**ev("USD"), "value": "USD"},
            "line_items": [{"description": {**ev("Widget"), "value": "Widget"},
                            "quantity": ev("2"), "unit_price": ev("50.00"),
                            "amount": ev("100.00")}]}


async def _seed(db):
    from datetime import datetime, timezone

    docs = [
        ("inv-a.jpg", "invoice", _invoice("INV-A", "108.00"), "PASS"),
        ("inv-b.jpg", "invoice", _invoice("INV-B", "216.00"), "PASS"),
        ("fail.jpg", "invoice", _invoice("INV-F", "999.00"), "FAIL"),
    ]
    for name, dtype, data, vstatus in docs:
        db.add(Document(document_name=name, document_type=dtype,
                       processing_status="PASS", file_validation={"status": "PASS"},
                       extracted_data=data,
                       validation={"overall_status": vstatus, "checks": [
                           {"name": "c1", "status": vstatus, "formula": "sum"}],
                           "summary": {}},
                       processing_metadata={},
                       created_at=datetime.now(timezone.utc)))
    await db.flush()


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def final_rank(candidates: list[dict], relevant: set[str]) -> int | None:
    """1-based position of the first relevant candidate in the FINAL fused
    ordering (the candidates list order after RRF + dedup + truncation).

    This must be computed from final positions — never from a component
    `rank` stored by one retrieval path, which reflects pre-fusion order.
    None when no relevant candidate was returned.
    """
    for i, c in enumerate(candidates, start=1):
        if (c.get("document_name") or "") in relevant:
            return i
    return None


async def main(provider=None, engine=None, out=None, live: bool = False,
               strict_retrieval: bool = False) -> dict:
    """provider: EmbeddingProvider (default offline hash). engine: SQLAlchemy
    async engine (default in-memory sqlite). live=True labels the report
    LIVE/SEMANTIC and rolls back synthetic seed rows instead of committing."""
    from backend.app.providers.embedding_provider import HashEmbeddingProvider as _Hash

    questions = json.loads(QUESTIONS.read_text())
    OUT_PATH = out or OUT
    # Offline runs must never touch a live provider, even when the shell's
    # .env carries keys. Blank them for the duration and restore afterwards.
    from backend.app.core.config import settings as _settings

    _saved_keys = {}
    if not live:
        for _k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GROQ_API_KEY", "GROQ_API_KEYS"):
            _saved_keys[_k] = getattr(_settings, _k, "")
            setattr(_settings, _k, "")
    if engine is None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        own_engine = True
    else:
        own_engine = False
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    emb = EmbeddingService(provider or _Hash())
    results = []
    latencies = []
    async with factory() as db:
        await _seed(db)
        docs = (await db.execute(select(Document))).scalars().all()
        if live:
            # Scope to OUR synthetic rows only — never touch real documents.
            docs = [d for d in docs if d.document_name in
                    {"inv-a.jpg", "inv-b.jpg", "fail.jpg"}]
        name_to_id = {d.document_name: d.id for d in docs}
        for d in docs:
            await IndexingService(db, embeddings=emb).index_document(d.id)
        if not live:
            await db.commit()

        router = QuestionRouter()
        filter_results = []
        for q in questions:
            t0 = time.time()
            decision = router.classify(q["question"])
            # ── Ranking retrieval: UNFILTERED, exactly as a real user call ──
            # The expected document must NOT be passed as a filter here; that
            # would pre-select the answer and inflate Recall/MRR. Filters are
            # exercised separately by dedicated filter_* questions below.
            ret = await RetrievalService(db, emb).retrieve(
                q["question"], strict=strict_retrieval)
            cands = ret["candidates"]
            if live:
                assert ret["vector_path"] == "pgvector", ret
                assert ret["keyword_path"] == "postgres-fts", ret
            relevant = set(q.get("relevant_documents") or [])
            rank = final_rank(cands, relevant)
            if relevant:
                hit = rank is not None and rank <= K
                rr = (1.0 / rank) if rank else 0.0
            else:
                # No ground-truth relevance (unsupported/absent questions):
                # excluded from recall/MRR; abstention is scored instead.
                hit, rr = None, None
            # ── Document-filter behavior (separate probe, not ranking) ──
            filter_ok = None
            if q.get("filter_document"):
                target = name_to_id.get(q["filter_document"])
                fret = await RetrievalService(db, emb).retrieve(
                    q["question"], document_id=target)
                fcands = fret["candidates"]
                filter_ok = bool(fcands) and all(
                    str(c.get("document_id")) == str(target) for c in fcands)
                filter_results.append(filter_ok)
            # Structured / validation paths — mirror the real QA route:
            # fetch SQL whenever a field applies (including MISSING markers),
            # fetch validation whenever the question is about it.
            sql_rs = []
            if q.get("field"):
                res = await FinancialQueryService(db).get_value(
                    field=q["field"], document_name=q.get("expected_document"),
                    document_type=q.get("document_type"))
                sql_rs = [res]
                fq_ok = (res.get("ok") and str(res.get("value")) == q["expected_value"]
                         ) if q.get("expected_value") else None
            else:
                fq_ok = None
            val_ok = None
            validation = None
            if q.get("expected_validation") or "RECONCILIATION" in decision.intents \
                    or "DISCREPANCY" in decision.intents:
                validation = await FinancialQueryService(db).validation_status(
                    document_name=q.get("expected_document"))
                val_ok = (validation.get("ok")
                          and validation.get("overall_status") == q["expected_validation"]
                          ) if q.get("expected_validation") else None
            # Evidence + answer (respect router abstention).
            if decision.intents == ["INSUFFICIENT_OR_UNSUPPORTED"]:
                ev = {"items": []}
            else:
                ev = EvidenceService().build(sql_results=sql_rs, validation=validation,
                                             candidates=cands)
            ans = await AnswerService().answer(question=q["question"], evidence=ev)
            # Claim-level support of the FINAL answer, measured with the same
            # production validator the pipeline enforces (not mere citation
            # existence, not overlap with a preselected snippet).
            abstained = ans.status == "INSUFFICIENT_EVIDENCE"
            if abstained:
                # Abstention carries no claims; it must not earn support or
                # faithfulness credit — abstention correctness is scored below.
                cit_score, faith = None, None
            else:
                check = AnswerService()._validate_claims(
                    ans.answer, ans.citations, ev.get("items", []))
                tot = check["factual_total"]
                cit_score = (check["factual_kept"] / tot) if tot else 1.0
                pool = " ".join(f"{i.get('content','')} {i.get('source_text','')}"
                                for i in ev.get("items", []))
                pool_nums = _numbers(pool)
                ans_nums = _numbers(ans.answer)
                faith = 1.0 if not ans_nums else len(ans_nums & pool_nums) / len(ans_nums)
            # Answer correctness: expected value/snippet present, or correct abstention.
            if q.get("requires_abstention"):
                ans_ok = ans.status == "INSUFFICIENT_EVIDENCE"
            elif q.get("expected_value"):
                ans_ok = q["expected_value"] in ans.answer
            elif q.get("expected_citation_contains"):
                ans_ok = q["expected_citation_contains"] in ans.answer
            elif q.get("expected_validation"):
                ans_ok = q["expected_validation"] in ans.answer
            else:
                ans_ok = ans.status == "ANSWERED" and bool(ans.citations)
            abst_ok = (ans.status == "INSUFFICIENT_EVIDENCE") == bool(q.get("requires_abstention"))
            lat = (time.time() - t0) * 1000
            latencies.append(lat)
            results.append({"id": q["id"], "rank": rank, "hit_at_5": hit, "rr": rr,
                            "fq_ok": fq_ok, "val_ok": val_ok, "filter_ok": filter_ok,
                            "cit_support": round(cit_score, 3) if cit_score is not None else None,
                            "ans_ok": ans_ok,
                            "faith": round(faith, 3) if faith is not None else None,
                            "abst_ok": abst_ok,
                            "vector_path": ret.get("vector_path"),
                            "keyword_path": ret.get("keyword_path"),
                            "answer_provider": ans.provider,
                            "status": ans.status, "latency_ms": round(lat, 1),
                            "intents": decision.intents})
        if live:
            # Synthetic rows never persist in prod (session close would roll
            # back anyway; this makes it explicit and immediate).
            await db.rollback()
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    native_ok = [r for r in results
                 if r.get("vector_path") == "pgvector" and r.get("keyword_path") == "postgres-fts"]
    providers = sorted({r.get("answer_provider") for r in results if r.get("answer_provider")})
    if live:
        label = "LIVE/SEMANTIC — gemini-embedding-2 + PostgreSQL (synthetic corpus, rolled back)"
        provider_name = f"GeminiEmbeddingProvider ({emb.model}, dim {emb.dim})"
        db_name = "postgresql+asyncpg (pgvector + FTS native paths asserted per question)"
        corpus = {"documents": len(docs), "synthetic": True, "rolled_back": True,
                  "prod_rows_untouched": True}
        limits = [
            "Synthetic 3-doc corpus; recall/MRR measure plumbing + live semantic ranking on synthetic data only.",
            "Ranking retrieval unfiltered; rank is the final fused position (see final_rank).",
            "Every ranking retrieval asserted pgvector + postgres-fts native paths (strict).",
            "Claim support re-validates the final answer with the production validator;",
            " token overlap remains a basic guard, not entailment proof.",
        ]
    else:
        label = "OFFLINE/HASH — NOT production semantic quality"
        provider_name = "HashEmbeddingProvider (deterministic, non-semantic)"
        db_name = "sqlite+aiosqlite (python cosine + token overlap; pgvector/FTS NOT exercised)"
        corpus = {"documents": 3, "synthetic": True}
        limits = [
            "Hash vectors are not semantic; recall/MRR here test plumbing + keyword overlap only.",
            "Ranking retrieval is unfiltered; rank is the final fused position (see final_rank).",
            "Claim support re-validates the final answer with the production validator;",
            " token overlap remains a basic guard, not entailment proof.",
            "Live semantic quality requires evaluation/rag_live_evaluation.py with GEMINI_API_KEY + PostgreSQL.",
        ]
    report = {
        "label": label,
        "provider": provider_name,
        "embedding_model": emb.model,
        "embedding_dim": emb.dim,
        "database": db_name,
        "corpus": corpus,
        "test_set": len(results),
        "k": K,
        "answer_providers_seen": providers,
        "native_paths": {"questions": len(native_ok), "of": len(results)},
        "recall_at_k": round(mean([1.0 if r["hit_at_5"] else 0.0 for r in results if r["hit_at_5"] is not None]), 3),
        "mrr": round(mean([r["rr"] for r in results if r["rr"] is not None]), 3),
        "citation_support": round(mean([r["cit_support"] for r in results if r["cit_support"] is not None]), 3),
        "answer_correctness": round(mean([1.0 if r["ans_ok"] else 0.0 for r in results]), 3),
        "faithfulness": round(mean([r["faith"] for r in results if r["faith"] is not None]), 3),
        "unsupported_claim_rate": round(mean(
            [1.0 - r["cit_support"] for r in results if r["cit_support"] is not None]), 3),
        "abstention_correctness": round(mean([1.0 if r["abst_ok"] else 0.0 for r in results]), 3),
        "filter_behavior": round(mean(
            [1.0 if r["filter_ok"] else 0.0 for r in results if r["filter_ok"] is not None]), 3),
        "financial_query_correctness": round(mean([1.0 if r["fq_ok"] else 0.0 for r in results if r["fq_ok"] is not None]), 3),
        "validation_path_correctness": round(mean([1.0 if r["val_ok"] else 0.0 for r in results if r["val_ok"] is not None]), 3),
        "latency_ms": {"mean": round(mean(latencies), 1), "max": round(max(latencies), 1) if latencies else 0},
        "limitations": limits,
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    for _k, _v in _saved_keys.items():
        setattr(_settings, _k, _v)
    if own_engine:
        await engine.dispose()
    return report


if __name__ == "__main__":
    asyncio.run(main())
