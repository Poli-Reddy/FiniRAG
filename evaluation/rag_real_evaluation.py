#!/usr/bin/env python
"""Real-corpus RAG evaluation — live PostgreSQL + real embeddings + real pipeline.

Runs evaluation/rag_real_questions.json (grounded in the actual 50-document
corpus) through the production QA path and measures:

Retrieval : hit rate (recall@5), MRR over relevant documents
Routing   : needs_sql / needs_retrieval / needs_validation vs expectation
Generation: faithfulness, groundedness (claim-level support),
            answer relevance (numeric-aware value presence + abstention)

Read-only: retrieval + QA make no database writes. LLM answers cost quota
(Gemini → Groq → template); degraded providers are recorded honestly per
question, never hidden.

Usage:
    python evaluation/rag_real_evaluation.py [--max-questions 60]
"""

import argparse
import asyncio
import json
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent
BANK = ROOT / "rag_real_questions.json"
OUT = ROOT / "rag_real_report.json"

K = 5


def _numbers(text: str) -> set[str]:
    # Signed: cash-flow decreases print as negative differences, and
    # stripping the sign would score "-672167" as missing.
    return set(re.findall(r"-?\d+(?:\.\d+)?", text or ""))


def _to_decimal(s: str) -> Decimal | None:
    try:
        return Decimal(str(s).replace(",", ""))
    except (InvalidOperation, ValueError, AttributeError):
        return None


def _value_present(expected: str | None, answer: str) -> bool | None:
    """Numeric-aware relevance: equal numbers count even when formatting
    differs ("108.00" vs "108"). Non-numeric expectations pass when every
    significant token appears (case-insensitive) — "HDFC Bank" answers
    "HDFC Bank Limited". None when there is nothing checkable."""
    if not expected:
        return None
    exp_num = _to_decimal(expected)
    if exp_num is not None:
        return any(_to_decimal(n) == exp_num for n in _numbers(answer))
    stop = {"the", "a", "an", "of", "and", "is", "in", "for", "for?"}
    toks = [t for t in re.findall(r"[a-z0-9]+", expected.lower())
            if len(t) > 2 and t not in stop]
    if not toks:
        return expected.lower() in (answer or "").lower()
    low = (answer or "").lower()
    return all(t in low for t in toks)


def _stratified(questions: list[dict], cap: int) -> list[dict]:
    """Deterministic per-category cap so a --max-questions run stays diverse."""
    if cap <= 0 or len(questions) <= cap:
        return questions
    by_cat: dict[str, list[dict]] = {}
    for q in questions:
        by_cat.setdefault(q["category"], []).append(q)
    per = max(1, cap // len(by_cat))
    out: list[dict] = []
    for cat in sorted(by_cat):
        out.extend(by_cat[cat][:per])
    return out[:cap]


def final_rank(candidates: list[dict], relevant: set[str]) -> int | None:
    for i, c in enumerate(candidates, start=1):
        if (c.get("document_name") or "") in relevant:
            return i
    return None


async def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Real-corpus RAG evaluation (live, read-only)")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="Stratified cap across categories (0 = all)")
    parser.add_argument("--out", type=str, default=str(OUT))
    args = parser.parse_args(argv)

    import os

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

    from backend.app.core.config import normalize_database_url, settings
    from backend.app.services.embedding_service import EmbeddingService
    from backend.app.providers.embedding_provider import GeminiEmbeddingProvider
    from backend.app.services.retrieval_service import RetrievalService
    from backend.app.services.question_router import QuestionRouter
    from backend.app.api.routes.qa import router_instance
    from backend.app.services.financial_query_service import FinancialQueryService
    from backend.app.services.evidence_service import EvidenceService
    from backend.app.services.answer_service import AnswerService
    from backend.app.api.routes.qa import _answer_pipeline
    from backend.app.schemas.qa import QAAskRequest

    db_url = normalize_database_url(
        os.environ.get("DATABASE_URL") or settings.DATABASE_URL)
    if not db_url or "user:pass@host" in db_url:
        print("ERROR: DATABASE_URL not configured")
        return {"error": "no DATABASE_URL"}

    questions = _stratified(json.loads(BANK.read_text(encoding="utf-8")),
                            args.max_questions)
    print(f"running {len(questions)} questions against live PostgreSQL")

    engine = create_async_engine(db_url, echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    emb = EmbeddingService(GeminiEmbeddingProvider())
    router = router_instance

    results: list[dict] = []
    latencies: list[float] = []
    async with factory() as db:
        for n, q in enumerate(questions, start=1):
            t0 = time.time()
            print(f"[{n}/{len(questions)}] {q['id']} ({q['category']})")
            try:
                decision = router.classify(q["question"])
                # Abstention questions are scored on OUTCOME only: whether
                # tools fired on the way to a correct abstention is not a
                # routing defect (e.g. a conceptual read that finds nothing).
                # A None expectation means "don't care" (exact questions may
                # legitimately take the SQL-only or the dual path).
                score_route = not q.get("requires_abstention")
                exp_sql, exp_ret = q.get("expect_sql"), q.get("expect_retrieval")
                exp_val = q.get("expect_validation", False)
                route_sql_ok = (decision.needs_sql == exp_sql) if (score_route and exp_sql is not None) else None
                route_ret_ok = (decision.needs_retrieval == exp_ret) if (score_route and exp_ret is not None) else None
                route_val_ok = (decision.needs_validation == exp_val) if score_route else None
                scored = [v for v in (route_sql_ok, route_ret_ok, route_val_ok) if v is not None]
                route_ok = all(scored) if scored else None

                ret = await RetrievalService(db, emb).retrieve(q["question"], strict=True)
                cands = ret["candidates"]
                relevant = set(q.get("relevant_documents") or [])
                rank = final_rank(cands, relevant)
                if relevant:
                    hit = rank is not None and rank <= K
                    rr = (1.0 / rank) if rank else 0.0
                else:
                    hit, rr = None, None

                body = QAAskRequest(question=q["question"])
                ans = await _answer_pipeline(body, q["question"], None,
                                             f"real-eval-{q['id']}", db)
                abstained = ans.status == "INSUFFICIENT_EVIDENCE"
                if abstained:
                    cit_score, faith = None, None
                else:
                    measure_ev = EvidenceService().build(candidates=cands)
                    check = AnswerService()._validate_claims(
                        ans.answer, ans.citations, measure_ev.get("items", []))
                    tot = check["factual_total"]
                    cit_score = (check["factual_kept"] / tot) if tot else 1.0
                    pool = " ".join(f"{i.get('content', '')} {i.get('source_text', '')}"
                                    for i in measure_ev.get("items", []))
                    pool_nums, ans_nums = _numbers(pool), _numbers(ans.answer)
                    faith = 1.0 if not ans_nums else len(ans_nums & pool_nums) / len(ans_nums)

                if q.get("requires_abstention"):
                    ans_ok = abstained
                elif q.get("expected_value"):
                    ans_ok = _value_present(q["expected_value"], ans.answer or "")
                else:
                    ans_ok = (ans.status == "ANSWERED" and bool(ans.citations))
                abst_ok = abstained == bool(q.get("requires_abstention"))
                results.append({
                    "id": q["id"], "category": q["category"],
                    "rank": rank, "hit_at_5": hit, "rr": rr,
                    "route_ok": route_ok, "route_sql_ok": route_sql_ok,
                    "route_ret_ok": route_ret_ok, "route_val_ok": route_val_ok,
                    "routed_sql": decision.needs_sql,
                    "routed_retrieval": decision.needs_retrieval,
                    "routed_validation": decision.needs_validation,
                    "intents": decision.intents,
                    "cit_support": round(cit_score, 3) if cit_score is not None else None,
                    "faith": round(faith, 3) if faith is not None else None,
                    "ans_ok": ans_ok if ans_ok is not None else None,
                    "abst_ok": abst_ok,
                    "vector_path": ret.get("vector_path"),
                    "keyword_path": ret.get("keyword_path"),
                    "degraded": ret.get("degraded"),
                    "answer_provider": ans.answer_provider,
                    "status": ans.status,
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "error": None,
                })
            except Exception as e:
                # Isolate per-question failures: a poisoned transaction must
                # never wedge the rest of the run (PendingRollbackError).
                try:
                    await db.rollback()
                except Exception:
                    pass
                results.append({
                    "id": q["id"], "category": q["category"],
                    "rank": None, "hit_at_5": None, "rr": None,
                    "route_ok": None, "route_sql_ok": None,
                    "route_ret_ok": None, "route_val_ok": None,
                    "routed_sql": None, "routed_retrieval": None,
                    "routed_validation": None, "intents": [],
                    "cit_support": None, "faith": None,
                    "ans_ok": None, "abst_ok": False,
                    "vector_path": None, "keyword_path": None,
                    "degraded": None, "answer_provider": "error",
                    "status": "ERROR",
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                    "error": f"{type(e).__name__}: {e}",
                })
            latencies.append(results[-1]["latency_ms"])

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    def rate(key, pred=lambda v: bool(v)):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(1.0 for v in vals if pred(v)) / len(vals), 3) if vals else 0.0

    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in results}):
        sub = [r for r in results if r["category"] == cat]

        def sub_rate(key, _sub=sub, pred=lambda v: bool(v)):
            vals = [r[key] for r in _sub if r[key] is not None]
            return round(sum(1.0 for v in vals if pred(v)) / len(vals), 3) if vals else 0.0

        hits = [r["hit_at_5"] for r in sub if r["hit_at_5"] is not None]
        by_cat[cat] = {
            "n": len(sub),
            "hit_rate": round(sum(1.0 for h in hits if h) / len(hits), 3) if hits else 0.0,
            "mrr": mean([r["rr"] for r in sub]),
            "routing": sub_rate("route_ok"),
            "answer": sub_rate("ans_ok"),
            "abstention": sub_rate("abst_ok"),
        }

    providers = sorted({r.get("answer_provider") for r in results
                        if r.get("answer_provider") and r.get("status") != "ERROR"})
    report = {
        "label": "REAL-CORPUS — 50 documents, live pgvector + real pipeline (read-only)",
        "provider": f"GeminiEmbeddingProvider ({emb.model}, dim {emb.dim})",
        "corpus": {"documents": 50, "synthetic": False},
        "test_set": len(results),
        "k": K,
        "answer_providers_seen": providers,
        "hit_rate": rate("hit_at_5"),
        "mrr": mean([r["rr"] for r in results]),
        "routing_accuracy": rate("route_ok"),
        "routing_sql": rate("route_sql_ok"),
        "routing_retrieval": rate("route_ret_ok"),
        "routing_validation": rate("route_val_ok"),
        "groundedness": mean([r["cit_support"] for r in results]),
        "faithfulness": mean([r["faith"] for r in results]),
        "answer_relevance": rate("ans_ok"),
        "abstention_correctness": rate("abst_ok"),
        "native_pgvector": rate("vector_path", lambda v: v == "pgvector"),
        "native_fts": rate("keyword_path", lambda v: v == "postgres-fts"),
        "degraded_rate": rate("degraded"),
        "errors": sum(1 for r in results if r.get("error")),
        "latency_ms": {"mean": round(sum(latencies) / len(latencies), 1) if latencies else 0,
                       "max": round(max(latencies), 1) if latencies else 0},
        "by_category": by_cat,
        "limitations": [
            "Expected values come from the recovered JSONs (model-transcribed), "
            "not independent human labels — answer_relevance is agreement with "
            "prior extraction, not certified truth.",
            "Numeric-aware matching: 108.00 == 108 counts as relevant.",
            "Groundedness reuses the production claim validator (heuristic); "
            "wrong-source/period/currency cases are covered by dedicated tests.",
            "LLM answers cost quota; degraded providers are recorded per question.",
        ],
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "results"}
    print(json.dumps(summary, indent=2))
    await engine.dispose()
    return report


if __name__ == "__main__":
    report = asyncio.run(main())
    sys.exit(0 if "error" not in report else 1)
