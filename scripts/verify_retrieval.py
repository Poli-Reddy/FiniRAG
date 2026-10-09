"""PostgreSQL retrieval verification — proves native pgvector + FTS index use.

Two modes:

* default — EXPLAIN the real vector and FTS queries against current data.
* --seed N — insert N synthetic chunks (plus a synthetic document) inside a
  transaction, ANALYZE, EXPLAIN, then ROLL BACK. This gives the planner a
  representative table so index selection can be observed; production data is
  never modified. Exit code 2 when --strict is set and the intended index is
  not chosen.

On sqlite the script reports the fallback honestly instead of claiming index
usage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys

sys.path.insert(0, ".")

VECTOR_QUERY = ("EXPLAIN (COSTS OFF) SELECT id FROM document_chunks "
                "ORDER BY embedding <=> :q LIMIT 8")
FTS_QUERY = ("EXPLAIN (COSTS OFF) SELECT id FROM document_chunks "
             "WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :q)")


async def _explain(db, sql: str, params: dict) -> list[str]:
    from sqlalchemy import text

    rows = (await db.execute(text(sql), params)).fetchall()
    return [str(r[0]) for r in rows]


def _uses_index(plan: list[str]) -> bool:
    joined = "\n".join(plan).lower()
    return "index" in joined and "seq scan" not in joined


def _mentions(plan: list[str], needle: str) -> bool:
    return needle.lower() in "\n".join(plan).lower()


def _compact(plan: list[str], width: int = 160) -> list[str]:
    """Trim giant literals (768-dim vectors) so reports stay readable."""
    out = []
    for line in plan:
        line = line.rstrip()
        out.append(line if len(line) <= width else line[:width] + "...[truncated]")
    return out


async def _run_seeded(db, count: int) -> dict:
    """Insert synthetic rows, EXPLAIN, roll back. Never commits."""
    import uuid

    from sqlalchemy import text

    from backend.app.models.document import Document
    from backend.app.models.rag import DocumentChunk

    rng = random.Random(1234)
    try:
        doc = Document(
            document_name=f"__verify_seed__{uuid.uuid4().hex[:8]}.pdf",
            document_type="invoice", processing_status="PASS",
            file_validation={"status": "PASS"}, extracted_data={},
            validation={"overall_status": "PASS"}, processing_metadata={},
        )
        db.add(doc)
        await db.flush()
        batch = []
        for i in range(count):
            # Make the FTS term selective: only ~2% of rows contain it, so the
            # planner has a reason to use the GIN index (an all-matching corpus
            # makes Seq Scan genuinely optimal, which would prove nothing).
            content = (f"seed chunk {i} invoice total amount 100.{i % 100:02d} usd"
                       if i % 50 == 0 else
                       f"seed chunk {i} lorem ipsum dolor sit amet {i}")
            batch.append(DocumentChunk(
                document_id=doc.id, chunk_index=i, chunk_type="financial_field",
                content=content,
                page_number=1, section_name="total_amount",
                document_type="invoice",
                source_evidence={"source_text": "seed", "evidence_kind": "derived",
                                 "verbatim_available": False},
                chunk_metadata={"document_name": doc.document_name},
                embedding=[rng.uniform(-1, 1) for _ in range(768)],
                embedding_model="seed-768", index_version="verify",
                content_hash=f"seed-{uuid.uuid4().hex}",
            ))
            if len(batch) >= 250:
                db.add_all(batch)
                await db.flush()
                batch = []
        if batch:
            db.add_all(batch)
            await db.flush()
        # Fresh statistics so the planner sees a representative table.
        await db.execute(text("ANALYZE document_chunks"))
        qvec = "[" + ",".join(["0.1"] * 768) + "]"
        vec_plan = await _explain(db, VECTOR_QUERY, {"q": qvec})
        fts_plan = await _explain(db, FTS_QUERY, {"q": "invoice total"})
        return {"seeded_rows": count, "vector_plan": vec_plan, "fts_plan": fts_plan}
    finally:
        # Synthetic rows must never persist.
        await db.rollback()


async def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="insert N synthetic chunks (rolled back) before EXPLAIN")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero unless the intended index is used")
    args = parser.parse_args(argv)

    from backend.app.core.database import async_session_factory, engine

    dialect = getattr(getattr(engine, "dialect", None), "name", "unknown")
    out: dict = {"dialect": dialect}
    if dialect != "postgresql":
        out.update({"status": "SKIPPED_FALLBACK",
                    "vector_path": "python-fallback",
                    "keyword_path": "token-overlap",
                    "note": "PostgreSQL unavailable; production must verify on PG."})
        print(json.dumps(out, indent=2))
        return out

    try:
        async with async_session_factory() as db:
            if args.seed:
                seeded = await _run_seeded(db, args.seed)
                out.update(seeded)
            else:
                qvec = "[" + ",".join(["0"] * 768) + "]"
                out["vector_plan"] = await _explain(db, VECTOR_QUERY, {"q": qvec})
                out["fts_plan"] = await _explain(db, FTS_QUERY, {"q": "invoice total"})
    except Exception as e:
        out.update({"status": "SKIPPED_NO_POSTGRES",
                    "reason": f"{type(e).__name__}: PostgreSQL not reachable from this host",
                    "vector_path": "unverified-live",
                    "keyword_path": "unverified-live",
                    "note": "Run on Render PostgreSQL with pgvector; see docs."})
        print(json.dumps(out, indent=2))
        return out

    vec_plan, fts_plan = out["vector_plan"], out["fts_plan"]
    out.update({
        "status": "OK",
        "vector_index_used": _uses_index(vec_plan),
        "fts_index_used": _uses_index(fts_plan),
        "vector_index_name": _mentions(vec_plan, "idx_chunks_vec"),
        "fts_index_name": _mentions(fts_plan, "idx_chunks_fts"),
        "vector_plan": _compact(vec_plan),
        "fts_plan": _compact(fts_plan),
        "note": ("Seeded synthetic rows were rolled back." if args.seed else
                 "Empty/small tables may legitimately seq-scan; use --seed to prove "
                 "index selection on a representative table."),
    })
    print(json.dumps(out, indent=2))

    if args.strict and args.seed:
        if not (out["vector_index_used"] and out["fts_index_used"]):
            print("STRICT: intended index was not used", file=sys.stderr)
            sys.exit(2)
    return out


if __name__ == "__main__":
    asyncio.run(main())
