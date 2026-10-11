#!/usr/bin/env python
"""End-to-end connectivity smoke check: frontend + backend + database.

Exercises the REAL app object (lifespan included: migrations + stale-job
recovery) through HTTP-level calls, proving:

- frontend HTML + static JS are served (dashboard reachable)
- backend boots against the configured DATABASE_URL
- health reports the database connected with native-path readiness
- document listing reads live rows from PostgreSQL
- the QA route answers through the real pipeline (an unsupported
  question abstains WITHOUT spending LLM quota)

Usage:
    python scripts/e2e_smoke_check.py [--api-url http://localhost:8000]

Without --api-url the checks run in-process via Starlette TestClient
(same routes, same middleware, same lifespan). With --api-url they run
over HTTP against a deployed instance.

The QA probe deliberately uses an unsupported question so no LLM call
is made — wiring proof, not quota spend.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def run_local() -> bool:
    from fastapi.testclient import TestClient

    from backend.app.main import app

    ok = True
    with TestClient(app) as client:
        r = client.get("/")
        ok &= _check("frontend HTML served", r.status_code == 200,
                     f"HTTP {r.status_code}, {len(r.text)} bytes")
        ok &= _check("dashboard markup present",
                     "uploadForm" in r.text and "historyTable" in r.text)
        ok &= _check("no key entry in customer UI",
                     "apiKeyInput" not in r.text and "api-key-box" not in r.text)
        ok &= _check("question-only Q&A (no filter inputs)",
                     "qaSamplesList" in r.text and "qaDocName" not in r.text)

        r = client.get("/static/js/app.js")
        ok &= _check("frontend JS served", r.status_code == 200,
                     f"HTTP {r.status_code}, {len(r.text)} bytes")
        ok &= _check("apiFetch wrapper shipped", "apiFetch" in r.text)

        r = client.get("/api/v1/health")
        body = r.json()
        ok &= _check("health 200", r.status_code == 200)
        ok &= _check("database connected", body.get("database") == "connected",
                     str(body.get("database")))
        ok &= _check("native paths ready",
                     body.get("native_paths_ready") is True,
                     f"vector={body.get('vector_ready')} "
                     f"fts={body.get('fts_ready')}")
        ok &= _check("retrieval mode hybrid", body.get("retrieval_mode") == "hybrid")
        ok &= _check("chunks indexed", (body.get("indexed_chunks") or 0) > 0,
                     str(body.get("indexed_chunks")))
        print(f"  info: indexed_documents={body.get('indexed_documents')} "
              f"model={body.get('embedding_model')}")
        # Production must never boot open: ENVIRONMENT=production with no
        # API_KEYS raises in lifespan, so reaching here means the gate or
        # the environment is sane for this instance.
        print(f"  info: environment gate ok (booted, health reachable)")

        r = client.get("/api/v1/documents?limit=2&offset=0")
        if r.status_code == 401:
            print("  info: API_KEYS gate is ON (anonymous list rejected as designed)")
        else:
            ok &= _check("document listing reads PG", r.status_code == 200,
                         f"HTTP {r.status_code}")
            if r.status_code == 200:
                payload = r.json()
                items = payload.get("documents", payload.get("items", []))
                ok &= _check("live rows returned", len(items) > 0,
                             f"{len(items)} row(s)")
                # Processing History must receive the full corpus, not a
                # truncated page: the UI fetches limit=1000.
                full = client.get("/api/v1/documents?limit=1000&offset=0").json()
                ok &= _check("history gets every processed document",
                             full.get("total", 0) >= 50
                             and len(full.get("documents", [])) == full.get("total", -1),
                             f"total={full.get('total')}")

        r = client.post("/api/v1/qa/ask",
                        json={"question": "Should I buy this stock?"})
        if r.status_code == 401:
            print("  info: API_KEYS gate is ON (anonymous QA rejected as designed)")
        else:
            ok &= _check("QA route answers", r.status_code == 200,
                         f"HTTP {r.status_code}")
            if r.status_code == 200:
                body = r.json()
                ok &= _check("unsupported abstains w/o LLM spend",
                             body.get("status") == "INSUFFICIENT_EVIDENCE",
                             str(body.get("status")))

        r = client.get("/api/v1/qa/samples")
        if r.status_code == 401:
            print("  info: API_KEYS gate is ON (anonymous samples rejected as designed)")
        else:
            ok &= _check("sample questions served", r.status_code == 200,
                         f"HTTP {r.status_code}")
            if r.status_code == 200:
                samples = r.json().get("samples", [])
                ok &= _check("samples without ground truth",
                             len(samples) > 50 and all(
                                 set(s) == {"id", "question", "category"}
                                 for s in samples[:20]),
                             f"{len(samples)} samples")
    return ok


def run_remote(base: str) -> bool:
    import httpx

    ok = True
    with httpx.Client(base_url=base.rstrip("/"), timeout=30.0) as client:
        r = client.get("/")
        ok &= _check("frontend HTML served", r.status_code == 200)
        r = client.get("/api/v1/health")
        ok &= _check("health 200", r.status_code == 200)
        if r.status_code == 200:
            ok &= _check("database connected",
                         r.json().get("database") == "connected")
        r = client.get("/api/v1/documents?limit=1&offset=0")
        ok &= _check("documents reachable", r.status_code in (200, 401),
                     f"HTTP {r.status_code}")
        r = client.post("/api/v1/qa/ask",
                        json={"question": "Should I buy this stock?"})
        ok &= _check("QA reachable", r.status_code in (200, 401),
                     f"HTTP {r.status_code}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2E frontend+backend+database smoke check")
    parser.add_argument("--api-url", default=None,
                        help="Remote base URL (else in-process TestClient)")
    args = parser.parse_args(argv)

    print("=== E2E SMOKE CHECK "
          f"({'remote ' + args.api_url if args.api_url else 'in-process'}) ===")
    ok = run_remote(args.api_url) if args.api_url else run_local()
    print(f"\n=== {'[OK] ALL CONNECTED' if ok else '[FAIL] CONNECTIVITY GAPS'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
