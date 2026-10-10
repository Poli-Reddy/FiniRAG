# FiniRAG Implementation Record

Complete record of everything designed, built, fixed, and verified from the
pre-RAG baseline (`68b8079`) to the current state. Compiled 2026-10-10 from
the actual code, tests, migration output, and live verification runs — not
from memory. The user-facing guide remains `docs/README.md`; this file is the
engineering history: what was done, why, what broke, and what was proven.

Commits: `7551d06` (RAG 2.0 + audit fixes) → `0e46a00` (hardening) →
`0ae3dda` (metrics + guards). 62 files, +7748/−37 lines.

---

## 1. Starting point

Baseline `68b8079` ("Pin Render to Python 3.12") was a document-intelligence
app with **no RAG**: FastAPI + Mistral OCR + Gemini/Groq extraction +
normalization + Decimal validation + PostgreSQL persistence + a dashboard.
The RAG work was built in four phases below.

---

## 2. Phase 1 — Hybrid RAG build (commit `7551d06`, first half)

Goal: genuine end-to-end RAG (index → hybrid retrieve → grounded answer),
not diagrams or a decorative chatbot. Hard constraints kept throughout:
separate ingestion/indexing/QA pipelines sharing persisted records; never
re-run OCR per question; deterministic validation never replaced by the LLM;
no LangChain/LlamaIndex/second database/agent framework.

### New modules (all under existing naming/import conventions)

| File | Responsibility |
|---|---|
| `backend/app/models/rag.py` | `DocumentChunk` (chunk text, metadata, `VECTOR(768)`, evidence, hashes) and `IndexingJob` (PENDING→PROCESSING→COMPLETED/FAILED, attempts, errors) |
| `backend/app/providers/embedding_provider.py` | `GeminiEmbeddingProvider` (`gemini-embedding-2`, 768d via `output_dimensionality`, `google-genai` SDK only) + offline-only `HashEmbeddingProvider` |
| `backend/app/services/chunking_service.py` | Structure-aware chunking: invoice header/line-items/tables; statement section+period+unit; narrative ~500 tokens with ~50 overlap; deterministic `content_hash`; verbatim-vs-derived provenance |
| `backend/app/services/embedding_service.py` | Bounded batches, reuse keyed `hash:model:dim` (embedding spaces are incompatible across models) |
| `backend/app/services/indexing_service.py` | Idempotent jobs, savepoint-atomic chunk replacement, stale recovery, max 5 attempts |
| `backend/app/services/retrieval_service.py` | pgvector `<=>` + FTS `ts_rank`, RRF k=60, pre-retrieval filters, reported execution paths, `strict=` verification mode |
| `backend/app/services/question_router.py` | Deterministic intent classification (lookup/conceptual/exact/comparison/reconciliation/discrepancy/multi/insufficient) |
| `backend/app/services/financial_query_service.py` | Whitelisted, parameterized SQL over JSONB with Decimal math; MISSING≠zero; unit/currency/period guards |
| `backend/app/services/evidence_service.py` | Single evidence package with stable `CIT-n` ids; contradictions preserved; failed checks stay visible |
| `backend/app/services/answer_service.py` | Gemini→Groq→template chain with `answer_provider` surfaced; claim-level citation validation; `INSUFFICIENT_EVIDENCE` abstention |
| `backend/app/repositories/rag_repository.py` | Chunk/job persistence with `SKIP LOCKED` claims |
| `backend/app/workers/indexing_worker.py` | Bounded worker + startup stale recovery; manual `scripts/run_indexing_worker.py` |
| `backend/app/api/routes/qa.py` | `POST /api/v1/qa/ask` with whole-pipeline `QA_REQUEST_TIMEOUT_S` budget (504 `QA_TIMEOUT`) |
| `backend/app/schemas/qa.py` | QA/indexing-job contracts |

### Extended (not rewritten)

- `services/document_service.py`: after a PASS persist, creates an indexing
  job and processes it inline; indexing failure never fails the document.
- `api/routes/documents.py`: `index-status`, `reindex`, job-status endpoints;
  richer `/health` (index counts, embedding config, breaker states).
- `frontend/`: Q&A section (question, filters, answer, sources, facts,
  validation, retrieval metadata) in existing styling.
- `render.yaml`, `.env.example`, `requirements.txt` (+pgvector, google-genai,
  alembic, numpy).

### Migrations

- `0000_documents`: adopts an existing `documents` table if present (data
  preserved), else creates it with JSONB columns matching the ORM exactly.
- `0001_rag`: enables the **`vector`** extension (correct name, not
  `pgvector`), creates `document_chunks(VECTOR(768))` + `indexing_jobs` with
  FKs, GIN FTS index, HNSW index. Fails loudly if the extension is missing.
  Downgrades drop RAG objects only — never `documents`.

---

## 3. Phase 2 — Audit fixes (commit `7551d06`, second half)

Each item below was a demonstrated defect or verification gap, fixed in code
and covered by a test — not papered over in prose.

1. **Retired embedding model.** Code configured `text-embedding-004` (shut
   down 2026-01-14). Migrated to **`gemini-embedding-2`** (GA Apr 2026):
   768 dims via `output_dimensionality` (auto-normalized MRL truncation),
   documents formatted `title: … | text: …`, queries
   `task: question answering | query: …` (embedding-2 has no `task_type`
   parameter). Legacy `google-generativeai` embedding fallback removed;
   hash embeddings are explicitly offline-only.
2. **Year-extraction regex bug.** `_guess_periods` used a capturing group, so
   "2024" parsed as "20". Fixed to `(?:19|20)\d{2}` with tests for 1999,
   2024, 2025, two-year and year-less questions.
3. **Worker never ran automatically.** Added bounded inline indexing,
   startup stale-job recovery, `SKIP LOCKED` claims, retry budget, and a
   manual worker script — documented as single-service limits, no paid queue.
4. **Weak provenance.** Chunks now distinguish `verbatim` OCR snippets from
   `derived` generated summaries; fabricated page/coordinate certainty is gone.
5. **Hash must not stand in for production.** Hash provider is opt-in-only;
   production falls back to keyword retrieval with `degraded` status, and
   reports label the backend/model explicitly.
6. **Evaluation methodology.** Recall@k/MRR computed from final fused ranks
   over unfiltered retrieval against ground-truth document sets; citation
   support measured claim-by-claim with the production validator; abstentions
   earn no support/faithfulness credit; offline and live reports are separate
   artifacts (`rag_report.json` vs `rag_live_report.json`).

### Genuine defects found *during* verification (not in the original audit)

- **Swallowed chunk metadata.** Chunk dicts used the bare `"metadata"` key,
  which `DeclarativeBase` silently absorbs (it already defines `.metadata`),
  so every stored chunk lost its `document_name`. Found because a test showed
  empty candidate names. Fix: emit `chunk_metadata`; repo normalizes legacy
  dicts. Test: `test_chunk_metadata_carries_document_identity`.
- **Missing vector comparator.** `DocumentChunk.embedding.cosine_distance`
  does not exist — `TypeDecorator` hides pgvector's comparator — so native
  vector search *always* fell back to Python on every backend. Fix:
  `embedding_cosine_distance()` renders `<=>` directly; compile-verified
  against the PG dialect and proven live (`Index Scan using idx_chunks_vec`).
- **Answer-model quota.** Live answers silently fell back to the template
  because `gemini-3.6-flash` was 429-exhausted (20/day free tier). Fix:
  single `GEMINI_MODEL` knob (now `gemini-3.8-flash`, verified working),
  Gemini→Groq chain for answers mirroring extraction, and `answer_provider`
  surfaced so degradation is visible instead of silent.

---

## 4. Phase 3 — Production hardening (commit `0e46a00`)

- **Retry/backoff+jitter** (`core/resilience.py`): 429/5xx/timeouts retried
  honouring provider `Retry-After`; other 4xx fail fast. Rate-limited keys
  are retired through the shared `KeyPool` so retries land on fresh keys.
- **Circuit breakers** per provider (open → half-open trial), exposed on
  `/health`.
- **Timeouts at every layer**: per-call `PROVIDER_TIMEOUT_S`, Postgres
  `statement_timeout` (15 s), whole-pipeline QA budget (504).
- **Rate limiting** on the two expensive endpoints (429 + `Retry-After`);
  `X-Request-ID` correlation on all responses.
- **Retrieval**: `hnsw.ef_search` raised per session (`SET LOCAL`),
  deterministic tie-breaking, Python fallback row-bounded.
- **Migration `0002_rag_hardening`**: HNSW rebuilt with explicit
  `m=16, ef_construction=64`; partial unique index `uq_chunks_active_hash`
  makes duplicate active chunks impossible even under concurrent indexing.
- **Citation repair**: uncited-but-supported claims keep a repaired citation;
  unsupported claims are dropped; fabricated ids rejected — abstention only
  when nothing survives.

## 5. Phase 4 — Production metrics (commit `0ae3dda`)

Dependency-free Prometheus registry (`core/metrics.py`; no SDK needed):
`GET /metrics` (text, `?format=json` for JSON). Series and their purpose:

| Area | Series | Watch for |
|---|---|---|
| HTTP | `…_http_requests_total{method,route,status}`, `…_http_request_duration_seconds` | error rate, p95 per route template |
| Retrieval | `…_retrieval_requests_total{vector_path,keyword_path,degraded}`, `…_retrieval_duration_seconds{stage}`, `…_retrieval_candidates{stage}`, `…_retrieval_empty_total{reason}` | degraded share, empty rate, stage latency |
| Embeddings | `…_embedding_requests_total{kind,model,outcome}`, `…_embedding_duration_seconds`, `…_embedding_batch_size` | error rate, batch sanity |
| Answers | `…_answer_requests_total{provider,outcome}`, `…_answer_duration_seconds{provider}`, `…_answer_fallbacks_total{from,to}`, `…_llm_tokens_total{provider,kind,direction}` | provider mix, fallback rate, spend |
| Grounding | `…_claims_dropped_total`, `…_claims_repaired_total`, `…_citations_emitted_total` | unsupported-claim pressure |
| Indexing | `…_indexing_jobs_total{status,error?}`, `…_indexing_chunks_total{document_type}`, `…_indexing_duration_seconds`, `…_indexing_pending_jobs` | failure by code, backlog |
| Documents | `…_documents_processed_total{…}`, `…_document_processing_duration_seconds` | pass/fail mix, latency |

Labels are enumerated values and route templates only — never ids, text,
amounts, or secrets. Alert starting points are in `README.md` §10.

---

## 6. Decisions log (why, not just what)

- **pgvector in the existing Postgres, not a vector DB.** One datastore to
  operate/backup; corpus size doesn't justify a second system; FTS GIN and
  HNSW cover both retrieval paths.
- **Hash embeddings offline-only.** Deterministic tests need no network, but
  hash vectors carry no semantics — production degrades to keyword + flag
  rather than pretending.
- **In-process pools/breakers/limiters.** Matches the single-instance Render
  deploy; documented as needing Redis/shared state if scaled out.
- **Template fallback kept.** A degraded-but-grounded answer beats an error
  page, *provided* it is labelled — hence `answer_provider` + limitations.
- **Token overlap is a guard, not proof.** Stated in code, docs, and eval
  limitations; exact values/periods/currencies are additionally cross-checked
  as strings against structured evidence.
- **Small synthetic eval corpus.** Proves plumbing, ranking math, index use
  and citation enforcement — not production answer quality. Stated plainly.

---

## 7. Test map (444 passed; 1 pre-existing failure)

| File | Tests | Proves |
|---|---|---|
| `test_validation.py` (59) | formulas, tolerances | financial correctness (untouched) |
| `test_ocr.py` (39) | OCR paths | extraction inputs (untouched) |
| `test_pipeline.py` (32) | pipeline | ingestion (untouched) |
| `test_rag.py` (39) | 20 RAG scenarios + claim-integrity adversarials + `final_rank` + answer chain | RAG behavior, citation enforcement |
| `test_audit_fixes.py` (19) | years, migration SQL, provenance, worker, PG paths, lifecycle | every audit fix |
| `test_embedding_provider.py` (16) | mocked SDK, exact wire strings, failures | embedding contract |
| `test_metrics.py` (8) | registry format, endpoint, instrumentation | observability |
| `test_resilience.py` (13) | retry classes, backoff, breaker, limiter | reliability primitives |
| `test_reliability_api.py` (4) | health fields, request ids, rate limit | API guards |
| `test_qa_api.py` (7) | contracts | QA API shape |
| others | extraction/normalization/config/api/repo | pre-existing behavior preserved |

Pre-existing failure: `test_invalid_document_type...[""]` expects 400, gets
422 (installed fastapi 0.136.3 vs pinned 0.115.0) — reproduces identically on
the clean tree via `git stash`; test left untouched. An autouse fixture
blanks LLM keys (plus freezes retries/rate limits) so the suite never makes
live calls even with a populated `.env`.

---

## 8. Live verification log (Render PostgreSQL + real keys, 2026-10-09)

All run for real; nothing mocked:

- `alembic upgrade head` → `0000`→`0001`→`0002` applied; **54 pre-existing
  `documents` rows preserved**; `vector`/`pgcrypto` present;
  `document_chunks(embedding vector(768))`, GIN + HNSW(`m=16,ef=64`) +
  unique active-hash index confirmed in `pg_indexes`.
- `live_embedding_smoke.py` → `gemini-embedding-2`, dim 768, finite,
  `real_api_call: true`.
- `verify_retrieval.py --seed 1500 --strict` (ANALYZE, rolled back) →
  `Index Scan using idx_chunks_vec … Order By (embedding <=> …)` and
  `Bitmap Index Scan on idx_chunks_fts`; strict exit 0; 0 synthetic rows left.
- `rag_live_evaluation.py` (15Q, rolled back) → native paths **15/15**,
  recall@5 **1.0**, MRR **0.958**, claim support **1.0**,
  unsupported-claim **0.0**, faithfulness **0.946**, fq **1.0**,
  validation **1.0**, filter **1.0**, answer correctness **0.867**
  (model over-abstains on some answerable questions; varies run to run).
  Answers served by Groq — Gemini generate quota (20/day) was exhausted.
- Offline eval (forced offline): recall 1.0, MRR 0.892, support 1.0,
  answer 0.8, ~13 ms mean.
- Production rows untouched after every run (54 docs, 0 synthetic chunks).

---

## 9. Operations

```powershell
# deploy / migrate
pip install -r backend/requirements.txt
alembic upgrade head
python scripts/render_smoke_check.py
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT

# verify live
python scripts/live_embedding_smoke.py
python scripts/verify_retrieval.py --seed 1500 --strict
$env:PYTHONPATH='.'; python evaluation/rag_live_evaluation.py
curl /metrics ; curl /api/v1/health

# operate
python scripts/run_indexing_worker.py          # drain pending jobs
python scripts/check_db_state.py               # counts, indexes, version
# reindex one document:
#   POST /api/v1/documents/{id}/reindex
# model/dim change => re-embed everything (spaces incompatible)
```

Secrets live only in `.env` (gitignored, verified absent from history and the
remote before every push). Rotate `GEMINI_API_KEYS`/billing when the free
generate quota binds; `GROQ_API_KEY` currently carries answers.

## 10. Open items (honest, not hidden)

1. Gemini generate quota (20/day) forces Groq for answers until reset/billing.
2. Live answer quality (0.667–0.867) varies with model behaviour on a tiny
   synthetic set; retrieval metrics are stable.
3. `docs/architecture.png` is still the stale React/Vercel drawing.
4. No auth (rate limiting only); single-instance state (pools, breakers,
   limiter) needs shared stores to scale out.
5. One pre-existing test failure (FastAPI version drift, §7).
