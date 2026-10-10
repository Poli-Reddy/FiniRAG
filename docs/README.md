# FiniRAG Installation and Submission Guide

## 1. Solution Overview

FiniRAG is an AI-powered financial document extraction and validation platform.
It accepts invoices, balance sheets, profit and loss statements, and cash flow
statements as PDF, JPG, or PNG files. The system validates the file, extracts
text with OCR or native PDF processing, uses structured LLM extraction,
normalizes values, performs deterministic financial checks, stores the result
in PostgreSQL, and displays it through the dashboard and REST API.

The processing flow is:

```text
Upload
  -> File validation
  -> Native text or OCR
  -> Structured AI extraction
  -> Normalization and evidence
  -> Deterministic financial validation
  -> PostgreSQL persistence
  -> Dashboard and REST response
```

## 2. Architecture

```text
User
 |
 v
FastAPI-served HTML/CSS/JavaScript frontend (same service, no React/Vercel)
 |
 v
POST /api/v1/documents/process
 |
 v
Document validation
 |
 +--> Native PDF text extraction
 |
 +--> Mistral OCR for scanned/image documents
 |
 v
Gemini structured extraction
 |
 v
Groq fallback extraction
 |
 v
Normalization + evidence
 |
 v
Decimal financial validation
 |
 v
PostgreSQL JSONB persistence (documents)
 |
 +--> Indexing jobs -> chunks -> gemini-embedding-2 -> pgvector + FTS
 |
 +--> POST /api/v1/qa/ask (router -> SQL/retrieval/validation -> evidence -> answer)
 |
 +--> Dashboard history + Q&A
 +--> GET document result API
 +--> Swagger/OpenAPI
```

The frontend and backend are hosted by the same Render web service. A separate
frontend service is not required. NOTE: `docs/architecture.png` is stale
(mentions React/Vite/Vercel/S3); the shipped system is FastAPI + PostgreSQL +
pgvector on a single Render service. Regenerate the diagram from this section.

Place the final visual architecture diagram in this directory as:

- `architecture.png`, or
- `architecture.pdf`

## 3. Technology Stack

- Python 3.12.10
- FastAPI 0.115.0
- Uvicorn 0.30.6
- SQLAlchemy 2.0.35 with async PostgreSQL support
- PostgreSQL with JSONB result storage + pgvector (`vector` extension) + FTS
- Alembic (migrations `0000_documents` baseline + `0001_rag`)
- PyMuPDF 1.24.10
- Pillow 10.4.0
- Mistral OCR SDK 2.10.0
- Google Generative AI SDK 0.8.2 (extraction/answers only)
- google-genai SDK (embeddings: `gemini-embedding-2`, 768d via output_dimensionality)
- Groq SDK 1.7.0
- Pydantic 2.11.2
- HTML, CSS, and browser JavaScript frontend
- Render web service and Render PostgreSQL

## 4. Why These Technologies Were Selected

- **FastAPI** provides typed request handling, multipart upload support, async
  service integration, and automatic Swagger/OpenAPI documentation.
- **HTML/CSS/JavaScript** keeps the frontend simple, deployable with the API,
  and aligned with the assignment, which does not require React or Node.js.
- **PostgreSQL** provides durable hosted persistence and JSONB for flexible
  document-specific fields, tables, periods, and evidence.
- **Mistral OCR** provides OCR text plus layout blocks and page coordinates for
  scanned documents and evidence support.
- **Gemini** is the primary structured extraction model.
- **Groq** is the fallback provider for provider quota or transport failures.
- **PyMuPDF** handles PDF validation, page counts, rendering, and native text
  probing.
- **Pydantic** provides consistent machine-readable schemas and API contracts.
- **Decimal arithmetic** prevents floating-point errors in financial checks.

## 5. Local Setup

Requirements:

- Python 3.12+
- PostgreSQL
- Gemini, Groq, and Mistral provider credentials

From the project root:

```powershell
python -m venv venv
venv/Scripts/python.exe -m pip install -r backend/requirements.txt
Copy-Item .env.example .env
```

Set the values in `.env`, then run:

```powershell
venv/Scripts/python.exe -m uvicorn backend.app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/api/v1/health
```

Run the automated tests:

```powershell
venv/Scripts/python.exe -m pytest backend/tests -q
```

The normal test run excludes live PostgreSQL integration tests. Those tests
can be run with the integration marker when the database is available.

## 6. Environment Variables

Use `.env.example` as the template. Never commit `.env` or real credentials.

Required production variables:

```env
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/database
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
MISTRAL_API_KEY=your_mistral_key
OCR_PROVIDER=mistral
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIM=768
EMBEDDING_BATCH_SIZE=8
INDEX_VERSION=v1
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
MAX_PAGES=3
```

A model or dimension change requires full re-embedding (embedding spaces are
incompatible). Hash embeddings (`hash-768-OFFLINE`) are test-only and never
production semantic search.

Render supplies `DATABASE_URL` from the Render PostgreSQL database. Provider
keys are entered through the Render environment settings and are not stored
in GitHub.

## 7. Frontend URL

After Render deployment:

```text
https://YOUR-APP.onrender.com/
```

The frontend is served directly by FastAPI from `frontend/templates` and
`frontend/static`.

## 8. Backend URL

The backend base URL is the same Render service:

```text
https://YOUR-APP.onrender.com
```

## 9. Swagger URL

FastAPI automatically serves Swagger UI at:

```text
https://YOUR-APP.onrender.com/docs
```

## 10. API Examples

### Process a document

```bash
curl -X POST https://YOUR-APP.onrender.com/api/v1/documents/process \
  -F "file=@Dataset/Invoices/batch3-1354.jpg" \
  -F "document_type=invoice"
```

Supported `document_type` values:

```text
invoice
balance_sheet
profit_and_loss
cash_flow_statement
```

### List persisted documents

```text
GET /api/v1/documents?limit=100&offset=0
```

### Retrieve the latest result by document name

```text
GET /api/v1/documents/Consolidated%20Balance%20Sheet%202026.pdf
```

If the same document name is processed multiple times, the newest attempt is
returned. Older attempts remain available in the database-backed history.

### Health check

```text
GET /api/v1/health
```

Healthy response includes readiness (no secrets):

```json
{
  "status": "healthy",
  "database": "connected",
  "timestamp": "...",
  "embedding_model": "gemini-embedding-2",
  "embedding_configured": true,
  "retrieval_mode": "hybrid",
  "index_version": "v1",
  "indexed_chunks": 128,
  "indexed_documents": 12,
  "provider_breakers": [{"provider": "gemini", "state": "closed", "failures": 0}]
}
```

Without a Gemini key, `retrieval_mode` is `keyword-only` (degraded semantic).

### Operational metrics

`GET /metrics` exposes a dependency-free Prometheus endpoint (also
`?format=json`). Series, all with low-cardinality labels and no document
content, amounts, or secrets:

| Area | Series | What to watch |
|---|---|---|
| HTTP | `finirag_http_requests_total{method,route,status}`, `finirag_http_request_duration_seconds` | error rate and p95 latency per route template |
| Retrieval | `finirag_retrieval_requests_total{vector_path,keyword_path,degraded}`, `finirag_retrieval_duration_seconds{stage}`, `finirag_retrieval_candidates{stage}`, `finirag_retrieval_empty_total{reason}` | degraded share, empty-result rate, stage latency |
| Embeddings | `finirag_embedding_requests_total{kind,model,outcome}`, `finirag_embedding_duration_seconds`, `finirag_embedding_batch_size` | error rate by kind, batch sizes |
| Answers | `finirag_answer_requests_total{provider,outcome}`, `finirag_answer_duration_seconds{provider}`, `finirag_answer_fallbacks_total{from,to}`, `finirag_llm_tokens_total{provider,kind,direction}` | provider mix, fallback rate, token spend and latency |
| Grounding | `finirag_claims_dropped_total`, `finirag_claims_repaired_total`, `finirag_citations_emitted_total` | unsupported-claim pressure, abstention drivers |
| Indexing | `finirag_indexing_jobs_total{status,error?}`, `finirag_indexing_chunks_total{document_type}`, `finirag_indexing_duration_seconds`, `finirag_indexing_pending_jobs` (gauge) | failure rate by error code, backlog growth |
| Documents | `finirag_documents_processed_total{document_type,processing_status,...}`, `finirag_document_processing_duration_seconds` | pass/fail mix, processing latency |

Alert runbook (starting thresholds, tune from a week of data):

- `finirag_retrieval_requests_total{degraded="true"}` share > 5% over 15m →
  a native path is failing; check `native_errors` in QA responses and the
  provider breakers on `/health`.
- `finirag_indexing_pending_jobs` growing monotonically → worker/jobs stuck;
  check `indexing_jobs` error codes via `GET /api/v1/indexing/jobs/{id}`.
- `finirag_answer_fallbacks_total` spike or a breaker `state="open"` →
  provider outage or quota; Groq serves answers until Gemini recovers.
- `finirag_llm_tokens_total` sudden jump → prompt/evidence bloat; check
  `QA_MAX_CONTEXT_CHUNKS` and chunk sizes.
- `finirag_claims_dropped_total` rising with stable retrieval → evidence
  quality or prompt drift; sample answers against `rag_live_evaluation.py`.

Annotate every change to models, prompts, chunking, or retrieval limits with
a timestamped marker so a metric shift can be attributed to a deploy.

### Q&A

```bash
curl -X POST https://YOUR-APP.onrender.com/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the total amount?","document_name":"batch3-1354.jpg"}'
```

### Indexing

```text
GET /api/v1/documents/{document_id}/index-status
POST /api/v1/documents/{document_id}/reindex
GET /api/v1/indexing/jobs/{job_id}
```

No auth is implemented; the expensive endpoints carry a per-client rate
limit (429 `RATE_LIMITED` with `Retry-After`) and every response carries
`X-Request-ID` for log correlation. Do not describe the public endpoints as
production-secure.

### Controlled error response

```json
{
  "error": {
    "code": "UNSUPPORTED_FILE_TYPE",
    "message": "Only PDF / JPG / PNG documents are supported."
  }
}
```

## 11. OCR Approach

The service first validates the upload and probes PDFs for usable native text.
If native text is unavailable, PDF pages are rendered and image documents are
preprocessed with EXIF orientation correction. Mistral OCR then extracts page
text and layout blocks.

The OCR path supports:

- Native PDFs
- Scanned PDFs
- JPEG invoices and receipts
- PNG documents
- Multi-page documents up to three pages
- Rotated images with EXIF orientation metadata
- Parenthesized financial values and OCR-noisy layouts

OCR output is treated as source material, not as unquestioned truth. The LLM
extracts structured fields, while deterministic validation checks arithmetic.

## 12. LLM / Model Used

- **Primary:** Gemini using `GEMINI_API_KEY` (model via `GEMINI_MODEL`)
- **Fallback:** Groq using `GROQ_API_KEY` — used for both extraction and
  answer generation when Gemini refuses (quota/transport)
- **OCR:** Mistral OCR using `MISTRAL_API_KEY`

The LLM returns structured document-type-specific JSON. It is instructed to:

- Extract all visible meaningful fields
- Preserve original values
- Preserve comparative periods
- Preserve line items and tables
- Return null for genuinely missing data
- Avoid inventing or silently correcting unreadable values
- Include evidence source text and page numbers where available

The LLM does not perform trusted financial arithmetic. Python Decimal
validation performs the financial checks after extraction.

## 13. Financial Validation Logic

### Invoice

- Quantity × unit price, adjusted for an explicit line discount, compared with
  line amount
- Sum of line amounts compared with subtotal or total where applicable
- Tax-inclusive and tax-exclusive interpretations supported
- Subtotal + tax compared with total when appropriate
- Cash paid − total compared with change
- Signed discounts handled by magnitude so a printed `-5.59` discount reduces
  the line amount by `5.59`

### Balance Sheet

- Total assets compared with total capital and liabilities
- Asset component sums compared with total assets
- Capital/liability component sums compared with total capital and liabilities
- Checks performed independently per extracted period

### Profit and Loss

- Interest earned + other income compared with total income
- Interest expended + operating expenses + provisions compared with total
  expenditure
- Total income − total expenditure compared with profit before minority
- Minority interest and associates handled where present
- Appropriation transfers reconciled with total available profit
- Carried-forward residual rows excluded from the transfer sum and disclosed

### Cash Flow

- Operating + investing + financing + FX compared with net increase
- Opening cash + net increase + applicable amalgamation adjustment compared with
  closing cash
- Parenthesized values normalized as negative
- Multi-page statement rows preserved
- Intermediate subtotal rows excluded from component sums and disclosed

Missing operands produce `NOT_APPLICABLE`, not an invented zero.

## 14. Numerical Tolerance

All financial checks use Python `Decimal` and:

```text
max(0.01, abs(reported_value) * 0.001)
```

This provides a one-cent absolute floor and a 0.1% relative tolerance for
rounding differences at larger scales. Every check includes its formula,
operands, calculated value, reported value, variance, tolerance, and status.

## 15. Database / Persistence

Render PostgreSQL stores one row per processing attempt in the `documents`
table. JSONB columns hold:

- File validation
- Complete extracted data
- Validation checks
- Processing metadata
- Error information

The database also stores document name, type, processing status, UUID, and
timestamp. The dashboard loads persisted records through
`GET /api/v1/documents`, and clicking a history row loads the latest complete
result by name.

The evaluation harness intentionally bypasses persistence to measure the
pipeline without polluting dashboard history. The measured results can be
seeded with:

```powershell
venv/Scripts/python.exe scripts/seed_evaluation_results.py
```

## 16. Known Limitations

- Evidence distinguishes verbatim OCR snippets (`evidence_kind=verbatim`)
  from generated summaries (`derived`). Bounding-box grounding is unreliable
  and not claimed. `source_text=""` indexing passes no narrative payload, so
  structured chunks are derived-only unless extraction evidence carries text.
- Hash embeddings are offline-only; without `GEMINI_API_KEY`, semantic search
  is degraded to keyword retrieval (`retrieval_mode=keyword-only`).
- SQLite tests use Python cosine + token overlap; pgvector/HNSW + FTS/GIN are
  exercised only on PostgreSQL (`scripts/verify_retrieval.py`).
- Provider free-tier quotas can affect latency or availability.
- Render free services can sleep and have cold-start latency.
- History filters operate in the browser over the loaded dashboard result set.
- Authentication, authorization, background queues, and distributed quota
  state are not included in this case-study prototype (rate limiting, retry,
  breakers and timeouts are in-process; see §10 "Operational metrics" and
  §17).
- Confidence is intentionally omitted because arbitrary LLM confidence values
  are not a defensible measurement.

## 17. Production Improvements

- Add authentication and authorization (upload/QA endpoints currently rely on
  rate limiting only; see §20 for the current guard rails).
- Store OCR block references and implement verified field-to-block matching.
- Add a background job queue for long-running multi-page processing.
- Add distributed key/quota state (key pools and breakers are in-process).
- Add staging deployment and end-to-end tests against deployed services.
- Use managed secret rotation and separate production/staging databases.

## 18. AI Tools Used

Generative AI assistants were used for codebase exploration, implementation
assistance, debugging, test generation, evaluation analysis, and documentation
drafting. The final pipeline architecture, prompts, financial formulas,
security behavior, deployment configuration, and evaluation results were
reviewed against the assignment and verified with automated tests and the
50-document evaluation dataset.

## 19. FiniRAG 2.0 — Hybrid RAG

Ingestion (`DocumentService`) is unchanged: file gate -> native/OCR ->
Gemini/Groq extraction -> normalization -> Decimal validation -> persist.
After a PASS persist, an indexing job is created for the exact attempt row;
`IndexingService` builds structure-aware chunks (invoice header/line-items,
statement section+period+unit, narrative ~500 tokens/50 overlap),
embeds with `gemini-embedding-2` (768d via output_dimensionality, google-genai
SDK; hash provider is OFFLINE-ONLY), and stores vectors + FTS content in
`document_chunks` (pgvector `vector` on Postgres, Python fallback on sqlite).
Failed indexing never fails the document. Model/dim change => re-embed all.

QA (`POST /api/v1/qa/ask`) routes deterministically
(DOCUMENT_LOOKUP/CONCEPTUAL/EXACT/COMPARISON/RECONCILIATION/DISCREPANCY/
MULTI/INSUFFICIENT): exact values and comparisons use whitelisted SQL over
JSONB + Decimal (MISSING never zero, units must match, ambiguous periods
rejected); reconciliation reads persisted validation checks; conceptual uses
hybrid retrieval (pgvector `<=>` + FTS ts_rank, RRF k=60, limit 8/path,
filters applied pre-retrieval, actual executed paths reported with degraded
flags and strict verification mode) + grounded answer with claim-level
citation enforcement (per-sentence support incl. numbers/periods/currency/
source, unsupported claims removed, abstention when none survive) and
INSUFFICIENT_EVIDENCE abstention. Year parsing returns full
years (`(?:19|20)\d{2}`). Chunk `chunk_metadata` carries document identity
(bare `metadata` kwarg is normalized at the repo boundary — DeclarativeBase
swallows it). Document text is untrusted; embedded instructions
are sanitized.

Tables: baseline `0000_documents` adopts/creates `documents`; `0001_rag`
enables `vector`, creates `document_chunks(vector(768), FTS GIN, HNSW)` +
`indexing_jobs`, FK to documents. Downgrades drop RAG only, never documents.
Worker: inline bounded indexing + startup stale recovery + SKIP LOCKED claims
+ max 5 attempts + savepoint-atomic replacement; manual
`python scripts/run_indexing_worker.py`. `GET /documents/{id}/index-status`,
`POST /documents/{id}/reindex`, `GET /indexing/jobs/{job_id}` expose status.

Setup: `pip install -r backend/requirements.txt` (adds pgvector,
google-genai, alembic, numpy), configure `.env` (see `.env.example`
EMBEDDING_*), `alembic upgrade head`, `uvicorn backend.app.main:app`.
Tests: `pytest backend/tests -q` (unit, mocked providers; an autouse
fixture blanks LLM keys so the unit suite never makes live calls);
`pytest -m integration` for live DB. Offline RAG eval:
`PYTHONPATH=. python evaluation/rag_evaluation.py` (15 questions,
sqlite+hash, forced-offline; see `evaluation/rag_report.json`). Live:
`python evaluation/rag_live_evaluation.py` (real PostgreSQL + live
embeddings, `evaluation/rag_live_report.json`), plus
`scripts/live_embedding_smoke.py` and `scripts/verify_retrieval.py`.

Answer generation uses the same provider chain as extraction: Gemini
(`GEMINI_MODEL`, default `gemini-3.8-flash`) -> Groq (`GROQ_MODEL`) ->
deterministic template. The provider that actually answered is surfaced as
`answer_provider` on `POST /api/v1/qa/ask`; a template fallback adds a
limitation rather than pretending the model answered.

### Reliability engineering

- **Retry with backoff + jitter** (`core/resilience.py`) for every outbound
  provider call: 429/5xx/timeouts are retried (honouring provider
  `Retry-After`); other 4xx fail fast. Rate-limited API keys are retired via
  the shared `KeyPool`, so a retry lands on a different key.
- **Circuit breakers** per provider: after N consecutive failures the provider
  is skipped until a cooldown half-open trial, so a dead provider cannot make
  every request pay the full timeout. Breaker state is exposed on `/health`.
- **Hard timeouts**: per-call `PROVIDER_TIMEOUT_S`, a 15 s Postgres
  `statement_timeout`, and a `QA_REQUEST_TIMEOUT_S` budget over the whole
  question-answering pipeline (504 `QA_TIMEOUT` on breach).
- **Rate limiting** on the two expensive endpoints (per-client fixed window,
  `Retry-After` on 429); `X-Request-ID` correlation on every response.
- **Retrieval integrity**: `hnsw.ef_search` raised per session for recall,
  deterministic tie-breaking on equal distances, and a bounded Python
  fallback (`RETRIEVAL_FALLBACK_MAX_ROWS`) so the vector path can never sweep
  an unbounded corpus in Python.
- **Database-level guarantees** (migration `0002_rag_hardening`): HNSW rebuilt
  with explicit `m=16, ef_construction=64`, plus a partial unique index
  (`uq_chunks_active_hash`) so two concurrent indexing attempts cannot leave
  duplicate active chunks.
- **Citation repair instead of reflexive abstention**: an uncited factual
  claim is kept only if a real evidence item supports it (id attached);
  otherwise it is dropped. Fabricated ids are still rejected.
- **Health readiness** (`/health`): DB, index counts, embedding configuration,
  retrieval mode and breaker states — no secrets.

### Verified live (2026-10-09, Render PostgreSQL + real providers)

- Migrations `0000_documents` -> `0001_rag` -> `0002_rag_hardening` applied to
  the live Render database; the pre-existing 54 `documents` rows were
  preserved throughout; `vector`/`pgcrypto` installed;
  `document_chunks(embedding vector(768))`, `idx_chunks_fts` (GIN),
  `idx_chunks_vec` (HNSW `m=16, ef_construction=64`) and
  `uq_chunks_active_hash` present.
- `scripts/live_embedding_smoke.py`: real API call, model `gemini-embedding-2`,
  returned dimension 768, finite values.
- `scripts/verify_retrieval.py --seed 1500 --strict` (synthetic rows, ANALYZE,
  then rolled back): `Index Scan using idx_chunks_vec` with
  `Order By (embedding <=> ...)` and `Bitmap Index Scan on idx_chunks_fts` —
  both native indexes selected on a representative table; strict exit 0.
- `evaluation/rag_live_evaluation.py` (15Q, synthetic 3-doc corpus, rolled
  back): 15/15 questions used native pgvector + postgres-fts; recall@5 1.0,
  MRR 0.958, claim_support 1.0, unsupported-claim rate 0.0, faithfulness
  0.946, abstention 0.867, financial-query 1.0, validation 1.0, filter 1.0,
  answer correctness 0.867. Answers were served by Groq because the Gemini
  free tier (20 generate requests/day/model) was exhausted; one call fell to
  the template when both providers refused. Production rows untouched
  (54 documents, 0 synthetic chunks after every run).
- Answer correctness and abstention vary run to run with model behaviour
  (observed 0.667-0.867 on this small synthetic set); retrieval metrics are
  stable across runs.

Offline eval (actual, sqlite/hash, 15Q, forced offline): recall@5 1.0,
MRR 0.892 (final fused positions), claim_support 1.0, answer 0.8,
faithfulness 0.70, abstention 0.8, financial-query 1.0, validation 1.0,
filter 1.0, latency ~13 ms mean. Token overlap is a documented basic guard,
not entailment proof.

Honest open items: the Gemini generate quota forces Groq for answers until the
key's quota resets or billing is enabled (rotate `GEMINI_API_KEYS` or add
billing for Gemini answers); the live eval corpus is small and synthetic, so
it proves plumbing and index behaviour rather than production answer quality;
`docs/architecture.png` is still the stale React/Vercel drawing. Full suite:
444 passed; 1 pre-existing failure (`test_invalid_document_type...[""]`
expects 400, gets 422 on installed fastapi 0.136.3 vs pinned 0.115.0 — fails
identically on the clean tree).

## Submission Files

Place the final submission artifacts in this directory:

- `architecture.png` or `architecture.pdf`
- `solution_presentation.pdf`
- Optional editable source: `solution_presentation.pptx`

The application runtime does not read these files. They are repository and
submission documentation only.
