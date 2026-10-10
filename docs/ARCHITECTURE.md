# FiniRAG Architecture (current implementation)

> `docs/architecture.png` / `architecture.pdf` are **stale**: they describe a
> React/Vite frontend, Vercel hosting, S3 file storage and extra database
> tables that do not exist in this codebase. This file is authoritative until
> those images are redrawn. Do not implement from the PNG.

Single FastAPI service + Render PostgreSQL. No separate frontend service, no
second database, no queue service, no LangChain/LlamaIndex/agent framework.

```mermaid
flowchart TB
    subgraph browser["Browser (same origin, no CORS needed)"]
        UI["Static HTML/CSS/JS dashboard\nfrontend/templates + frontend/static"]
    end

    subgraph app["FastAPI app (backend/app/main.py)"]
        MW["Middleware: X-Request-ID, RED metrics"]
        AUTH["require_api_key (all /api/v1/* + /metrics;\nopen only when API_KEYS is empty)"]
        RL["rate_limit (all document + QA routes)"]
        DOC["documents.py:\nprocess · list · get-by-name\nindex-status · reindex · job-status"]
        QA["qa.py: POST /qa/ask\n(classify → SQL/retrieval/evidence → answer)"]
        HEALTH["public_router: GET /health\n(liveness + cheap readiness, no secrets)"]
        METRICS["GET /metrics\n(Prometheus text or ?format=json)"]
    end

    subgraph pipeline["Ingestion pipeline (DocumentService)"]
        direction TB
        GATE["file gate\n(size cap pre-read, MIME, pages,\ndecoded-pixel caps)"]
        TEXT["native PDF text or Mistral OCR\n(EXIF fix, truncated-JPEG retry)"]
        LLM["Gemini extraction\n(Groq vision/text fallback)"]
        NORM["normalization (Decimal)\n+ deterministic validation"]
    end

    subgraph rag["RAG pipelines (separate, share persisted rows)"]
        direction TB
        IDX["IndexingService\n(chunk → embed → upsert,\nSKIP LOCKED jobs, bounded retries)"]
        RET["RetrievalService\n(pgvector + FTS + RRF)"]
        ANS["AnswerService\n(Gemini → Groq → template,\ncitation repair, abstention)"]
    end

    subgraph db["Render PostgreSQL"]
        TABLES["documents (+source_text)\ndocument_chunks (VECTOR(768))\nindexing_jobs\nalembic_version"]
    end

    UI --> MW --> AUTH --> RL --> DOC & QA
    MW --> HEALTH & METRICS
    DOC --> pipeline
    pipeline -->|persist attempt| db
    IDX -->|chunks/vectors| db
    RET -->|reads| db
    QA --> RET & ANS
    ANS -->|Gemini/Groq APIs| EXT["provider APIs"]
    LLM -->|Gemini/Groq APIs| EXT
```

## Request flows

**Upload** `POST /api/v1/documents/process`
`auth → rate-limit → Content-Length pre-check → bounded read (limit+1) →
MIME/pages/decoded-pixel validation → native text or OCR → Gemini/Groq
extraction → normalization → Decimal validation → persist attempt (+bounded
retained source text) → inline bounded indexing job → 200 with result
(PASS or FAILED envelope).`

**Ask** `POST /api/v1/qa/ask` (60 s budget, 504 `QA_TIMEOUT` on breach)
`auth → rate-limit → classify (rules-first; unsupported abstains unless a
document scope is given) → resolve filename-in-question → whitelisted SQL
(with retrieval fallback when SQL yields nothing) → hybrid retrieval
(filters applied pre-retrieval) → evidence package → answer chain →
citation repair/abstention → response with provider, citations (verbatim vs
derived), facts, calculations, limitations, retrieval metadata.`

**Indexing** Inline at upload; `POST …/reindex` only *queues* a job (returns
`PENDING` + job id) — `python scripts/run_indexing_worker.py` drains the
queue. Startup recovers stale jobs. Job state: `GET /indexing/jobs/{id}`.

## Data model

- `documents`: one row per processing attempt (never updated); JSONB payloads;
  `source_text` (nullable, bounded) holds the OCR/native text for
  verification and narrative chunks. Raw source text is never in API
  responses except inside verbatim evidence snippets.
- `document_chunks`: `embedding VECTOR(768)`, GIN FTS index, HNSW cosine
  index (m=16, ef_construction=64), partial unique index on active
  `(document_id, index_version, content_hash)`, canonical `company_norm`.
- `indexing_jobs`: PENDING/PROCESSING/COMPLETED/FAILED with attempts, error
  codes and timestamps; partial unique index gives one active job per
  document. Inactive chunks are pruned only by `scripts/prune_chunks.py`.

## Trust boundaries

- Everything under `/api/v1/*` (except `/health`) requires `API_KEYS` when
  configured; without it the service is open (dev default).
- `GET /health` is public by design (platform probes); it carries status
  counts and config labels only.
- Uploaded bytes are untrusted: bounded reads, MIME sniffing, decoded-pixel
  caps, Pydantic-validated extraction, prompt-delimited document text, and
  sanitized logs/errors throughout.
- Provider SDKs: generation via unified `google-genai`; embeddings via
  `google-genai`; OCR via Mistral; fallback LLM via Groq. The retired
  `google-generativeai` package is gone.
