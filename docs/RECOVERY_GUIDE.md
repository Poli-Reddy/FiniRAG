# FiniRAG Database Recovery Guide

Reconstruction of recoverable state from `evaluation/results/*.json` into a
new PostgreSQL database. This is a **reconstruction from files, not a
backup restore** — read the limitations section before promising anything.

## What the source files contain

- **54 result JSON files** in `evaluation/results/`: **50 PASS** records
  (20 invoices, 10 balance sheets, 10 profit & loss, 10 cash flow
  statements) plus **4 FAILED edge-case fixtures** (`edge_corrupted`,
  `edge_empty_file`, `edge_four_pages`, `edge_unsupported_type`).
- All 54 files **lack** `id`, `created_at`, and `source_text`. New UUIDs
  and current timestamps are generated on import and reported as such;
  historical IDs/timestamps cannot be recovered.
- Because no `source_text` is retained, rebuilt chunks are **derived
  structured summaries**; narrative/verbatim source chunks are absent
  until source text is re-extracted and the corpus reindexed.

## Prerequisites

1. **Render PostgreSQL** provisioned (pgvector available).
2. `DATABASE_URL` — `postgres://`, `postgresql://`, or
   `postgresql+asyncpg://` are all accepted (one shared normalizer).
3. `GEMINI_API_KEY` (or `GEMINI_API_KEYS`) — **required** for the index
   rebuild. The rebuild **aborts** rather than writing non-semantic hash
   vectors. `--allow-hash-embeddings` works **only** against localhost
   test databases and is refused for remote hosts.
4. Python 3.12 with `pip install -r backend/requirements.txt`.

## Quick start

```bash
export DATABASE_URL="postgresql://user:password@host:5432/dbname"

# Strictly read-only preview: no migrations, no inserts, no jobs.
python scripts/recover_database.py --dry-run

# Full recovery (requires --confirm for every write step).
python scripts/recover_database.py --confirm

# Small-scale test (both import and index honor --limit).
python scripts/recover_database.py --confirm --limit 10
```

`--dry-run` and `--confirm` are mutually exclusive. Skips are granular:
`--skip-db-check`, `--skip-migrations` (connectivity check still runs),
`--skip-import`, `--skip-index`, `--skip-verify`.

## Step-by-step

### 1. Connectivity check (read-only)

```bash
python scripts/check_db_connection.py
```

Verifies SELECT 1, database identity, PostgreSQL version, existing
tables, and pgvector availability. Fails closed when the `vector`
extension is unavailable.

### 2. Migrations

```bash
alembic upgrade head
```

Creates `documents`, `document_chunks` (`VECTOR(768)`, GIN FTS
`idx_chunks_fts`, HNSW `idx_chunks_vec`), `indexing_jobs`, and
`alembic_version`. An existing `documents` table is **adopted only**
after a name+type+nullability diff passes; drift fails loudly without
dropping data, and missing baseline indexes are backfilled.

### 3. Import (50 PASS + 4 FAILED historical attempts)

```bash
python scripts/restore_project_data.py --dry-run
python scripts/restore_project_data.py --confirm
```

- Expected result: **54 records** (50 PASS + 4 FAILED edge fixtures
  imported as historical failed attempts).
- Idempotent on the stable `import_hash`: re-runs skip existing rows.
- `--force` **upserts in place** (same row updated, chunks/jobs keep
  pointing at it) — it never creates duplicates.
- Each record writes inside its own savepoint: one bad record cannot
  wedge the batch.
- Writes `evaluation/recovery_manifest.json` (file → record mapping;
  override with `--manifest`). The verifier compares the database
  against this manifest.

### 4. Index rebuild (real embeddings or abort)

```bash
python scripts/rebuild_rag_index.py --dry-run
python scripts/rebuild_rag_index.py --confirm
```

- Uses `IndexingService` at the current `INDEX_VERSION`; skips
  documents that already have active chunks (override with `--force`).
- **Fail-closed**: without working Gemini embeddings the run aborts
  with `ABORTED_NO_EMBEDDINGS` and imported data untouched.
- Vector search filters by configured model + index version, so mixed
  vector spaces can never silently compare.

### 5. Verify (strict)

```bash
python scripts/verify_recovery.py [--api-url http://localhost:8000] [--allow-pending]
```

Fails (non-zero exit) on: manifest mismatch, wrong/mixed embedding
models, dimension mismatches, missing `vector` extension or required
index names, failed inspection queries, any FAILED job, leftover
pending jobs (unless `--allow-pending`), degraded or empty strict
retrieval, wrong-document filtered results, or failed API smoke tests.
Without `--api-url`, HTTP checks report SKIPPED — never passed.

## Indexing model (what "queued" means)

There is **no automatic background loop** on the single-service
deployment. Indexing happens (a) inline at upload, (b) via
`python scripts/run_indexing_worker.py [--limit N]`, or (c) by startup
stale-job recovery. `POST /documents/{id}/reindex` only **queues** a
job (`queued: true`); the dashboard says so explicitly and polls the
job until it completes or instructs the operator to run the worker.

## Production checklist

- Set `DATABASE_URL` (internal URL when co-located on Render).
- Set `API_KEYS` — with `ENVIRONMENT=production` the app **refuses to
  boot** without them. The dashboard header has a session-only key
  field (never embedded in source); keys live in `sessionStorage`.
- Optional: `TRUST_X_FORWARDED_FOR=true` only behind a proxy verified
  to overwrite client-supplied forwarding headers (default off).
- `EMBEDDING_DIM` must equal the migrated `VECTOR(768)` or boot fails;
  a model/dimension change needs a versioned migration + re-embed.

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| `DATABASE_URL not configured` | Export it; all three scheme forms accepted |
| `pgvector extension NOT available` | Enable pgvector on the database |
| `ABORTED_NO_EMBEDDINGS` | Set `GEMINI_API_KEY(S)` / fix quota; re-run |
| `Import shows N skipped` | Already imported (idempotent); `--force` upserts in place |
| `manifest not found` | Re-run restore (non-dry-run) to regenerate it |
| Verification lists failures | Each line names the exact defect; fix and re-run |

## Limitations (not recoverable from these files)

- Historical UUIDs, creation timestamps, prior job history, and
  previous embeddings are **gone** with the old database.
- No `source_text` → no verbatim narrative chunks or page-level
  citations until OCR/native text is re-extracted and reindexed.
- Claim checking remains a lexical heuristic plus deterministic
  exact-value/source/currency guards — not semantic entailment proof.

## Files

| Script | Purpose |
|--------|---------|
| `scripts/check_db_connection.py` | Read-only connectivity + pgvector check |
| `scripts/restore_project_data.py` | JSON → documents (+ manifest) |
| `scripts/rebuild_rag_index.py` | Chunks + real embeddings (fail-closed) |
| `scripts/verify_recovery.py` | Strict manifest/model/index/retrieval/API checks |
| `scripts/recover_database.py` | Orchestration (direct calls, read-only dry-run) |
| `scripts/run_indexing_worker.py` | Manual pending-job drain |
| `docs/RECOVERY_GUIDE.md` | This document |
