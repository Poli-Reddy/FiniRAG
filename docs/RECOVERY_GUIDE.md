# FiniRAG Database Recovery Guide

This document describes the complete process to recover FiniRAG data to a new Render PostgreSQL database.

## Prerequisites

1. **Render PostgreSQL Database** - Provisioned and running
2. **DATABASE_URL** - External connection string from Render Dashboard
3. **API Keys** - GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY in environment
4. **Python 3.12+** with project dependencies installed (`pip install -r backend/requirements.txt`)

## Quick Start

```bash
# 1. Set environment variables
export DATABASE_URL="postgresql+asyncpg://user:password@host:5432/dbname"
export GEMINI_API_KEY="your-gemini-key"
export GROQ_API_KEY="your-groq-key"  
export MISTRAL_API_KEY="your-mistral-key"

# 2. Run complete recovery (dry-run first)
python scripts/recover_database.py --dry-run

# 3. If dry-run looks good, run for real
python scripts/recover_database.py --confirm
```

## Step-by-Step Recovery

### Step 1: Verify Database Connection

```bash
python scripts/check_db_connection.py
```

This verifies:
- DATABASE_URL is configured
- Connection works (SELECT 1)
- PostgreSQL version
- Database identity
- pgvector extension availability
- Existing tables/data

**Expected output:** All checks pass with `[OK]`

### Step 2: Run Migrations

```bash
alembic upgrade head
```

This creates:
- `documents` table (JSONB columns for extracted data)
- `document_chunks` table (VECTOR(768) + GIN + HNSW indexes)
- `indexing_jobs` table
- `alembic_version` tracking table

### Step 3: Import Processed Records

```bash
# Dry-run first
python scripts/restore_project_data.py --dry-run

# Actual import
python scripts/restore_project_data.py --confirm
```

This imports 54 JSON files from `evaluation/results/`:
- 10 invoices
- 10 balance sheets (2017-2026)
- 10 profit & loss statements (2017-2026)
- 10 cash flow statements (2017-2026)
- 4 additional invoices (batch1, batch2, batch3, etc.)

**Import features:**
- Content-hash deduplication (re-runs are safe)
- Preserves original UUIDs when valid
- Stores import metadata in `processing_metadata.import_hash`
- Reports: imported, skipped (existing), skipped (invalid), failed

### Step 4: Rebuild RAG Index

```bash
# Dry-run first
python scripts/rebuild_rag_index.py --dry-run

# Actual rebuild
python scripts/rebuild_rag_index.py --confirm
```

This:
- Processes all PASS-status documents
- Creates chunks from `extracted_data` using `IndexingService`
- Generates embeddings via `GeminiEmbeddingProvider` (or `HashEmbeddingProvider` offline)
- Uses idempotent indexing (skips already-indexed documents)
- Reports: indexed, already_indexed, failed

### Step 5: Verify Recovery

```bash
python scripts/verify_recovery.py
```

Reports:
- Document counts by type and status
- Active chunks and embedding coverage
- Indexing job statuses
- pgvector version and indexes
- Vector search + FTS retrieval tests

## Individual Script Reference

### `scripts/check_db_connection.py`
```bash
python scripts/check_db_connection.py
```
Pre-migration connectivity and compatibility check.

### `scripts/restore_project_data.py`
```bash
python scripts/restore_project_data.py [--dry-run] [--force] [--confirm]
```
- `--dry-run`: Preview what would be imported
- `--force`: Re-import even if content hash exists
- `--confirm`: Required for actual writes

### `scripts/rebuild_rag_index.py`
```bash
python scripts/rebuild_rag_index.py [--limit N] [--force] [--all] [--dry-run] [--confirm]
```
- `--limit N`: Process only N documents
- `--force`: Reindex even if chunks exist
- `--all`: Process all PASS docs (not just missing chunks)
- `--dry-run`: Preview
- `--confirm`: Required for actual writes

### `scripts/verify_recovery.py`
```bash
python scripts/verify_recovery.py
```
Comprehensive verification report.

### `scripts/recover_database.py` (Master)
```bash
python scripts/recover_database.py [--dry-run] [--confirm] [--limit N]
```
Orchestrates all steps with proper error handling.

## Expected Results

After successful recovery:
- **54 documents** imported (matching evaluation/results/*.json)
- **~400+ chunks** created (depending on document content)
- **Embeddings generated** for all chunks (if API keys valid)
- **Vector search** and **FTS** both operational
- **Zero pending/failed indexing jobs** (all COMPLETED)

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| `DATABASE_URL not configured` | Export `DATABASE_URL` from Render Dashboard |
| `pgvector extension NOT available` | Enable pgvector in Render PostgreSQL settings |
| `Connection refused` | Check Render firewall/allowlist; use External URL from local |
| `GeminiEmbeddingProvider not available` | Set `GEMINI_API_KEY` or ensure quota available |
| `Import shows 0 imported, 54 skipped` | Re-run with `--force` or clear `import_hash` from metadata |
| `Index rebuild shows 0 indexed` | Check `GEMINI_API_KEY` and embedding provider logs |

## Important Notes

- **No historical data fabricated**: Only data present in `evaluation/results/*.json` is recovered
- **Original document IDs preserved** when valid UUIDs exist in JSON
- **Content-hash deduplication** prevents duplicate imports
- **Offline mode**: Without `GEMINI_API_KEY`, uses deterministic `HashEmbeddingProvider`
- **Secrets**: Never commit `.env`, database URLs, or API keys to Git

## Files Created

| Script | Purpose |
|--------|---------|
| `scripts/check_db_connection.py` | Pre-migration connectivity check |
| `scripts/restore_project_data.py` | Import JSON → documents table |
| `scripts/rebuild_rag_index.py` | Build chunks + embeddings |
| `scripts/verify_recovery.py` | Post-recovery verification |
| `scripts/recover_database.py` | Master orchestration |
| `docs/RECOVERY_GUIDE.md` | This document |

## Next Steps After Recovery

1. Deploy application to Render with new `DATABASE_URL`
2. Set `API_KEYS` for production access control
3. Run `python -m pytest backend/tests -q` to verify API
4. Run `python scripts/render_smoke_check.py` for end-to-end test
5. Monitor `GET /metrics` for operational metrics