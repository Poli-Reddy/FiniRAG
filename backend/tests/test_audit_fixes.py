"""Audit-fix tests: year parsing, migrations, provenance, worker, retrieval paths."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from backend.app.api.routes.qa import _guess_periods
from backend.app.core.database import Base
import backend.app.models  # noqa
from backend.app.models.document import Document
from backend.app.providers.embedding_provider import HashEmbeddingProvider
from backend.app.services.chunking_service import ChunkingService
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.indexing_service import IndexingService
from backend.app.services.retrieval_service import (
    RetrievalService, RetrievalError, _dialect_name, rrf_fuse,
)


def test_rrf_tiebreak_is_deterministic():
    """Finding #20: equal fused scores order by chunk id, not run order."""
    a = [{"chunk_id": "b", "content": "x", "content_hash": "h1"}]
    b = [{"chunk_id": "a", "content": "y", "content_hash": "h2"}]
    first = [c["chunk_id"] for c in rrf_fuse([a, b], k=60)]
    second = [c["chunk_id"] for c in rrf_fuse([b, a], k=60)]
    assert first == second == ["a", "b"]


def test_company_normalization():
    from backend.app.services.chunking_service import normalize_company

    assert normalize_company("HDFC Bank Ltd.") == "hdfc bank"
    assert normalize_company("hdfc bank") == "hdfc bank"
    assert normalize_company("  ") is None
    assert normalize_company(None) is None
    # Meaningful words are never stripped.
    assert normalize_company("Tata Steel") == "tata steel"


@pytest.mark.asyncio
async def test_company_filter_matches_canonical_name(db):
    """Finding #19: 'HDFC Bank Ltd' finds chunks indexed as 'hdfc bank'."""
    from backend.app.services.chunking_service import ChunkingService

    def _statement(company):
        return {"company_name": {"value": company, "original_value": company,
                                 "status": "EXPLICIT",
                                 "evidence": {"source_text": company, "page_number": 1}},
                "periods": {"2024": {
                    "total_assets": {"value": "100", "original_value": "100",
                                     "status": "EXPLICIT",
                                     "evidence": {"source_text": "100", "page_number": 1},
                                     "unit": "INR_CRORE"}}}}

    for name, company in (("hdfc.pdf", "HDFC Bank Ltd."),
                          ("tata.pdf", "Tata Steel Limited")):
        db.add(Document(document_name=name, document_type="balance_sheet",
                       processing_status="PASS", file_validation={"status": "PASS"},
                       extracted_data=_statement(company),
                       validation={"overall_status": "PASS", "checks": [], "summary": {}},
                       processing_metadata={}, created_at=datetime.now(timezone.utc)))
    await db.flush()
    docs = (await db.execute(select(Document))).scalars().all()
    emb = EmbeddingService(HashEmbeddingProvider())
    for d in docs:
        await IndexingService(db, embeddings=emb).index_document(d.id)
    await db.commit()
    out = await RetrievalService(db, emb).retrieve("total assets", company="hdfc bank")
    names = {c["document_name"] for c in out["candidates"]}
    assert names == {"hdfc.pdf"}


@pytest.mark.asyncio
async def test_company_filter_never_mixes_known_companies(db):
    """A chunk with a different canonical company must not match."""
    from backend.app.services.chunking_service import ChunkingService

    def _statement(company):
        return {"company_name": {"value": company, "original_value": company,
                                 "status": "EXPLICIT",
                                 "evidence": {"source_text": company, "page_number": 1}},
                "periods": {"2024": {
                    "total_assets": {"value": "100", "original_value": "100",
                                     "status": "EXPLICIT",
                                     "evidence": {"source_text": "100", "page_number": 1},
                                     "unit": "INR_CRORE"}}}}

    db.add(Document(document_name="tata.pdf", document_type="balance_sheet",
                   processing_status="PASS", file_validation={"status": "PASS"},
                   extracted_data=_statement("Tata Steel Limited"),
                   validation={"overall_status": "PASS", "checks": [], "summary": {}},
                   processing_metadata={}, created_at=datetime.now(timezone.utc)))
    await db.flush()
    docs = (await db.execute(select(Document))).scalars().all()
    emb = EmbeddingService(HashEmbeddingProvider())
    for d in docs:
        await IndexingService(db, embeddings=emb).index_document(d.id)
    await db.commit()
    out = await RetrievalService(db, emb).retrieve("total assets", company="HDFC Bank")
    assert out["candidates"] == []


class _FakeScalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _FakeScalars(self._rows)


class _FakePGDB:
    """Pretends to be PostgreSQL. Raises on native vector/FTS statements
    when fail_native=True; plain ORM selects succeed."""

    def __init__(self, rows=None, fail_native=False):
        from backend.app.models.rag import DocumentChunk

        self._rows = rows or []
        self._fail_native = fail_native

    async def execute(self, stmt, *a, **k):
        text = str(stmt)
        # Native vector renders the <=> distance operator; native FTS renders
        # to_tsvector. Plain fallback selects merely list the embedding column.
        native = ("<=>" in text) or ("to_tsvector" in text)
        if native and self._fail_native:
            raise RuntimeError("boom-native")
        return _FakeResult(self._rows)


def _pg_chunk(doc_id, name="inv-a.jpg", content="Total 108.00 USD"):
    from backend.app.models.rag import DocumentChunk

    return DocumentChunk(
        id=uuid.uuid4(), document_id=doc_id, chunk_index=0,
        chunk_type="financial_field", content=content, page_number=1,
        section_name="total_amount", document_type="invoice",
        reporting_period=None,
        source_evidence={"source_text": content, "evidence_kind": "verbatim",
                         "verbatim_available": True},
        chunk_metadata={"document_name": name},
        embedding=[0.1] * 768, embedding_model="gemini-embedding-2",
        index_version="v1", content_hash="h1")


@pytest.mark.asyncio
async def test_pg_native_paths_reported_on_success(monkeypatch):
    import backend.app.services.retrieval_service as rs

    monkeypatch.setattr(rs, "_dialect_name", lambda db: "postgresql")
    did = uuid.uuid4()
    db = _FakePGDB(rows=[_pg_chunk(did)])
    out = await RetrievalService(db, EmbeddingService(HashEmbeddingProvider())).retrieve(
        "total amount")
    assert out["vector_path"] == "pgvector"
    assert out["keyword_path"] == "postgres-fts"
    assert out["degraded"] is False
    assert out["native_errors"] == {}
    assert out["candidates"]


@pytest.mark.asyncio
async def test_pg_native_failure_falls_back_with_metadata(monkeypatch):
    import backend.app.services.retrieval_service as rs

    monkeypatch.setattr(rs, "_dialect_name", lambda db: "postgresql")
    did = uuid.uuid4()
    db = _FakePGDB(rows=[_pg_chunk(did)], fail_native=True)
    out = await RetrievalService(db, EmbeddingService(HashEmbeddingProvider())).retrieve(
        "total amount")
    # Fallback executed AND reported honestly — never claims pgvector.
    assert out["vector_path"] == "python-fallback"
    assert out["keyword_path"] == "token-overlap"
    assert out["degraded"] is True
    assert out["native_errors"] == {"vector": "RuntimeError", "keyword": "RuntimeError"}


@pytest.mark.asyncio
async def test_strict_mode_raises_on_pg_failure(monkeypatch):
    import backend.app.services.retrieval_service as rs

    monkeypatch.setattr(rs, "_dialect_name", lambda db: "postgresql")
    db = _FakePGDB(rows=[_pg_chunk(uuid.uuid4())], fail_native=True)
    with pytest.raises(RetrievalError):
        await RetrievalService(db, EmbeddingService(HashEmbeddingProvider())).retrieve(
            "total amount", strict=True)


@pytest.mark.asyncio
async def test_keyword_only_reports_degraded(monkeypatch):
    import backend.app.services.retrieval_service as rs

    monkeypatch.setattr(rs, "_dialect_name", lambda db: "postgresql")

    class NoVec(EmbeddingService):
        async def embed_query(self, text):
            raise RuntimeError("no vec")

    db = _FakePGDB(rows=[_pg_chunk(uuid.uuid4())])
    out = await RetrievalService(db, NoVec(HashEmbeddingProvider())).retrieve("total")
    assert out["vector_path"] == "disabled"
    assert out["keyword_path"] == "postgres-fts"
    assert out["degraded"] is True
    assert out["native_errors"]["embedding"] == "RuntimeError"


def test_guess_periods_full_years():
    assert _guess_periods("What were total assets in 1999?") == ["1999"]
    assert _guess_periods("Compare 2024 vs 2025 totals") == ["2024", "2025"]
    assert _guess_periods("total for 2024?") == ["2024"]
    assert _guess_periods("no year here") == []
    # Order preserved.
    assert _guess_periods("2025 then 2024") == ["2025", "2024"]


def test_migration_chain_valid():
    from alembic.config import Config  # noqa
    import ast

    v0 = open("alembic/versions/0000_documents.py").read()
    v1 = open("alembic/versions/0001_rag.py").read()
    assert 'revision = "0000_documents"' in v0
    assert 'revision = "0001_rag"' in v1
    assert 'down_revision = "0000_documents"' in v1
    assert 'CREATE EXTENSION IF NOT EXISTS "vector"' in v1
    assert '"pgvector"' not in v1.replace("pgvector SQLAlchemy", "")
    # Downgrade never drops documents.
    assert "op.drop_table(\"documents\")" not in v0 and "drop_table(\"documents\")" not in v1


def test_migration_offline_sql():
    """Generated SQL must contain the extension, VECTOR(768), FKs, indexes."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        capture_output=True, text=True, cwd=".")
    assert proc.returncode == 0, proc.stderr[-2000:]
    sql = proc.stdout
    assert 'CREATE EXTENSION IF NOT EXISTS "vector"' in sql
    assert "VECTOR(768)" in sql
    assert sql.count("REFERENCES documents (id) ON DELETE CASCADE") == 2
    assert "USING gin (to_tsvector('english', content))" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql
    # Chain order: documents before chunks.
    assert sql.index("CREATE TABLE documents") < sql.index("CREATE TABLE document_chunks")
    # Baseline matches ORM schema types.
    assert "file_validation JSONB NOT NULL" in sql
    # Hardening migration: explicit HNSW params + active-chunk uniqueness.
    assert "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)" in sql
    assert "uq_chunks_active_hash" in sql and "WHERE is_active" in sql
    # Correctness migration: retained source text + atomic job claims.
    assert "ADD COLUMN source_text" in sql
    assert "uq_active_job_per_document" in sql
    assert "WHERE status IN ('PENDING', 'PROCESSING')" in sql
    assert "idx_chunks_company" in sql


def test_chunk_provenance_derived_vs_verbatim():
    svc = ChunkingService()
    data = {"invoice_number": {"value": "INV-1", "original_value": "INV-1",
                               "status": "EXPLICIT",
                               "evidence": {"source_text": "INV-1", "page_number": 2}},
            "total_amount": {"value": "10", "original_value": "10",
                             "status": "EXPLICIT",
                             "evidence": {"source_text": "", "page_number": 1}},
            "line_items": []}
    chunks = svc.chunk_document(document_id=str(uuid.uuid4()), document_type="invoice",
                                extracted_data=data, document_name="a.jpg")
    header = next(c for c in chunks if c["section_name"] == "invoice_header")
    # Generated header is not a verbatim quotation of itself.
    assert header["content"] != (header["source_evidence"].get("source_text") or header["content"]) or \
        header["source_evidence"].get("evidence_kind") in ("verbatim", "derived")
    assert "evidence_kind" in header["source_evidence"]
    # Section chunks are derived.
    assert all("evidence_kind" in c["source_evidence"] for c in chunks)
    # Nothing is verifiable without retained source text.
    assert all(c["source_evidence"].get("evidence_kind") == "derived" for c in chunks)
    assert all(c["source_evidence"].get("page_number") is None for c in chunks)


def _invoice_evidence_data():
    return {
        "invoice_number": {"value": "INV-9", "original_value": "INV-9", "status": "EXPLICIT",
                           "evidence": {"source_text": "Invoice No. INV-9", "page_number": 1}},
        "total_amount": {"value": "108.00", "original_value": "108.00", "status": "EXPLICIT",
                         "evidence": {"source_text": "Total: 108.00", "page_number": 2}},
        "line_items": [],
    }


def test_snippet_verified_against_retained_source():
    """A snippet found on the claimed page becomes verbatim with that page."""
    svc = ChunkingService()
    source = ("Invoice No. INV-9\n--- PAGE BREAK ---\n"
              "Line items\nTotal: 108.00\nThank you")
    chunks = svc.chunk_document(document_id=str(uuid.uuid4()), document_type="invoice",
                                extracted_data=_invoice_evidence_data(),
                                source_text=source, page_count=2,
                                document_name="inv-9.jpg")
    by_section = {c["section_name"]: c for c in chunks}
    total = by_section["total_amount"]
    assert total["source_evidence"]["evidence_kind"] == "verbatim"
    assert total["source_evidence"]["page_number"] == 2
    assert total["page_number"] == 2


def test_snippet_absent_from_source_stays_derived():
    """A snippet that cannot be found is kept as context, never quoted."""
    svc = ChunkingService()
    chunks = svc.chunk_document(document_id=str(uuid.uuid4()), document_type="invoice",
                                extracted_data=_invoice_evidence_data(),
                                source_text="Totally unrelated OCR text here",
                                page_count=1, document_name="inv-9.jpg")
    for c in chunks:
        if c["chunk_type"] == "narrative":
            continue
        assert c["source_evidence"]["evidence_kind"] == "derived"
        assert c["source_evidence"]["page_number"] is None
        assert c["page_number"] is None


def test_wrong_claimed_page_is_corrected():
    """A snippet found on a different page than claimed is corrected."""
    svc = ChunkingService()
    source = "Page one has nothing\n--- PAGE BREAK ---\nTotal: 108.00 here"
    chunks = svc.chunk_document(document_id=str(uuid.uuid4()), document_type="invoice",
                                extracted_data={
                                    "total_amount": {
                                        "value": "108.00", "original_value": "108.00",
                                        "status": "EXPLICIT",
                                        "evidence": {"source_text": "Total: 108.00",
                                                     "page_number": 1}}},
                                source_text=source, page_count=2,
                                document_name="inv-9.jpg")
    total = next(c for c in chunks if c["section_name"] == "total_amount")
    assert total["source_evidence"]["evidence_kind"] == "verbatim"
    assert total["source_evidence"]["page_number"] == 2


def test_out_of_range_page_is_dropped():
    """An unlocatable snippet with an out-of-range page keeps no page."""
    svc = ChunkingService()
    chunks = svc.chunk_document(document_id=str(uuid.uuid4()), document_type="invoice",
                                extracted_data={
                                    "total_amount": {
                                        "value": "108.00", "original_value": "108.00",
                                        "status": "EXPLICIT",
                                        "evidence": {"source_text": "Total: 999.99",
                                                     "page_number": 5}}},
                                source_text="Invoice No. INV-9 Total: 108.00",
                                page_count=1, document_name="inv-9.jpg")
    total = next(c for c in chunks if c["section_name"] == "total_amount")
    # "Total: 999.99" is nowhere in the source and page 5 exceeds page_count=1.
    assert total["source_evidence"]["evidence_kind"] == "derived"
    assert total["source_evidence"]["page_number"] is None
    assert total["page_number"] is None


def test_evidence_supports_claim_level():
    verb = {"citation_id": "CIT-1", "content": "Total 108.00 USD",
            "source_text": "Total: 108.00", "evidence_kind": "verbatim"}
    assert EvidenceService.supports(verb, "total is 108.00") is True
    assert EvidenceService.supports(verb, "total is 999.99") is False
    assert EvidenceService.supports({"citation_id": "x", "content": "abc",
                                     "source_text": ""}, "unrelated 12345") is False
    # Negation polarity cannot be established from a source that lacks it.
    assert EvidenceService.supports(verb, "total is not 108.00") is False
    assert EvidenceService.supports(
        {"citation_id": "y", "content": "Total was never reported",
         "source_text": "never reported"}, "total was never reported") is True


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


def _doc(name="a.jpg", dtype="invoice", data=None):
    from datetime import datetime, timezone

    return Document(document_name=name, document_type=dtype, processing_status="PASS",
                    file_validation={"status": "PASS"}, extracted_data=data or {},
                    validation={"overall_status": "PASS"}, processing_metadata={},
                    created_at=datetime.now(timezone.utc))


@pytest.mark.asyncio
async def test_retrieval_reports_fallback_paths(db):
    doc = _doc(data={"invoice_number": {"value": "INV", "original_value": "INV",
                                        "status": "EXPLICIT",
                                        "evidence": {"source_text": "INV", "page_number": 1}},
                     "line_items": []})
    db.add(doc)
    await db.flush()
    await IndexingService(db, embeddings=EmbeddingService(HashEmbeddingProvider())).index_document(doc.id)
    out = await RetrievalService(db, EmbeddingService(HashEmbeddingProvider())).retrieve("INV invoice")
    assert out["dialect"] in ("unknown", "sqlite", "postgresql")
    assert out["vector_path"] in ("python-fallback", "pgvector", "disabled")
    assert out["keyword_path"] in ("token-overlap", "postgres-fts")
    assert _dialect_name(db) != "postgresql"  # sqlite in tests


@pytest.mark.asyncio
async def test_cosine_fallback_is_deterministic_on_ties(monkeypatch):
    """Equal-distance rows must order by chunk id, not DB insertion order."""
    did = uuid.uuid4()
    rows = [_pg_chunk(did, name="b.jpg"), _pg_chunk(did, name="a.jpg"),
            _pg_chunk(did, name="c.jpg")]
    for r in rows:
        r.embedding = [0.5] * 768  # identical vectors => identical cosine

    svc = RetrievalService(_FakePGDB(), EmbeddingService(HashEmbeddingProvider()))
    async def fake_load(filters, limit=None):
        return rows, False
    monkeypatch.setattr(svc, "_load_eligible", fake_load)

    out, truncated = await svc._cosine_fallback([0.5] * 768, [], 3)
    assert truncated is False
    ids = [c["chunk_id"] for c in out]
    assert ids == sorted(ids)  # stable id ordering
    # And repeated calls return the same order.
    out2, _ = await svc._cosine_fallback([0.5] * 768, [], 3)
    assert [c["chunk_id"] for c in out2] == ids


@pytest.mark.asyncio
async def test_fallback_rows_are_bounded():
    """_load_eligible fetches one past the cap so truncation is detectable."""
    from sqlalchemy.dialects import postgresql
    from backend.app.core.config import settings

    captured = {}

    class CapDB:
        async def execute(self, stmt, *a, **k):
            captured["stmt"] = stmt

            class R:
                def scalars(self):
                    class S:
                        def all(self):
                            return []
                    return S()
            return R()

    svc = RetrievalService(CapDB(), EmbeddingService(HashEmbeddingProvider()))
    rows, truncated = await svc._load_eligible([])
    assert rows == [] and truncated is False
    sql = str(captured["stmt"].compile(
        dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert f"LIMIT {settings.RETRIEVAL_FALLBACK_MAX_ROWS + 1}" in sql


@pytest.mark.asyncio
async def test_fallback_truncation_is_reported(monkeypatch):
    """A capped fallback sweep must surface as degraded, not silent."""
    from backend.app.core.config import settings

    did = uuid.uuid4()
    rows = [_pg_chunk(did, content=f"some query text here row {i}") for i in range(3)]
    svc = RetrievalService(_FakePGDB(), EmbeddingService(HashEmbeddingProvider()))

    async def fake_load(filters, limit=None):
        cap = limit or settings.RETRIEVAL_FALLBACK_MAX_ROWS
        assert len(rows) <= cap  # fixture sanity: truncation comes from the flag
        return rows, True  # truncated sweep of a larger corpus

    monkeypatch.setattr(svc, "_load_eligible", fake_load)
    out = await svc.retrieve("some query text here")
    assert out["degraded"] is True
    assert "fallback-truncated" in str(out["native_errors"])


@pytest.mark.asyncio
async def test_session_guards_set_ef_search_on_postgres(monkeypatch):
    import backend.app.services.retrieval_service as rs

    executed = []

    class GuardDB:
        async def execute(self, stmt, *a, **k):
            executed.append(str(stmt))
            class R:
                def scalars(self):
                    class S:
                        def all(self):
                            return []
                    return S()
            return R()

    monkeypatch.setattr(rs, "_dialect_name", lambda db: "postgresql")
    svc = RetrievalService(GuardDB(), EmbeddingService(HashEmbeddingProvider()))
    await svc._apply_session_guards()
    assert any("hnsw.ef_search" in s for s in executed)


@pytest.mark.asyncio
async def test_stale_recovery_and_retry_bound(db):
    from backend.app.repositories.rag_repository import IndexingJobRepository

    doc = _doc(data={"invoice_number": {"value": "x", "original_value": "x",
                                        "status": "EXPLICIT",
                                        "evidence": {"source_text": "x", "page_number": 1}},
                     "line_items": []})
    db.add(doc)
    await db.flush()
    repo = IndexingJobRepository(db)
    job = await repo.create_pending(doc.id, "v1")
    # Simulate stale PROCESSING.
    job.status = "PROCESSING"
    job.attempt_count = 1
    job.started_at = datetime.now(timezone.utc) - timedelta(minutes=60)
    await db.flush()
    n = await repo.recover_stale(stale_minutes=30)
    assert n == 1
    refreshed = await repo.get_by_id(job.id)
    assert refreshed.status == "PENDING"
    # Exhausted retries go terminal.
    refreshed.status = "PROCESSING"
    refreshed.attempt_count = 5
    refreshed.started_at = datetime.now(timezone.utc) - timedelta(minutes=60)
    await db.flush()
    await repo.recover_stale(stale_minutes=30, max_attempts=5)
    assert (await repo.get_by_id(job.id)).status == "FAILED"


@pytest.mark.asyncio
async def test_chunk_write_failure_preserves_document_and_job(db):
    """Finding #6: a chunk-write failure must not roll back the document.

    The document row and the retryable job row are created in the caller's
    transaction BEFORE chunk replacement; only the savepoint may roll back.
    """
    from sqlalchemy import select

    from backend.app.models.document import Document
    from backend.app.models.rag import IndexingJob
    from backend.app.repositories.rag_repository import ChunkRepository

    doc = _doc(name="survivor.jpg", data={
        "invoice_number": {"value": "INV-9", "original_value": "INV-9", "status": "EXPLICIT",
                           "evidence": {"source_text": "INV-9", "page_number": 1}},
        "line_items": []})
    db.add(doc)
    await db.flush()

    svc = IndexingService(db, embeddings=EmbeddingService(HashEmbeddingProvider()))
    job = await svc.ensure_job(doc.id)

    async def _boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ChunkRepository, "upsert_chunks", _boom)
    try:
        res = await svc.process_job(job)
    finally:
        monkeypatch.undo()
    assert res["status"] == "FAILED" and res["error"] == "INDEX_WRITE_FAILED"

    # Fresh reads (not the possibly-detached in-memory objects).
    db.expunge_all()
    assert (await db.execute(select(Document).where(Document.id == doc.id))
            ).scalar_one_or_none() is not None
    fresh_job = (await db.execute(select(IndexingJob).where(IndexingJob.id == job.id))
                 ).scalar_one_or_none()
    assert fresh_job is not None
    assert fresh_job.status == "FAILED" and fresh_job.error_code == "INDEX_WRITE_FAILED"


@pytest.mark.asyncio
async def test_ensure_job_is_idempotent(db):
    doc = _doc(data={"invoice_number": {"value": "x", "original_value": "x",
                                        "status": "EXPLICIT",
                                        "evidence": {"source_text": "x", "page_number": 1}},
                     "line_items": []})
    db.add(doc)
    await db.flush()
    svc = IndexingService(db, embeddings=EmbeddingService(HashEmbeddingProvider()))
    j1 = await svc.ensure_job(doc.id)
    j2 = await svc.ensure_job(doc.id)
    assert str(j1.id) == str(j2.id)


@pytest.mark.asyncio
async def test_ensure_job_lost_race_returns_winner(db, monkeypatch):
    """Finding #8: a unique-conflict loser re-reads instead of duplicating.

    The first read is forced to miss (as if a concurrent request inserted
    between our read and our write); the partial unique index then rejects
    the duplicate insert and ensure_job must return the existing winner.
    """
    from sqlalchemy import func, select
    from sqlalchemy.exc import IntegrityError

    from backend.app.models.rag import IndexingJob
    from backend.app.repositories.rag_repository import IndexingJobRepository

    doc = _doc(name="racy.jpg", data={
        "invoice_number": {"value": "INV-9", "original_value": "INV-9", "status": "EXPLICIT",
                           "evidence": {"source_text": "INV-9", "page_number": 1}},
        "line_items": []})
    db.add(doc)
    await db.flush()

    svc = IndexingService(db, embeddings=EmbeddingService(HashEmbeddingProvider()))
    first = await svc.ensure_job(doc.id)

    orig_get = IndexingJobRepository.get_active_for_document
    calls = {"n": 0}

    async def _flaky_get(self, document_id):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # emulate the stale read of a concurrent loser
        return await orig_get(self, document_id)

    monkeypatch.setattr(IndexingJobRepository, "get_active_for_document", _flaky_get)
    recovered = await svc.ensure_job(doc.id)
    assert str(recovered.id) == str(first.id)
    # Still exactly one active job — no duplicate was created.
    n_active = (await db.execute(
        select(func.count(IndexingJob.id)).where(
            IndexingJob.document_id == doc.id,
            IndexingJob.status.in_(["PENDING", "PROCESSING"])))).scalar_one()
    assert n_active == 1


def _invoice_data(status="EXPLICIT", value="108.00", unit="USD"):
    return {
        "invoice_number": {"value": "INV-9", "original_value": "INV-9",
                           "status": "EXPLICIT",
                           "evidence": {"source_text": "INV-9", "page_number": 1}},
        "total_amount": {"value": value, "original_value": value, "status": status,
                         "evidence": {"source_text": value, "page_number": 1},
                         "unit": unit},
        "line_items": []}


@pytest.mark.asyncio
async def test_failed_replacement_preserves_active_chunks(db):
    """A failed reindex must not destroy the previous active chunk set."""
    from backend.app.repositories.rag_repository import ChunkRepository

    doc = _doc(name="keep.jpg", data=_invoice_data())
    db.add(doc)
    await db.flush()
    good = IndexingService(db, embeddings=EmbeddingService(HashEmbeddingProvider()))
    assert (await good.index_document(doc.id))["status"] == "COMPLETED"
    n_before = await ChunkRepository(db).count_active(doc.id)
    assert n_before > 0

    class Boom(HashEmbeddingProvider):
        def __init__(self):
            super().__init__(dim=768, model="boom-768-DOWN")

        async def embed_texts(self, texts, titles=None):
            raise RuntimeError("provider down")

    bad = IndexingService(db, embeddings=EmbeddingService(Boom()))
    job = await bad.ensure_job(doc.id)
    res = await bad.process_job(job)
    assert res["status"] == "FAILED"
    assert await ChunkRepository(db).count_active(doc.id) == n_before


@pytest.mark.asyncio
async def test_model_change_forces_reembed(db):
    """Vectors from another model/dim must not be reused silently."""
    from backend.app.repositories.rag_repository import ChunkRepository

    doc = _doc(name="remodel.jpg", data=_invoice_data())
    db.add(doc)
    await db.flush()
    v1 = IndexingService(db, embeddings=EmbeddingService(HashEmbeddingProvider()))
    assert (await v1.index_document(doc.id))["status"] == "COMPLETED"

    other = HashEmbeddingProvider(dim=768, model="other-model-768")
    v2 = IndexingService(db, embeddings=EmbeddingService(other))
    job = await v2.ensure_job(doc.id)
    assert (await v2.process_job(job))["status"] == "COMPLETED"
    models = {c.embedding_model for c in await ChunkRepository(db).active_for_document(doc.id)}
    assert models == {"other-model-768"}


@pytest.mark.asyncio
async def test_chunk_metadata_carries_document_identity(db):
    """Candidates must resolve document_name from stored chunk metadata
    (regression: bare 'metadata' kwarg was swallowed by DeclarativeBase)."""
    from sqlalchemy import select
    from backend.app.models.rag import DocumentChunk

    doc = _doc(name="identity.jpg", data=_invoice_data())
    db.add(doc)
    await db.flush()
    await IndexingService(db, embeddings=EmbeddingService(HashEmbeddingProvider())).index_document(doc.id)
    chunks = (await db.execute(select(DocumentChunk))).scalars().all()
    assert chunks and all(
        (c.chunk_metadata or {}).get("document_name") == "identity.jpg" for c in chunks)


@pytest.mark.asyncio
async def test_failed_validation_stays_visible_in_evidence(db):
    """Contradictory/failed checks are preserved, never silently merged."""
    doc = _doc(name="failvis.jpg", data=_invoice_data())
    db.add(doc)
    await db.flush()
    await IndexingService(db, embeddings=EmbeddingService(HashEmbeddingProvider())).index_document(doc.id)
    out = await RetrievalService(db, EmbeddingService(HashEmbeddingProvider())).retrieve(
        "total amount", document_id=doc.id)
    ev = EvidenceService().build(
        validation={"ok": True, "document_id": str(doc.id), "document_name": "failvis.jpg",
                    "overall_status": "FAIL",
                    "checks": [{"name": "sum_lines_vs_subtotal", "status": "FAIL",
                                "formula": "sum(line_amounts)"}]},
        candidates=out["candidates"])
    kinds = [i["kind"] for i in ev["items"]]
    assert "validation_check" in kinds
    assert any("FAIL" in i["content"] for i in ev["items"] if i["kind"] == "validation_check")


def test_oversized_image_rejected_before_ocr(monkeypatch):
    """Finding #22: decoded pixels are capped, not just compressed bytes."""
    from PIL import Image

    from backend.app.core.config import settings
    from backend.app.services.image_preprocessor import ImagePreprocessor

    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 100)
    img = Image.new("RGB", (10, 10), "white")  # 100 px: at the cap, passes
    ImagePreprocessor._check_pixels(img.width, img.height, "test")
    with pytest.raises(ValueError, match="over the .* px cap"):
        ImagePreprocessor._check_pixels(11, 10, "test")


def test_truncated_flag_restored_and_locked():
    """Finding #22: the process-global Pillow flag is restored and guarded."""
    import threading

    from PIL import ImageFile

    from backend.app.services import image_preprocessor as ip_mod
    from backend.app.services.image_preprocessor import ImagePreprocessor

    assert isinstance(ip_mod._TRUNCATED_FLAG_LOCK, type(threading.Lock()))
    before = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImagePreprocessor._open(b"definitely not an image at all!!!!")
    except Exception:
        pass
    assert ImageFile.LOAD_TRUNCATED_IMAGES is before


def test_render_page_cap(monkeypatch, sample_pdf):
    """Finding #22: absurd page dimensions fail before rendering."""
    from backend.app.core.config import settings
    from backend.app.services.image_preprocessor import ImagePreprocessor

    monkeypatch.setattr(settings, "MAX_RENDER_PIXELS_PER_PAGE", 1)
    with pytest.raises(ValueError, match="render beyond"):
        ImagePreprocessor().render_pdf_pages(sample_pdf)


@pytest.mark.asyncio
async def test_prune_keeps_active_and_recent(db):
    """Finding #23: only old INACTIVE chunks are prunable."""
    import sys
    from datetime import timedelta

    sys.path.insert(0, ".")
    from scripts.prune_chunks import prune

    from backend.app.models.rag import DocumentChunk

    doc = _doc(name="prune.jpg", data=_invoice_data())
    db.add(doc)
    await db.flush()
    old = datetime.now(timezone.utc) - timedelta(days=60)
    rows = [
        # Old inactive: prunable.
        DocumentChunk(document_id=doc.id, chunk_index=0, chunk_type="section",
                      content="old", page_number=1, document_type="invoice",
                      source_evidence={}, chunk_metadata={}, embedding=None,
                      embedding_model="", index_version="v1", content_hash="h-old",
                      is_active=False, created_at=old, updated_at=old),
        # Recent inactive: retained (recovery window).
        DocumentChunk(document_id=doc.id, chunk_index=1, chunk_type="section",
                      content="recent", page_number=1, document_type="invoice",
                      source_evidence={}, chunk_metadata={}, embedding=None,
                      embedding_model="", index_version="v1", content_hash="h-new",
                      is_active=False),
        # Old but ACTIVE: never touched.
        DocumentChunk(document_id=doc.id, chunk_index=2, chunk_type="section",
                      content="live", page_number=1, document_type="invoice",
                      source_evidence={}, chunk_metadata={}, embedding=None,
                      embedding_model="", index_version="v1", content_hash="h-live",
                      is_active=True, created_at=old, updated_at=old),
    ]
    db.add_all(rows)
    await db.flush()

    # NB: factory itself is sync (returns the context manager); only the
    # protocol methods are async.
    class _Ctx:
        async def __aenter__(self):
            return db

        async def __aexit__(self, *args):
            return False

    def _sync_factory():
        return _Ctx()

    result = await prune(30, dry_run=False, session_factory=_sync_factory)
    assert result == {"eligible": 1, "deleted": 1, "dry_run": False,
                      "older_than_days": 30}
    remaining = (await db.execute(select(DocumentChunk))).scalars().all()
    assert sorted(c.content_hash for c in remaining) == ["h-live", "h-new"]


def test_offline_guard_blocks_sockets():
    """Finding #17: the unit suite cannot open TCP connections."""
    import socket

    with pytest.raises(AssertionError, match="must not open network connections"):
        socket.create_connection(("example.com", 80))


def test_key_pools_drained_by_fixture():
    """Finding #17: module-singleton pools hold no live keys under tests."""
    from backend.app.core import config as _config

    assert _config.GEMINI_KEYS.keys == []
    assert _config.GROQ_KEYS.keys == []
    assert _config.MISTRAL_KEYS.keys == []
    from backend.app.core.config import settings

    assert settings.MISTRAL_API_KEY == ""


def test_migration_0003_offline_sql():
    """Finding #8/#9: source_text column + active-job uniqueness in DDL."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        capture_output=True, text=True, cwd=".")
    assert proc.returncode == 0, proc.stderr[-2000:]
    sql = proc.stdout
    assert "ADD COLUMN source_text" in sql
    assert "uq_active_job_per_document" in sql
    assert "WHERE status IN ('PENDING', 'PROCESSING')" in sql
    assert "idx_chunks_company" in sql


def test_baseline_required_columns_match_orm():
    """Finding #27: adoption validation covers the deployed schema."""
    import importlib.util

    from backend.app.models.document import Document

    spec = importlib.util.spec_from_file_location(
        "m0000", "alembic/versions/0000_documents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    orm_columns = {c.name for c in Document.__table__.columns}
    # source_text arrives via 0003; the baseline must require everything else.
    assert module.REQUIRED_DOCUMENT_COLUMNS == orm_columns - {"source_text"}
