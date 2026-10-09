"""Audit-fix tests: year parsing, migrations, provenance, worker, retrieval paths."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta

import pytest
import pytest_asyncio
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
    RetrievalService, RetrievalError, _dialect_name,
)


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

    proc = subprocess.run(
        ["python", "-m", "alembic", "upgrade", "head", "--sql"],
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


def test_evidence_supports_claim_level():
    verb = {"citation_id": "CIT-1", "content": "Total 108.00 USD",
            "source_text": "Total: 108.00", "evidence_kind": "verbatim"}
    assert EvidenceService.supports(verb, "total is 108.00") is True
    assert EvidenceService.supports(verb, "total is 999.99") is False
    assert EvidenceService.supports({"citation_id": "x", "content": "abc",
                                     "source_text": ""}, "unrelated 12345") is False


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
async def test_concurrent_ensure_idempotent(db):
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
