"""QA + indexing endpoint contract tests (stubbed services, no network)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from backend.app.core.database import Base, get_db
import backend.app.models  # noqa: F401 - register all tables for create_all
from backend.app.main import app


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


def _invoice_doc(name, total="108.00"):
    from backend.app.models.document import Document

    def ev(v):
        return {"value": v, "original_value": v, "status": "EXPLICIT",
                "evidence": {"source_text": v, "page_number": 1}, "unit": "USD"}

    return Document(
        document_name=name, document_type="invoice", processing_status="PASS",
        file_validation={"status": "PASS"},
        extracted_data={
            "invoice_number": {**ev(name), "value": name},
            "total_amount": ev(total), "subtotal": ev("100.00"),
            "currency": {**ev("USD"), "value": "USD"}, "line_items": []},
        validation={"overall_status": "PASS", "checks": [], "summary": {}},
        processing_metadata={}, created_at=datetime.now(timezone.utc))


class FakeSession:
    async def execute(self, *a, **k):
        from types import SimpleNamespace

        return SimpleNamespace(scalar=lambda: 1)


def _db_override():
    async def _fake():
        yield FakeSession()

    return _fake


@pytest.fixture()
def client(monkeypatch):
    app.dependency_overrides[get_db] = _db_override()
    # Stub retrieval/answer stack at the route level by patching services.
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def test_qa_rejects_empty_question(client):
    r = client.post("/api/v1/qa/ask", json={"question": ""})
    assert r.status_code in (400, 422)


def test_qa_rejects_bad_document_type(client):
    r = client.post("/api/v1/qa/ask", json={"question": "What is total?",
                                            "document_type": "receipt"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_DOCUMENT_TYPE"


def test_qa_rejects_bad_uuid(client):
    r = client.post("/api/v1/qa/ask", json={"question": "What is total?",
                                            "document_id": "nope"})
    assert r.status_code == 400


def test_qa_unsupported_returns_insufficient(client):
    r = client.post("/api/v1/qa/ask", json={"question": "Should I buy this stock?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "INSUFFICIENT_EVIDENCE"
    assert body["answer"]


def test_index_status_rejects_bad_uuid(client):
    r = client.get("/api/v1/documents/not-a-uuid/index-status")
    assert r.status_code == 400


def test_job_status_rejects_bad_uuid(client):
    r = client.get("/api/v1/indexing/jobs/not-a-uuid")
    assert r.status_code == 400


def test_qa_routes_registered():
    paths = {getattr(r, "path", "") for r in app.routes}
    assert "/api/v1/qa/ask" in paths


# ── Real-pipeline behavior (findings #12/#13/#14) ──────────────────────
# These drive the actual route pipeline (_answer_pipeline), not hand-built
# evidence, so routing, scope resolution and fallbacks are really tested.

@pytest.mark.asyncio
async def test_filename_in_question_resolves_scope(db):
    """Finding #13: 'total on invoice X?' works with no separate filter."""
    from backend.app.api.routes.qa import _answer_pipeline
    from backend.app.schemas.qa import QAAskRequest

    db.add(_invoice_doc("batch3-1354.jpg", total="250.00"))
    await db.flush()
    body = QAAskRequest(question="What is the total amount on invoice batch3-1354.jpg?")
    res = await _answer_pipeline(body, body.question, None, "req-1", db)
    assert res.retrieval_metadata.get("document_resolved_from_question") == "batch3-1354.jpg"
    assert res.status == "ANSWERED"
    assert "250.00" in res.answer


@pytest.mark.asyncio
async def test_unknown_field_falls_back_to_retrieval(db):
    """Finding #13/#18: unmapped wording retrieves instead of stopping."""
    from sqlalchemy import select

    from backend.app.api.routes.qa import _answer_pipeline
    from backend.app.models.document import Document
    from backend.app.schemas.qa import QAAskRequest
    from backend.app.providers.embedding_provider import HashEmbeddingProvider
    from backend.app.services.embedding_service import EmbeddingService
    from backend.app.services.indexing_service import IndexingService

    db.add(_invoice_doc("batch3-1354.jpg", total="250.00"))
    await db.flush()
    doc_id = (await db.execute(select(Document.id))).scalar_one()
    await IndexingService(db, embeddings=EmbeddingService(HashEmbeddingProvider())).index_document(doc_id)
    await db.commit()
    body = QAAskRequest(question="What is the widget surcharge total on batch3-1354.jpg?")
    res = await _answer_pipeline(body, body.question, None, "req-2", db)
    assert res.retrieval_metadata.get("retrieval_fallback") is not None
    assert res.status == "ANSWERED" and res.citations


@pytest.mark.asyncio
async def test_comparison_carries_operand_provenance(db):
    """Finding #14: comparison facts resolve to real records, not blanks."""
    from backend.app.api.routes.qa import _answer_pipeline
    from backend.app.schemas.qa import QAAskRequest

    def ev(v):
        return {"value": v, "original_value": v, "status": "EXPLICIT",
                "evidence": {"source_text": v, "page_number": 2}, "unit": "INR_CRORE"}

    from backend.app.models.document import Document

    db.add(Document(
        document_name="bs.pdf", document_type="balance_sheet", processing_status="PASS",
        file_validation={"status": "PASS"},
        extracted_data={"periods": {"2023": {"total_assets": ev("100")},
                                    "2024": {"total_assets": ev("120")}}},
        validation={"overall_status": "PASS", "checks": [], "summary": {}},
        processing_metadata={}, created_at=datetime.now(timezone.utc)))
    await db.flush()
    body = QAAskRequest(question="Compare total assets 2023 vs 2024 on bs.pdf")
    res = await _answer_pipeline(body, body.question, None, "req-3", db)
    assert res.calculations, res.retrieval_metadata
    assert res.calculations[0]["difference"] == "20"
    facts = {f["field"]: f for f in res.financial_facts if isinstance(f, dict)}
    for period in ("2023", "2024"):
        fact = facts.get(f"total_assets@{period}")
        assert fact is not None
        assert fact["document_name"] == "bs.pdf"
        assert fact["period"] == period
        assert fact["document_id"]  # resolvable, never ""


@pytest.mark.asyncio
async def test_unsupported_with_scope_answers_grounded(db):
    """Finding #18: a scoped 'forecast' question is a lookup, not advice."""
    from backend.app.api.routes.qa import _answer_pipeline
    from backend.app.schemas.qa import QAAskRequest

    db.add(_invoice_doc("batch3-1354.jpg", total="250.00"))
    await db.flush()
    body = QAAskRequest(question="What does the forecast section of batch3-1354.jpg say?",
                        document_name="batch3-1354.jpg")
    res = await _answer_pipeline(body, body.question, None, "req-4", db)
    assert "CONCEPTUAL_QUESTION" in res.intents
    assert res.retrieval_metadata.get("route_reason") == "unsupported-topic-with-scope"


def test_auth_gate_blocks_without_key(client, monkeypatch):
    """Finding P0-2: configured API_KEYS rejects anonymous callers."""
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "API_KEYS", "secret-1,secret-2")
    r = client.post("/api/v1/qa/ask", json={"question": "What is total?"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHORIZED"


def test_auth_gate_allows_bearer_and_x_api_key(client, monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "API_KEYS", "secret-1")
    for headers in ({"Authorization": "Bearer secret-1"}, {"X-API-Key": "secret-1"}):
        r = client.post("/api/v1/qa/ask", json={"question": "Should I buy this stock?"},
                        headers=headers)
        # Authenticated: past the gate (unsupported topic still abstains).
        assert r.status_code == 200
        assert r.json()["status"] == "INSUFFICIENT_EVIDENCE"


def test_auth_open_by_default(client):
    r = client.post("/api/v1/qa/ask", json={"question": "Should I buy this stock?"})
    assert r.status_code == 200


def test_health_stays_public_without_key(client, monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "API_KEYS", "secret-1")
    r = client.get("/api/v1/health")
    assert r.status_code == 200


def test_health_reports_native_path_readiness():
    from backend.app.schemas.document import HealthResponse

    body = HealthResponse().model_dump()
    for key in ("native_paths_ready", "vector_ready", "fts_ready"):
        assert key in body
        assert body[key] is False  # sqlite/offline: honestly unverified


def test_init_db_migrates_postgres_and_creates_otherwise(monkeypatch):
    """Finding #5: Alembic on Postgres, create_all only for dev backends."""
    import backend.app.core.database as db_module
    from backend.app.core.config import settings
    from sqlalchemy.ext.asyncio import create_async_engine

    calls = {}
    real_upgrade = db_module._run_alembic_upgrade

    async def fake_upgrade():
        calls["alembic"] = True

    # Postgres URL -> migrations, never create_all.
    monkeypatch.setattr(settings, "DATABASE_URL",
                        "postgresql+asyncpg://u:p@host:5432/db")
    monkeypatch.setattr(db_module, "_run_alembic_upgrade", fake_upgrade)
    import asyncio

    asyncio.run(db_module.init_db())
    assert calls.get("alembic") is True

    # Non-Postgres URL -> create_all on a throwaway sqlite engine.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(settings, "DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setattr(db_module, "_run_alembic_upgrade", real_upgrade)
    asyncio.run(db_module.init_db())

    async def _tables():
        from sqlalchemy import inspect as _inspect

        async with engine.connect() as conn:
            return await conn.run_sync(
                lambda c: _inspect(c).get_table_names())

    assert "documents" in asyncio.run(_tables())
    asyncio.run(engine.dispose())
