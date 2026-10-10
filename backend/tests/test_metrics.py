"""Metrics tests: registry correctness, exposition format, and instrumentation."""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from backend.app.core.database import Base
import backend.app.models  # noqa: F401 - register all tables for create_all
from backend.app.core.metrics import (
    MetricsRegistry, metrics,
)


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    await engine.dispose()


def test_counter_and_render():
    reg = MetricsRegistry()
    reg.inc("req_total", {"route": "/x", "status": "200"})
    reg.inc("req_total", {"route": "/x", "status": "200"}, amount=2)
    text = reg.render_prometheus()
    assert "# TYPE req_total counter" in text
    assert 'req_total{route="/x",status="200"} 3' in text


def test_gauge_set_and_render():
    reg = MetricsRegistry()
    reg.set_gauge("backlog", 7)
    assert "backlog 7" in reg.render_prometheus()


def test_histogram_buckets_cumulative():
    reg = MetricsRegistry()
    reg.observe("lat", 0.03)
    reg.observe("lat", 5.0)
    text = reg.render_prometheus()
    assert "# TYPE lat histogram" in text
    assert 'lat_bucket{le="0.025"} 0' in text
    assert 'lat_bucket{le="0.05"} 1' in text
    assert 'lat_bucket{le="+Inf"} 2' in text
    assert "lat_count 2" in text
    assert "lat_sum" in text


def test_label_values_are_escaped():
    reg = MetricsRegistry()
    reg.inc("c", {"q": 'a"b\\c'})
    text = reg.render_prometheus()
    assert 'q="a\\"b\\\\c"' in text


def test_snapshot_shape():
    reg = MetricsRegistry()
    reg.inc("c")
    snap = reg.snapshot()
    assert set(snap) == {"counters", "gauges", "histograms"}


@pytest.mark.asyncio
async def test_retrieval_records_paths_and_candidates(db):
    from backend.app.models.document import Document
    from backend.app.providers.embedding_provider import HashEmbeddingProvider
    from backend.app.services.chunking_service import ChunkingService
    from backend.app.services.embedding_service import EmbeddingService
    from backend.app.services.indexing_service import IndexingService
    from backend.app.services.retrieval_service import RetrievalService
    from backend.app.core import metrics as m

    from datetime import datetime, timezone

    doc = Document(
        document_name="metrics.jpg", document_type="invoice",
        processing_status="PASS", file_validation={"status": "PASS"},
        extracted_data={"invoice_number": {"value": "M1", "original_value": "M1",
                                           "status": "EXPLICIT",
                                           "evidence": {"source_text": "M1", "page_number": 1}},
                        "line_items": []},
        validation={"overall_status": "PASS"},
        processing_metadata={}, created_at=datetime.now(timezone.utc))
    db.add(doc)
    await db.flush()
    await IndexingService(db, embeddings=EmbeddingService(HashEmbeddingProvider())).index_document(doc.id)

    before = dict(m.metrics.snapshot()["counters"])
    out = await RetrievalService(db, EmbeddingService(HashEmbeddingProvider())).retrieve("M1 invoice")
    snap = m.metrics.snapshot()["counters"]
    assert out["candidates"]
    key = str(tuple(sorted({"vector_path": out["vector_path"],
                            "keyword_path": out["keyword_path"],
                            "degraded": "false"}.items())))
    assert snap["finirag_retrieval_requests_total"].get(key, 0) == \
        before.get("finirag_retrieval_requests_total", {}).get(key, 0) + 1


@pytest.mark.asyncio
async def test_answer_outcome_recorded():
    from backend.app.services.answer_service import AnswerService
    from backend.app.core import metrics as m

    before = dict(m.metrics.snapshot()["counters"])
    svc = AnswerService()
    ans = await svc.answer(question="Total?", evidence={"items": []})
    assert ans.status == "INSUFFICIENT_EVIDENCE"
    snap = m.metrics.snapshot()["counters"]
    key = str(tuple(sorted({"provider": "none", "outcome": "abstained"}.items())))
    assert snap["finirag_answer_requests_total"].get(key, 0) == \
        before.get("finirag_answer_requests_total", {}).get(key, 0) + 1


def test_metrics_endpoint_serves_prometheus():
    from fastapi.testclient import TestClient

    from backend.app.core.database import get_db
    from backend.app.main import app

    async def _fake_db():
        from types import SimpleNamespace

        yield SimpleNamespace(execute=_boom)

    async def _boom(*a, **k):
        raise RuntimeError("no db in this test")

    # /metrics must work even when the database is down.
    from backend.app.main import app as _app

    _app.dependency_overrides[get_db] = _fake_db
    try:
        client = TestClient(_app, raise_server_exceptions=False)
        client.get("/metrics")  # warms the HTTP series (middleware records post-response)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "finirag_http_requests_total" in r.text
        assert 'route="/metrics"' in r.text  # templates, never raw paths
        r2 = client.get("/metrics", params={"format": "json"})
        assert set(r2.json()) == {"counters", "gauges", "histograms"}
    finally:
        _app.dependency_overrides.clear()
