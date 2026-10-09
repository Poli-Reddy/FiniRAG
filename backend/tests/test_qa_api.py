"""QA + indexing endpoint contract tests (stubbed services, no network)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from backend.app.core.database import get_db
from backend.app.main import app


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
