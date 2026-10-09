"""API reliability guards: health readiness, request-id, rate limiting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.api.rate_limit import _limiter, rate_limit
from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.main import app


class _FakeSession:
    async def execute(self, *a, **k):
        return SimpleNamespace(scalar=lambda *a, **k: 7)


def _client() -> TestClient:
    async def _fake_db():
        yield _FakeSession()

    app.dependency_overrides[get_db] = _fake_db
    return TestClient(app, raise_server_exceptions=False)


def test_health_reports_operational_fields():
    client = _client()
    try:
        body = client.get("/api/v1/health").json()
        for key in ("status", "database", "embedding_model", "embedding_configured",
                    "retrieval_mode", "index_version", "indexed_chunks",
                    "indexed_documents", "provider_breakers"):
            assert key in body, f"health missing {key}"
    finally:
        app.dependency_overrides.clear()


def test_response_carries_request_id_header():
    client = _client()
    try:
        r = client.get("/api/v1/health")
        assert r.headers.get("x-request-id")
        # A supplied id is echoed, enabling end-to-end correlation.
        r2 = client.get("/api/v1/health", headers={"X-Request-ID": "abc-123"})
        assert r2.headers["x-request-id"] == "abc-123"
    finally:
        app.dependency_overrides.clear()


class _Req:
    def __init__(self, host: str):
        self.headers = {}
        self.client = SimpleNamespace(host=host)


@pytest.mark.asyncio
async def test_rate_limit_dependency_blocks(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 2)
    key = "10.0.0.9"
    _limiter._hits.pop(key, None)
    await rate_limit(_Req(key))  # 1
    await rate_limit(_Req(key))  # 2
    with pytest.raises(HTTPException) as ei:
        await rate_limit(_Req(key))  # 3 -> blocked
    assert ei.value.status_code == 429
    assert ei.value.detail["code"] == "RATE_LIMITED"
    assert ei.value.headers.get("Retry-After")


@pytest.mark.asyncio
async def test_rate_limit_respects_forwarded_for(monkeypatch):
    """Client identity comes from X-Forwarded-For behind a proxy."""
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 1)
    _limiter._hits.pop("203.0.113.5", None)

    class ProxyReq:
        def __init__(self):
            self.headers = {"x-forwarded-for": "203.0.113.5, 10.0.0.1"}
            self.client = SimpleNamespace(host="10.0.0.1")

    await rate_limit(ProxyReq())
    with pytest.raises(HTTPException):
        await rate_limit(ProxyReq())
