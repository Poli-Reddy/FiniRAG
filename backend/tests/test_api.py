"""
Unit tests for the REST API layer.

The database and the processing pipeline are both stubbed: what is under
test here is the HTTP contract — status codes, the document-type gate,
filename sanitization, error shape, and the promise that a stack trace
never reaches the client.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.core.database import get_db
from backend.app.main import app
from backend.app.schemas.document import (
    VALID_DOCUMENT_TYPES,
    DocumentProcessResponse,
    FileValidationResult,
    ProcessingMetadata,
)
from backend.app.api.routes.documents import _sanitize_filename


# ─────────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────────

class FakeSession:
    """Stands in for AsyncSession. Raises on execute when `healthy` is False."""

    def __init__(self, healthy: bool = True):
        self.healthy = healthy

    async def execute(self, *args, **kwargs):
        if not self.healthy:
            raise RuntimeError("connection refused")
        return SimpleNamespace(scalar=lambda: 1)


def _ok_response(filename: str, document_type: str) -> DocumentProcessResponse:
    return DocumentProcessResponse(
        document_name=filename,
        document_type=document_type,
        processing_status="PASS",
        file_validation=FileValidationResult(
            file_type="application/pdf", is_supported=True,
            is_readable=True, page_count=1, status="PASS",
        ),
        extracted_data={"company_name": {"value": "HDFC Bank Limited"}},
        validation={
            "checks": [{
                "name": "2021_sum_assets",
                "calculated_value": "17995066442",
                "reported_value": "17995066442",
                "variance": "0",
                "tolerance": "17995066.442",
                "status": "PASS",
            }],
            "overall_status": "PASS",
            "summary": {"pass": 1, "fail": 0, "not_applicable": 0},
        },
        processing_metadata=ProcessingMetadata(
            text_extraction_method="native_pdf", llm_provider="gemini",
            processed_at=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=1234,
        ),
    )


class StubService:
    """Records what the route handed the pipeline, without running it."""

    calls: list[dict] = []
    documents: list = []
    lookup = None

    def __init__(self, db):
        self._db = db

    async def process_document(self, file_bytes, filename, document_type):
        StubService.calls.append({
            "filename": filename,
            "document_type": document_type,
            "size": len(file_bytes),
        })
        return _ok_response(filename, document_type)

    async def list_documents(self, limit=100, offset=0):
        return StubService.documents, len(StubService.documents)

    async def get_document_by_name(self, document_name):
        return StubService.lookup


@pytest.fixture()
def client(monkeypatch):
    """A TestClient with the DB and the pipeline replaced by stubs."""
    StubService.calls = []
    StubService.documents = []
    StubService.lookup = None

    async def _fake_db():
        yield FakeSession(healthy=True)

    monkeypatch.setattr(
        "backend.app.api.routes.documents.DocumentService", StubService
    )
    app.dependency_overrides[get_db] = _fake_db
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def _upload(document_type: str = "balance_sheet", filename: str = "bs.pdf"):
    return {
        "files": {"file": (filename, io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
        "data": {"document_type": document_type},
    }


# ─────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────

def test_health_reports_healthy_when_database_answers(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["database"] == "connected"
    assert body["timestamp"]


def test_health_degrades_instead_of_crashing_when_database_is_down(monkeypatch):
    """A dead database is a reportable state, not a 500."""
    async def _dead_db():
        yield FakeSession(healthy=False)

    app.dependency_overrides[get_db] = _dead_db
    try:
        r = TestClient(app).get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"
        assert r.json()["database"] == "disconnected"
    finally:
        app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────────────
# Document type gate (Trap P10 — the user selects, we never classify)
# ─────────────────────────────────────────────────────────────────────

def test_valid_document_types_are_exactly_the_four_in_the_brief():
    assert VALID_DOCUMENT_TYPES == {
        "invoice", "balance_sheet", "profit_and_loss", "cash_flow_statement",
    }


@pytest.mark.parametrize("doc_type", sorted(VALID_DOCUMENT_TYPES))
def test_every_valid_document_type_is_accepted(client, doc_type):
    up = _upload(doc_type)
    r = client.post("/api/v1/documents/process", files=up["files"], data=up["data"])
    assert r.status_code == 200
    assert r.json()["document_type"] == doc_type


@pytest.mark.parametrize("doc_type", ["", "receipt", "BALANCE_SHEET", "tax_return"])
def test_invalid_document_type_is_rejected_with_a_code(client, doc_type):
    up = _upload(doc_type)
    r = client.post("/api/v1/documents/process", files=up["files"], data=up["data"])
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_DOCUMENT_TYPE"


def test_document_type_is_required_not_inferred(client):
    """Omitting the type must fail, never fall back to guessing from content."""
    r = client.post(
        "/api/v1/documents/process",
        files={"file": ("bs.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert r.status_code == 422


def test_document_type_is_not_inferred_from_the_filename(client):
    """A file called invoice.pdf submitted as a balance sheet stays one."""
    up = _upload("balance_sheet", filename="invoice.pdf")
    r = client.post("/api/v1/documents/process", files=up["files"], data=up["data"])
    assert r.json()["document_type"] == "balance_sheet"
    assert StubService.calls[-1]["document_type"] == "balance_sheet"


def test_file_is_required(client):
    r = client.post("/api/v1/documents/process", data={"document_type": "invoice"})
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────
# Filename sanitization
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd", "passwd"),
    ("/absolute/path/report.pdf", "report.pdf"),
    ("plain.pdf", "plain.pdf"),
    ("", "unnamed"),
    ("   ", "unnamed"),
    ("bad\x00name.pdf", "badname.pdf"),
])
def test_sanitize_filename_strips_paths_and_null_bytes(raw, expected):
    assert _sanitize_filename(raw) == expected


def test_uploaded_path_traversal_never_reaches_the_pipeline(client):
    up = _upload("invoice", filename="../../../etc/passwd")
    r = client.post("/api/v1/documents/process", files=up["files"], data=up["data"])
    assert r.status_code == 200
    assert "/" not in StubService.calls[-1]["filename"]
    assert ".." not in StubService.calls[-1]["filename"]


# ─────────────────────────────────────────────────────────────────────
# Response contract
# ─────────────────────────────────────────────────────────────────────

def test_process_response_has_every_contract_key(client):
    up = _upload()
    body = client.post(
        "/api/v1/documents/process", files=up["files"], data=up["data"]
    ).json()
    for key in (
        "document_name", "document_type", "processing_status",
        "file_validation", "extracted_data", "validation",
        "processing_metadata", "error",
    ):
        assert key in body, f"missing response key: {key}"


def test_processing_failure_is_a_200_result_not_an_http_error(client, monkeypatch):
    """
    A rejected file is a valid processing outcome the client must be able
    to read, not a transport-level failure.
    """
    async def _failing(self, file_bytes, filename, document_type):
        return DocumentProcessResponse(
            document_name=filename,
            document_type=document_type,
            processing_status="FAILED",
            file_validation=FileValidationResult(
                status="FAIL", error="PAGE_LIMIT_EXCEEDED", page_count=7,
            ),
            processing_metadata=ProcessingMetadata(),
            error={"code": "PAGE_LIMIT_EXCEEDED", "message": "PAGE_LIMIT_EXCEEDED"},
        )

    monkeypatch.setattr(StubService, "process_document", _failing)
    up = _upload()
    r = client.post("/api/v1/documents/process", files=up["files"], data=up["data"])
    assert r.status_code == 200
    body = r.json()
    assert body["processing_status"] == "FAILED"
    assert body["error"]["code"] == "PAGE_LIMIT_EXCEEDED"
    assert body["extracted_data"] is None


def test_money_crosses_the_wire_as_exact_strings_not_floats(client):
    """
    17995066.442 as a JSON float is a rounding hazard. The API must hand
    the client the same digits the validator computed.
    """
    up = _upload()
    body = client.post(
        "/api/v1/documents/process", files=up["files"], data=up["data"]
    ).json()
    check = body["validation"]["checks"][0]
    assert check["tolerance"] == "17995066.442"
    assert isinstance(check["calculated_value"], str)


# ─────────────────────────────────────────────────────────────────────
# Listing and lookup
# ─────────────────────────────────────────────────────────────────────

def _stub_doc(name: str, status: str = "PASS"):
    return SimpleNamespace(
        id="11111111-2222-3333-4444-555555555555",
        document_name=name,
        document_type="balance_sheet",
        processing_status=status,
        file_validation={"status": "PASS"},
        extracted_data={"x": 1},
        validation={"overall_status": "PASS"},
        processing_metadata={"llm_provider": "gemini"},
        error=None,
        created_at=datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc),
    )


def test_list_documents_returns_items_and_total(client):
    StubService.documents = [_stub_doc("a.pdf"), _stub_doc("b.pdf", "FAILED")]
    body = client.get("/api/v1/documents").json()
    assert body["total"] == 2
    assert [d["document_name"] for d in body["documents"]] == ["a.pdf", "b.pdf"]
    assert body["documents"][1]["processing_status"] == "FAILED"


def test_list_documents_is_empty_not_null_when_nothing_stored(client):
    body = client.get("/api/v1/documents").json()
    assert body == {"documents": [], "total": 0}


def test_lookup_by_name_returns_the_stored_record(client):
    StubService.lookup = _stub_doc("Consolidated Balance Sheet 2021.pdf")
    r = client.get("/api/v1/documents/Consolidated Balance Sheet 2021.pdf")
    assert r.status_code == 200
    body = r.json()
    assert body["document_name"] == "Consolidated Balance Sheet 2021.pdf"
    assert body["validation"]["overall_status"] == "PASS"
    assert body["created_at"].startswith("2026-09-10")


def test_lookup_of_unknown_document_is_404_with_a_code(client):
    StubService.lookup = None
    r = client.get("/api/v1/documents/never-uploaded.pdf")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"


# ─────────────────────────────────────────────────────────────────────
# Bounded uploads + access control (audit findings #4, P0-2, #21)
# ─────────────────────────────────────────────────────────────────────

def test_oversized_upload_rejected_before_processing(client, monkeypatch):
    """Finding #4: the size gate fires on a bounded read, not after buffering."""
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 0)
    # MAX_FILE_SIZE_MB=0 -> max_bytes=0, so any content exceeds the cap.
    r = client.post(
        "/api/v1/documents/process",
        files={"file": ("big.pdf", io.BytesIO(b"%PDF-1.4 " + b"x" * 100), "application/pdf")},
        data={"document_type": "balance_sheet"},
    )
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert StubService.calls == []  # pipeline never ran


def test_upload_at_limit_passes_size_gate(client, monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    body = b"%PDF-1.4 " + b"x" * 100
    assert len(body) <= 1 * 1024 * 1024
    up = {
        "files": {"file": ("bs.pdf", io.BytesIO(body), "application/pdf")},
        "data": {"document_type": "balance_sheet"},
    }
    r = client.post("/api/v1/documents/process", files=up["files"], data=up["data"])
    assert r.status_code == 200
    assert len(StubService.calls) == 1


def test_read_endpoints_require_key_when_configured(client, monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "API_KEYS", "secret-1")
    assert client.get("/api/v1/documents").status_code == 401
    assert client.get("/api/v1/documents/whatever.pdf").status_code == 401
    assert client.get(
        "/api/v1/documents/11111111-2222-3333-4444-555555555555/index-status"
    ).status_code == 401
    assert client.get(
        "/api/v1/indexing/jobs/11111111-2222-3333-4444-555555555555"
    ).status_code == 401
    authed = {"Authorization": "Bearer secret-1"}
    assert client.get("/api/v1/documents", headers=authed).status_code == 200


def test_rate_limit_applies_to_reads(client, monkeypatch):
    from backend.app.api import rate_limit as rl_module
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 1)
    rl_module._limiter._hits.clear()
    assert client.get("/api/v1/documents").status_code == 200
    r = client.get("/api/v1/documents")
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "RATE_LIMITED"
    rl_module._limiter._hits.clear()


# ─────────────────────────────────────────────────────────────────────
# Error handling (Trap P14, Correction #10)
# ─────────────────────────────────────────────────────────────────────

def test_unhandled_errors_never_leak_internals(client, monkeypatch):
    async def _explode(self, file_bytes, filename, document_type):
        raise RuntimeError(
            "psycopg: password=hunter2 host=db.internal failed at line 402"
        )

    monkeypatch.setattr(StubService, "process_document", _explode)
    up = _upload()
    r = client.post("/api/v1/documents/process", files=up["files"], data=up["data"])

    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"

    leaked = r.text.lower()
    for secret in ("hunter2", "psycopg", "traceback", "db.internal", "line 402"):
        assert secret not in leaked, f"error response leaked {secret!r}"


# ─────────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────────

def test_root_serves_the_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_static_assets_are_mounted(client):
    r = client.get("/static/js/app.js")
    assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────
# Ingress body cap + production auth fail-closed (findings #11/#12)
# ─────────────────────────────────────────────────────────────────────

def test_ingress_rejects_lying_content_length_before_parsing(client, monkeypatch):
    """Finding #12: an oversized declared body is rejected at ingress,
    before multipart parsing — the pipeline must never run."""
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    up = _upload()
    # Small actual body, lying huge Content-Length: ingress must 413.
    r = client.post("/api/v1/documents/process",
                    files=up["files"], data=up["data"],
                    headers={"content-length": str(500 * 1024 * 1024)})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert StubService.calls == []


@pytest.mark.asyncio
async def test_production_boot_refuses_without_api_keys(monkeypatch):
    """Finding #11: ENVIRONMENT=production with no API_KEYS refuses boot."""
    import backend.app.main as main_mod

    monkeypatch.setattr(main_mod.settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(main_mod.settings, "API_KEYS", "")
    monkeypatch.setattr(main_mod.settings, "ALLOW_OPEN_API", False)

    async def fake_init():
        raise AssertionError("must not reach init_db")

    monkeypatch.setattr(main_mod, "init_db", fake_init)
    with pytest.raises(RuntimeError, match="API_KEYS"):
        async with main_mod.lifespan(main_mod.app):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_production_boot_open_only_with_explicit_opt_in(monkeypatch):
    """ALLOW_OPEN_API=true is the single deliberate path to an open prod boot."""
    import backend.app.main as main_mod

    monkeypatch.setattr(main_mod.settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(main_mod.settings, "API_KEYS", "")
    monkeypatch.setattr(main_mod.settings, "ALLOW_OPEN_API", True)

    booted = {}

    async def fake_init():
        booted["yes"] = True

    async def fake_recover():
        return 0

    monkeypatch.setattr(main_mod, "init_db", fake_init)
    monkeypatch.setattr("backend.app.workers.indexing_worker.recover_stale_jobs",
                        fake_recover)
    async with main_mod.lifespan(main_mod.app):
        pass
    assert booted.get("yes") is True


def test_untrusted_forwarded_header_ignored_by_default(monkeypatch):
    """Finding #13: with TRUST_X_FORWARDED_FOR off (the default), a forged
    header does not change client identity."""
    from fastapi import Request

    from backend.app.api.rate_limit import client_key
    from backend.app.core.config import settings

    assert settings.TRUST_X_FORWARDED_FOR is False
    scope = {"type": "http", "headers": [(b"x-forwarded-for", b"9.9.9.9")],
             "client": ("10.0.0.7", 1234)}
    assert client_key(Request(scope)) == "10.0.0.7"
