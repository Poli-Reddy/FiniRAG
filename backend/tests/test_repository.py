"""
Integration tests for DocumentRepository against the live PostgreSQL
database (master_prompt.md — TEST STRATEGY / DATABASE).

These write real rows through the real SQLAlchemy layer and read them
back, so JSONB round-tripping and ORDER BY are exercised by Postgres
rather than asserted against a stub.

Every test runs inside a transaction that is rolled back afterwards, so
the database is left exactly as it was found — the rows already there
from earlier uploads are never touched, and nothing new survives.

Marked `integration`: deselected by default. Run with
    pytest -m integration
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from backend.app.core.config import settings
from backend.app.models.document import Document
from backend.app.repositories import DocumentRepository

pytestmark = pytest.mark.integration


# ─────────────────────────────────────────────────────────────────────
# Live session, rolled back
# ─────────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture()
async def session() -> AsyncSession:
    """
    A session bound to an open transaction that is rolled back on teardown.

    The repository flushes rather than commits, so rows are visible to
    every query inside this transaction and disappear when it unwinds.
    A fresh engine per test with NullPool keeps asyncpg connections from
    outliving the event loop pytest-asyncio creates for each test.
    """
    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    try:
        conn = await engine.connect()
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"database unreachable: {type(exc).__name__}")

    trans = await conn.begin()
    db = AsyncSession(bind=conn, expire_on_commit=False)
    try:
        yield db
    finally:
        await db.close()
        await trans.rollback()
        await conn.close()
        await engine.dispose()


@pytest.fixture()
def unique_name() -> str:
    """A document name no earlier upload can collide with."""
    return f"pytest-{uuid.uuid4().hex[:12]}.pdf"


def _doc(
    name: str,
    status: str = "PASS",
    created_at: datetime | None = None,
    extracted_data: dict | None = None,
    validation: dict | None = None,
    error: dict | None = None,
) -> Document:
    return Document(
        document_name=name,
        document_type="balance_sheet",
        processing_status=status,
        file_validation={"file_type": "application/pdf", "page_count": 1, "status": "PASS"},
        extracted_data=extracted_data,
        validation=validation,
        processing_metadata={"llm_provider": "gemini", "processing_time_ms": 1234},
        error=error,
        created_at=created_at or datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_created_row_is_readable_back(session, unique_name):
    repo = DocumentRepository(session)
    created = await repo.create(_doc(unique_name, extracted_data={"company": "HDFC"}))

    assert created.id is not None
    fetched = await repo.get_by_id(created.id)

    assert fetched is not None
    assert fetched.document_name == unique_name
    assert fetched.document_type == "balance_sheet"
    assert fetched.processing_status == "PASS"
    assert fetched.extracted_data == {"company": "HDFC"}
    assert fetched.processing_metadata["processing_time_ms"] == 1234


@pytest.mark.asyncio
async def test_jsonb_returns_money_as_the_exact_string_it_was_given(session, unique_name):
    """
    The whole Decimal pipeline is pointless if Postgres hands the digits
    back as a float. JSONB stores JSON numbers as numeric, so amounts are
    written as strings — this proves they survive the round trip intact.
    """
    amounts = {
        "total_assets": "17995066442.55",
        "tolerance": "17995066.44255",
        "zero": "0",
        "negative": "-1234.05",
    }
    repo = DocumentRepository(session)
    created = await repo.create(_doc(unique_name, extracted_data=amounts))

    session.expunge(created)
    fetched = await repo.get_by_id(created.id)

    assert fetched.extracted_data == amounts
    assert all(isinstance(v, str) for v in fetched.extracted_data.values())


@pytest.mark.asyncio
async def test_nullable_columns_come_back_as_none_not_empty(session, unique_name):
    """A failed attempt has no extraction — that must read as NULL."""
    repo = DocumentRepository(session)
    created = await repo.create(_doc(
        unique_name,
        status="FAILED",
        extracted_data=None,
        validation=None,
        error={"code": "PAGE_LIMIT_EXCEEDED", "message": "PAGE_LIMIT_EXCEEDED"},
    ))

    session.expunge(created)
    fetched = await repo.get_by_id(created.id)

    assert fetched.extracted_data is None
    assert fetched.validation is None
    assert fetched.error == {"code": "PAGE_LIMIT_EXCEEDED", "message": "PAGE_LIMIT_EXCEEDED"}


@pytest.mark.asyncio
async def test_nested_validation_structure_survives_the_round_trip(session, unique_name):
    validation = {
        "checks": [
            {"name": "2021_sum_assets", "calculated_value": "17995066442",
             "reported_value": "17995066442", "variance": "0", "status": "PASS"},
            {"name": "2021_sum_liabilities", "status": "NOT_APPLICABLE",
             "reason": "total_liabilities not readable"},
        ],
        "overall_status": "PASS",
        "summary": {"pass": 1, "fail": 0, "not_applicable": 1},
    }
    repo = DocumentRepository(session)
    created = await repo.create(_doc(unique_name, validation=validation))

    session.expunge(created)
    fetched = await repo.get_by_id(created.id)

    assert fetched.validation == validation
    assert fetched.validation["checks"][0]["variance"] == "0"


@pytest.mark.asyncio
async def test_get_by_id_of_an_unknown_uuid_is_none(session):
    assert await DocumentRepository(session).get_by_id(uuid.uuid4()) is None


# ─────────────────────────────────────────────────────────────────────
# Latest-by-name (every attempt is a new row, no upsert)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reprocessing_appends_a_row_instead_of_overwriting(session, unique_name):
    """
    The core persistence rule. Uploading the same document twice must
    leave two rows — the earlier attempt is part of the audit trail.
    """
    now = datetime.now(timezone.utc)
    repo = DocumentRepository(session)

    first = await repo.create(_doc(unique_name, "FAILED", created_at=now - timedelta(minutes=5)))
    second = await repo.create(_doc(unique_name, "PASS", created_at=now))

    count = await session.execute(
        select(func.count(Document.id)).where(Document.document_name == unique_name)
    )
    assert count.scalar_one() == 2
    assert first.id != second.id

    still_there = await repo.get_by_id(first.id)
    assert still_there is not None
    assert still_there.processing_status == "FAILED"


@pytest.mark.asyncio
async def test_lookup_by_name_returns_the_newest_attempt(session, unique_name):
    now = datetime.now(timezone.utc)
    repo = DocumentRepository(session)

    await repo.create(_doc(unique_name, "FAILED", created_at=now - timedelta(hours=2)))
    await repo.create(_doc(unique_name, "FAILED", created_at=now - timedelta(hours=1)))
    newest = await repo.create(_doc(unique_name, "PASS", created_at=now))

    latest = await repo.get_latest_by_name(unique_name)

    assert latest.id == newest.id
    assert latest.processing_status == "PASS"


@pytest.mark.asyncio
async def test_lookup_by_name_is_unaffected_by_newer_rows_for_other_documents(session):
    now = datetime.now(timezone.utc)
    repo = DocumentRepository(session)
    mine = f"pytest-{uuid.uuid4().hex[:12]}.pdf"
    other = f"pytest-{uuid.uuid4().hex[:12]}.pdf"

    wanted = await repo.create(_doc(mine, "PASS", created_at=now - timedelta(hours=1)))
    await repo.create(_doc(other, "FAILED", created_at=now))

    latest = await repo.get_latest_by_name(mine)
    assert latest.id == wanted.id


@pytest.mark.asyncio
async def test_lookup_of_a_name_never_uploaded_is_none(session):
    repo = DocumentRepository(session)
    assert await repo.get_latest_by_name(f"never-{uuid.uuid4().hex}.pdf") is None


@pytest.mark.asyncio
async def test_lookup_matches_the_name_exactly(session, unique_name):
    """
    No prefix or fuzzy matching. "Balance Sheet 2021.pdf" must not answer
    a request for "Balance Sheet 2021".
    """
    repo = DocumentRepository(session)
    await repo.create(_doc(unique_name))

    assert await repo.get_latest_by_name(unique_name.removesuffix(".pdf")) is None
    assert await repo.get_latest_by_name(unique_name.upper()) is None
    assert await repo.get_latest_by_name(unique_name) is not None


# ─────────────────────────────────────────────────────────────────────
# Listing
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_listing_total_counts_the_whole_table_not_the_page(session, unique_name):
    """
    The dashboard shows "showing N of TOTAL". If total were len(page) it
    would always equal the page size and the pager would never appear.
    """
    repo = DocumentRepository(session)
    _, before = await repo.list_all(limit=1)

    for i in range(3):
        await repo.create(_doc(f"{unique_name}-{i}"))

    page, after = await repo.list_all(limit=1)

    assert after == before + 3
    assert len(page) == 1


@pytest.mark.asyncio
async def test_listing_is_newest_first(session, unique_name):
    now = datetime.now(timezone.utc)
    repo = DocumentRepository(session)

    # Dated into the future so they sort above the rows already stored.
    await repo.create(_doc(f"{unique_name}-old", created_at=now + timedelta(days=1)))
    await repo.create(_doc(f"{unique_name}-mid", created_at=now + timedelta(days=2)))
    await repo.create(_doc(f"{unique_name}-new", created_at=now + timedelta(days=3)))

    page, _ = await repo.list_all(limit=3)

    assert [d.document_name for d in page] == [
        f"{unique_name}-new", f"{unique_name}-mid", f"{unique_name}-old",
    ]


@pytest.mark.asyncio
async def test_offset_pages_through_without_repeating_rows(session, unique_name):
    now = datetime.now(timezone.utc)
    repo = DocumentRepository(session)
    for i in range(3):
        await repo.create(_doc(
            f"{unique_name}-{i}", created_at=now + timedelta(days=10, seconds=i)
        ))

    first, _ = await repo.list_all(limit=2, offset=0)
    second, _ = await repo.list_all(limit=2, offset=2)

    ids = [d.id for d in first] + [d.id for d in second]
    assert len(set(ids)) == len(ids)
    assert first[0].document_name == f"{unique_name}-2"


@pytest.mark.asyncio
async def test_listing_returns_full_rows_not_just_names(session, unique_name):
    repo = DocumentRepository(session)
    await repo.create(_doc(
        f"{unique_name}",
        created_at=datetime.now(timezone.utc) + timedelta(days=20),
        extracted_data={"company": "HDFC"},
    ))

    page, _ = await repo.list_all(limit=1)
    row = page[0]

    assert row.document_name == unique_name
    assert row.extracted_data == {"company": "HDFC"}
    assert row.file_validation["page_count"] == 1
    assert row.created_at is not None
