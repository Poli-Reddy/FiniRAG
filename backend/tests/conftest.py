"""
Shared pytest fixtures and builders for the FiniRAG test suite.

Tests are split into two groups:
  - Unit tests (default): no network, no database, fast.
  - Integration tests (@pytest.mark.integration): hit Gemini / the dataset.
    Run with:  pytest -m integration
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from backend.app.schemas.extraction import (
    Evidence,
    ExtractedValue,
    FieldStatus,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "Dataset"


# ─────────────────────────────────────────────────────────────────────
# No live provider calls in the unit suite
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _no_live_llm_keys(monkeypatch):
    """Blank LLM credentials for every test.

    The development shell may carry real keys in .env; unit tests must be
    deterministic and offline. Blanking forces the documented offline paths
    (template answers, hash-style behavior where applicable). Live checks
    live in scripts/ and evaluation/rag_live_evaluation.py instead.
    """
    from backend.app.core.config import settings
    from backend.app.core import resilience

    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS",
                 "GROQ_API_KEY", "GROQ_API_KEYS"):
        monkeypatch.setattr(settings, name, "", raising=False)
    # No retries or rate limiting in the unit suite: fast and deterministic.
    # The retry/breaker and limiter logic has its own focused tests.
    monkeypatch.setattr(settings, "PROVIDER_MAX_RETRIES", 0)
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 1_000_000)
    resilience.reset_breakers()


# ─────────────────────────────────────────────────────────────────────
# ExtractedValue builders
# ─────────────────────────────────────────────────────────────────────

def ev(value, status: FieldStatus = FieldStatus.EXPLICIT, unit: str | None = None) -> ExtractedValue:
    """Build an EXPLICIT numeric ExtractedValue from a number-like input."""
    dec = Decimal(str(value))
    if dec == 0 and status is FieldStatus.EXPLICIT:
        status = FieldStatus.EXPLICIT_ZERO
    return ExtractedValue(
        value=dec,
        original_value=str(value),
        status=status,
        evidence=Evidence(source_text=str(value), page_number=1),
        unit=unit,
    )


def ev_missing() -> ExtractedValue:
    """Build a MISSING ExtractedValue (field absent from the document)."""
    return ExtractedValue(value=None, original_value="", status=FieldStatus.MISSING)


def ev_unreadable(original: str = "###") -> ExtractedValue:
    """Build an UNREADABLE ExtractedValue (present but not legible)."""
    return ExtractedValue(
        value=None,
        original_value=original,
        status=FieldStatus.UNREADABLE,
        evidence=Evidence(source_text=original, page_number=1),
    )


def ev_zero() -> ExtractedValue:
    """Build an EXPLICIT_ZERO ExtractedValue (document shows 0 or a nil dash)."""
    return ExtractedValue(
        value=Decimal("0"),
        original_value="-",
        status=FieldStatus.EXPLICIT_ZERO,
        evidence=Evidence(source_text="-", page_number=1),
    )


def check_by_name(result, name: str):
    """Find a single ValidationCheck by name, or None."""
    for c in result.checks:
        if c.name == name:
            return c
    return None


# ─────────────────────────────────────────────────────────────────────
# Dataset fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def dataset_dir() -> Path:
    if not DATASET.is_dir():
        pytest.skip(f"Dataset directory not found at {DATASET}")
    return DATASET


@pytest.fixture(scope="session")
def sample_pdf(dataset_dir: Path) -> bytes:
    path = dataset_dir / "Balance Sheet" / "Consolidated Balance Sheet 2021.pdf"
    if not path.is_file():
        pytest.skip(f"Missing sample PDF: {path}")
    return path.read_bytes()


@pytest.fixture(scope="session")
def sample_jpeg(dataset_dir: Path) -> bytes:
    path = dataset_dir / "Invoices" / "batch1-1109.jpg"
    if not path.is_file():
        pytest.skip(f"Missing sample JPEG: {path}")
    return path.read_bytes()


@pytest.fixture(scope="session")
def rotated_jpeg(dataset_dir: Path) -> bytes:
    """The EXIF-rotated phone photo (dataset trap)."""
    path = dataset_dir / "Invoices" / "20251118_000612.jpg"
    if not path.is_file():
        pytest.skip(f"Missing rotated JPEG: {path}")
    return path.read_bytes()
