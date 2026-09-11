"""
Unit tests for Settings.

Small surface, but the DATABASE_URL rewrite is what stands between a
working Render deploy and a driver error at startup: Render hands out
`postgres://` URLs and SQLAlchemy's async engine needs
`postgresql+asyncpg://`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.core.config import Settings, settings


# ─────────────────────────────────────────────────────────────────────
# DATABASE_URL rewriting
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("given,expected", [
    # Render's own format
    ("postgres://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
    # The commoner spelling, still missing a driver
    ("postgresql://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
    # Already correct — must be left exactly as-is
    ("postgresql+asyncpg://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
])
def test_database_url_is_normalised_to_the_async_driver(given, expected):
    assert Settings(DATABASE_URL=given).DATABASE_URL == expected


def test_only_the_scheme_is_rewritten():
    """A password or database name containing the scheme text must survive."""
    given = "postgres://user:postgres://weird@host:5432/postgres"
    result = Settings(DATABASE_URL=given).DATABASE_URL

    assert result.startswith("postgresql+asyncpg://")
    assert result.count("postgresql+asyncpg://") == 1
    assert result.endswith("@host:5432/postgres")


def test_a_non_postgres_url_is_left_alone():
    given = "sqlite+aiosqlite:///./test.db"
    assert Settings(DATABASE_URL=given).DATABASE_URL == given


def test_the_live_setting_uses_the_async_driver():
    """Whatever is in .env must be usable by create_async_engine."""
    assert settings.DATABASE_URL.startswith("postgresql+asyncpg://")


# ─────────────────────────────────────────────────────────────────────
# Derived values
# ─────────────────────────────────────────────────────────────────────

def test_file_size_limit_is_reported_in_bytes():
    assert Settings(MAX_FILE_SIZE_MB=10).max_file_size_bytes == 10 * 1024 * 1024
    assert Settings(MAX_FILE_SIZE_MB=1).max_file_size_bytes == 1_048_576


def test_production_is_recognised_only_by_the_exact_value():
    assert Settings(ENVIRONMENT="production").is_production is True
    assert Settings(ENVIRONMENT="development").is_production is False
    assert Settings(ENVIRONMENT="Production").is_production is False


# ─────────────────────────────────────────────────────────────────────
# Defaults and secrets
# ─────────────────────────────────────────────────────────────────────

def test_the_documented_limits_are_in_effect():
    """
    MAX_PAGES=3 is a stated requirement, not a tunable. This reads the
    live setting — .env included — because what matters is the value the
    running gate uses, not the class default it may have overridden.
    """
    assert settings.MAX_PAGES == 3
    assert settings.MAX_FILE_SIZE_MB == 10


def test_no_credential_is_hardcoded_as_a_default():
    """
    Defaults are what a fresh checkout gets. A key baked in here would
    be a key committed to the repository.
    """
    fresh = Settings.model_fields
    assert fresh["GEMINI_API_KEY"].default == ""
    assert fresh["GOOGLE_VISION_CREDENTIALS"].default == ""


def test_the_dataset_path_resolves_to_the_real_folder():
    assert isinstance(settings.DATASET_DIR, Path)
    assert settings.DATASET_DIR.name == "Dataset"


def test_provider_selection_falls_back_to_mistral_and_gemini():
    """The class defaults, with .env taken out of the picture."""
    fields = Settings.model_fields
    assert fields["OCR_PROVIDER"].default == "mistral"
    assert fields["LLM_PROVIDER"].default == "gemini"


def test_the_configured_providers_are_ones_the_code_knows_about():
    """A typo in .env would leave OCRService with no provider at all."""
    assert settings.OCR_PROVIDER in {"mistral", "google_vision"}
    assert settings.LLM_PROVIDER == "gemini"
