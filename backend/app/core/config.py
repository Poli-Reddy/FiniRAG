"""
FiniRAG Configuration Module.

Reads all settings from environment variables with sensible defaults.
Secrets are NEVER hardcoded. See .env.example for reference.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

from backend.app.core.key_pool import KeyPool, parse_keys


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        # Resolve .env from project root (FiniRAG/.env)
        env_file=str(Path(__file__).resolve().parent.parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Database ──
    DATABASE_URL: str = "postgresql+asyncpg://localhost:5432/finirag"

    # ── API Keys (NEVER hardcoded) ──
    #
    # Each provider takes either one key or several. The plural form accepts a
    # comma- or newline-separated list and exists for one reason: the free
    # tiers are capped per day, and 50 documents outlast a single key. The
    # singular form is still read and still sufficient — a deployment with one
    # key per provider needs no other change, and that is the intended steady
    # state once a run is finished.
    #
    # Keys are pooled in the order given: the first is used until it refuses.
    GEMINI_API_KEY: str = ""
    GEMINI_API_KEYS: str = ""
    GROQ_API_KEY: str = ""
    GROQ_API_KEYS: str = ""
    MISTRAL_API_KEY: str = ""
    MISTRAL_API_KEYS: str = ""

    # Retained so an existing deployment's env does not break on upgrade.
    # Vision is no longer the OCR backend — it required billing to be enabled
    # on the GCP project, which this project does not have.
    GOOGLE_VISION_CREDENTIALS: str = ""

    # ── Provider Selection ──
    # "mistral" is the OCR backend. It is the only one here that returns
    # per-block bounding boxes and page indices, which is what makes evidence
    # traceable to the source instead of an echo of the extracted value.
    OCR_PROVIDER: str = "mistral"
    LLM_PROVIDER: str = "gemini"

    MISTRAL_OCR_MODEL: str = "mistral-ocr-latest"

    # Extraction/answer LLM. gemini-3.8-flash is the currently supported model
    # (2.5-series is closed to new users; 3.6 is quota-capped on this account).
    # One knob governs the extraction provider, the answer service and the
    # LLM question classifier so they cannot drift apart.
    GEMINI_MODEL: str = "gemini-3.8-flash"

    # ── Embeddings (RAG) ──
    # gemini-embedding-2 (GA April 2026): default 3072 dims, MRL-truncated via
    # output_dimensionality to 768/1536/3072 (auto-normalized). text-embedding-004
    # was retired Jan 2026 and must not be used. A model/dim change requires
    # full re-embedding (spaces are incompatible).
    EMBEDDING_MODEL: str = "gemini-embedding-2"
    EMBEDDING_DIM: int = 768
    EMBEDDING_BATCH_SIZE: int = 8
    EMBEDDING_TIMEOUT_S: float = 30.0
    EMBEDDING_MAX_RETRIES: int = 3

    # ── RAG / Retrieval ──
    INDEX_VERSION: str = "v1"
    CHUNK_TARGET_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 50
    RETRIEVAL_LIMIT_PER_PATH: int = 8
    RETRIEVAL_RRF_K: int = 60
    QA_MAX_CONTEXT_CHUNKS: int = 8
    QA_REQUEST_TIMEOUT_S: float = 60.0

    # Second LLM backend, tried when the primary refuses a call. Empty
    # GROQ_API_KEY disables it and the pipeline runs on Gemini alone.
    #
    # The vision model is not interchangeable with the text one: it is the
    # model on the account that accepts image parts *and* honours
    # response_format=json_object. Since no document in the dataset has a
    # native text layer, losing that property disables the fallback entirely.
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_VISION_MODEL: str = "qwen/qwen3.8-27b"

    # ── Application ──
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    MAX_FILE_SIZE_MB: int = 10
    MAX_PAGES: int = 3

    # ── Paths ──
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATASET_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "Dataset"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def fix_database_url(cls, v: str) -> str:
        """Render provides postgres:// but SQLAlchemy needs postgresql+asyncpg://"""
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgresql://") and "+asyncpg" not in v:
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


# Singleton instance
settings = Settings()


# ── Key pools ──
#
# Module-level singletons on purpose. A pool carries the record of which keys
# are spent, and that record has to outlive the request that discovered it.
# Rebuilding a pool per request would hand every request the key that was
# already refused, and a 50-document run would make no progress at all.
#
# Empty pools are fine and mean "this provider is not configured" — the
# provider layer checks `if pool:` and skips itself rather than failing.
GEMINI_KEYS = KeyPool(
    keys=parse_keys(settings.GEMINI_API_KEYS, settings.GEMINI_API_KEY),
    provider="gemini",
)
GROQ_KEYS = KeyPool(
    keys=parse_keys(settings.GROQ_API_KEYS, settings.GROQ_API_KEY),
    provider="groq",
)
MISTRAL_KEYS = KeyPool(
    keys=parse_keys(settings.MISTRAL_API_KEYS, settings.MISTRAL_API_KEY),
    provider="mistral",
)