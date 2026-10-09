"""
FiniRAG Embedding provider — gemini-embedding-2 via official google-genai SDK.

Model: gemini-embedding-2 (GA April 2026). Default 3072 dims, MRL-truncated
with output_dimensionality=768 (auto-normalized). Input limit 8192 tokens.
Task formatting (embedding-2 has NO task_type param):
  documents: "title: {title} | text: {content}"
  queries:   "task: question answering | query: {content}"
Each text requires its own embed_content call (multi-input aggregates).
HashEmbeddingProvider is TEST/OFFLINE ONLY — never production semantic search.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from abc import ABC, abstractmethod

from backend.app.core.config import settings

logger = logging.getLogger("finirag")

EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768
MAX_INPUT_CHARS = 30000  # ~8192 tokens safety bound
MAX_BATCH_ITEMS = 64


class EmbeddingError(Exception):
    def __init__(self, message: str, code: str = "EMBEDDING_FAILED"):
        super().__init__(message)
        self.code = code


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str],
                          titles: list[str] | None = None) -> list[list[float]]:
        """Embed document texts. Each input is formatted as a document
        (`title: {title} | text: {content}`) before the API call. When
        titles are omitted, "none" is used — never invented metadata."""
        ...

    @property
    def model(self) -> str:
        return getattr(self, "_model", "")

    @property
    def dim(self) -> int:
        return getattr(self, "_dim", 768)


def validate_vectors(vectors: list[list[float]], dim: int) -> None:
    if not vectors:
        raise EmbeddingError("empty embedding response", "EMPTY_RESPONSE")
    for v in vectors:
        if len(v) != dim:
            raise EmbeddingError(f"embedding dim mismatch {len(v)} != {dim}",
                                 "DIM_MISMATCH")
        for x in v:
            if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                raise EmbeddingError("non-finite embedding value", "NON_FINITE")


def format_document(content: str, title: str | None = None) -> str:
    t = (title or "none").strip() or "none"
    return f"title: {t} | text: {(content or '').strip()}"


def format_query(content: str) -> str:
    return f"task: question answering | query: {(content or '').strip()}"


class HashEmbeddingProvider(EmbeddingProvider):
    """TEST/OFFLINE ONLY. Deterministic hash vectors, NOT semantic embeddings.

    Must never silently stand in for production. Label any report using it
    as OFFLINE/HASH, never as live semantic quality.
    """

    def __init__(self, dim: int = EMBEDDING_DIM, model: str = "hash-768-OFFLINE"):
        self._model = model
        self._dim = dim

    async def embed_texts(self, texts: list[str],
                          titles: list[str] | None = None) -> list[list[float]]:
        return [self._embed_one(t or "") for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for block in range(self._dim // 32):
            digest = hashlib.sha256(f"{text}\x00{block}".encode()).digest()
            for i in range(32):
                a = digest[(i * 2) % len(digest)]
                b = digest[(i * 2 + 1) % len(digest)]
                vec[block * 32 + i] = ((a * 256 + b) / 65535.0) * 2.0 - 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Live adapter. Uses ONLY google-genai SDK (no legacy fallback)."""

    def __init__(self, model: str | None = None, dim: int | None = None,
                 batch_size: int | None = None, timeout_s: float | None = None,
                 max_retries: int | None = None, api_key: str = ""):
        self._model = model or settings.EMBEDDING_MODEL or EMBEDDING_MODEL
        self._dim = dim or settings.EMBEDDING_DIM or EMBEDDING_DIM
        self.batch_size = max(1, min(batch_size or settings.EMBEDDING_BATCH_SIZE, 16))
        self.timeout_s = timeout_s or settings.EMBEDDING_TIMEOUT_S
        self.max_retries = max_retries if max_retries is not None else settings.EMBEDDING_MAX_RETRIES
        self._api_key = api_key or settings.GEMINI_API_KEY
        if not self._api_key:
            logger.warning("Gemini embedding API key not configured")

    async def embed_texts(self, texts: list[str],
                          titles: list[str] | None = None) -> list[list[float]]:
        cleaned = [(t or "")[:MAX_INPUT_CHARS] for t in texts]
        if any(not t.strip() for t in cleaned):
            raise EmbeddingError("empty embedding input", "EMPTY_INPUT")
        if len(cleaned) > MAX_BATCH_ITEMS:
            raise EmbeddingError("batch too large", "BATCH_TOO_LARGE")
        titles = list(titles) if titles is not None else ["none"] * len(cleaned)
        if len(titles) != len(cleaned):
            raise EmbeddingError("titles/texts length mismatch", "BAD_BATCH")
        formatted = [format_document(t, title=title) for t, title in zip(cleaned, titles)]
        out: list[list[float]] = []
        for i in range(0, len(formatted), self.batch_size):
            batch = formatted[i:i + self.batch_size]
            out.extend(await self._embed_batch(batch))
        validate_vectors(out, self._dim)
        return out

    async def embed_query(self, text: str) -> list[float]:
        cleaned = (text or "").strip()[:MAX_INPUT_CHARS]
        if not cleaned:
            raise EmbeddingError("empty query", "EMPTY_INPUT")
        formatted = format_query(cleaned)
        vecs = await self._embed_batch([formatted])
        validate_vectors(vecs, self._dim)
        return vecs[0]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._embed_sync, batch), timeout=self.timeout_s)
            except EmbeddingError:
                raise
            except Exception as e:
                last = e
                if attempt < self.max_retries and _retryable(e):
                    await asyncio.sleep(min(2.0 ** attempt, 8.0))
                    continue
                raise EmbeddingError(f"embedding failed: {type(e).__name__}",
                                     _error_code(e)) from e
        raise EmbeddingError("embedding failed", "EMBEDDING_FAILED") from last

    def _embed_sync(self, batch: list[str]) -> list[list[float]]:
        try:
            from google import genai as new_genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise EmbeddingError("google-genai SDK not installed", "SDK_MISSING") from e
        if not self._api_key:
            raise EmbeddingError("missing GEMINI_API_KEY", "NO_CREDENTIALS")
        client = new_genai.Client(api_key=self._api_key)
        vectors: list[list[float]] = []
        for text in batch:
            try:
                cfg = genai_types.EmbedContentConfig(output_dimensionality=self._dim)
            except Exception:
                cfg = {"output_dimensionality": self._dim}
            resp = client.models.embed_content(
                model=self._model, contents=text, config=cfg)
            embs = getattr(resp, "embeddings", None) or []
            if not embs:
                raise EmbeddingError("empty embedding response", "EMPTY_RESPONSE")
            vals = getattr(embs[0], "values", None)
            if vals is None:
                raise EmbeddingError("malformed embedding response", "MALFORMED_RESPONSE")
            vectors.append([float(x) for x in list(vals)])
        return vectors


def _error_code(exc: Exception) -> str:
    text = f"{type(exc).__name__} {exc}".lower()
    if "timeout" in text or "deadline" in text:
        return "EMBEDDING_TIMEOUT"
    if "429" in text or "rate" in text or "quota" in text:
        return "EMBEDDING_RATE_LIMITED"
    if "auth" in text or "api key" in text or "401" in text or "403" in text:
        return "EMBEDDING_UNAUTHORIZED"
    if "500" in text or "502" in text or "503" in text or "unavailable" in text:
        return "EMBEDDING_TRANSIENT"
    return "EMBEDDING_FAILED"


def _retryable(exc: Exception) -> bool:
    return _error_code(exc) in ("EMBEDDING_TIMEOUT", "EMBEDDING_RATE_LIMITED",
                                "EMBEDDING_TRANSIENT")


def build_embedding_provider() -> EmbeddingProvider:
    """Production: always live Gemini adapter (fails loudly without key).

    Hash provider is opt-in only: HashEmbeddingProvider() directly in tests.
    """
    return GeminiEmbeddingProvider()
