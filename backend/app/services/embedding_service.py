"""
FiniRAG Embedding service — bounded batches, reuse guarded by model+dim.
"""

from __future__ import annotations

import logging

from backend.app.core.config import settings
from backend.app.core.metrics import (
    COUNT_BUCKETS, EMBEDDING_BATCH, EMBEDDING_LATENCY, EMBEDDING_REQUESTS,
    metrics,
)
from backend.app.providers.embedding_provider import (
    EmbeddingProvider, build_embedding_provider, validate_vectors, EmbeddingError,
)

logger = logging.getLogger("finirag")


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider | None = None):
        self._provider = provider or build_embedding_provider()
        self.dim = getattr(self._provider, "dim", settings.EMBEDDING_DIM)
        self.model = getattr(self._provider, "model", settings.EMBEDDING_MODEL)
        self.semantic_available = "hash" not in str(self.model).lower()

    async def embed_chunks(self, chunks: list[dict],
                           existing: dict[str, list[float]] | None = None) -> list[dict]:
        """Attach embeddings. Reuse ONLY when content_hash+model+dim match.

        Document input formatting lives in exactly one place:
        `EmbeddingProvider.embed_texts` (title/text per the model contract).
        This service passes raw contents + titles, never pre-formatted text.
        """
        existing = existing or {}
        todo_idx: list[int] = []
        todo_texts: list[str] = []
        todo_titles: list[str] = []
        for i, c in enumerate(chunks):
            key = f"{c['content_hash']}:{self.model}:{self.dim}"
            if key in existing:
                c["embedding"] = existing[key]
                c["embedding_model"] = self.model
            else:
                todo_idx.append(i)
                meta = c.get("chunk_metadata") or c.get("metadata") or {}
                title = meta.get("document_name") or c.get("section_name") or "none"
                todo_texts.append(c["content"])
                todo_titles.append(title)
        if todo_texts:
            import time as _time

            _start = _time.perf_counter()
            try:
                vectors = await self._provider.embed_texts(todo_texts, titles=todo_titles)
            except Exception as e:
                metrics.inc(EMBEDDING_REQUESTS, {"kind": "document", "outcome": "error",
                                                 "model": self.model})
                logger.error("Embedding batch failed", extra={
                    "error_type": type(e).__name__, "count": len(todo_texts),
                    "code": getattr(e, "code", "EMBEDDING_FAILED")})
                raise
            validate_vectors(vectors, self.dim)
            metrics.inc(EMBEDDING_REQUESTS, {"kind": "document", "outcome": "ok",
                                             "model": self.model})
            metrics.observe(EMBEDDING_LATENCY, _time.perf_counter() - _start,
                            {"kind": "document", "model": self.model})
            metrics.observe(EMBEDDING_BATCH, float(len(todo_texts)),
                            {"kind": "document"}, buckets=COUNT_BUCKETS)
            for i, v in zip(todo_idx, vectors):
                chunks[i]["embedding"] = v
                chunks[i]["embedding_model"] = self.model
        return chunks

    async def embed_query(self, text: str) -> list[float]:
        cleaned = (text or "").strip()
        if not cleaned:
            raise EmbeddingError("empty query", "EMPTY_INPUT")
        import time as _time

        _start = _time.perf_counter()
        try:
            # Live provider applies query task-prefix internally.
            if hasattr(self._provider, "embed_query"):
                vecs = [await self._provider.embed_query(cleaned)]
            else:
                vecs = await self._provider.embed_texts([cleaned])
        except Exception:
            metrics.inc(EMBEDDING_REQUESTS, {"kind": "query", "outcome": "error",
                                             "model": self.model})
            raise
        validate_vectors(vecs, self.dim)
        metrics.inc(EMBEDDING_REQUESTS, {"kind": "query", "outcome": "ok",
                                         "model": self.model})
        metrics.observe(EMBEDDING_LATENCY, _time.perf_counter() - _start,
                        {"kind": "query", "model": self.model})
        return vecs[0]
