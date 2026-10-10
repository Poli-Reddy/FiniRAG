"""Hybrid retrieval — pgvector + PostgreSQL FTS with RRF (k=60)."""

from __future__ import annotations

import logging
import math
import re

from sqlalchemy import select, or_

from backend.app.core.config import settings
from backend.app.core.metrics import (
    COUNT_BUCKETS, RETRIEVAL_CANDIDATES, RETRIEVAL_EMPTY, RETRIEVAL_LATENCY,
    RETRIEVAL_REQUESTS, metrics,
)
from backend.app.models.rag import DocumentChunk, embedding_cosine_distance
from backend.app.services.embedding_service import EmbeddingService

logger = logging.getLogger("finirag")

_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def normalize_query(q: str) -> str:
    """Lowercase + collapse whitespace, but keep identifiers/periods/units."""
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _dialect_name(db) -> str:
    """Best-effort dialect name for AsyncSession (postgres vs sqlite)."""
    try:
        bind = getattr(db, "bind", None)
        if bind is not None and getattr(bind, "dialect", None) is not None:
            return bind.dialect.name
    except Exception:
        pass
    try:
        sync = getattr(db, "sync_session", None)
        bind = getattr(sync, "bind", None) if sync is not None else None
        if bind is not None and getattr(bind, "dialect", None) is not None:
            return bind.dialect.name
    except Exception:
        pass
    try:
        getb = getattr(db, "get_bind", None)
        if callable(getb):
            b = getb()
            if b is not None and getattr(getattr(b, "dialect", None), "name", None):
                return b.dialect.name
    except Exception:
        pass
    return "unknown"


def cosine(a: list[float], b: list[float]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if not denom:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / denom


def rrf_fuse(rank_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    ranks: dict[str, dict] = {}
    for li, ranked in enumerate(rank_lists):
        for rank, item in enumerate(ranked, start=1):
            cid = str(item["chunk_id"])
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            by_id[cid] = item
            ranks.setdefault(cid, {})[f"path_{li}"] = rank
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for cid, score in ordered:
        item = dict(by_id[cid])
        item["rrf_score"] = score
        item["component_ranks"] = ranks[cid]
        out.append(item)
    return out


class RetrievalError(Exception):
    """Strict-mode failure: a native PostgreSQL retrieval path failed."""


class RetrievalService:
    def __init__(self, db, embeddings: EmbeddingService | None = None):
        self._db = db
        self._embeddings = embeddings or EmbeddingService()

    async def retrieve(self, query: str, *, document_id=None, document_type: str | None = None,
                       reporting_period: str | None = None, limit_per_path: int | None = None,
                       company: str | None = None, strict: bool = False) -> dict:
        """Hybrid retrieval. Paths reported reflect what ACTUALLY executed.

        strict=True (used by verification): on PostgreSQL, a failed native
        vector/FTS query raises RetrievalError instead of silently falling
        back, so broken indexes/queries cannot masquerade as healthy.
        """
        from uuid import UUID as _UUID

        import time as _time

        _start = _time.perf_counter()
        q = normalize_query(query)
        if not q:
            return {"candidates": [], "insufficient": True, "reason": "EMPTY_QUERY"}
        if document_id is not None:
            try:
                document_id = document_id if isinstance(document_id, _UUID) else _UUID(str(document_id))
            except Exception:
                pass
        limit = limit_per_path or settings.RETRIEVAL_LIMIT_PER_PATH
        native_errors: dict[str, str] = {}
        try:
            qvec = await self._embeddings.embed_query(q)
            vec_ok = True
        except Exception as e:
            logger.warning("Query embedding failed, keyword-only", extra={
                "error_type": type(e).__name__})
            native_errors["embedding"] = type(e).__name__
            qvec = None
            vec_ok = False

        filters = []
        if document_id is not None:
            filters.append(DocumentChunk.document_id == document_id)
        if document_type:
            filters.append(DocumentChunk.document_type == document_type)
        if reporting_period:
            # Match period stored on chunk or inside content.
            filters.append(or_(DocumentChunk.reporting_period == reporting_period,
                               DocumentChunk.content.contains(reporting_period)))
        if company:
            # Pre-filter (not only post-filter): never silently mix companies.
            # Company must appear in content; otherwise the query is
            # insufficient rather than cross-contaminated.
            filters.append(DocumentChunk.content.contains(company))

        sem: list[dict] = []
        vector_path = "disabled"
        if vec_ok and qvec is not None:
            _t0 = _time.perf_counter()
            sem, vector_path, vec_err = await self._semantic(
                qvec, filters, limit, strict=strict)
            self._record_stage("vector", _time.perf_counter() - _t0)
            if vec_err:
                native_errors["vector"] = vec_err
        elif strict and _dialect_name(self._db) == "postgresql":
            raise RetrievalError("strict: query embedding unavailable")
        _t0 = _time.perf_counter()
        kw, keyword_path, kw_err = await self._keyword(q, filters, limit, strict=strict)
        self._record_stage("keyword", _time.perf_counter() - _t0)
        if kw_err:
            native_errors["keyword"] = kw_err
        dialect = _dialect_name(self._db)
        degraded = bool(native_errors)
        fused = rrf_fuse([sem, kw], k=settings.RETRIEVAL_RRF_K)
        # Dedupe identical content.
        seen: set[str] = set()
        deduped = []
        for c in fused:
            h = c.get("content_hash") or c["content"][:200]
            if h in seen:
                continue
            seen.add(h)
            deduped.append(c)
        top = deduped[: settings.QA_MAX_CONTEXT_CHUNKS]
        out = {"candidates": top, "semantic_count": len(sem), "keyword_count": len(kw),
               "query": q, "semantic_available": vec_ok, "dialect": dialect,
               "vector_path": vector_path, "keyword_path": keyword_path,
               "degraded": degraded, "native_errors": native_errors,
               "insufficient": len(top) == 0,
               "reason": None if top else ("NO_EVIDENCE" if vec_ok else "NO_EVIDENCE_SEMANTIC_DEGRADED")}
        import time as _time

        self._record_stage("total", _time.perf_counter() - _start)
        metrics.inc(RETRIEVAL_REQUESTS, {"vector_path": vector_path,
                                         "keyword_path": keyword_path,
                                         "degraded": str(degraded).lower()})
        metrics.observe(RETRIEVAL_CANDIDATES, float(len(top)), {"stage": "fused"},
                        buckets=COUNT_BUCKETS)
        if not top:
            metrics.inc(RETRIEVAL_EMPTY, {"reason": out["reason"] or "unknown"})
        return out

    @staticmethod
    def _record_stage(stage: str, seconds: float) -> None:
        metrics.observe(RETRIEVAL_LATENCY, seconds, {"stage": stage})

    async def _apply_session_guards(self) -> None:
        """Set per-session Postgres tuning before retrieval.

        hnsw.ef_search controls how many candidates HNSW considers; the
        pgvector default (40) under-serves recall, so we raise it. SET LOCAL
        scopes the change to the current transaction, so pooled connections
        are never left modified. Values are int-coerced — never interpolated
        from user input.
        """
        if _dialect_name(self._db) != "postgresql":
            return
        try:
            from sqlalchemy import text as _text

            ef = int(settings.HNSW_EF_SEARCH)
            await self._db.execute(_text(f"SET LOCAL hnsw.ef_search = {ef}"))
        except Exception as e:  # pragma: no cover - tuning is best-effort
            logger.warning("Could not set hnsw.ef_search", extra={
                "error_type": type(e).__name__})

    async def _load_eligible(self, filters, limit: int | None = None) -> list:
        stmt = select(DocumentChunk).where(DocumentChunk.is_active == True)  # noqa: E712
        for f in filters:
            stmt = stmt.where(f)
        stmt = stmt.order_by(DocumentChunk.id).limit(
            limit or settings.RETRIEVAL_FALLBACK_MAX_ROWS)
        return list((await self._db.execute(stmt)).scalars().all())

    async def _semantic(self, qvec, filters, limit: int,
                        strict: bool = False) -> tuple[list[dict], str, str | None]:
        """Returns (items, actual_path, native_error). Native pgvector on
        Postgres (indexed cosine_distance); Python cosine fallback otherwise.
        strict=True raises RetrievalError instead of falling back."""
        if _dialect_name(self._db) == "postgresql":
            try:
                await self._apply_session_guards()
                stmt = select(DocumentChunk).where(DocumentChunk.is_active == True)  # noqa: E712
                for f in filters:
                    stmt = stmt.where(f)
                # Deterministic tie-break on chunk id so equal distances return
                # a stable order (important for reproducible evaluation).
                stmt = stmt.order_by(
                    embedding_cosine_distance(DocumentChunk.embedding, qvec),
                    DocumentChunk.id,
                ).limit(limit)
                rows = list((await self._db.execute(stmt)).scalars().all())
                if rows:
                    return ([self._to_candidate(c, "vector", i + 1) for i, c in enumerate(rows)],
                            "pgvector", None)
                return ([], "pgvector", None)
            except Exception as e:
                logger.warning("pgvector query failed, python fallback", extra={
                    "error_type": type(e).__name__})
                if strict:
                    raise RetrievalError(f"strict: pgvector failed: {type(e).__name__}") from e
                fallback = await self._cosine_fallback(qvec, filters, limit)
                return (fallback, "python-fallback", type(e).__name__)
        return (await self._cosine_fallback(qvec, filters, limit), "python-fallback", None)

    async def _cosine_fallback(self, qvec, filters, limit: int) -> list[dict]:
        # Bounded and deterministic: never sweep an unbounded corpus in Python,
        # and break score ties on chunk id so ordering is reproducible.
        rows = await self._load_eligible(filters)
        scored = []
        for c in rows:
            if not c.embedding:
                continue
            try:
                scored.append((cosine(qvec, list(c.embedding)), str(c.id), c))
            except Exception:
                continue
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [self._to_candidate(c, "vector", i + 1, score=s)
                for i, (s, _cid, c) in enumerate(scored[:limit])]

    async def _keyword(self, query: str, filters, limit: int,
                       strict: bool = False) -> tuple[list[dict], str, str | None]:
        """Returns (items, actual_path, native_error). Postgres FTS with GIN
        index (to_tsvector @@ plainto_tsquery, ts_rank); token fallback.
        strict=True raises RetrievalError instead of falling back."""
        if _dialect_name(self._db) == "postgresql":
            try:
                from sqlalchemy import text as _text

                # Parameterized; no string interpolation of user input into SQL.
                stmt = select(DocumentChunk).where(DocumentChunk.is_active == True)  # noqa: E712
                for f in filters:
                    stmt = stmt.where(f)
                fts = _text("to_tsvector('english', document_chunks.content) "
                            "@@ plainto_tsquery('english', :q)")
                stmt = stmt.where(fts).order_by(
                    _text("ts_rank(to_tsvector('english', document_chunks.content), "
                          "plainto_tsquery('english', :q)) DESC")).limit(limit).params(q=query)
                rows = list((await self._db.execute(stmt)).scalars().all())
                if rows:
                    return ([self._to_candidate(c, "keyword", i + 1) for i, c in enumerate(rows)],
                            "postgres-fts", None)
                return ([], "postgres-fts", None)
            except Exception as e:
                logger.warning("FTS query failed, token fallback", extra={
                    "error_type": type(e).__name__})
                if strict:
                    raise RetrievalError(f"strict: FTS failed: {type(e).__name__}") from e
                return (await self._overlap_fallback(query, filters, limit),
                        "token-overlap", type(e).__name__)
        return (await self._overlap_fallback(query, filters, limit), "token-overlap", None)

    async def _overlap_fallback(self, query: str, filters, limit: int) -> list[dict]:
        rows = await self._load_eligible(filters)
        qtokens = set(tokenize(query))
        scored = []
        for c in rows:
            ctokens = set(tokenize(c.content or ""))
            overlap = len(qtokens & ctokens)
            if overlap:
                # Boost exact identifier/period matches; stable tie-break.
                scored.append((overlap, str(c.id), c))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [self._to_candidate(c, "keyword", i + 1, score=float(s))
                for i, (s, _cid, c) in enumerate(scored[:limit])]

    def _to_candidate(self, c: DocumentChunk, method: str, rank: int, score=None) -> dict:
        meta = dict(c.chunk_metadata or {})
        return {"chunk_id": str(c.id), "document_id": str(c.document_id),
                "document_name": meta.get("document_name", ""),
                "document_type": c.document_type,
                "page_number": c.page_number, "section_name": c.section_name,
                "reporting_period": c.reporting_period, "content": c.content,
                "retrieval_method": method, "rank": rank, "score": score,
                "source_evidence": c.source_evidence or {},
                "index_version": c.index_version,
                "content_hash": c.content_hash,
                "chunk_type": c.chunk_type}
