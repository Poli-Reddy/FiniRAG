"""Embedding provider tests — mocked google-genai SDK, no network."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from backend.app.providers.embedding_provider import (
    GeminiEmbeddingProvider, HashEmbeddingProvider, EmbeddingError,
    validate_vectors, format_document, format_query,
    EMBEDDING_MODEL, EMBEDDING_DIM,
)


def _install_fake_genai(values=None, exc: Exception | None = None):
    fake_genai = types.ModuleType("google.genai")
    fake_models = types.SimpleNamespace()
    calls = {"n": 0, "contents": [], "models": []}

    class FakeEmb:
        def __init__(self, vals):
            self.values = vals

    class FakeResp:
        def __init__(self, embs):
            self.embeddings = embs

    def embed_content(*, model, contents, config=None):
        calls["n"] += 1
        calls["contents"].append(contents)
        calls["models"].append(model)
        assert model == "gemini-embedding-2"
        # output_dimensionality must be propagated.
        dim = getattr(config, "output_dimensionality", None) or (config or {}).get("output_dimensionality")
        assert dim == 768
        if exc is not None:
            raise exc
        vals = values if values is not None else [0.1] * 768
        return FakeResp([FakeEmb(vals)])

    fake_models.embed_content = embed_content
    fake_genai.Client = lambda api_key=None: types.SimpleNamespace(models=fake_models)
    fake_types = types.ModuleType("google.genai.types")

    class EmbedContentConfig:
        def __init__(self, output_dimensionality=None):
            self.output_dimensionality = output_dimensionality

    fake_types.EmbedContentConfig = EmbedContentConfig
    google_mod = types.ModuleType("google")
    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = fake_types
    return calls


def _clear_fake_genai():
    for m in ("google", "google.genai", "google.genai.types"):
        sys.modules.pop(m, None)


def test_model_defaults_are_supported():
    assert EMBEDDING_MODEL == "gemini-embedding-2"
    assert EMBEDDING_DIM == 768
    p = GeminiEmbeddingProvider(api_key="k")
    assert p.model == "gemini-embedding-2" or "gemini-embedding-2" in p.model


def test_format_helpers():
    assert format_document("hello", title="inv").startswith("title: inv | text:")
    assert format_query("q").startswith("task: question answering | query:")


@pytest.mark.asyncio
async def test_live_adapter_mocked_success():
    calls = _install_fake_genai()
    try:
        p = GeminiEmbeddingProvider(api_key="k")
        vecs = await p.embed_texts(["doc one", "doc two"])
        assert len(vecs) == 2 and len(vecs[0]) == 768
        assert calls["n"] == 2  # one call per text (no aggregation)
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_embed_texts_applies_document_format():
    """The exact strings sent to the SDK must carry title/text formatting."""
    calls = _install_fake_genai()
    try:
        p = GeminiEmbeddingProvider(api_key="k")
        await p.embed_texts(["hello world"], titles=["inv-a.jpg"])
        assert calls["contents"] == ["title: inv-a.jpg | text: hello world"]
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_embed_texts_default_title_is_none():
    calls = _install_fake_genai()
    try:
        p = GeminiEmbeddingProvider(api_key="k")
        await p.embed_texts(["hello world"])
        assert calls["contents"] == ["title: none | text: hello world"]
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_embed_query_applies_task_prefix():
    calls = _install_fake_genai()
    try:
        p = GeminiEmbeddingProvider(api_key="k")
        await p.embed_query("total amount?")
        assert calls["contents"] == ["task: question answering | query: total amount?"]
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_embed_texts_rejects_title_mismatch():
    _install_fake_genai()
    try:
        p = GeminiEmbeddingProvider(api_key="k")
        with pytest.raises(EmbeddingError):
            await p.embed_texts(["a", "b"], titles=["only-one"])
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_embed_texts_truncates_long_input():
    calls = _install_fake_genai()
    try:
        p = GeminiEmbeddingProvider(api_key="k")
        await p.embed_texts(["x" * 100000], titles=["t"])
        sent = calls["contents"][0]
        assert len(sent) <= 30000 + len("title: t | text: ")
        assert sent.startswith("title: t | text: ")
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_live_adapter_rejects_dim_mismatch():
    _install_fake_genai(values=[0.1] * 100)
    try:
        p = GeminiEmbeddingProvider(api_key="k")
        with pytest.raises(EmbeddingError):
            await p.embed_texts(["x"])
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_live_adapter_rejects_empty_response():
    class FakeResp:
        embeddings = []

    def _setup():
        import sys as _sys
        import types as _t
        fg = _t.ModuleType("google.genai")
        fg.Client = lambda api_key=None: _t.SimpleNamespace(
            models=_t.SimpleNamespace(embed_content=lambda **kw: FakeResp()))
        ft = _t.ModuleType("google.genai.types")
        ft.EmbedContentConfig = lambda **kw: _t.SimpleNamespace(**kw)
        _sys.modules["google"] = _t.ModuleType("google")
        _sys.modules["google.genai"] = fg
        _sys.modules["google.genai.types"] = ft

    _setup()
    try:
        with pytest.raises(EmbeddingError):
            await GeminiEmbeddingProvider(api_key="k").embed_texts(["x"])
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_live_adapter_empty_input():
    _install_fake_genai()
    try:
        with pytest.raises(EmbeddingError):
            await GeminiEmbeddingProvider(api_key="k").embed_texts(["   "])
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_live_adapter_timeout_code():
    _install_fake_genai(exc=TimeoutError("deadline exceeded"))
    try:
        with pytest.raises(EmbeddingError) as ei:
            await GeminiEmbeddingProvider(api_key="k", max_retries=0, timeout_s=5).embed_texts(["x"])
        assert ei.value.code in ("EMBEDDING_TIMEOUT", "EMBEDDING_FAILED")
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_live_adapter_no_credentials(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    _install_fake_genai()
    try:
        with pytest.raises(EmbeddingError) as ei:
            await GeminiEmbeddingProvider(api_key="").embed_texts(["x"])
        assert ei.value.code in ("NO_CREDENTIALS", "EMBEDDING_UNAUTHORIZED", "EMBEDDING_FAILED")
    finally:
        _clear_fake_genai()


@pytest.mark.asyncio
async def test_service_passes_raw_content_plus_titles_once():
    """Single formatting path: service sends raw text+titles, provider formats."""
    from backend.app.services.embedding_service import EmbeddingService

    seen = {}

    class Capture(HashEmbeddingProvider):
        async def embed_texts(self, texts, titles=None):
            seen["texts"] = list(texts)
            seen["titles"] = list(titles) if titles is not None else None
            return await super().embed_texts(texts, titles)

    svc = EmbeddingService(Capture())
    chunks = [{"content": "Total 108.00", "content_hash": "h1",
               "metadata": {"document_name": "inv-a.jpg"}}]
    await svc.embed_chunks(chunks)
    assert seen["texts"] == ["Total 108.00"]  # raw, not pre-formatted
    assert seen["titles"] == ["inv-a.jpg"]
    assert chunks[0]["embedding_model"] == svc.model


def test_hash_is_offline_labelled():
    h = HashEmbeddingProvider()
    assert "hash" in h.model.lower() or "offline" in h.model.lower()


def test_validate_vectors_rejects():
    with pytest.raises(EmbeddingError):
        validate_vectors([[float("inf")] * 768], 768)
    with pytest.raises(EmbeddingError):
        validate_vectors([[]], 768)
