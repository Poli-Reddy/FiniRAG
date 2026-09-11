"""
Unit tests for the LLM fallback chain and the Groq provider.

No network: both providers are doubles. What is under test is the policy —
when the chain moves on, when it refuses to, which backend gets credited for
the answer, and whether a Groq reply survives the shapes that model family
actually emits.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from backend.app.providers.errors import LLMOutputTruncatedError, LLMRateLimitError
from backend.app.providers.fallback_provider import FallbackLLMProvider
from backend.app.providers.llm_provider import LLMProvider


# ─────────────────────────────────────────────────────────────────────
# Doubles
# ─────────────────────────────────────────────────────────────────────

class ScriptedProvider(LLMProvider):
    """Returns `result`, or raises `error` if one is set."""

    def __init__(self, result=None, error=None, name="scripted"):
        self.result = result if result is not None else {"company_name": name}
        self.error = error
        self.text_calls = 0
        self.image_calls = 0

    async def extract_structured(self, text, document_type, system_prompt):
        self.text_calls += 1
        if self.error:
            raise self.error
        return self.result

    async def extract_from_images(self, images, document_type, system_prompt):
        self.image_calls += 1
        if self.error:
            raise self.error
        return self.result


class TextOnlyProvider(LLMProvider):
    """A backend with no vision support, to exercise the skip path."""

    def __init__(self):
        self.text_calls = 0

    async def extract_structured(self, text, document_type, system_prompt):
        self.text_calls += 1
        return {"company_name": "text-only"}

    extract_from_images = None  # type: ignore[assignment]


def chain(*providers):
    return FallbackLLMProvider([(p_name, p) for p_name, p in providers])


PROMPT = "Extract the balance sheet."


# ─────────────────────────────────────────────────────────────────────
# The happy path stays on the primary
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_primary_answers_and_the_fallback_is_never_called():
    """A working primary must not cost a second API call."""
    primary = ScriptedProvider(name="gemini")
    backup = ScriptedProvider(name="groq")
    c = chain(("gemini", primary), ("groq", backup))

    result = await c.extract_structured("t", "balance_sheet", PROMPT)

    assert result == {"company_name": "gemini"}
    assert backup.text_calls == 0
    assert c.last_provider == "gemini"


# ─────────────────────────────────────────────────────────────────────
# Failover
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_quota_rejection_moves_to_the_fallback():
    """The case this exists for: the 2026-09-10 daily-quota outage."""
    primary = ScriptedProvider(error=LLMRateLimitError(31.0))
    backup = ScriptedProvider(name="groq")
    c = chain(("gemini", primary), ("groq", backup))

    result = await c.extract_structured("t", "balance_sheet", PROMPT)

    assert result == {"company_name": "groq"}
    assert c.last_provider == "groq"


@pytest.mark.asyncio
async def test_vision_falls_over_too():
    """No dataset document has a native text layer, so vision is the path."""
    primary = ScriptedProvider(error=LLMRateLimitError())
    backup = ScriptedProvider(name="groq")
    c = chain(("gemini", primary), ("groq", backup))

    pages = [Image.new("RGB", (8, 8))]
    assert await c.extract_from_images(pages, "invoice", PROMPT) == {"company_name": "groq"}
    assert backup.image_calls == 1


@pytest.mark.asyncio
async def test_a_transport_error_also_fails_over():
    primary = ScriptedProvider(error=ConnectionError("connection reset"))
    backup = ScriptedProvider(name="groq")
    c = chain(("gemini", primary), ("groq", backup))

    assert await c.extract_structured("t", "invoice", PROMPT) == {"company_name": "groq"}


@pytest.mark.asyncio
async def test_unparseable_json_fails_over():
    """
    A model that will not produce JSON has already burned its own retry
    budget. Another backend is the only remaining move.
    """
    primary = ScriptedProvider(error=ValueError("LLM returned invalid JSON after 2 attempts"))
    backup = ScriptedProvider(name="groq")
    c = chain(("gemini", primary), ("groq", backup))

    assert await c.extract_structured("t", "invoice", PROMPT) == {"company_name": "groq"}


# ─────────────────────────────────────────────────────────────────────
# What must NOT fail over
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_caller_bug_is_not_retried_on_another_backend():
    """
    An unknown document type is a bad request, not a provider outage. Asking
    a second model the same malformed question hides the bug and doubles the
    latency of a failure that cannot succeed.
    """
    primary = ScriptedProvider(error=ValueError("Unknown document type: ledger"))
    backup = ScriptedProvider(name="groq")
    c = chain(("gemini", primary), ("groq", backup))

    with pytest.raises(ValueError, match="Unknown document type"):
        await c.extract_structured("t", "ledger", PROMPT)

    assert backup.text_calls == 0


# ─────────────────────────────────────────────────────────────────────
# Exhaustion
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_when_every_backend_refuses_the_last_error_survives():
    """
    The pipeline maps the exception to an error code. If the chain flattened
    a quota rejection into a generic error, a throttled document would once
    again look like a bad document.
    """
    c = chain(
        ("gemini", ScriptedProvider(error=ConnectionError("reset"))),
        ("groq", ScriptedProvider(error=LLMRateLimitError(60.0))),
    )

    with pytest.raises(LLMRateLimitError):
        await c.extract_structured("t", "invoice", PROMPT)


@pytest.mark.asyncio
async def test_a_provider_without_vision_is_skipped_not_failed():
    primary = ScriptedProvider(error=LLMRateLimitError())
    backup = TextOnlyProvider()
    c = chain(("gemini", primary), ("text_only", backup))

    with pytest.raises(LLMRateLimitError):
        await c.extract_from_images([Image.new("RGB", (8, 8))], "invoice", PROMPT)


@pytest.mark.asyncio
async def test_an_empty_chain_is_rejected_at_construction():
    with pytest.raises(ValueError):
        FallbackLLMProvider([])


# ─────────────────────────────────────────────────────────────────────
# Groq reply handling
# ─────────────────────────────────────────────────────────────────────

class FakeGroqClient:
    """Mimics `client.chat.completions.create` with scripted replies."""

    def __init__(self, *replies, finish_reason="stop", completion_tokens=42):
        self.replies = list(replies)
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                reply = outer.replies[
                    min(len(outer.calls) - 1, len(outer.replies) - 1)
                ]
                if isinstance(reply, Exception):
                    raise reply
                return _Response(reply, finish_reason, completion_tokens)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class _Response:
    def __init__(self, text, finish_reason="stop", completion_tokens=42):
        message = type("M", (), {"content": text})()
        self.choices = [
            type("C", (), {"message": message, "finish_reason": finish_reason})()
        ]
        self.usage = type("U", (), {"completion_tokens": completion_tokens})()


@pytest.fixture()
def groq_provider(monkeypatch):
    """A GroqProvider with its HTTP client replaced."""
    from backend.app.providers import groq_provider as gp_module

    p = gp_module.GroqProvider.__new__(gp_module.GroqProvider)
    p._client = FakeGroqClient(json.dumps({"company_name": "HDFC Bank Limited"}))
    p._text_model = "text-model"
    p._vision_model = "vision-model"
    return p


EXPECTED = {"company_name": "HDFC Bank Limited"}


@pytest.mark.asyncio
async def test_groq_parses_a_plain_json_reply(groq_provider):
    assert await groq_provider.extract_structured("t", "balance_sheet", PROMPT) == EXPECTED


@pytest.mark.asyncio
async def test_groq_strips_a_reasoning_block(groq_provider):
    """
    qwen models narrate before answering even under JSON mode. A stray
    <think> block would turn a correct extraction into a parse failure.
    """
    groq_provider._client = FakeGroqClient(
        '<think>\nThe total is on the last line.\n</think>\n'
        '{"company_name": "HDFC Bank Limited"}'
    )
    assert await groq_provider.extract_structured("t", "balance_sheet", PROMPT) == EXPECTED


@pytest.mark.asyncio
async def test_groq_strips_markdown_fences(groq_provider):
    groq_provider._client = FakeGroqClient(
        '```json\n{"company_name": "HDFC Bank Limited"}\n```'
    )
    assert await groq_provider.extract_structured("t", "balance_sheet", PROMPT) == EXPECTED


@pytest.mark.asyncio
async def test_groq_retries_once_on_bad_json(groq_provider):
    groq_provider._client = FakeGroqClient("not json", json.dumps(EXPECTED))

    assert await groq_provider.extract_structured("t", "balance_sheet", PROMPT) == EXPECTED
    assert len(groq_provider._client.calls) == 2


@pytest.mark.asyncio
async def test_groq_gives_up_after_two_bad_replies(groq_provider):
    groq_provider._client = FakeGroqClient("nope", "still nope")

    with pytest.raises(ValueError, match="invalid JSON"):
        await groq_provider.extract_structured("t", "balance_sheet", PROMPT)

    assert len(groq_provider._client.calls) == 2


@pytest.mark.asyncio
async def test_groq_translates_its_own_throttling(groq_provider):
    """So a chain ending on Groq still reports LLM_RATE_LIMITED."""
    class RateLimitError(Exception):
        pass

    groq_provider._client = FakeGroqClient(RateLimitError("429"))

    with pytest.raises(LLMRateLimitError):
        await groq_provider.extract_structured("t", "balance_sheet", PROMPT)


@pytest.mark.asyncio
async def test_groq_requests_json_mode(groq_provider):
    """Without it the model prepends prose and every reply needs salvaging."""
    await groq_provider.extract_structured("t", "balance_sheet", PROMPT)

    [call] = groq_provider._client.calls
    assert call["response_format"] == {"type": "json_object"}
    assert call["model"] == "text-model"


@pytest.mark.asyncio
async def test_groq_vision_sends_the_prompt_then_every_page_in_order(groq_provider):
    pages = [Image.new("RGB", (8, 8), c) for c in ("red", "green", "blue")]

    await groq_provider.extract_from_images(pages, "cash_flow_statement", PROMPT)

    [call] = groq_provider._client.calls
    parts = call["messages"][0]["content"]
    assert call["model"] == "vision-model"
    assert parts[0]["type"] == "text" and PROMPT in parts[0]["text"]
    assert [p["type"] for p in parts[1:]] == ["image_url"] * 3
    assert all(p["image_url"]["url"].startswith("data:image/png;base64,") for p in parts[1:])


@pytest.mark.asyncio
async def test_groq_downscales_an_oversized_page(groq_provider):
    """
    A 4.4 MB scan base64-encodes past the request size limit. The cap has to
    apply before encoding, not after.
    """
    from backend.app.providers.groq_provider import MAX_IMAGE_EDGE, _encode

    encoded = _encode(Image.new("RGB", (5000, 4000)))
    small = _encode(Image.new("RGB", (100, 80)))

    assert encoded.startswith("data:image/png;base64,")
    assert len(encoded) < 2_000_000
    assert len(small) < len(encoded)
    assert MAX_IMAGE_EDGE == 1600


@pytest.mark.asyncio
async def test_groq_converts_a_mode_png_cannot_encode(groq_provider):
    """Rendered pages arrive in whatever mode the preprocessor left them."""
    from backend.app.providers.groq_provider import _encode

    assert _encode(Image.new("CMYK", (40, 40))).startswith("data:image/png;base64,")


# ─────────────────────────────────────────────────────────────────────
# Output truncation
#
# The account's qwen models cap output at 1,000 tokens per minute. A large
# balance sheet exceeds that, and the reply comes back *parseable but short* —
# measured against `Consolidated Balance Sheet 2024.pdf`, which returned
# finish_reason=length at 999 tokens with the 2023 period's total_assets
# missing. Nothing downstream can tell that from a document that genuinely
# lacks the row, so it has to be caught here.
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_truncated_reply_is_rejected_even_though_it_parses(groq_provider):
    groq_provider._client = FakeGroqClient(
        '{"company_name": "HDFC Bank Limited", "periods": {}}',
        finish_reason="length",
        completion_tokens=999,
    )

    with pytest.raises(LLMOutputTruncatedError) as excinfo:
        await groq_provider.extract_structured("t", "balance_sheet", PROMPT)

    assert excinfo.value.completion_tokens == 999


@pytest.mark.asyncio
async def test_json_mode_refusal_falls_back_to_free_form(groq_provider):
    """
    gpt-oss constrained generation can refuse long prompts outright
    (json_validate_failed). The retry budget is then spent on a free-form
    reply instead of failing the document.
    """
    groq_provider._client = FakeGroqClient(
        Exception("400 json_validate_failed: Failed to validate JSON"),
        json.dumps(EXPECTED),
    )

    assert await groq_provider.extract_structured("t", "balance_sheet", PROMPT) == EXPECTED

    first, second = groq_provider._client.calls
    assert first["response_format"] == {"type": "json_object"}
    assert "response_format" not in second


@pytest.mark.asyncio
async def test_a_truncated_reply_is_not_retried(groq_provider):
    """The second attempt would stop at the same ceiling."""
    groq_provider._client = FakeGroqClient(
        '{"company_name": "x"}', finish_reason="length"
    )

    with pytest.raises(LLMOutputTruncatedError):
        await groq_provider.extract_from_images(
            [Image.new("RGB", (8, 8))], "balance_sheet", PROMPT
        )

    assert len(groq_provider._client.calls) == 1


@pytest.mark.asyncio
async def test_groq_always_sends_an_explicit_output_ceiling(groq_provider):
    """
    Leaving max_tokens unset is accepted only when the preceding minute left
    enough budget, so an unset ceiling makes success depend on run history.
    Ceilings are per-model: the qwen vision tier caps at 999 while the
    gpt-oss text model accepts 8000.
    """
    from backend.app.providers.groq_provider import (
        MAX_OUTPUT_TOKENS,
        MAX_TEXT_OUTPUT_TOKENS,
    )

    await groq_provider.extract_structured("t", "balance_sheet", PROMPT)

    [call] = groq_provider._client.calls
    assert call["max_tokens"] == MAX_TEXT_OUTPUT_TOKENS
    assert MAX_TEXT_OUTPUT_TOKENS <= 8000
    assert MAX_OUTPUT_TOKENS < 1000


@pytest.mark.asyncio
async def test_groq_vision_keeps_the_capped_ceiling(groq_provider):
    """The qwen vision models reject anything above ~1000 output tokens."""
    from backend.app.providers.groq_provider import MAX_OUTPUT_TOKENS

    await groq_provider.extract_from_images(
        [Image.new("RGB", (8, 8))], "balance_sheet", PROMPT
    )

    [call] = groq_provider._client.calls
    assert call["max_tokens"] == MAX_OUTPUT_TOKENS
    assert MAX_OUTPUT_TOKENS < 1000


@pytest.mark.asyncio
async def test_a_complete_reply_at_the_ceiling_is_still_accepted(groq_provider):
    """Spending the whole budget is only a problem if generation was cut off."""
    groq_provider._client = FakeGroqClient(
        json.dumps(EXPECTED), finish_reason="stop", completion_tokens=999
    )

    assert await groq_provider.extract_structured("t", "balance_sheet", PROMPT) == EXPECTED


@pytest.mark.asyncio
async def test_truncation_moves_the_chain_to_a_roomier_backend():
    """
    A truncation is a limit of that backend, not of the document, so it must
    fail over rather than end the extraction.
    """
    primary = ScriptedProvider(error=LLMOutputTruncatedError(999))
    backup = ScriptedProvider(name="gemini")
    c = chain(("groq", primary), ("gemini", backup))

    assert await c.extract_structured("t", "balance_sheet", PROMPT) == {
        "company_name": "gemini"
    }
    assert c.last_provider == "gemini"
