"""
Unit tests for GeminiProvider.

No network: the model object is replaced with a scripted double. What is
under test is everything around the call — how the prompt and images are
assembled, how a fenced or malformed JSON reply is handled, how many
times a bad reply is retried, and what reaches the log.
"""

from __future__ import annotations

import json
import logging

import pytest
from PIL import Image

from backend.app.providers.errors import LLMRateLimitError
from backend.app.providers.gemini_provider import GeminiProvider


# ─────────────────────────────────────────────────────────────────────
# Doubles
# ─────────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, text: str):
        self.text = text


class FakeModel:
    """
    Replays `replies` one per call.

    An entry that is an Exception is raised instead of returned, so a
    test can script "fails once, then succeeds".
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list = []
        self.configs: list = []

    def generate_content(self, content, generation_config=None):
        self.calls.append(content)
        self.configs.append(generation_config)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)


@pytest.fixture()
def provider() -> GeminiProvider:
    """
    A real GeminiProvider with its model swapped out.

    Construction is left real on purpose: if building the provider ever
    starts requiring a live key or a network round trip, the app would
    stop booting without credentials and this fixture would catch it.
    """
    p = GeminiProvider()
    p._model = FakeModel(json.dumps({"company_name": "HDFC Bank Limited"}))
    return p


PROMPT = "Extract the balance sheet."
EXPECTED = {"company_name": "HDFC Bank Limited"}


# ─────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plain_json_is_parsed(provider):
    result = await provider.extract_structured("text", "balance_sheet", PROMPT)
    assert result == EXPECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [
    '```json\n{"company_name": "HDFC Bank Limited"}\n```',
    '```\n{"company_name": "HDFC Bank Limited"}\n```',
    '```json{"company_name": "HDFC Bank Limited"}```',
    '  \n ```json\n{"company_name": "HDFC Bank Limited"}\n```  \n ',
])
async def test_markdown_fences_are_stripped(provider, wrapped):
    """
    The model is asked for application/json but still fences its reply
    often enough that failing on it would be a routine outage.
    """
    provider._model = FakeModel(wrapped)
    assert await provider.extract_structured("t", "balance_sheet", PROMPT) == EXPECTED


@pytest.mark.asyncio
async def test_surrounding_whitespace_is_tolerated(provider):
    provider._model = FakeModel('\n\n  {"company_name": "HDFC Bank Limited"}  \n')
    assert await provider.extract_structured("t", "balance_sheet", PROMPT) == EXPECTED


@pytest.mark.asyncio
async def test_nested_structures_survive_parsing(provider):
    payload = {
        "unit_code": "INR_THOUSANDS",
        "periods": {"2020": {"net_cash_from_operating": "-168690920"}},
    }
    provider._model = FakeModel(json.dumps(payload))
    assert await provider.extract_structured("t", "cash_flow_statement", PROMPT) == payload


# ─────────────────────────────────────────────────────────────────────
# Retry behaviour
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_malformed_json_is_retried_once_then_succeeds(provider):
    provider._model = FakeModel("not json at all", json.dumps(EXPECTED))

    result = await provider.extract_structured("t", "balance_sheet", PROMPT)

    assert result == EXPECTED
    assert len(provider._model.calls) == 2


@pytest.mark.asyncio
async def test_two_malformed_replies_raise_rather_than_returning_junk(provider):
    """
    Better a clean failure the pipeline records as LLM_EXTRACTION_FAILED
    than a half-parsed dict presented as an extraction.
    """
    provider._model = FakeModel("not json", "still not json")

    with pytest.raises(ValueError, match="invalid JSON"):
        await provider.extract_structured("t", "balance_sheet", PROMPT)

    assert len(provider._model.calls) == 2


@pytest.mark.asyncio
async def test_retries_stop_at_two_attempts(provider):
    provider._model = FakeModel(*["nope"] * 10)

    with pytest.raises(ValueError):
        await provider.extract_structured("t", "balance_sheet", PROMPT)

    assert len(provider._model.calls) == 2


@pytest.mark.asyncio
async def test_a_transport_error_is_not_retried(provider):
    """
    Retrying a 403 or a quota rejection just doubles the latency of a
    failure that was never going to succeed. Only bad JSON is retried.
    """
    provider._model = FakeModel(RuntimeError("429 quota exceeded"), json.dumps(EXPECTED))

    with pytest.raises(RuntimeError, match="429"):
        await provider.extract_structured("t", "balance_sheet", PROMPT)

    assert len(provider._model.calls) == 1


# ─────────────────────────────────────────────────────────────────────
# Throttling
# ─────────────────────────────────────────────────────────────────────

class ResourceExhausted(Exception):
    """
    Shaped like `google.api_core.exceptions.ResourceExhausted`.

    Reconstructed locally rather than imported so the test does not depend on
    the SDK's exception layout, and so a `RetryInfo` proto does not have to be
    built to exercise the delay path.
    """

    def __init__(self, seconds=None):
        super().__init__("429 You exceeded your current quota")
        self.code = 429
        if seconds is not None:
            self.retry_delay = type("RetryInfo", (), {"seconds": seconds})()


@pytest.mark.asyncio
async def test_a_quota_rejection_becomes_a_rate_limit_error(provider):
    """
    The real 429 from the 2026-09-10 dataset run. It has to arrive at the
    pipeline as its own type, or the only thing distinguishing "we were
    throttled" from "this document is unreadable" is a vendor string the
    logging policy forbids recording.
    """
    provider._model = FakeModel(ResourceExhausted(seconds=31))

    with pytest.raises(LLMRateLimitError) as caught:
        await provider.extract_structured("t", "balance_sheet", PROMPT)

    assert caught.value.retry_after_seconds == 31.0


@pytest.mark.asyncio
async def test_a_quota_rejection_is_not_retried(provider):
    """A per-day quota does not clear in the time a retry takes."""
    provider._model = FakeModel(ResourceExhausted(), json.dumps(EXPECTED))

    with pytest.raises(LLMRateLimitError):
        await provider.extract_structured("t", "balance_sheet", PROMPT)

    assert len(provider._model.calls) == 1


@pytest.mark.asyncio
async def test_a_daily_quota_has_no_retry_delay_to_report(provider):
    """
    Per-day quotas arrive without RetryInfo — the answer is "tomorrow", not
    a number of seconds. None must not become 0.0, which would read as
    "retry immediately".
    """
    provider._model = FakeModel(ResourceExhausted())

    with pytest.raises(LLMRateLimitError) as caught:
        await provider.extract_structured("t", "balance_sheet", PROMPT)

    assert caught.value.retry_after_seconds is None


@pytest.mark.asyncio
async def test_the_rate_limit_error_carries_no_vendor_text(provider):
    """Correction #10 — the quota reply names the project and the model."""
    provider._model = FakeModel(ResourceExhausted(seconds=31))

    with pytest.raises(LLMRateLimitError) as caught:
        await provider.extract_structured("t", "balance_sheet", PROMPT)

    assert "429" not in str(caught.value)
    assert "quota" in str(caught.value).lower()


@pytest.mark.asyncio
async def test_vision_calls_translate_throttling_too(provider):
    """The vision path is the one that actually hit this in production."""
    provider._model = FakeModel(ResourceExhausted(seconds=31))

    with pytest.raises(LLMRateLimitError):
        await provider.extract_from_images(
            [Image.new("RGB", (8, 8))], "balance_sheet", PROMPT
        )


@pytest.mark.asyncio
async def test_a_blocked_response_surfaces_rather_than_retrying(provider):
    """`response.text` raises when the model returns no candidate."""
    class Blocked:
        @property
        def text(self):
            raise ValueError("no candidates: finish_reason SAFETY")

    class BlockingModel(FakeModel):
        def generate_content(self, content, generation_config=None):
            self.calls.append(content)
            return Blocked()

    provider._model = BlockingModel()

    with pytest.raises(ValueError, match="SAFETY"):
        await provider.extract_structured("t", "invoice", PROMPT)

    assert len(provider._model.calls) == 1


# ─────────────────────────────────────────────────────────────────────
# Request assembly
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_text_request_carries_the_prompt_and_the_document(provider):
    await provider.extract_structured("TOTAL ASSETS 17995066442", "balance_sheet", PROMPT)

    [sent] = provider._model.calls
    assert PROMPT in sent
    assert "TOTAL ASSETS 17995066442" in sent


@pytest.mark.asyncio
async def test_document_text_is_delimited_from_the_instructions(provider):
    """
    Without a boundary, a document containing instruction-like text
    reads as part of the prompt.
    """
    await provider.extract_structured("body", "balance_sheet", PROMPT)

    [sent] = provider._model.calls
    assert "--- DOCUMENT TEXT ---" in sent
    assert "--- END DOCUMENT TEXT ---" in sent


@pytest.mark.asyncio
async def test_vision_request_sends_the_prompt_then_every_page_in_order(provider):
    pages = [Image.new("RGB", (8, 8), c) for c in ("red", "green", "blue")]

    await provider.extract_from_images(pages, "cash_flow_statement", PROMPT)

    [sent] = provider._model.calls
    assert isinstance(sent, list)
    assert len(sent) == 4
    assert PROMPT in sent[0]
    assert sent[1:] == pages


@pytest.mark.asyncio
async def test_vision_request_with_no_images_is_still_well_formed(provider):
    await provider.extract_from_images([], "invoice", PROMPT)

    [sent] = provider._model.calls
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_json_output_is_requested_at_low_temperature(provider):
    """
    Asking for application/json is what keeps fenced prose rare, and a
    low temperature is what makes re-running a document reproducible.
    """
    await provider.extract_structured("t", "balance_sheet", PROMPT)

    [config] = provider._model.configs
    assert config.response_mime_type == "application/json"
    assert config.temperature == 0.1


# ─────────────────────────────────────────────────────────────────────
# Logging (Correction #10)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neither_the_prompt_nor_the_reply_is_logged(provider, caplog):
    secret_amount = "17995066442.55"
    provider._model = FakeModel(json.dumps({"total_assets": secret_amount}))

    with caplog.at_level(logging.DEBUG, logger="finirag"):
        await provider.extract_structured(
            f"TOTAL ASSETS {secret_amount}", "balance_sheet", PROMPT
        )

    assert secret_amount not in caplog.text
    assert PROMPT not in caplog.text


@pytest.mark.asyncio
async def test_the_extraction_log_records_type_and_attempt_only(provider, caplog):
    with caplog.at_level(logging.INFO, logger="finirag"):
        await provider.extract_structured("t", "balance_sheet", PROMPT)

    record = next(r for r in caplog.records if r.message == "LLM extraction")
    assert record.provider == "gemini"
    assert record.document_type == "balance_sheet"
    assert record.attempt == 1


@pytest.mark.asyncio
async def test_a_failing_call_logs_the_error_type_not_its_message(provider, caplog):
    provider._model = FakeModel(RuntimeError("api_key=AIzaSyBADSECRET rejected"))

    with caplog.at_level(logging.ERROR, logger="finirag"):
        with pytest.raises(RuntimeError):
            await provider.extract_structured("t", "invoice", PROMPT)

    record = next(r for r in caplog.records if r.message == "LLM extraction failed")
    assert record.error_type == "RuntimeError"
    assert "AIzaSyBADSECRET" not in caplog.text
