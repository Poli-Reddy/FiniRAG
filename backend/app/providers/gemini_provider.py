"""
FiniRAG Gemini LLM Provider.

Uses the unified Google GenAI SDK (`google.genai`) for structured document
extraction — the legacy `google.generativeai` package is end-of-support and
is no longer used anywhere in this codebase. Supports TWO modes:
  1. Text-based: send OCR/native text to Gemini
  2. Vision-based: send rendered images directly to Gemini (for when OCR fails)

The LLM extracts; deterministic Python validates.
Logging: does NOT log full prompt or response (Correction #10).
"""

from __future__ import annotations

import io
import json
import logging
import re

from PIL import Image

from backend.app.core.config import GEMINI_KEYS, settings
from backend.app.core.key_pool import (
    AllKeysCooling,
    AllKeysExhausted,
    KeyPool,
    is_daily_quota,
)
from backend.app.providers.errors import LLMOutputTruncatedError, LLMRateLimitError
from backend.app.providers.genai_text import GenerationConfig, build_text_model
from backend.app.providers.llm_provider import LLMProvider

logger = logging.getLogger("finirag")


def _retry_after(exc: Exception) -> float | None:
    """
    The delay the provider asked us to wait, in seconds, if it gave one.

    Read defensively from several shapes (legacy RetryInfo protos, new-SDK
    error details, Retry-After headers) because none of them is a stable
    contract. Returns None when the error carries no usable hint.
    """
    delay = getattr(exc, "retry_delay", None)
    seconds = getattr(delay, "seconds", None)
    if seconds:
        return float(seconds)
    for attr in ("retry_after", "retry_after_seconds"):
        value = getattr(exc, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    details = getattr(exc, "details", None)
    if isinstance(details, (list, tuple)):
        for entry in details:
            secs = getattr(entry, "seconds", None) or (
                entry.get("retryDelay", "") if isinstance(entry, dict) else None)
            try:
                if secs:
                    return float(str(secs).rstrip("s"))
            except (TypeError, ValueError):
                continue
    return None


def _is_rate_limit(exc: Exception) -> bool:
    """
    True for a quota or rate-limit rejection.

    Matched on class names and structured status codes rather than an
    isinstance check against one SDK's hierarchy, so the provider keeps
    working across SDK generations and stays testable without constructing
    a real API error. Never matches validation/auth errors: those must fail
    fast, not retry.
    """
    if type(exc).__name__ in ("ResourceExhausted", "TooManyRequests",
                              "RateLimitError", "RateLimited"):
        return True
    if getattr(exc, "code", None) == 429 or getattr(
        getattr(exc, "response", None), "status_code", None
    ) == 429:
        return True
    # Structured signals only. Message wording alone (including "429" or
    # "quota" in free text) does NOT count: retiring a key is destructive,
    # and unstructured text stays a transport error (pinned by
    # test_a_transport_error_is_not_retried). The new SDK surfaces real
    # quota rejections with an int `code`, so nothing real is lost.
    return False


def _is_truncated(response) -> bool:
    """True when Gemini stopped at its output limit (finish_reason MAX_TOKENS)."""
    try:
        for cand in (getattr(response, "candidates", None) or []):
            reason = getattr(cand, "finish_reason", None)
            # Legacy SDK: int enum (2 = MAX_TOKENS) or string. New SDK:
            # FinishReason enum (has .name). Accept all three shapes.
            if reason in (2, "MAX_TOKENS", "LENGTH", "length"):
                return True
            name = getattr(reason, "name", "")
            if name in ("MAX_TOKENS",):
                return True
    except Exception:
        return False
    return False


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider for structured extraction."""

    def __init__(self, model: str | None = None, pool: KeyPool | None = None):
        import os

        # `pool=None` preserves the legacy single-key behaviour the unit
        # tests pin down. Production passes the shared GEMINI_KEYS pool so
        # a 50-document run spreads across keys: each call borrows the
        # lowest live key, and a refused key is retired (permanently for a
        # daily quota, briefly for a minute throttle) without retrying the
        # same call — a retry could not succeed on a spent quota.
        self._pool = pool
        self._key_index: int | None = None
        # Shared config knob (GEMINI_MODEL); one default for the whole app.
        self._model_name = model or os.getenv("GEMINI_MODEL") or settings.GEMINI_MODEL

        if pool is None:
            # Singular key first, then first key from the plural pool so a
            # plural-only deployment does not silently run unauthenticated.
            # Bound lazily on first call so construction never needs a live
            # key or a network round trip (the unit fixture asserts this).
            self._single_key = settings.GEMINI_API_KEY or (
                GEMINI_KEYS.keys[0] if len(GEMINI_KEYS) else "")
            if not self._single_key and os.getenv("PYTEST_CURRENT_TEST") is None \
                    and os.getenv("CI") is None:
                logger.warning("Gemini API key not configured")
            self._model = None
        else:
            self._single_key = ""
            # Model is bound lazily per call from whichever pooled key is
            # borrowed, so key deaths never strand the provider.
            self._model = None

    def _bind(self, api_key: str) -> None:
        """(Re)bind the model client to one explicit key. No SDK globals."""
        self._model = build_text_model(api_key, self._model_name)

    def _ensure_model(self) -> None:
        """
        Bind the model to the pool's current lowest-live key.

        Reconfigures only when the borrowed key changed, so the common
        case (same key alive across documents) pays no rebuild. Raises
        LLMRateLimitError when the whole pool is spent — with the pool's
        own wait estimate when the keys merely need a minute.
        """
        assert self._pool is not None
        try:
            index, api_key = self._pool.acquire()
        except AllKeysCooling as e:
            raise LLMRateLimitError(e.seconds_until_next) from e
        except AllKeysExhausted as e:
            raise LLMRateLimitError() from e
        if index != self._key_index or self._model is None:
            self._bind(api_key)
            self._key_index = index

    def _get_model(self):
        """The test double wins when one was injected; otherwise bind lazily.

        Unit tests swap `provider._model` with scripted fakes — that seam is
        preserved exactly (same `generate_content(content,
        generation_config=None)` call shape).
        """
        if self._model is not None:
            return self._model
        if self._pool is not None:
            self._ensure_model()
            return self._model
        self._bind(self._single_key)
        return self._model

    async def extract_structured(
        self,
        text: str,
        document_type: str,
        system_prompt: str,
    ) -> dict:
        """
        Send aggregated OCR text to Gemini and get structured extraction.

        Retries once on JSON parse failure.
        """
        full_prompt = f"{system_prompt}\n\n--- DOCUMENT TEXT ---\n{text}\n--- END DOCUMENT TEXT ---"

        return await self._call_gemini(full_prompt, document_type)

    async def extract_from_images(
        self,
        images: list[Image.Image],
        document_type: str,
        system_prompt: str,
    ) -> dict:
        """
        Send rendered document images directly to Gemini for extraction.

        Used when OCR fails or returns empty text. Gemini reads text
        directly from the rendered page images.
        """
        # Build multimodal content: prompt + images
        content_parts = [system_prompt + "\n\nExtract all data from the following document image(s):"]

        for i, img in enumerate(images):
            content_parts.append(img)

        return await self._call_gemini(content_parts, document_type)

    async def _call_gemini(
        self,
        content,
        document_type: str,
    ) -> dict:
        """
        Call Gemini with either text or multimodal content.

        Retries once on JSON parse failure.
        """
        import asyncio

        model = self._get_model()

        for attempt in range(2):
            try:
                response = await asyncio.to_thread(
                    model.generate_content,
                    content,
                    generation_config=GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                # Truncation guard: a MAX_TOKENS finish still returns
                # parseable JSON missing its later rows. Must not be
                # accepted as a complete extraction.
                if _is_truncated(response):
                    usage = getattr(response, "usage_metadata", None)
                    tokens = getattr(usage, "candidates_token_count", None)
                    logger.error("LLM output truncated", extra={
                        "provider": "gemini",
                        "completion_tokens": tokens,
                    })
                    raise LLMOutputTruncatedError(tokens)

                logger.info("LLM extraction", extra={
                    "provider": "gemini",
                    "document_type": document_type,
                    "attempt": attempt + 1,
                })

                if self._pool is not None and self._key_index is not None:
                    self._pool.report_success(self._key_index)

                # Parse the JSON response
                response_text = response.text.strip()

                # Handle markdown code fences around JSON
                if response_text.startswith("```"):
                    response_text = re.sub(
                        r"^```(?:json)?\s*\n?", "", response_text
                    )
                    response_text = re.sub(r"\n?```\s*$", "", response_text)

                result = json.loads(response_text)
                return result

            except json.JSONDecodeError as e:
                logger.warning("LLM JSON parse failed, retrying", extra={
                    "attempt": attempt + 1,
                    "error_type": "JSONDecodeError",
                })
                if attempt == 1:
                    raise ValueError(
                        f"LLM returned invalid JSON after 2 attempts"
                    ) from e

            except Exception as e:
                logger.error("LLM extraction failed", extra={
                    "attempt": attempt + 1,
                    "error_type": type(e).__name__,
                })
                if _is_rate_limit(e):
                    # Deliberately not retried here. The free tier's binding
                    # limit is per *day*, so a second attempt inside this call
                    # would fail identically — and even a per-minute limit is
                    # measured in tens of seconds, far longer than a request
                    # should hold a worker. Raising a distinct type lets the
                    # pipeline label it as throttling rather than as a bad
                    # document, which is the difference between the user
                    # rescanning a perfectly good file and simply waiting.
                    if self._pool is not None and self._key_index is not None:
                        self._pool.retire(
                            self._key_index,
                            permanent=is_daily_quota(str(e)),
                        )
                        self._key_index = None
                    raise LLMRateLimitError(_retry_after(e)) from e
                raise

        raise RuntimeError("LLM extraction failed after retries")
