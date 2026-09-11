"""
FiniRAG Groq LLM Provider.

A second structured-extraction backend, used as a fallback when the primary
provider refuses a call. Groq's free tier is far more generous than Gemini's
20-requests-per-day, and on this workload it is roughly ten times faster, so
it turns a quota outage into a slowdown rather than a stoppage.

Supports the same two modes as GeminiProvider:
  1. Text-based: send OCR/native text
  2. Vision-based: send rendered page images

Model choice is deliberate. Only the two qwen models on this account accept
image parts at all — every gpt-oss, compound and allam model rejects
multimodal content with "messages[0].content must be a string". Of the two,
`qwen/qwen3.8-27b` handles both text and vision under
`response_format=json_object`; `qwen3.6-27b` does vision but fails JSON
validation on text-only prompts. If the vision model is ever changed, re-check
both properties — losing either silently disables the fallback on the path
that actually needs it, since no document in the dataset has a native text
layer.

Output budget is the real constraint on this tier, not speed or input size.
See MAX_OUTPUT_TOKENS.

Logging: does NOT log the prompt or the response (Correction #10).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re

from PIL import Image

from backend.app.core.config import GROQ_KEYS, settings
from backend.app.core.key_pool import (
    AllKeysCooling,
    AllKeysExhausted,
    KeyPool,
    is_daily_quota,
)
from backend.app.providers.errors import LLMOutputTruncatedError, LLMRateLimitError
from backend.app.providers.llm_provider import LLMProvider

logger = logging.getLogger("finirag")

# Reasoning-tuned models narrate before answering. JSON mode usually suppresses
# it, but qwen3.6 emits it even so, and a stray <think> block turns a valid
# reply into a parse failure.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Groq rejects oversized request bodies. Pages render well above what the model
# needs to read them, and a 4.4 MB scan base64-encodes to ~6 MB, so the longest
# edge is capped before encoding. 1600 px keeps statement digits legible.
# Input is not the binding limit either way: a rendered page costs ~2,200
# prompt tokens against a 8,000 TPM allowance.
MAX_IMAGE_EDGE = 1600

# The account's qwen models enforce a 1,000 output-tokens-per-minute cap, and
# it is checked per request against the *expected* output — the call is
# rejected with a 429 before the model reads anything. Measured on this key:
# max_tokens=999 is always accepted, 2000 and 4000 are always rejected, and
# leaving max_tokens unset is accepted only when the preceding minute left
# enough budget. That last case is the dangerous one: it makes success depend
# on run history rather than on the request, so the ceiling is always sent
# explicitly.
#
# 999 is the most the qwen vision tier will accept, not a tuning choice.
# The text model (gpt-oss-120b) accepts max_tokens=8000 on the same key,
# so the text path gets a roomy budget while vision keeps the capped one.
# The two ceilings are per-model, not per-document: verified live against
# the API (an 8000-token text call is accepted; 2000/4000 on qwen are not).
MAX_OUTPUT_TOKENS = 999
MAX_TEXT_OUTPUT_TOKENS = 8000


def _is_rate_limit(exc: Exception) -> bool:
    """True for a Groq quota or rate-limit rejection."""
    if type(exc).__name__ in ("RateLimitError", "TooManyRequests"):
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    # Groq TPM messages often arrive without status_code (e.g. "rate limit
    # exceeded ... tokens per minute"). Match on text, never logged.
    text = str(exc).lower()
    return "rate limit" in text or "too many requests" in text or "429" in text


def _is_quota_error(exc: Exception) -> bool:
    """
    True when a 400-class rejection is really a capacity/quota refusal.

    Groq returns budget overruns (TPM/RPM spend caps, model capacity) as
    400 BadRequest rather than 429, so the status-code check alone files
    them as document failures. The message is inspected but never logged.
    Matching is deliberately narrow: content/shape errors must keep
    failing the document, not masquerade as throttling.
    """
    text = str(exc).lower()
    return any(
        cue in text
        for cue in (
            "quota", "tokens per minute", "requests per minute", "tpm", "rpm",
            "token budget", "exceeded", "capacity", "overloaded",
            "try again", "rate_limit_exceeded",
        )
    )


def _encode(image: Image.Image) -> str:
    """One page as a base64 PNG data URL, downscaled if oversized."""
    if max(image.size) > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / max(image.size)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.LANCZOS,
        )
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class GroqProvider(LLMProvider):
    """Groq LLM provider for structured extraction."""

    def __init__(self, pool: KeyPool | None = None):
        # Imported here so the app still boots when the optional dependency is
        # absent — a missing fallback must not take the primary path down.
        from groq import Groq

        # `pool=None` keeps the legacy single-key behaviour. Production
        # passes the shared GROQ_KEYS pool; the client is then bound lazily
        # per call to the lowest live key (see _ensure_client), so spent
        # keys fall out of rotation without any in-call retry.
        self._pool = pool
        self._key_index: int | None = None
        if pool is None:
            api_key = settings.GROQ_API_KEY or (GROQ_KEYS.keys[0] if len(GROQ_KEYS) else "")
            self._client = Groq(api_key=api_key)
        else:
            self._client = None
        self._text_model = settings.GROQ_MODEL
        self._vision_model = settings.GROQ_VISION_MODEL

    def _ensure_client(self):
        """Bind the client to the pool's current lowest-live key."""
        from groq import Groq

        assert self._pool is not None
        try:
            index, api_key = self._pool.acquire()
        except AllKeysCooling as e:
            raise LLMRateLimitError(e.seconds_until_next) from e
        except AllKeysExhausted as e:
            raise LLMRateLimitError() from e
        if index != self._key_index or self._client is None:
            self._client = Groq(api_key=api_key)
            self._key_index = index
        return self._client

    async def extract_structured(
        self,
        text: str,
        document_type: str,
        system_prompt: str,
    ) -> dict:
        """Send aggregated document text to Groq and get structured extraction."""
        prompt = (
            f"{system_prompt}\n\n--- DOCUMENT TEXT ---\n{text}\n--- END DOCUMENT TEXT ---"
        )
        return await self._call(
            model=self._text_model,
            content=prompt,
            document_type=document_type,
            max_tokens=MAX_TEXT_OUTPUT_TOKENS,
        )

    async def extract_from_images(
        self,
        images: list[Image.Image],
        document_type: str,
        system_prompt: str,
    ) -> dict:
        """Send rendered page images to Groq and get structured extraction."""
        parts: list[dict] = [{
            "type": "text",
            "text": system_prompt
            + "\n\nExtract all data from the following document image(s):",
        }]
        for image in images:
            parts.append({
                "type": "image_url",
                "image_url": {"url": _encode(image)},
            })

        return await self._call(
            model=self._vision_model,
            content=parts,
            document_type=document_type,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    async def _call(
        self, model: str, content, document_type: str,
        max_tokens: int = MAX_OUTPUT_TOKENS,
    ) -> dict:
        """
        One extraction, retried once if the reply will not parse as JSON.

        Mirrors GeminiProvider deliberately: same retry budget, same fence
        handling, same refusal to retry a transport error. Two backends that
        fail differently would make the pipeline's behaviour depend on which
        one happened to answer.
        """
        import asyncio

        # Pool-backed rotation. `getattr` (not direct access) because the
        # unit doubles build instances via __new__ without running __init__.
        pool = getattr(self, "_pool", None)
        if pool is not None:
            self._ensure_client()

        # JSON mode is an optimization, not a requirement: on long prompts
        # the model can refuse constrained generation outright
        # (json_validate_failed). Dropping the constraint and parsing the
        # free-form reply with the same fence-stripping below recovers
        # documents that would otherwise fail the whole chain.
        json_mode = True

        for attempt in range(2):
            try:
                create_kwargs: dict = {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    create_kwargs["response_format"] = {"type": "json_object"}
                response = await asyncio.to_thread(
                    self._client.chat.completions.create,
                    **create_kwargs,
                )

                logger.info("LLM extraction", extra={
                    "provider": "groq",
                    "document_type": document_type,
                    "attempt": attempt + 1,
                })

                if pool is not None and getattr(self, "_key_index", None) is not None:
                    pool.report_success(self._key_index)

                choice = response.choices[0]

                # A truncated reply still parses: JSON mode closes the braces
                # it opened, so the result looks like a clean extraction that
                # happens to be missing its later rows and totals. Checked
                # before parsing, because parsing succeeds and is exactly what
                # makes this dangerous.
                if getattr(choice, "finish_reason", None) == "length":
                    usage = getattr(response, "usage", None)
                    logger.error("LLM output truncated at the token ceiling", extra={
                        "provider": "groq",
                        "document_type": document_type,
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                    })
                    raise LLMOutputTruncatedError(
                        getattr(usage, "completion_tokens", None)
                    )

                reply = (choice.message.content or "").strip()
                reply = _THINK_BLOCK.sub("", reply).strip()

                if reply.startswith("```"):
                    reply = re.sub(r"^```(?:json)?\s*\n?", "", reply)
                    reply = re.sub(r"\n?```\s*$", "", reply)

                return json.loads(reply)

            except LLMOutputTruncatedError:
                # Already logged with its token count, and retrying would only
                # truncate again at the same ceiling.
                raise

            except json.JSONDecodeError as e:
                logger.warning("LLM JSON parse failed, retrying", extra={
                    "provider": "groq",
                    "attempt": attempt + 1,
                    "error_type": "JSONDecodeError",
                })
                if attempt == 1:
                    raise ValueError(
                        "LLM returned invalid JSON after 2 attempts"
                    ) from e

            except Exception as e:
                logger.error("LLM extraction failed", extra={
                    "provider": "groq",
                    "attempt": attempt + 1,
                    "error_type": type(e).__name__,
                })
                if "json_validate_failed" in str(e).lower() and json_mode and attempt == 0:
                    # Constrained generation refused; spend the remaining
                    # attempt on a free-form reply instead of failing.
                    json_mode = False
                    logger.info("Groq JSON mode refused, retrying free-form", extra={
                        "provider": "groq",
                        "document_type": document_type,
                    })
                    continue
                if _is_rate_limit(e) or _is_quota_error(e):
                    if pool is not None and getattr(self, "_key_index", None) is not None:
                        pool.retire(
                            self._key_index,
                            permanent=is_daily_quota(str(e)),
                        )
                        self._key_index = None
                        self._client = None
                    raise LLMRateLimitError() from e
                raise

        raise RuntimeError("LLM extraction failed after retries")
