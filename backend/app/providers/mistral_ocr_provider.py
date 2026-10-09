"""
FiniRAG Mistral OCR Provider.

The OCR backend. Chosen over Google Cloud Vision for one hard reason and one
good one:

  * **Hard:** Vision requires a billing account to be linked before it will
    serve a single request, free tier or not. This project has none, and the
    API stayed disabled for the whole of the previous run.
  * **Good:** Mistral returns `blocks` — laid-out regions with real pixel
    coordinates and the text inside them. That is what turns evidence from an
    echo of the extracted value into something a reviewer can trace back to a
    place on the page. It is the single biggest correctness gap this project
    had, and no other free OCR option here closes it.

Cost is $4 per 1,000 pages against a $10/month free credit, so the 50-document
dataset costs roughly two cents and the credit covers ~2,500 pages a month.

Key rotation
────────────
Every call borrows a key from the shared MISTRAL_KEYS pool and hands it back.
A refused key is retired — permanently for a per-day quota, briefly for a
per-minute throttle — and the *same* call retries on the next key. The caller
never sees the rotation unless every key is spent.

Logging: page and block counts only. No document text, no key material
(Correction #10).
"""

from __future__ import annotations

import base64
import io
import logging

from PIL import Image

from backend.app.core.config import MISTRAL_KEYS, settings
from backend.app.core.key_pool import AllKeysCooling, AllKeysExhausted, KeyPool
from backend.app.providers.errors import LLMRateLimitError
from backend.app.providers.ocr_provider import OCRBlock, OCRProvider, OCRResult, OCRWord

logger = logging.getLogger("finirag")

# Rendered pages run well above what the model needs to read them, and the
# image is base64-encoded into a JSON body. 2000 px on the long edge keeps
# statement digits legible while keeping the request modest — higher than the
# LLM vision path's 1600 px because OCR is the one stage that should see the
# finest detail available.
MAX_IMAGE_EDGE = 2000

# Block types that carry no readable financial text. Kept out of the text
# stream so a signature scribble or a logo caption cannot be matched as
# evidence for a number.
SKIP_BLOCK_TYPES = {"image", "signature"}


def _is_rate_limit(exc: Exception) -> bool:
    """True for a Mistral quota or rate-limit rejection."""
    if type(exc).__name__ in ("RateLimitError", "TooManyRequests", "SDKError"):
        # SDKError is the SDK's generic HTTP wrapper; only a 429 counts.
        return getattr(exc, "status_code", None) == 429 or type(exc).__name__ != "SDKError"
    return getattr(exc, "status_code", None) == 429


def _is_daily_quota(exc: Exception) -> bool:
    """
    True when the refusal is a per-*day* limit rather than a per-minute one.

    The distinction decides whether a key is retired for the run or merely
    rested, and getting it wrong is expensive in both directions: resting a
    dead key wastes a round trip on every document, and retiring a live one
    throws away free capacity.

    The message is inspected but never logged — Correction #10 keeps vendor
    text out of logs, not out of control flow.
    """
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("per day", "perday", "daily", "per-day", "monthly", "credit")
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


def _block_confidence(block) -> float | None:
    """
    The block's confidence, if the response carried one.

    Returns None rather than 0.0 when absent. A real zero means "the model is
    certain this is wrong"; a missing score means "the model did not say".
    Collapsing them would let an unscored block look like a rejected one.
    """
    scores = getattr(block, "confidence_scores", None)
    if scores is None:
        return None
    for attr in ("block", "mean", "score", "confidence"):
        value = getattr(scores, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


class MistralOCRProvider(OCRProvider):
    """Mistral OCR provider with multi-key rotation."""

    def __init__(self, pool: KeyPool | None = None, model: str | None = None):
        # The pool is injected so tests can supply their own without touching
        # module state, and defaults to the shared one in production.
        self._pool = pool if pool is not None else MISTRAL_KEYS
        self._model = model or settings.MISTRAL_OCR_MODEL

    def _client(self, api_key: str):
        """
        A client bound to one key.

        Built per call rather than cached: the key changes on rotation, and a
        cached client would keep using the key that was just retired.
        Construction is cheap — it configures an httpx client, nothing more.
        """
        try:
            from mistralai import Mistral  # v3 SDK

            return Mistral(api_key=api_key)
        except ImportError:
            from mistralai.client import Mistral  # v2 SDK

            return Mistral(api_key=api_key)

    async def extract_text(self, image: Image.Image, page_number: int) -> OCRResult:
        """
        Extract text and laid-out blocks from one preprocessed page image.

        Rotates through the key pool on refusal. Raises LLMRateLimitError only
        when every key is spent — at that point the pipeline's vision fallback
        is the remaining option, and it needs to be told this was throttling
        and not an unreadable document.
        """
        encoded = _encode(image)
        attempts = 0

        # Bounded by the pool size: every iteration either returns or retires
        # a key, so the loop cannot outlive the pool. The +1 leaves room for
        # the final key's attempt.
        while attempts <= len(self._pool):
            attempts += 1

            try:
                index, api_key = self._pool.acquire()
            except AllKeysCooling as e:
                # Recoverable, but not by holding a request open for a minute.
                # The batch runner catches this and sleeps; an API request
                # surfaces it as throttling.
                logger.warning("All OCR keys cooling", extra={
                    "provider": "mistral",
                    "seconds_until_next": round(e.seconds_until_next, 1),
                })
                raise LLMRateLimitError(e.seconds_until_next) from e
            except AllKeysExhausted as e:
                logger.error("All OCR keys exhausted", extra={
                    "provider": "mistral",
                    "keys_total": e.total,
                })
                raise LLMRateLimitError() from e

            try:
                response = await self._client(api_key).ocr.process_async(
                    model=self._model,
                    document={"type": "image_url", "image_url": encoded},
                    include_blocks=True,
                    confidence_scores_granularity="block",
                )
                self._pool.report_success(index)
                return self._to_result(response, page_number)

            except Exception as e:
                if not _is_rate_limit(e):
                    logger.error("OCR extraction failed", extra={
                        "provider": "mistral",
                        "page_number": page_number,
                        "key_index": index,
                        "error_type": type(e).__name__,
                    })
                    raise

                self._pool.retire(index, permanent=_is_daily_quota(e))
                logger.info("OCR key refused, rotating", extra={
                    "provider": "mistral",
                    "page_number": page_number,
                    "key_index": index,
                    "keys_live": self._pool.live_count,
                })

        # Every key was tried and refused within this call.
        logger.error("OCR exhausted every key", extra={
            "provider": "mistral",
            "page_number": page_number,
            "keys_total": len(self._pool),
        })
        raise LLMRateLimitError()

    def _to_result(self, response, page_number: int) -> OCRResult:
        """
        Map one OCR response onto the pipeline's OCRResult.

        A single page image is sent per call, so `pages` holds one entry. It
        is read defensively anyway: an empty `pages` list is a valid response
        shape for a page with nothing on it, and indexing it blindly would
        turn a blank page into a crash.
        """
        pages = getattr(response, "pages", None) or []
        if not pages:
            return OCRResult(text="", provider="mistral", page_number=page_number)

        page = pages[0]

        blocks: list[OCRBlock] = []
        for raw in (getattr(page, "blocks", None) or []):
            block_type = getattr(raw, "type", "text")
            content = (getattr(raw, "content", "") or "").strip()
            if not content or block_type in SKIP_BLOCK_TYPES:
                continue

            box = None
            x1 = getattr(raw, "top_left_x", None)
            if x1 is not None:
                box = [
                    float(x1),
                    float(getattr(raw, "top_left_y", 0)),
                    float(getattr(raw, "bottom_right_x", 0)),
                    float(getattr(raw, "bottom_right_y", 0)),
                ]

            blocks.append(OCRBlock(
                text=content,
                bounding_box=box,
                block_type=block_type,
                confidence=_block_confidence(raw),
            ))

        # The markdown is the provider's own reading order and keeps table
        # structure intact, which matters for a balance sheet. Blocks are
        # joined only as a fallback, because a blocks-only join loses the
        # column alignment that makes a statement legible to the LLM.
        text = (getattr(page, "markdown", "") or "").strip()
        if not text and blocks:
            text = "\n".join(b.text for b in blocks)

        dimensions = getattr(page, "dimensions", None)
        size = None
        if dimensions is not None:
            width = getattr(dimensions, "width", None)
            height = getattr(dimensions, "height", None)
            if width and height:
                size = (int(width), int(height))

        logger.info("OCR extraction", extra={
            "provider": "mistral",
            "page_number": page_number,
            "chars": len(text),
            "blocks": len(blocks),
        })

        return OCRResult(
            text=text,
            words=[
                # Block text under the word field as well, so any consumer
                # written against `words` still sees content. The real
                # spatial unit here is the block; see OCRBlock.
                OCRWord(text=b.text, confidence=b.confidence or 0.0,
                        bounding_box=b.bounding_box)
                for b in blocks
            ],
            blocks=blocks,
            provider="mistral",
            page_number=page_number,
            page_dimensions=size,
        )
