"""
FiniRAG OCR Service.

Orchestrates OCR extraction with a primary/fallback provider chain.
Mistral OCR is the primary backend (OCR_PROVIDER=mistral); Google Vision
remains supported for legacy deployments (OCR_PROVIDER=google_vision).
The chain keeps its second slot because the failure path below is what
the pipeline depends on.

Fallback is per-page, not per-document — a failure on page 1 does NOT
prevent page 2 from using the primary provider.

When every provider fails on a page the service returns empty text with
provider "none" rather than raising. That is deliberate: the real second
line of defence lives one layer up, where DocumentService sends the
rendered pages to the LLM's vision endpoint. An exception here would
abort the document before that fallback ever got its turn.
"""

from __future__ import annotations

import logging
from PIL import Image

from backend.app.core.config import settings
from backend.app.providers.ocr_provider import OCRProvider, OCRResult

logger = logging.getLogger("finirag")


class OCRService:
    """Orchestrates OCR with primary/fallback providers."""

    def __init__(self):
        self._primary: OCRProvider | None = None
        self._fallback: OCRProvider | None = None
        self._init_providers()

    def _init_providers(self):
        """
        Initialize OCR providers based on configuration.

        An unrecognised OCR_PROVIDER leaves the chain empty instead of
        guessing. Every page then reports provider "none" and the document
        goes to the LLM vision fallback — degraded but correct, and the
        error log names the typo. Silently substituting a default would
        hide a misconfigured deployment behind plausible-looking results.
        """
        if settings.OCR_PROVIDER == "mistral":
            from backend.app.providers.mistral_ocr_provider import MistralOCRProvider

            try:
                self._primary = MistralOCRProvider()
            except Exception as e:
                logger.error("Mistral OCR unavailable, OCR disabled", extra={
                    "error_type": type(e).__name__,
                })
                self._primary = None
            self._fallback = None
        elif settings.OCR_PROVIDER == "google_vision":
            from backend.app.providers.google_vision_provider import GoogleVisionProvider

            self._primary = GoogleVisionProvider(settings.GOOGLE_VISION_CREDENTIALS)
            self._fallback = None
        else:
            logger.error("Unknown OCR_PROVIDER, OCR disabled", extra={
                "ocr_provider": settings.OCR_PROVIDER,
            })
            self._primary = None
            self._fallback = None

    async def extract_text_from_images(
        self, images: list[Image.Image]
    ) -> list[OCRResult]:
        """
        Extract text from a list of preprocessed images.

        Uses primary provider first; falls back per-page on failure.
        """
        results = []

        for i, img in enumerate(images):
            page_number = i + 1
            result = await self._extract_with_fallback(img, page_number)
            results.append(result)

        return results

    async def _extract_with_fallback(
        self, image: Image.Image, page_number: int
    ) -> OCRResult:
        """
        Extract text from a single image with primary/fallback.

        Falls back per-page — does NOT abort the entire document.
        """
        # Try primary provider
        if self._primary:
            try:
                result = await self._primary.extract_text(image, page_number)
                if result.text.strip():
                    return result
                logger.warning("Primary OCR returned empty text", extra={
                    "page_number": page_number,
                    "provider": result.provider,
                })
            except Exception as e:
                logger.warning("Primary OCR failed, trying fallback", extra={
                    "page_number": page_number,
                    "error_type": type(e).__name__,
                })

        # Try fallback provider
        if self._fallback:
            try:
                result = await self._fallback.extract_text(image, page_number)
                return result
            except Exception as e:
                logger.error("Fallback OCR also failed", extra={
                    "page_number": page_number,
                    "error_type": type(e).__name__,
                })

        # Both failed — return empty result
        return OCRResult(
            text="",
            words=[],
            provider="none",
            page_number=page_number,
        )
