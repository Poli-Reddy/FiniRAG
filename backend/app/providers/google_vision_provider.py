"""
FiniRAG Google Vision OCR Provider (legacy / compatibility shim).

Kept so existing deployments and tests referencing Google Vision keep
working. New deployments should use Mistral OCR (OCR_PROVIDER=mistral),
which returns layout blocks for traceable evidence.

Client is lazy: constructing the provider makes no network call.
"""

from __future__ import annotations

import logging

from PIL import Image

from backend.app.providers.ocr_provider import OCRProvider, OCRResult

logger = logging.getLogger("finirag")


class GoogleVisionProvider(OCRProvider):
    """Google Cloud Vision OCR backend (legacy)."""

    def __init__(self, credentials_path: str = ""):
        self._credentials_path = credentials_path
        # Lazy: tests assert this is None until first use.
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google.cloud import vision

            if self._credentials_path:
                import os

                os.environ.setdefault(
                    "GOOGLE_APPLICATION_CREDENTIALS", self._credentials_path
                )
            self._client = vision.ImageAnnotatorClient()
        return self._client

    async def extract_text(self, image: Image.Image, page_number: int) -> OCRResult:
        import asyncio
        import io

        def _call() -> str:
            client = self._ensure_client()
            from google.cloud import vision

            buf = io.BytesIO()
            img = image.convert("RGB") if image.mode not in ("RGB", "L") else image
            img.save(buf, format="PNG")
            vision_image = vision.Image(content=buf.getvalue())
            response = client.document_text_detection(image=vision_image)
            if response.error.message:
                raise RuntimeError(response.error.message)
            return (response.full_text_annotation.text or "") if response.full_text_annotation else ""

        text = await asyncio.to_thread(_call)
        logger.info(
            "OCR extraction",
            extra={"provider": "google_vision", "page_number": page_number, "chars": len(text)},
        )
        return OCRResult(text=text, provider="google_vision", page_number=page_number)
