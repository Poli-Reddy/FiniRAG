"""
FiniRAG Image Preprocessor.

Application-level image preprocessing that runs BEFORE any OCR provider.
This is an application responsibility, NOT delegated to the OCR provider
(Gate 2.1 architectural principle).

Handles:
  - EXIF orientation correction (Trap D19)
  - PDF → image rendering at 300 DPI (for vector-outlined PDFs)
  - Missing JPEG EOI marker (Samsung SEFT trailer)
"""

from __future__ import annotations

import io
import logging
import threading

import fitz  # PyMuPDF
from PIL import Image, ImageFile, ImageOps

from backend.app.core.config import settings

logger = logging.getLogger("finirag")

# EXIF tag 0x0112. Values 2-8 all describe a rotation and/or a mirror;
# 1 (and an absent tag) mean the pixels are already the right way up.
_EXIF_ORIENTATION = 0x0112

# Serializes the LOAD_TRUNCATED_IMAGES toggle below. The flag is
# process-global; without the lock, two concurrent decodes could observe
# each other's setting. The critical section is short and synchronous.
_TRUNCATED_FLAG_LOCK = threading.Lock()


class ImagePreprocessor:
    """Application-level image preprocessing before OCR."""

    RENDER_DPI = 300  # High enough for OCR accuracy

    def preprocess_images(self, file_bytes: bytes, file_type: str) -> list[Image.Image]:
        """
        Preprocess JPG/PNG images for OCR.

        1. Open with Pillow, tolerating a truncated tail
        2. Apply EXIF orientation correction (Trap D19)
        3. Normalise the colour mode
        4. Return list of PIL Images (always 1 for images)
        """
        img = self._open(file_bytes)

        # Decoded-size gate: compressed bytes say nothing about pixel cost.
        # Reject before EXIF/transpose work allocates anything further.
        self._check_pixels(img.width, img.height, "upload")

        # Read the tag before transposing: exif_transpose strips it from
        # the image it returns, so afterwards there is nothing left to
        # report and the log line would claim no rotation ever happened.
        orientation = img.getexif().get(_EXIF_ORIENTATION, 1)
        exif_rotated = orientation not in (0, 1)

        # EXIF orientation correction — critical for rotated images
        img = ImageOps.exif_transpose(img)

        # Convert to RGB if needed (some PNGs are RGBA, some are palette).
        # L is left alone: greyscale is already what OCR wants, and
        # widening it to RGB triples the bytes sent to the provider.
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        logger.info("Image preprocessing", extra={
            "pages": 1,
            "exif_rotated": exif_rotated,
            "size": f"{img.width}x{img.height}",
        })

        return [img]

    @staticmethod
    def _check_pixels(width: int, height: int, what: str) -> None:
        """Reject absurd decoded dimensions before they cost memory/CPU."""
        pixels = width * height
        if pixels > settings.MAX_IMAGE_PIXELS:
            raise ValueError(
                f"{what} decodes to {width}x{height} "
                f"({pixels} px), over the {settings.MAX_IMAGE_PIXELS} px cap"
            )

    @staticmethod
    def _open(file_bytes: bytes) -> Image.Image:
        """
        Decode the upload, tolerating a truncated tail on a second attempt.

        Phone cameras and mail gateways both emit JPEGs whose data stops
        before the EOI marker. Pillow refuses those by default, and the
        first attempt keeps that strictness so genuine garbage still
        raises. Only after a real decode failure do we retry with
        LOAD_TRUNCATED_IMAGES — and restore it afterwards, because the
        flag is process-global and leaving it on would quietly weaken
        the corruption gate for every later upload.

        Both attempts call load(): Image.open only reads the header, so
        truncation does not surface until the pixels are decoded.
        """
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.load()
            return img
        except Exception:
            pass

        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            # Locked: the flag is process-global, and without serialization a
            # concurrent decode could observe the other request's setting. The
            # section is synchronous (no awaits), so the hold time is bounded.
            with _TRUNCATED_FLAG_LOCK:
                img = Image.open(io.BytesIO(file_bytes))
                img.load()
                return img
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous

    def render_pdf_pages(self, file_bytes: bytes) -> list[Image.Image]:
        """
        Render each PDF page to a 300 DPI PIL Image using PyMuPDF.

        Used only when native text extraction fails
        (vector-outlined PDFs, scanned PDFs).
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        images = []

        # Scale factor: 300 DPI / 72 DPI (PDF default)
        zoom = self.RENDER_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Rendered-pixel gate first: a huge page rect at 300 DPI can
            # decode to gigapixels from a small compressed file.
            rect = page.rect
            if rect.width * zoom * rect.height * zoom > settings.MAX_RENDER_PIXELS_PER_PAGE:
                doc.close()
                raise ValueError(
                    f"page {page_num + 1} would render beyond the "
                    f"{settings.MAX_RENDER_PIXELS_PER_PAGE} px cap"
                )
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            # Convert PyMuPDF pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()

        logger.info("PDF rendering", extra={
            "pages": len(images),
            "dpi": self.RENDER_DPI,
        })

        return images
