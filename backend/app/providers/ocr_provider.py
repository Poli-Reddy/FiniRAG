"""
FiniRAG OCR Provider Interface.

Abstract base class for all OCR providers. Swappable without
touching the pipeline (Gate 2.1 architectural principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class OCRWord:
    """A single word with optional spatial information."""
    text: str
    confidence: float = 0.0
    bounding_box: list[float] | None = None  # [x1, y1, x2, y2]


@dataclass
class OCRBlock:
    """
    A laid-out region of a page — a paragraph, a table, a title — with the
    box it occupies.

    Distinct from OCRWord on purpose. A word box is too small to be evidence
    for a financial figure: "557.97" on its own proves nothing, while the
    block it sits in ("Capital 557.97") shows what the number *is*. Evidence
    is built from blocks for that reason.

    Coordinates are pixels in the coordinate space of `page_dimensions` on
    the owning OCRResult, top-left origin.
    """
    text: str
    bounding_box: list[float] | None = None  # [x1, y1, x2, y2]
    block_type: str = "text"                 # text | title | table | list | ...
    confidence: float | None = None          # None when the provider gave none


@dataclass
class OCRResult:
    """Result of OCR extraction for a single page."""
    text: str = ""
    words: list[OCRWord] = field(default_factory=list)
    blocks: list[OCRBlock] = field(default_factory=list)
    provider: str = ""
    page_number: int = 1
    page_dimensions: tuple[int, int] | None = None  # (width, height) in px
    raw_response: Any = None  # For DEBUG-level logging only (Correction #10)


class OCRProvider(ABC):
    """Abstract OCR provider. All providers must implement this interface."""

    @abstractmethod
    async def extract_text(self, image: Image.Image, page_number: int) -> OCRResult:
        """
        Extract text from a single preprocessed image.

        Args:
            image: PIL Image, already EXIF-corrected and oriented
            page_number: 1-indexed page number

        Returns:
            OCRResult with text, words, and provider metadata
        """
        pass
