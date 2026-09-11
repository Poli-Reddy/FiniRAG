"""
FiniRAG LLM Provider Interface.

Abstract base class for LLM providers. Gemini is the primary and Groq the
fallback; the two are interchangeable behind this interface, and
FallbackLLMProvider implements it as well so a chain can stand wherever a
single provider can.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


class LLMProvider(ABC):
    """Abstract LLM provider for structured extraction."""

    @abstractmethod
    async def extract_structured(
        self,
        text: str,
        document_type: str,
        system_prompt: str,
    ) -> dict:
        """
        Send aggregated text to LLM and get structured extraction.

        Args:
            text: Full document text (all pages concatenated)
            document_type: "invoice", "balance_sheet", etc.
            system_prompt: The extraction prompt with schema

        Returns:
            Dict matching the document type's extraction schema
        """
        pass

    @abstractmethod
    async def extract_from_images(
        self,
        images: list["Image.Image"],
        document_type: str,
        system_prompt: str,
    ) -> dict:
        """
        Send rendered page images to the LLM and get structured extraction.

        Used when OCR yields no usable text. Declared here rather than left
        to each provider because it is not an optional extra: no document in
        the dataset has a native text layer, so this is the path that runs.

        Args:
            images: One rendered image per page, in page order
            document_type: "invoice", "balance_sheet", etc.
            system_prompt: The extraction prompt with schema

        Returns:
            Dict matching the document type's extraction schema
        """
        pass
