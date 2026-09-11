"""
FiniRAG fallback LLM provider.

Wraps an ordered list of providers and tries each in turn until one returns an
extraction. Its whole purpose is to keep a provider-side outage — a quota
rejection, a transport error, a model that will not produce parseable JSON —
from ending a document's processing when another backend could have answered.

What it does NOT do is retry the *same* provider. Each provider already owns
its own retry budget; a second layer on top would multiply latency on failures
that were never going to succeed.

Two rules keep the chain honest:

  1. **Caller errors are not failover conditions.** A ValueError for an unknown
     document type is a bug in the request, and asking a second provider the
     same malformed question just hides it. Only provider-side failures move
     the chain forward.

  2. **The answer records who gave it.** `last_provider` names the backend that
     actually produced the extraction, and it is reported in the response
     metadata. A silent fallback would leave a document's numbers attributed to
     a model that never saw it.
"""

from __future__ import annotations

import logging

from PIL import Image

from backend.app.providers.llm_provider import LLMProvider

logger = logging.getLogger("finirag")


class FallbackLLMProvider(LLMProvider):
    """Tries each provider in order until one succeeds."""

    def __init__(self, providers: list[tuple[str, LLMProvider]]):
        """
        Args:
            providers: (name, provider) in priority order. The first is the
                primary; the rest are only consulted when their predecessors
                fail.
        """
        if not providers:
            raise ValueError("FallbackLLMProvider needs at least one provider")
        self._providers = providers
        self.last_provider: str | None = None

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self._providers]

    async def extract_structured(
        self,
        text: str,
        document_type: str,
        system_prompt: str,
    ) -> dict:
        return await self._attempt(
            "extract_structured",
            document_type,
            text=text,
            document_type_=document_type,
            system_prompt=system_prompt,
        )

    async def extract_from_images(
        self,
        images: list[Image.Image],
        document_type: str,
        system_prompt: str,
    ) -> dict:
        return await self._attempt(
            "extract_from_images",
            document_type,
            images=images,
            document_type_=document_type,
            system_prompt=system_prompt,
        )

    async def _attempt(self, method: str, document_type: str, **kwargs) -> dict:
        """Call `method` on each provider in turn; raise the last failure."""
        payload = dict(kwargs)
        payload["document_type"] = payload.pop("document_type_")

        last_error: Exception | None = None

        for index, (name, provider) in enumerate(self._providers):
            call = getattr(provider, method, None)
            if call is None:
                # A provider that cannot do vision is not a failure — it is
                # simply not a candidate for this call.
                logger.info("Provider does not support this mode, skipping", extra={
                    "provider": name,
                    "mode": method,
                })
                continue

            try:
                result = await call(**payload)
                self.last_provider = name
                if index > 0:
                    logger.info("LLM fallback succeeded", extra={
                        "provider": name,
                        "document_type": document_type,
                        "position": index + 1,
                    })
                return result

            except ValueError as e:
                # Raised for an unknown document type (a caller bug) and for a
                # reply that would not parse after retries (a provider fault).
                # Only the second is worth another backend, and the two are
                # distinguishable by message because neither carries vendor
                # text under Correction #10.
                if "invalid JSON" not in str(e):
                    raise
                last_error = e
                logger.warning("Provider returned unparseable JSON, falling back", extra={
                    "provider": name,
                    "document_type": document_type,
                })

            except Exception as e:
                last_error = e
                logger.warning("Provider failed, falling back", extra={
                    "provider": name,
                    "document_type": document_type,
                    "error_type": type(e).__name__,
                    "remaining": len(self._providers) - index - 1,
                })

        if last_error is not None:
            # Every backend refused. The pipeline sees the last failure, so a
            # chain that ended on a quota rejection still surfaces as
            # LLM_RATE_LIMITED rather than as a generic error.
            raise last_error

        raise RuntimeError(f"No provider supports {method}")
