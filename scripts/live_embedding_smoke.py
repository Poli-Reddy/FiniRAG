"""Live embedding smoke test — gemini-embedding-2 via google-genai.

Reports model ID, returned dims, real-API-call flag. Never prints the key.
Exit 0 on success or documented skip; exit 2 on unexpected failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, ".")


async def main() -> dict:
    from backend.app.core.config import settings

    has_key = bool(os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY)
    if not has_key:
        result = {"status": "SKIPPED", "reason": "GEMINI_API_KEY not set",
                  "model": settings.EMBEDDING_MODEL, "dim": settings.EMBEDDING_DIM,
                  "real_api_call": False}
        print(json.dumps(result, indent=2))
        return result
    from backend.app.providers.embedding_provider import GeminiEmbeddingProvider

    provider = GeminiEmbeddingProvider()
    try:
        vecs = await provider.embed_texts(["smoke test: invoice total 108.00"])
        result = {"status": "OK", "model": provider.model, "dim": len(vecs[0]),
                  "expected_dim": provider.dim, "real_api_call": True,
                  "finite": all(__import__("math").isfinite(x) for x in vecs[0][:8])}
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        result = {"status": "FAILED", "model": provider.model,
                  "code": getattr(e, "code", type(e).__name__),
                  "real_api_call": True}
        print(json.dumps(result, indent=2))
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
