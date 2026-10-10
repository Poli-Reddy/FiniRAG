"""
Unified Google GenAI text-generation adapter.

All text generation (extraction, answers) goes through google-genai
(`google.genai`, installed as `google-genai`). The legacy
`google.generativeai` package is end-of-support and must not be used for
new calls — grep the tree for `google.generativeai` if this needs proof.

The adapter preserves the exact call seam the unit tests pin down:
`generate_content(content, generation_config=None)` where content is a
string or a list mixing strings and PIL images, and the response exposes
`.text`, `.candidates` (with `finish_reason`) and `.usage_metadata`.
Tests keep swapping `provider._model` with scripted doubles.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("finirag")


class GenerationConfig:
    """Subset of generation knobs this codebase uses. Plain data so tests
    and providers never touch SDK types directly."""

    def __init__(self, response_mime_type: str | None = None,
                 temperature: float | None = None,
                 max_output_tokens: int | None = None):
        self.response_mime_type = response_mime_type
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens


class GenaiTextModel:
    """Thin wrapper over `client.models.generate_content`.

    Constructed with an explicit client (built from an explicit API key) and
    model name — never from ambient SDK-global configuration, so pooled keys
    actually take effect per call.
    """

    def __init__(self, client: Any, model_name: str):
        self._client = client
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_content(self, content, generation_config: GenerationConfig | None = None):
        from google.genai import types

        if isinstance(content, str):
            contents = content
        else:
            contents = [_to_part(p) for p in content]
        kwargs: dict[str, Any] = {}
        if generation_config is not None:
            cfg: dict[str, Any] = {}
            if generation_config.response_mime_type is not None:
                cfg["response_mime_type"] = generation_config.response_mime_type
            if generation_config.temperature is not None:
                cfg["temperature"] = generation_config.temperature
            if generation_config.max_output_tokens is not None:
                cfg["max_output_tokens"] = generation_config.max_output_tokens
            if cfg:
                kwargs["config"] = types.GenerateContentConfig(**cfg)
        return self._client.models.generate_content(
            model=self._model_name, contents=contents, **kwargs)


def _to_part(piece):
    """A string stays a string; a PIL image becomes an inline PNG Part."""
    if isinstance(piece, str):
        return piece
    import io

    from google.genai import types

    if hasattr(piece, "save"):
        buf = io.BytesIO()
        piece.save(buf, format="PNG")
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
    return str(piece)


def build_text_client(api_key: str):
    """A google-genai client bound to one explicit key (no globals)."""
    from google import genai

    if not api_key:
        raise RuntimeError("genai client requires an API key")
    return genai.Client(api_key=api_key)


def build_text_model(api_key: str, model_name: str) -> GenaiTextModel:
    return GenaiTextModel(build_text_client(api_key), model_name)
