"""
Provider-level exceptions the pipeline is allowed to reason about.

The pipeline must not import `google.api_core.exceptions` to find out that a
request was throttled — that would tie `DocumentService` to one vendor's
exception hierarchy and make the branch untestable without the SDK installed.
Providers translate their own failures into these instead.

Only distinctions the pipeline acts on differently belong here. A throttle
earns its own type because it is the one LLM failure that says nothing about
the document: the same file retried later will succeed, so the error code the
user sees has to point at the service rather than at their upload.
"""

from __future__ import annotations


class LLMRateLimitError(Exception):
    """
    The LLM provider rejected the call for quota or rate-limit reasons.

    Carries `retry_after_seconds` when the provider supplied it (per-minute
    limits usually do; per-day quotas usually do not, because the answer is
    "tomorrow"). The message deliberately holds no vendor text — Correction
    #10 keeps provider replies out of logs and responses alike.
    """

    def __init__(self, retry_after_seconds: float | None = None):
        self.retry_after_seconds = retry_after_seconds
        super().__init__("LLM provider rate limit or quota exceeded")


class LLMOutputTruncatedError(Exception):
    """
    The provider stopped generating because it hit its output-token ceiling.

    This has its own type because the reply is *not* a failure the caller can
    see: a truncated extraction still parses as JSON — the provider closes the
    structure — it is simply missing rows and totals. Returning it would put
    partial financials in front of a user with nothing marking them partial,
    and the validation layer would then report a real figure as MISSING.

    Treated as a provider fault rather than a caller error, so a chain moves on
    to a backend with a larger output budget instead of giving up.
    """

    def __init__(self, completion_tokens: int | None = None):
        self.completion_tokens = completion_tokens
        super().__init__("LLM output truncated at the provider's token limit")
