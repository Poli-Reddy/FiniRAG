"""
FiniRAG resilience primitives.

Shared by every outbound provider call (embeddings, answers, LLM router):

  * retry with exponential backoff + full jitter, honouring the provider's
    own Retry-After hint when present. Only 429 / 5xx / timeouts are retried;
    other 4xx (bad request, auth) fail fast — retrying them is pointless and
    just burns quota.
  * a per-provider circuit breaker that opens after N consecutive failures
    and half-opens after a cooldown, so a dead provider is skipped instead of
    making every request pay the full timeout budget.

Both are deliberately tiny and dependency-free. State is in-process, which
matches the single-instance deployment; see get_breaker().
"""

from __future__ import annotations

import asyncio
import logging
import random
import time

logger = logging.getLogger("finirag")

# Provider exception class names that are always retryable. Matched by name so
# the module does not depend on any single SDK's exception hierarchy.
_RETRYABLE_NAMES = {
    "ResourceExhausted", "TooManyRequests", "RateLimitError", "RateLimited",
    "InternalServerError", "ServiceUnavailable", "DeadlineExceeded",
}

# Names that look retryable but are not, to avoid accidental retries.
_NON_RETRYABLE_NAMES = {"Unauthenticated", "PermissionDenied", "InvalidArgument",
                        "NotFound", "BadRequest", "FailedPrecondition"}


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    v = getattr(resp, "status_code", None)
    if isinstance(v, int):
        return v
    return None


def is_retryable(exc: Exception) -> bool:
    """True for transient failures: rate limits, 5xx, timeouts, dropped conns.

    Never retries authentication, validation, or not-found errors.
    """
    name = type(exc).__name__
    if name in _NON_RETRYABLE_NAMES:
        return False
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    if name in _RETRYABLE_NAMES:
        return True
    code = _status_code(exc)
    if code is not None:
        return code == 429 or 500 <= code < 600
    text = str(exc).lower()
    return any(m in text for m in (
        "rate limit", "rate_limit", "429", "too many requests", "temporarily unavailable",
        "timeout", "timed out", "deadline", "502", "503", "500", "overloaded", "capacity",
        "connection reset", "connection aborted",
    ))


def is_rate_limit(exc: Exception) -> bool:
    """True when the failure is a rate limit / quota rejection specifically.

    Used to decide whether an *API key* is at fault (retire it) versus a
    generic transient failure (retry the same key).
    """
    name = type(exc).__name__
    if name in ("ResourceExhausted", "TooManyRequests", "RateLimitError", "RateLimited"):
        return True
    if _status_code(exc) == 429:
        return True
    text = str(exc).lower()
    return any(m in text for m in ("rate limit", "quota", "429", "too many requests"))


def retry_after_seconds(exc: Exception) -> float | None:
    """The delay the provider explicitly asked for, if it gave one."""
    delay = getattr(exc, "retry_delay", None)
    secs = getattr(delay, "seconds", None)
    if isinstance(secs, (int, float)) and secs > 0:
        return float(secs)
    for attr in ("retry_after", "retry_after_seconds"):
        v = getattr(exc, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        raw = headers.get("retry-after") if hasattr(headers, "get") else None
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return None


async def call_with_retry(fn, *, retries: int | None = None,
                          base_delay: float | None = None,
                          max_delay: float | None = None,
                          jitter: float = 0.5):
    """Await fn(), retrying transient failures with backoff + full jitter.

    `fn` is a zero-arg coroutine factory so each attempt is a fresh call.
    """
    from backend.app.core.config import settings

    retries = settings.PROVIDER_MAX_RETRIES if retries is None else retries
    base_delay = settings.PROVIDER_BASE_DELAY_S if base_delay is None else base_delay
    max_delay = settings.PROVIDER_MAX_DELAY_S if max_delay is None else max_delay

    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            if attempt >= retries or not is_retryable(exc):
                raise
            hint = retry_after_seconds(exc)
            delay = hint if hint is not None else min(max_delay, base_delay * (2 ** attempt))
            delay = max(0.0, delay) + random.uniform(0, jitter)
            logger.info("Provider call failed, retrying", extra={
                "attempt": attempt + 1, "max_retries": retries,
                "delay_s": round(delay, 2), "error_type": type(exc).__name__})
            await asyncio.sleep(delay)
            attempt += 1


class CircuitBreaker:
    """Trip after `failure_threshold` consecutive failures; half-open after
    `cooldown_s`. In-process, matching the single-instance deployment."""

    def __init__(self, name: str, *, failure_threshold: int = 5,
                 cooldown_s: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at = 0.0
        self._open = False

    def allow(self) -> bool:
        if not self._open:
            return True
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            return True  # half-open trial
        return False

    def record_success(self) -> None:
        if self._open:
            logger.info("Circuit breaker closed", extra={"provider": self.name})
        self._failures = 0
        self._open = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold and not self._open:
            self._open = True
            self._opened_at = time.monotonic()
            logger.error("Circuit breaker opened", extra={
                "provider": self.name, "failures": self._failures,
                "cooldown_s": self.cooldown_s})

    @property
    def state(self) -> str:
        if not self._open:
            return "closed"
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            return "half_open"
        return "open"

    def snapshot(self) -> dict:
        return {"provider": self.name, "state": self.state, "failures": self._failures}


_BREAKERS: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    breaker = _BREAKERS.get(name)
    if breaker is None:
        from backend.app.core.config import settings

        breaker = CircuitBreaker(
            name,
            failure_threshold=settings.CIRCUIT_FAILURE_THRESHOLD,
            cooldown_s=settings.CIRCUIT_COOLDOWN_S,
        )
        _BREAKERS[name] = breaker
    return breaker


def breakers_snapshot() -> list[dict]:
    return [b.snapshot() for b in _BREAKERS.values()]


def reset_breakers() -> None:
    """Test helper: forget all breaker state."""
    _BREAKERS.clear()
