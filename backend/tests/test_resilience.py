"""Reliability tests: retry/backoff, circuit breaker, rate limiter, API guards."""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.app.core.ratelimit import RateLimiter
from backend.app.core.resilience import (
    CircuitBreaker, call_with_retry, get_breaker, is_rate_limit, is_retryable,
    reset_breakers, retry_after_seconds,
)


# ── retry classification ─────────────────────────────────────────────────

class _429(Exception):
    status_code = 429


class _500(Exception):
    status_code = 500


class _400(Exception):
    status_code = 400


class ResourceExhausted(Exception):
    pass


def test_retryable_classification():
    assert is_retryable(_429())
    assert is_retryable(_500())
    assert is_retryable(asyncio.TimeoutError())
    assert is_retryable(ResourceExhausted("quota"))
    assert not is_retryable(_400())          # validation: never retried
    assert not is_retryable(ValueError("bad"))


def test_rate_limit_classification():
    assert is_rate_limit(_429())
    assert is_rate_limit(ResourceExhausted("Resource exhausted"))
    assert not is_rate_limit(_500())


def test_retry_after_hint():
    class _WithHint(Exception):
        retry_delay = type("D", (), {"seconds": 7})()

    assert retry_after_seconds(_WithHint()) == 7.0
    assert retry_after_seconds(_400()) is None


@pytest.mark.asyncio
async def test_call_with_retry_succeeds_after_transient(monkeypatch):
    monkeypatch.setattr("backend.app.core.config.settings.PROVIDER_BASE_DELAY_S", 0.0)
    monkeypatch.setattr("backend.app.core.config.settings.PROVIDER_MAX_DELAY_S", 0.0)
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _500("server error")
        return "ok"

    result = await call_with_retry(fn, retries=3, base_delay=0, max_delay=0, jitter=0)
    assert result == "ok" and attempts["n"] == 3


@pytest.mark.asyncio
async def test_call_with_retry_does_not_retry_permanent(monkeypatch):
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        raise _400("bad request")

    with pytest.raises(_400):
        await call_with_retry(fn, retries=3, base_delay=0, max_delay=0, jitter=0)
    assert attempts["n"] == 1  # 4xx (non-429) fails fast


@pytest.mark.asyncio
async def test_call_with_retry_exhausts(monkeypatch):
    attempts = {"n": 0}

    async def fn():
        attempts["n"] += 1
        raise _429("rate limited")

    with pytest.raises(_429):
        await call_with_retry(fn, retries=2, base_delay=0, max_delay=0, jitter=0)
    assert attempts["n"] == 3  # initial + 2 retries


# ── circuit breaker ──────────────────────────────────────────────────────

def test_breaker_opens_and_half_opens(monkeypatch):
    breaker = CircuitBreaker("t", failure_threshold=3, cooldown_s=0.05)
    assert breaker.state == "closed" and breaker.allow()
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow()
    time.sleep(0.06)
    assert breaker.state == "half_open"
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed"


def test_breaker_success_resets_failures():
    breaker = CircuitBreaker("t2", failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == "closed"  # streak was reset


def test_get_breaker_is_singleton_and_resettable():
    reset_breakers()
    a = get_breaker("unit-test-provider")
    b = get_breaker("unit-test-provider")
    assert a is b
    reset_breakers()
    assert get_breaker("unit-test-provider") is not a


# ── rate limiter ─────────────────────────────────────────────────────────

def test_rate_limiter_blocks_within_window():
    rl = RateLimiter(per_minute=3, window_seconds=60)
    for _ in range(3):
        allowed, _ = rl.check("client", now=100.0)
        assert allowed
    allowed, retry = rl.check("client", now=100.0)
    assert not allowed and retry >= 1


def test_rate_limiter_recovers_after_window():
    rl = RateLimiter(per_minute=2, window_seconds=10)
    assert rl.check("c", now=0.0)[0]
    assert rl.check("c", now=0.0)[0]
    assert not rl.check("c", now=0.0)[0]
    assert rl.check("c", now=11.0)[0]  # window rolled over


def test_rate_limiter_isolates_clients():
    rl = RateLimiter(per_minute=1, window_seconds=60)
    assert rl.check("a", now=0.0)[0]
    assert rl.check("b", now=0.0)[0]  # different key, own budget


def test_rate_limiter_disabled_when_zero():
    rl = RateLimiter(per_minute=0)
    for _ in range(100):
        assert rl.check("c", now=0.0)[0]
