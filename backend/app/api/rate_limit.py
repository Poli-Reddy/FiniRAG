"""
Shared API guards.

`rate_limit` is a FastAPI dependency applied to the expensive endpoints
(document processing, question answering). It is not global middleware, so
cheap reads (health, listings) are never throttled, and tests can raise the
limit through settings without touching the app wiring.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from backend.app.core.config import settings
from backend.app.core.ratelimit import RateLimiter

_limiter = RateLimiter(settings.RATE_LIMIT_PER_MINUTE)


def client_key(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    Uses the first X-Forwarded-For hop only when TRUST_X_FORWARDED_FOR is on
    (the Render deployment, where the platform proxy appends the real client
    address). Otherwise the direct peer IP is used: a client-supplied first
    hop is spoofable whenever the proxy chain is unknown, and trusting it
    would let an attacker rotate identities past the limiter.
    """
    if settings.TRUST_X_FORWARDED_FOR:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request) -> None:
    # Pick up runtime overrides (tests raise the limit; deployments tune it).
    _limiter.per_minute = settings.RATE_LIMIT_PER_MINUTE
    allowed, retry_after = _limiter.check(client_key(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={"code": "RATE_LIMITED",
                    "message": "Too many requests. Please slow down and retry."},
            headers={"Retry-After": str(retry_after)},
        )
