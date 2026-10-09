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
    """Best-effort client identity: first X-Forwarded-For hop, else peer IP.

    Render sits behind a proxy, so the peer IP is the proxy; the forwarded
    header carries the real client.
    """
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
