"""
API access control — optional shared-secret gate.

When `API_KEYS` is empty (the default), every route stays open so local
development and the existing test suite work unchanged. Set `API_KEYS`
before exposing the service publicly: the API serves full extracted
financial records (vendor/customer identity, tax IDs, amounts) and rate
limiting alone is not authorization.

Scope, stated plainly: this gates *who may call* (single shared secret,
constant-time compared). It does NOT implement per-user or per-tenant row
ownership — every authorized caller can still read every document. That
remains an open item (see docs), and this module must not be mistaken for it.
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

from backend.app.core.config import settings
from backend.app.core.key_pool import parse_keys


def _configured_keys() -> list[str]:
    return parse_keys(settings.API_KEYS)


def auth_enabled() -> bool:
    """True when at least one API key is configured."""
    return bool(_configured_keys())


async def require_api_key(request: Request) -> None:
    """FastAPI dependency: reject unauthenticated callers when enabled."""
    keys = _configured_keys()
    if not keys:
        return
    presented = request.headers.get("authorization", "")
    if presented.lower().startswith("bearer "):
        presented = presented[7:].strip()
    else:
        presented = request.headers.get("x-api-key", "").strip()
    for key in keys:
        if presented and hmac.compare_digest(presented, key):
            return
    raise HTTPException(
        status_code=401,
        detail={"code": "UNAUTHORIZED",
                "message": "Valid API credentials are required."},
        headers={"WWW-Authenticate": "Bearer"},
    )
