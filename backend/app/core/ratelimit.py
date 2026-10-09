"""
FiniRAG in-process rate limiter.

A fixed-window counter keyed by client identifier, applied to the expensive
endpoints (document processing, question answering). This protects against
accidental loops and small-scale abuse on a single-instance deploy.

Scope and honesty: state lives in process memory, so it is per-instance and
resets on restart. A multi-instance deployment needs a shared store (Redis)
or an edge limiter; this is a guard rail, not a distributed quota system.
"""

from __future__ import annotations

import threading
import time

WINDOW_SECONDS = 60.0
# Cap tracked keys so a spray of client ids cannot grow memory without bound.
_MAX_KEYS = 10_000


class RateLimiter:
    def __init__(self, per_minute: int, window_seconds: float = WINDOW_SECONDS):
        self.per_minute = per_minute
        self.window = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds).

        A zero `per_minute` disables limiting entirely (used by tests).
        """
        if self.per_minute <= 0:
            return True, 0
        now = time.monotonic() if now is None else now
        cutoff = now - self.window
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= _MAX_KEYS:
                    self._prune_stale(cutoff)
                hits = self._hits.setdefault(key, [])
            if hits and hits[0] < cutoff:
                hits[:] = [t for t in hits if t >= cutoff]
            if len(hits) >= self.per_minute:
                retry = int(self.window - (now - hits[0])) + 1
                return False, max(0, retry)
            hits.append(now)
            return True, 0

    def _prune_stale(self, cutoff: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or v[-1] < cutoff]:
            self._hits.pop(key, None)
