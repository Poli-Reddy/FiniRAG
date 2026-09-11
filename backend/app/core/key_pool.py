"""
FiniRAG API key pool.

One provider, several keys, used one at a time. The problem this solves is
specific: the free tiers this project runs on are capped per *day*, and the
dataset is 50 documents. A single Gemini key runs out roughly 20 documents in,
which is exactly what happened on 2026-09-10 — 43 readable documents were
recorded as failures because the key died six documents into the run.

A pool turns that from a wall into a speed bump. When a key is refused, the
next one takes over and the document is retried immediately, in the same call,
so the pipeline above never sees the rotation.

Two kinds of refusal, handled differently
─────────────────────────────────────────
Collapsing them is the mistake that makes a pool useless:

  * **Per-minute** throttling clears on its own. The key is put on a cooldown
    and comes back. Retiring it permanently would throw away most of the
    quota the user is paying nothing for.
  * **Per-day** quota does not clear on any timescale this run cares about.
    The key is retired for the life of the process. Leaving it in rotation
    would make every future document pay a wasted round trip to a key that
    is certain to refuse.

When every key is cooling, the caller is told how long until the soonest one
returns — that is a wait. When every key is dead, the caller is told to stop.
Those are different answers and the runner acts on them differently.

No key material is ever logged. Keys are identified by their index in the
configured list, which is enough to tell the user *which* of their keys died
and leaks nothing if a log is shared.

Concurrency: rotation is synchronous and contains no `await`, so it is atomic
with respect to the event loop and needs no lock. Two concurrent requests can
still both be handed the same key and both discover it is spent, which costs
one redundant round trip and nothing else — the pool stays the single source
of truth for what is live.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("finirag")

# How long a key sits out after a per-minute refusal that carried no explicit
# retry hint. Provider per-minute windows are 60 s; the extra 5 s keeps a
# key from coming back a hair early and burning a request to find out.
DEFAULT_COOLDOWN_SECONDS = 65.0


class AllKeysExhausted(Exception):
    """
    Every key for this provider is spent for the rest of the process.

    Distinct from AllKeysCooling because the answer is different: this one
    means stop, not wait.
    """

    def __init__(self, provider: str, total: int):
        self.provider = provider
        self.total = total
        super().__init__(f"all {total} {provider} key(s) exhausted")


class AllKeysCooling(Exception):
    """
    Every live key is inside a per-minute cooldown; the pool recovers shortly.

    `seconds_until_next` is what the caller should wait before asking again.
    """

    def __init__(self, provider: str, seconds_until_next: float):
        self.provider = provider
        self.seconds_until_next = seconds_until_next
        super().__init__(
            f"all {provider} keys cooling, {seconds_until_next:.0f}s until next"
        )


def is_daily_quota(message: str) -> bool:
    """
    True when a refusal reads as a per-*day* (or monthly/credit) limit
    rather than a per-minute throttle.

    Shared by providers so daily deaths retire permanently while minute
    throttles merely cool down. Inspects text only — never logged.
    """
    text = (message or "").lower()
    return any(
        marker in text
        for marker in ("per day", "perday", "daily", "per-day", "monthly", "credit")
    )


def parse_keys(*values: str) -> list[str]:
    """
    Split configured key strings into an ordered, de-duplicated list.

    Accepts comma- or newline-separated values so a `.env` line can hold a
    list, and takes several values so the plural setting and the legacy
    singular one can be merged without the caller caring which was set.

    De-duplication is not cosmetic: a key pasted twice would otherwise be
    retired twice, and the second retirement would be blamed on a key that
    was still good.
    """
    seen: dict[str, None] = {}
    for value in values:
        if not value:
            continue
        for chunk in value.replace("\n", ",").split(","):
            key = chunk.strip()
            if key:
                seen.setdefault(key, None)
    return list(seen)


@dataclass
class _KeyState:
    """Per-key bookkeeping. Holds no key material."""
    dead: bool = False
    cooling_until: float = 0.0
    successes: int = 0
    refusals: int = 0

    def available(self, now: float) -> bool:
        return not self.dead and now >= self.cooling_until


@dataclass
class KeyPool:
    """
    An ordered set of interchangeable API keys for one provider.

    Keys are handed out one at a time, lowest usable index first, so a run
    with quota to spare always lands on the same key and the user's spend
    stays predictable rather than smeared across every key they own.
    """

    keys: list[str]
    provider: str
    _state: list[_KeyState] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._state = [_KeyState() for _ in self.keys]

    def __len__(self) -> int:
        return len(self.keys)

    def __bool__(self) -> bool:
        """A pool with no keys is falsey, so `if pool:` reads naturally."""
        return bool(self.keys)

    @property
    def live_count(self) -> int:
        """Keys not permanently retired, including ones currently cooling."""
        return sum(1 for s in self._state if not s.dead)

    def acquire(self) -> tuple[int, str]:
        """
        The next usable key, as (index, key).

        Raises AllKeysCooling when the pool will recover shortly, and
        AllKeysExhausted when it will not.
        """
        if not self.keys:
            raise AllKeysExhausted(self.provider, 0)

        now = time.monotonic()

        for index, state in enumerate(self._state):
            if state.available(now):
                return index, self.keys[index]

        # Nothing available. Which of the two answers applies depends on
        # whether anything is merely cooling.
        waits = [
            s.cooling_until - now
            for s in self._state
            if not s.dead and s.cooling_until > now
        ]
        if waits:
            raise AllKeysCooling(self.provider, min(waits))

        raise AllKeysExhausted(self.provider, len(self.keys))

    def report_success(self, index: int) -> None:
        """Record that a key answered, for the end-of-run summary."""
        self._state[index].successes += 1

    def retire(
        self,
        index: int,
        *,
        permanent: bool,
        cooldown_seconds: float | None = None,
    ) -> None:
        """
        Take a key out of rotation.

        `permanent=True` is for a per-day quota: the key is gone for the life
        of the process. Otherwise the key sits out `cooldown_seconds` and
        returns. A permanent retirement is logged at ERROR because it costs
        the user real capacity and they should see it; a cooldown is INFO
        because it is routine.
        """
        state = self._state[index]
        state.refusals += 1

        if permanent:
            state.dead = True
            logger.error("API key exhausted, retired for this run", extra={
                "provider": self.provider,
                "key_index": index,
                "keys_live": self.live_count,
                "keys_total": len(self.keys),
            })
            return

        wait = DEFAULT_COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds
        state.cooling_until = time.monotonic() + wait
        logger.info("API key throttled, cooling down", extra={
            "provider": self.provider,
            "key_index": index,
            "cooldown_seconds": round(wait, 1),
            "keys_live": self.live_count,
        })

    def status(self) -> dict:
        """
        A log- and report-safe snapshot. Contains no key material.

        Per-key entries are positional so the user can map a dead key back to
        the line they pasted it on.
        """
        now = time.monotonic()
        return {
            "provider": self.provider,
            "total": len(self.keys),
            "live": self.live_count,
            "available_now": sum(1 for s in self._state if s.available(now)),
            "keys": [
                {
                    "index": i,
                    "state": (
                        "dead" if s.dead
                        else "cooling" if now < s.cooling_until
                        else "live"
                    ),
                    "successes": s.successes,
                    "refusals": s.refusals,
                }
                for i, s in enumerate(self._state)
            ],
        }
