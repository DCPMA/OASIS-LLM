"""Rate limiting for OpenRouter ``:free``-tier models.

OpenRouter caps the free tier at:
  • 20 requests per minute
  • 1000 requests per day (per account)

This module provides an async-safe gate that workers acquire BEFORE issuing
a free-tier request. The RPM limit is enforced via an in-memory sliding
window. The daily limit is enforced via a DuckDB row keyed by UTC date so
that the count survives restarts and is shared across processes.

Usage:
    from .rate_limit import OPENROUTER_FREE_LIMITER, is_openrouter_free_model
    if is_openrouter_free_model(cfg.provider, cfg.model):
        await OPENROUTER_FREE_LIMITER.acquire(con)

If the daily quota is exhausted, ``acquire`` raises ``DailyQuotaExceeded``.
The runner should treat this as a permanent error so the affected config is
auto-cancelled instead of looping forever.
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import date, datetime, timezone

import duckdb


RPM_LIMIT = 20
DAILY_LIMIT = 1000
WINDOW_S = 60.0


class DailyQuotaExceeded(Exception):
    """Raised when the OpenRouter free-tier daily cap (1000/day) is reached."""


def is_openrouter_free_model(provider: str, model: str) -> bool:
    """Return True for OpenRouter models with the ``:free`` suffix.

    Accepts both bare ids (``google/gemma-3-27b-it:free``) and litellm-prefixed
    ids (``openrouter/google/gemma-3-27b-it:free``).
    """
    return provider == "openrouter" and model.rstrip().endswith(":free")


class _OpenRouterFreeLimiter:
    """Single-process gate enforcing 20 rpm + 1000/day for OpenRouter :free."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._stamps: deque[float] = deque()
        # Cached daily count to skip a DB read on every request.
        self._cached_day: date | None = None
        self._cached_count: int = 0

    @staticmethod
    def _today() -> date:
        return datetime.now(timezone.utc).date()

    def _refresh_daily_count(self, con: duckdb.DuckDBPyConnection) -> int:
        today = self._today()
        if self._cached_day != today:
            row = con.execute(
                "SELECT count FROM or_free_daily WHERE day = ?", [today]
            ).fetchone()
            self._cached_count = int(row[0]) if row else 0
            self._cached_day = today
        return self._cached_count

    def _bump_daily_count(self, con: duckdb.DuckDBPyConnection) -> int:
        today = self._today()
        # Upsert: insert-or-add. DuckDB ON CONFLICT supports DO UPDATE.
        con.execute(
            """
            INSERT INTO or_free_daily (day, count) VALUES (?, 1)
            ON CONFLICT (day) DO UPDATE SET count = or_free_daily.count + 1
            """,
            [today],
        )
        # Re-read to keep the in-memory cache honest across processes.
        row = con.execute(
            "SELECT count FROM or_free_daily WHERE day = ?", [today]
        ).fetchone()
        self._cached_count = int(row[0]) if row else 0
        self._cached_day = today
        return self._cached_count

    def daily_remaining(self, con: duckdb.DuckDBPyConnection) -> int:
        """Return how many free requests are still available today."""
        return max(0, DAILY_LIMIT - self._refresh_daily_count(con))

    async def acquire(self, con: duckdb.DuckDBPyConnection) -> None:
        """Block until a free-tier slot is available; record one usage.

        Raises ``DailyQuotaExceeded`` if the 1000/day cap is reached. The
        runner treats this as a permanent error so the config is cancelled.
        """
        async with self._lock:
            # Daily cap (cheap path: cached count first).
            count = self._refresh_daily_count(con)
            if count >= DAILY_LIMIT:
                raise DailyQuotaExceeded(
                    f"OpenRouter :free daily quota exhausted ({count}/{DAILY_LIMIT})"
                )

            # RPM via 60s sliding window.
            loop = asyncio.get_event_loop()
            now = loop.time()
            cutoff = now - WINDOW_S
            while self._stamps and self._stamps[0] < cutoff:
                self._stamps.popleft()
            if len(self._stamps) >= RPM_LIMIT:
                wait = self._stamps[0] + WINDOW_S - now
                if wait > 0:
                    await asyncio.sleep(wait)
                # Drain expired stamps after sleep.
                now2 = loop.time()
                cutoff2 = now2 - WINDOW_S
                while self._stamps and self._stamps[0] < cutoff2:
                    self._stamps.popleft()

            # Count this request.
            self._stamps.append(loop.time())
            self._bump_daily_count(con)


# Module-level singleton — workers share the same in-memory window.
OPENROUTER_FREE_LIMITER = _OpenRouterFreeLimiter()
