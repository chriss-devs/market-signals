from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CollectorResult:
    """Normalized output from any collector."""

    source: str
    symbol: str
    data: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str | None = None
    cached: bool = False

    @property
    def is_ok(self) -> bool:
        return self.error is None


class BaseCollector(ABC):
    """Abstract base for all data collectors.

    Provides rate limiting, TTL caching, and error resilience.
    Subclasses implement _fetch() — the base handles the rest.
    """

    source: str = "base"
    base_url: str = ""
    rate_limit_sec: float = 1.0  # min seconds between requests
    cache_ttl_sec: float = 30.0  # how long cached results are valid

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._last_request: float = 0.0
        self._cache: dict[str, tuple[float, CollectorResult]] = {}

    # ── Public API ───────────────────────────────────────────────────────

    def collect(self, symbol: str, force: bool = False) -> CollectorResult:
        """Fetch data for a symbol, with caching and rate limiting."""
        cache_key = f"{self.source}:{symbol}"

        if not force and cache_key in self._cache:
            ts, result = self._cache[cache_key]
            if time.monotonic() - ts < self.cache_ttl_sec:
                result.cached = True
                return result

        self._rate_limit()
        try:
            result = self._fetch(symbol)
            result.source = self.source
            self._cache[cache_key] = (time.monotonic(), result)
            return result
        except Exception as exc:
            result = CollectorResult(
                source=self.source,
                symbol=symbol,
                error=f"{type(exc).__name__}: {exc}",
            )
            self._cache[cache_key] = (time.monotonic(), result)
            return result

    def collect_batch(self, symbols: list[str], force: bool = False) -> list[CollectorResult]:
        """Fetch multiple symbols sequentially with rate limiting."""
        return [self.collect(s, force) for s in symbols]

    # ── Subclass contract ────────────────────────────────────────────────

    @abstractmethod
    def _fetch(self, symbol: str) -> CollectorResult:
        """Implement the actual API call. Raise on failure."""
        ...

    # ── Internal helpers ─────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)
        self._last_request = time.monotonic()

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def clear_cache(self) -> None:
        self._cache.clear()
