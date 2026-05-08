"""Finnhub collector — earnings surprises, analyst targets, insider trading, news.

Free tier: 60 API calls/min. Requires API key (set FINNHUB_KEY env var or pass).
"""

from __future__ import annotations

import json
import os
import urllib.request

from market_signal_engine.collectors.base import BaseCollector, CollectorResult


class FinnhubCollector(BaseCollector):
    source = "finnhub"
    base_url = "https://finnhub.io/api/v1"
    rate_limit_sec = 1.1  # 60/min on free tier
    cache_ttl_sec = 600.0

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("FINNHUB_KEY", "")
        super().__init__(api_key=key)

    def _fetch(self, symbol: str) -> CollectorResult:
        if not self.api_key:
            return CollectorResult(
                source=self.source, symbol=symbol,
                error="No API key. Set FINNHUB_KEY env var or pass api_key.",
            )

        clean = symbol.upper()
        data: dict = {}

        # Quote
        try:
            data["quote"] = self._get(f"/quote?symbol={clean}")
        except Exception:
            pass

        # Company profile
        try:
            data["profile"] = self._get(f"/stock/profile2?symbol={clean}")
        except Exception:
            pass

        # Analyst recommendations
        try:
            data["recommendations"] = self._get(f"/stock/recommendation?symbol={clean}")
        except Exception:
            pass

        # Earnings surprises
        try:
            data["earnings"] = self._get(f"/stock/earnings?symbol={clean}")
        except Exception:
            pass

        # Insider transactions
        try:
            data["insider"] = self._get(f"/stock/insider-transactions?symbol={clean}")
        except Exception:
            pass

        return CollectorResult(source=self.source, symbol=symbol, data=data)

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}&token={self.api_key}"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]
