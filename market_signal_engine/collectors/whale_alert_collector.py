"""Whale Alert collector — large cryptocurrency transactions.

Free tier: Limited to recent transactions. API key available at whale-alert.io.
Works with or without API key (free endpoint has limited data).
"""

from __future__ import annotations

import json
import os
import urllib.request

from market_signal_engine.collectors.base import BaseCollector, CollectorResult


class WhaleAlertCollector(BaseCollector):
    source = "whale_alert"
    base_url = "https://api.whale-alert.io/v1"
    rate_limit_sec = 5.0  # very limited on free tier
    cache_ttl_sec = 300.0

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("WHALE_ALERT_KEY", "")
        super().__init__(api_key=key)

    def _fetch(self, symbol: str) -> CollectorResult:
        data: dict = {"transactions": [], "summary": {}}
        clean = symbol.upper()

        # If no API key, return a helpful error
        if not self.api_key:
            return CollectorResult(
                source=self.source, symbol=symbol,
                data=data,
                error="No API key. Get one at whale-alert.io or set WHALE_ALERT_KEY.",
            )

        try:
            raw = self._fetch_transactions(min_value=100000)
            txns = raw.get("transactions", [])

            # Filter for the requested symbol
            matching = [
                t for t in txns
                if clean in (t.get("symbol", "").upper()) or clean in (t.get("amount_usd", 0))
            ]

            # Summary statistics
            total_value = sum(
                float(t.get("amount_usd", 0)) for t in matching if t.get("amount_usd")
            )
            tx_types: dict[str, int] = {}
            for t in matching:
                ttype = t.get("transaction_type", "unknown")
                tx_types[ttype] = tx_types.get(ttype, 0) + 1

            data = {
                "transactions": matching[:20],
                "summary": {
                    "total_count": len(matching),
                    "total_value_usd": round(total_value, 2),
                    "by_type": tx_types,
                    "largest_txn_usd": max(
                        (float(t.get("amount_usd", 0)) for t in matching), default=0
                    ),
                },
            }

            return CollectorResult(source=self.source, symbol=symbol, data=data, raw=raw)

        except Exception:
            # Fallback: describe what data would be available
            return CollectorResult(
                source=self.source, symbol=symbol,
                data=data,
                error="Whale Alert API unavailable or rate limited. Try again later.",
            )

    def _fetch_transactions(self, min_value: int = 100000, limit: int = 100) -> dict:
        url = (
            f"{self.base_url}/transactions"
            f"?api_key={self.api_key}"
            f"&min_value={min_value}"
            f"&limit={limit}"
        )
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())  # type: ignore[no-any-return]

    def fetch_whales_by_symbol(self, symbol: str, min_value: int = 500000) -> CollectorResult:
        """Convenience method: get only large whale transactions for a symbol."""
        return self.collect(f"{symbol}:min{min_value}", force=True)
