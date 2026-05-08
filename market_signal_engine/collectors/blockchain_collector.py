"""Blockchain.com collector — on-chain metrics for Bitcoin.

Free API, no key required.
Endpoints:
  - /stats — hash rate, tx count, mempool, difficulty, block size, etc.
  - /charts/{metric}?timespan={}&format=json — historical chart data
"""

from __future__ import annotations

import json
import urllib.request

from market_signal_engine.collectors.base import BaseCollector, CollectorResult


class BlockchainCollector(BaseCollector):
    source = "blockchain.com"
    base_url = "https://api.blockchain.info"
    rate_limit_sec = 1.0  # ~10 sec between calls per docs, but generous in practice
    cache_ttl_sec = 120.0

    # Available metrics for chart data
    METRICS = [
        "hash-rate", "market-price", "n-transactions", "mempool-size",
        "mempool-count", "miners-revenue", "difficulty", "trade-volume",
        "estimated-transaction-volume-usd", "total-bitcoins",
        "n-unique-addresses", "avg-block-size",
    ]

    def _fetch(self, symbol: str) -> CollectorResult:
        # Primarily designed for BTC on-chain
        data: dict = {}

        # Stats overview
        try:
            data["stats"] = self._fetch_stats()
        except Exception:
            pass

        # Mempool
        try:
            data["mempool"] = self._fetch_mempool()
        except Exception:
            pass

        # Selected chart metrics (keep it light — 3 key metrics)
        for metric in ["hash-rate", "n-transactions", "n-unique-addresses"]:
            try:
                data[metric] = self._fetch_chart(metric, timespan="7days")
            except Exception:
                pass

        return CollectorResult(source=self.source, symbol=symbol, data=data)

    def _fetch_stats(self) -> dict:
        url = f"{self.base_url}/stats"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode())

        return {
            "hash_rate": raw.get("hash_rate", 0),
            "market_price_usd": raw.get("market_price_usd", 0),
            "total_btc_sent": raw.get("total_btc_sent", 0),
            "total_fees_btc": raw.get("total_fees_btc", 0),
            "n_transactions": raw.get("n_transactions", 0),
            "miners_revenue_usd": raw.get("miners_revenue_usd", 0),
            "difficulty": raw.get("difficulty", 0),
            "trade_volume_btc": raw.get("trade_volume_btc", 0),
            "trade_volume_usd": raw.get("trade_volume_usd", 0),
            "estimated_btc_sent": raw.get("estimated_btc_sent", 0),
        }

    def _fetch_mempool(self) -> dict:
        url = f"{self.base_url}/mempool/fees"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode())

        return {
            "fastest_fee": raw.get("fastestFee", 0),
            "half_hour_fee": raw.get("halfHourFee", 0),
            "hour_fee": raw.get("hourFee", 0),
            "economy_fee": raw.get("economyFee", 0),
            "minimum_fee": raw.get("minimumFee", 0),
        }

    def _fetch_chart(self, metric: str, timespan: str = "7days") -> list[dict]:
        url = f"{self.base_url}/charts/{metric}?timespan={timespan}&format=json"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode())

        values = raw.get("values", [])
        return [{"x": v.get("x"), "y": v.get("y")} for v in values[-48:]]  # last 48 data points
