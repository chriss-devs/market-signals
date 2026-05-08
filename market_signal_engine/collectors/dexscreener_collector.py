"""DEX Screener collector — DEX pair prices, volume, liquidity, transactions.

Free API, no key required. Rate limit: ~300 req/min.
"""

from __future__ import annotations

import urllib.request
import json

from market_signal_engine.collectors.base import BaseCollector, CollectorResult


class DEXScreenerCollector(BaseCollector):
    source = "dexscreener"
    base_url = "https://api.dexscreener.com/latest/dex"
    rate_limit_sec = 0.3  # generous rate limit
    cache_ttl_sec = 60.0

    def _headers_impl(self) -> dict[str, str]:
        h = super()._headers()
        h["User-Agent"] = "MarketSignalEngine/0.2"
        return h

    def _fetch(self, symbol: str) -> CollectorResult:
        if symbol.startswith("0x") and len(symbol) >= 40:
            url = f"{self.base_url}/pairs/ethereum/{symbol}"
        else:
            encoded = urllib.request.quote(symbol)
            url = f"{self.base_url}/search?q={encoded}"

        req = urllib.request.Request(url, headers=self._headers_impl())
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode())

        pairs = raw.get("pairs", []) if isinstance(raw, dict) else []
        if not pairs and isinstance(raw, dict) and "pair" in raw:
            pairs = [raw["pair"]]

        if not pairs:
            return CollectorResult(
                source=self.source,
                symbol=symbol,
                data={"pairs_found": 0},
                raw=raw,
            )

        pair = pairs[0]
        data = {
            "pairs_found": len(pairs),
            "chain": pair.get("chainId"),
            "dex": pair.get("dexId"),
            "price_usd": float(pair.get("priceUsd", 0)),
            "price_native": float(pair.get("priceNative", 0)),
            "volume_24h": float(pair.get("volume", {}).get("h24", 0)),
            "volume_6h": float(pair.get("volume", {}).get("h6", 0)),
            "volume_1h": float(pair.get("volume", {}).get("h1", 0)),
            "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0)),
            "txns_24h_buys": int(pair.get("txns", {}).get("h24", {}).get("buys", 0)),
            "txns_24h_sells": int(pair.get("txns", {}).get("h24", {}).get("sells", 0)),
            "price_change_5m": float(pair.get("priceChange", {}).get("m5", 0)),
            "price_change_1h": float(pair.get("priceChange", {}).get("h1", 0)),
            "price_change_6h": float(pair.get("priceChange", {}).get("h6", 0)),
            "price_change_24h": float(pair.get("priceChange", {}).get("h24", 0)),
            "fdv": float(pair.get("fdv", 0)),
            "pair_created_at": pair.get("pairCreatedAt"),
            "url": pair.get("url"),
        }

        # If multiple pairs, include top 5
        if len(pairs) > 1:
            data["top_pairs"] = [
                {
                    "dex": p.get("dexId"),
                    "price": float(p.get("priceUsd", 0)),
                    "volume": float(p.get("volume", {}).get("h24", 0)),
                    "liquidity": float(p.get("liquidity", {}).get("usd", 0)),
                    "chain": p.get("chainId"),
                }
                for p in pairs[:5]
            ]

        return CollectorResult(source=self.source, symbol=symbol, data=data, raw=raw)
