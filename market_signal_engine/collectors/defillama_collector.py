"""DeFi Llama collector — TVL, protocol data, chain metrics, stablecoins, yields.

Free API, no key required.
Endpoints:
  - /v2/chains — TVL per chain
  - /protocol/{slug} — protocol detail
  - /v2/protocol/{slug} — protocol with TVL history
  - /stablecoins — stablecoin market caps
  - /pools/yields — yield farming APY data (may be large)
"""

from __future__ import annotations

import json
import urllib.request

from market_signal_engine.collectors.base import BaseCollector, CollectorResult


class DefiLlamaCollector(BaseCollector):
    source = "defillama"
    base_url = "https://api.llama.fi"
    rate_limit_sec = 0.5
    cache_ttl_sec = 300.0  # TVL doesn't change second-to-second

    def _fetch(self, symbol: str) -> CollectorResult:
        # Multipurpose: "chains" → all chains, "stablecoins" → stablecoins,
        # "protocol:<slug>" → protocol detail, otherwise treat as chain name
        if symbol == "chains":
            return self._fetch_chains()
        elif symbol == "stablecoins":
            return self._fetch_stablecoins()
        elif symbol.startswith("protocol:"):
            return self._fetch_protocol(symbol.split(":", 1)[1])
        else:
            return self._fetch_chain(symbol)

    def _fetch_chains(self) -> CollectorResult:
        url = f"{self.base_url}/v2/chains"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode())

        chains = [
            {
                "name": c.get("name"),
                "tvl": c.get("tvl", 0),
                "token_symbol": c.get("tokenSymbol"),
                "chain_id": c.get("chainId"),
                "protocols": c.get("protocols", 0),
            }
            for c in raw
            if isinstance(c, dict)
        ]

        return CollectorResult(
            source=self.source,
            symbol="chains",
            data={"count": len(chains), "chains": chains[:50]},
            raw=raw,
        )

    def _fetch_stablecoins(self) -> CollectorResult:
        url = f"{self.base_url}/stablecoins"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode())

        pegged = raw.get("peggedAssets", [])
        total_circ = sum(
            float(p.get("circulating", {}).get("peggedUSD", 0)) for p in pegged if isinstance(p, dict)
        )

        stablecoins = [
            {"name": p.get("name"), "symbol": p.get("symbol"),
             "circulating_usd": float(p.get("circulating", {}).get("peggedUSD", 0))}
            for p in pegged[:20] if isinstance(p, dict)
        ]

        return CollectorResult(
            source=self.source,
            symbol="stablecoins",
            data={"total_circulating": total_circ, "count": len(pegged), "stablecoins": stablecoins},
            raw=raw,
        )

    def _fetch_protocol(self, slug: str) -> CollectorResult:
        url = f"{self.base_url}/protocol/{slug}"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode())

        data = {
            "name": raw.get("name"),
            "slug": slug,
            "tvl": raw.get("tvl", 0),
            "chain": raw.get("chain"),
            "category": raw.get("category"),
            "url": raw.get("url"),
            "audits": raw.get("audits", 0),
            "chains": raw.get("chains", []),
        }

        # Try to get TVL history for trend
        try:
            tvl_url = f"{self.base_url}/v2/protocol/{slug}"
            tvl_req = urllib.request.Request(tvl_url, headers=self._headers())
            with urllib.request.urlopen(tvl_req, timeout=15) as tvl_resp:
                tvl_raw = json.loads(tvl_resp.read().decode())
                tvl_series = tvl_raw.get("tvl", [])
                if tvl_series:
                    data["tvl_history"] = tvl_series[-30:]  # last 30 data points
        except Exception:
            pass

        return CollectorResult(source=self.source, symbol=f"protocol:{slug}", data=data, raw=raw)

    def _fetch_chain(self, chain_name: str) -> CollectorResult:
        url = f"{self.base_url}/v2/chains"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode())

        chain_data = None
        for c in raw:
            if isinstance(c, dict) and c.get("name", "").lower() == chain_name.lower():
                chain_data = c
                break

        if not chain_data:
            return CollectorResult(
                source=self.source,
                symbol=chain_name,
                data={"found": False},
                raw=raw,
            )

        data = {
            "name": chain_data.get("name"),
            "tvl": chain_data.get("tvl", 0),
            "token_symbol": chain_data.get("tokenSymbol"),
            "chain_id": chain_data.get("chainId"),
            "protocols": chain_data.get("protocols", 0),
            "change_1d": chain_data.get("change_1d"),
            "change_7d": chain_data.get("change_7d"),
            "change_1m": chain_data.get("change_1m"),
        }

        return CollectorResult(source=self.source, symbol=chain_name, data=data, raw=chain_data)
