"""FeatureBuilder — transforms collector output into FeatureSets for agents.

Maps raw collector data into standardized feature dictionaries consumed by agents.
"""

from __future__ import annotations

from collections import defaultdict

from market_signal_engine.agents.base import FeatureSet
from market_signal_engine.collectors.base import CollectorResult


class FeatureBuilder:
    """Builds FeatureSet dictionaries from multiple collector results."""

    def build(self, results: list[CollectorResult], symbol: str) -> FeatureSet:
        """Merge all collector results into a single FeatureSet."""
        features: dict[str, float] = {}
        metadata: dict[str, object] = {}

        for result in results:
            if not result.is_ok or not result.data:
                continue

            source = result.source.lower()

            if source == "yfinance":
                self._extract_yfinance(result.data, features, metadata)
            elif source == "dexscreener":
                self._extract_dexscreener(result.data, features, metadata)
            elif source == "defillama":
                self._extract_defillama(result.data, features, metadata)
            elif source == "binance":
                self._extract_binance(result.data, features, metadata)
            elif source == "blockchain.com":
                self._extract_blockchain(result.data, features, metadata)
            elif source == "finnhub":
                self._extract_finnhub(result.data, features, metadata)
            elif source == "whale_alert":
                self._extract_whale_alert(result.data, features, metadata)

        # Inject price/volume history into metadata for AnalysisContext
        for result in results:
            if result.data:
                if "prices_6mo" in result.data:
                    metadata["prices"] = result.data["prices_6mo"]
                if "volumes_6mo" in result.data:
                    metadata["volumes"] = result.data["volumes_6mo"]

        return FeatureSet(features=features, metadata=metadata)

    # ── Per-collector extractors ────────────────────────────────────────

    def _extract_yfinance(self, data: dict, features: dict, meta: dict) -> None:
        _set_if(features, "price", data.get("price"))
        _set_if(features, "volume", data.get("volume"))
        _set_if(features, "market_cap", data.get("market_cap"))
        _set_if(features, "pe_ratio", data.get("pe_ratio"))
        _set_if(features, "forward_pe", data.get("forward_pe"))
        _set_if(features, "eps", data.get("eps"))
        _set_if(features, "dividend_yield", float(data.get("dividend_yield", 0) or 0) * 100)
        _set_if(features, "beta", data.get("beta"))
        _set_if(features, "52w_high", data.get("52w_high"))
        _set_if(features, "52w_low", data.get("52w_low"))
        _set_if(features, "peg_ratio", data.get("peg_ratio"))
        _set_if(features, "debt_to_equity", data.get("debt_to_equity"))
        _set_if(features, "free_cashflow", data.get("free_cashflow"))
        _set_if(features, "sector", 0.0)  # Non-numeric, stored separately
        if data.get("sector"):
            features["sector_str"] = float(hash(data["sector"]) % 100)
        if "prices_6mo" in data:
            meta["prices"] = data["prices_6mo"]
        if "volumes_6mo" in data:
            meta["volumes"] = data["volumes_6mo"]

    def _extract_dexscreener(self, data: dict, features: dict, meta: dict) -> None:
        _set_if(features, "dex_volume_24h", data.get("volume_24h"))
        _set_if(features, "dex_liquidity", data.get("liquidity_usd"))
        _set_if(features, "dex_buys_24h", data.get("txns_24h_buys"))
        _set_if(features, "dex_sells_24h", data.get("txns_24h_sells"))
        _set_if(features, "dex_volume_change", data.get("price_change_24h"))
        if data.get("txns_24h_buys", 0) + data.get("txns_24h_sells", 0) > 0:
            buy_ratio = data["txns_24h_buys"] / (data["txns_24h_buys"] + data["txns_24h_sells"])
            features["buy_sell_ratio"] = buy_ratio

    def _extract_defillama(self, data: dict, features: dict, meta: dict) -> None:
        _set_if(features, "tvl", data.get("tvl"))
        _set_if(features, "tvl_change_1d", data.get("change_1d"))
        _set_if(features, "tvl_change_7d", data.get("change_7d"))
        _set_if(features, "total_circulating", data.get("total_circulating"))
        if data.get("tvl_history"):
            tvl_hist = data["tvl_history"]
            if len(tvl_hist) > 1:
                tvl_vals = [t.get("totalLiquidityUSD", 0) for t in tvl_hist if isinstance(t, dict)]
                if len(tvl_vals) > 1 and tvl_vals[0] > 0:
                    features["tvl_change_7d"] = (tvl_vals[-1] / tvl_vals[0] - 1) * 100

    def _extract_binance(self, data: dict, features: dict, meta: dict) -> None:
        _set_if(features, "funding_rate", data.get("funding_rate"))
        _set_if(features, "open_interest", data.get("open_interest"))
        _set_if(features, "bid_ask_ratio", data.get("bid_ask_ratio"))

    def _extract_blockchain(self, data: dict, features: dict, meta: dict) -> None:
        stats = data.get("stats", {})
        _set_if(features, "hash_rate", stats.get("hash_rate"))
        _set_if(features, "tx_count", stats.get("n_transactions"))
        _set_if(features, "active_addresses", self._extract_addr_count(data))
        mempool = data.get("mempool", {})
        _set_if(features, "avg_fee", mempool.get("fastest_fee"))

    def _extract_finnhub(self, data: dict, features: dict, meta: dict) -> None:
        recs = data.get("recommendations", [])
        if recs and isinstance(recs, list) and len(recs) > 0:
            latest = recs[0]
            strong_buy = latest.get("strongBuy", 0)
            buy = latest.get("buy", 0)
            hold = latest.get("hold", 0)
            sell = latest.get("sell", 0)
            strong_sell = latest.get("strongSell", 0)
            total = strong_buy + buy + hold + sell + strong_sell
            if total > 0:
                score = (strong_buy * 1 + buy * 0.5 + hold * 0 - sell * 0.5 - strong_sell * 1) / total
                features["analyst_score"] = score

        earnings = data.get("earnings", [])
        if earnings and isinstance(earnings, list) and len(earnings) > 0:
            surprises = [e.get("surprise", 0) for e in earnings[:4] if isinstance(e, dict)]
            if surprises:
                features["earnings_surprise"] = sum(surprises) / len(surprises)

    def _extract_whale_alert(self, data: dict, features: dict, meta: dict) -> None:
        _set_if(features, "large_tx_count", data.get("tx_count"))
        _set_if(features, "large_tx_volume", data.get("total_volume"))

    @staticmethod
    def _extract_addr_count(data: dict) -> float:
        chart = data.get("n-unique-addresses", [])
        if chart and isinstance(chart, list) and len(chart) > 0:
            last = chart[-1]
            if isinstance(last, dict):
                return float(last.get("y", 0))
        return 0.0


def _set_if(features: dict, key: str, value: object) -> None:
    """Set feature if value is non-None and numeric."""
    if value is None:
        return
    try:
        features[key] = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        pass
