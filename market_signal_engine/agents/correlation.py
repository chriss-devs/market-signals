"""Correlation Agent — analyzes asset correlations and regime changes.

Detects correlation breakdowns, flight-to-quality, and risk-on/off rotations
across crypto and equities baskets.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class CorrelationAgent(BaseAgent):
    name = "Correlation"
    agent_id = 16
    tier = 2
    category = "Cross-Market"
    data_sources = ["yfinance", "binance"]

    # Reference baskets for correlation analysis
    CRYPTO_BASKET = ["BTC", "ETH", "SOL"]
    EQUITY_BASKET = ["SPY", "AAPL", "NVDA"]
    HYBRID_BASKET = ["BTC", "SPY", "ETH"]

    def __init__(self) -> None:
        super().__init__()
        self._correlation_memory: dict[str, list[float]] = defaultdict(list)
        self._regime_history: list[dict] = []

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. Crypto-equity correlation
        crypto_equity_corr = features.get("crypto_equity_corr", 0.0)
        if crypto_equity_corr > 0.7:
            score += 0.06
            reasons.append(f"High crypto-equity correlation ({crypto_equity_corr:.2f}) — risk-on alignment")
        elif crypto_equity_corr < -0.3:
            score -= 0.04
            reasons.append(f"Negative crypto-equity correlation ({crypto_equity_corr:.2f}) — decoupling")
        elif -0.3 <= crypto_equity_corr <= 0.3:
            score += 0.03
            reasons.append(f"Low correlation ({crypto_equity_corr:.2f}) — diversification benefit")

        # 2. Intra-crypto correlation
        intra_crypto = features.get("intra_crypto_corr", 0.0)
        if intra_crypto > 0.85:
            score += 0.04
            reasons.append(f"Tight crypto basket correlation ({intra_crypto:.2f}) — sector move")
        elif intra_crypto < 0.5:
            score -= 0.05
            reasons.append(f"Crypto basket fragmentation ({intra_crypto:.2f}) — idiosyncratic")

        # 3. Correlation trend (is correlation rising or falling?)
        corr_trend = features.get("correlation_trend", 0.0)
        if corr_trend > 0.1:
            score += 0.03
            reasons.append(f"Correlation rising ({corr_trend:+.2f}) — macro-driven market")
        elif corr_trend < -0.1:
            score -= 0.03
            reasons.append(f"Correlation falling ({corr_trend:+.2f}) — dispersion increasing")

        # 4. BTC dominance effect
        btc_dominance = features.get("btc_dominance", 50.0)
        if sym != "BTC" and btc_dominance > 55:
            score -= 0.04
            reasons.append(f"BTC dominance high ({btc_dominance:.0f}%) — alts underperforming")
        elif sym == "BTC" and btc_dominance > 55:
            score += 0.04
            reasons.append(f"BTC dominance strong ({btc_dominance:.0f}%) — flight to safety")

        # 5. Volatility spillover
        vol_spillover = features.get("vol_spillover", 0.0)
        if vol_spillover > 0.5:
            score -= 0.06
            reasons.append(f"High vol spillover ({vol_spillover:.2f}) — contagion risk")
        elif vol_spillover < 0.15:
            score += 0.03
            reasons.append(f"Low vol spillover ({vol_spillover:.2f}) — calm regime")

        # 6. Correlation stability
        corr_stability = features.get("correlation_stability", 0.5)
        if corr_stability < 0.3:
            score -= 0.05
            reasons.append(f"Unstable correlations ({corr_stability:.2f}) — regime shift possible")
        elif corr_stability > 0.7:
            score += 0.02
            reasons.append(f"Stable correlation regime ({corr_stability:.2f})")

        # 7. Flight-to-quality detection
        flight_quality = features.get("flight_to_quality", 0.0)
        if flight_quality > 0.5:
            score -= 0.07
            reasons.append(f"Flight-to-quality detected ({flight_quality:.2f}) — risk aversion")
        elif flight_quality < -0.5:
            score += 0.05
            reasons.append(f"Risk-on rotation ({abs(flight_quality):.2f}) — risk appetite")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.04 else "bearish" if score < -0.04 else "neutral"
        confidence = min(0.75, 0.35 + abs(score) * 0.45)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "Correlation structure neutral",
            supporting_features=[
                f"crypto_equity_corr={crypto_equity_corr:.3f}",
                f"intra_crypto={intra_crypto:.3f}",
                f"corr_trend={corr_trend:.3f}",
                f"vol_spillover={vol_spillover:.3f}",
            ],
        )

    def compute_correlation(self, series_a: list[float], series_b: list[float]) -> float:
        """Pearson correlation between two equal-length series."""
        n = min(len(series_a), len(series_b))
        if n < 5:
            return 0.0
        a, b = series_a[-n:], series_b[-n:]

        mean_a = sum(a) / n
        mean_b = sum(b) / n

        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
        var_a = sum((x - mean_a) ** 2 for x in a)
        var_b = sum((x - mean_b) ** 2 for x in b)

        denom = math.sqrt(var_a * var_b)
        return cov / denom if denom > 0 else 0.0

    def compute_returns(self, prices: list[float]) -> list[float]:
        """Convert prices to log returns."""
        if len(prices) < 2:
            return []
        return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            sym = o.prediction.symbol
            conf = o.prediction.confidence
            self._correlation_memory.setdefault(sym, []).append(conf)
            if len(self._correlation_memory[sym]) > 200:
                self._correlation_memory[sym] = self._correlation_memory[sym][-200:]
