"""Momentum Agent — time-series momentum (1d/7d/30d/90d), cross-sectional, crash detection.

Self-improvement: optimizes lookback periods per market.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import adx, last_valid, pct_change


class MomentumAgent(BaseAgent):
    name = "Momentum"
    agent_id = 9
    tier = 1
    category = "Momentum"
    data_sources = ["price_history"]

    def __init__(self) -> None:
        super().__init__()
        # Per-asset lookback weights — tuned over time
        self._lookback_weights: dict[str, dict[int, float]] = defaultdict(
            lambda: {1: 0.15, 7: 0.25, 30: 0.35, 90: 0.25}
        )
        self._crash_zscore_threshold: dict[str, float] = defaultdict(lambda: -2.0)

    def analyze(self, context: AnalysisContext) -> Prediction:
        prices = context.price_history
        if len(prices) < 90:
            return Prediction(
                agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
                direction="neutral", confidence=0.3,
                reasoning="Insufficient price history (<90 points)",
            )

        sym = context.symbol
        weights = self._lookback_weights[sym]

        # Time-series momentum (multi-timeframe)
        mom_1d = pct_change(prices, 1)
        mom_7d = pct_change(prices, 7)
        mom_30d = pct_change(prices, 30)
        mom_90d = pct_change(prices, 90)

        # Weighted aggregate momentum
        weighted_mom = (
            weights[1] * mom_1d
            + weights[7] * mom_7d
            + weights[30] * mom_30d
            + weights[90] * mom_90d
        )

        # Momentum consistency: are all timeframes aligned?
        signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in [mom_1d, mom_7d, mom_30d, mom_90d]]
        aligned = len(set(signs)) == 1 and signs[0] != 0

        # Crash detection: z-score of 7d return vs 90d window
        ret_7d_series = []
        for i in range(90, len(prices)):
            ret_7d_series.append((prices[i] / prices[i - 7] - 1) * 100 if prices[i - 7] else 0)
        if len(ret_7d_series) > 5:
            mean_ret = sum(ret_7d_series) / len(ret_7d_series)
            std_ret = math.sqrt(sum((r - mean_ret)**2 for r in ret_7d_series) / len(ret_7d_series))
            z_score = (mom_7d - mean_ret) / std_ret if std_ret > 0 else 0
        else:
            z_score = 0

        crash_thresh = self._crash_zscore_threshold[sym]
        crash_risk = z_score < crash_thresh

        # ADX for trend strength
        highs = [p * 1.005 for p in prices]
        lows = [p * 0.995 for p in prices]
        adx_now = last_valid(adx(highs, lows, prices, 14))

        # ── Scoring ─────────────────────────────────────────────────────
        reasons: list[str] = []

        if crash_risk:
            direction = "bearish"
            confidence = min(0.85, 0.5 + abs(z_score) * 0.08)
            reasons.append(f"Momentum crash: 7d z-score={z_score:.2f}")
        elif weighted_mom > 2.0:
            direction = "bullish"
            confidence = min(0.80, 0.5 + weighted_mom * 0.04)
            consistency = "aligned" if aligned else "mixed"
            reasons.append(f"Strong positive momentum ({weighted_mom:.1f}%, {consistency})")
        elif weighted_mom < -2.0:
            direction = "bearish"
            confidence = min(0.80, 0.5 + abs(weighted_mom) * 0.04)
            consistency = "aligned" if aligned else "mixed"
            reasons.append(f"Strong negative momentum ({weighted_mom:.1f}%, {consistency})")
        elif weighted_mom > 0.5:
            direction = "bullish"
            confidence = 0.55
            reasons.append(f"Positive momentum ({weighted_mom:.1f}%)")
        elif weighted_mom < -0.5:
            direction = "bearish"
            confidence = 0.55
            reasons.append(f"Negative momentum ({weighted_mom:.1f}%)")
        else:
            direction = "neutral"
            confidence = 0.40
            reasons.append(f"Flat momentum ({weighted_mom:.1f}%)")

        if adx_now > 25:
            reasons.append(f"Trending (ADX={adx_now:.1f})")
        if aligned and not crash_risk:
            confidence = min(0.90, confidence * 1.2)
            reasons.append("All timeframes aligned")

        confidence = self.calibrate_confidence(confidence)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons),
            supporting_features=[
                f"mom_1d={mom_1d:.2f}", f"mom_7d={mom_7d:.2f}",
                f"mom_30d={mom_30d:.2f}", f"mom_90d={mom_90d:.2f}",
                f"z_score={z_score:.2f}",
            ],
        )

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            sym = o.prediction.symbol
            w = self._lookback_weights[sym]
            for feat in o.prediction.supporting_features:
                if feat.startswith("mom_") and o.was_correct:
                    # Increase weight for the lookback that matched
                    parts = feat.split("=")
                    if parts:
                        lb = int(parts[0].replace("mom_", "").replace("d", ""))
                        if lb in w:
                            w[lb] = min(0.50, w[lb] + 0.02)
                            # Normalize
                            total = sum(w.values())
                            for k in w:
                                w[k] /= total
                impact = 0.04 if o.was_correct else -0.02
                self.update_feature_importance(feat, impact)
