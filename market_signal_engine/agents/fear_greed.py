"""Fear & Greed Agent — composite index from multiple sources.

Self-improvement: learns extreme thresholds that signal reversals.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import last_valid, pct_change


class FearGreedAgent(BaseAgent):
    name = "Fear & Greed"
    agent_id = 11
    tier = 1
    category = "Sentiment"
    data_sources = ["price_history", "volume_history", "social_sentiment"]

    # Composite component weights
    COMPONENTS = {
        "momentum": 0.25,
        "volatility": 0.25,
        "volume": 0.15,
        "social": 0.15,
        "dominance": 0.10,
        "flows": 0.10,
    }

    def __init__(self) -> None:
        super().__init__()
        # Per-asset extreme thresholds (index values 0-100)
        self._extreme_fear: dict[str, float] = defaultdict(lambda: 20.0)
        self._extreme_greed: dict[str, float] = defaultdict(lambda: 80.0)
        # Component weight tuning
        self._component_weights: dict[str, dict[str, float]] = defaultdict(
            lambda: dict(self.COMPONENTS)
        )

    def analyze(self, context: AnalysisContext) -> Prediction:
        prices = context.price_history
        volumes = context.volume_history
        features = context.features.features

        if len(prices) < 30:
            return Prediction(
                agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
                direction="neutral", confidence=0.3,
                reasoning="Insufficient price history (<30 points)",
            )

        sym = context.symbol
        w = self._component_weights[sym]
        reasons: list[str] = []

        # ── 1. Price Momentum (0-100) ────────────────────────────────────
        mom_1d = pct_change(prices, 1)
        mom_7d = pct_change(prices, 7)
        mom_30d = pct_change(prices, 30) if len(prices) > 30 else mom_7d
        mom_raw = mom_1d * 0.3 + mom_7d * 0.4 + mom_30d * 0.3
        # Map to 0-100: -5% -> 0 (fear), +5% -> 100 (greed)
        momentum_score = self._map_to_index(mom_raw, -5, 5)
        reasons.append(f"Momentum: {momentum_score:.0f}")

        # ── 2. Volatility (0-100, inverted) ──────────────────────────────
        vol = self._compute_vol(prices)
        vol_score = self._map_to_index(-vol, -40, -5)  # inverted: high vol = fear
        reasons.append(f"Volatility: {vol_score:.0f}")

        # ── 3. Volume / participation ────────────────────────────────────
        vol_change = pct_change(volumes, 10) if len(volumes) > 10 else 0
        volume_score = self._map_to_index(vol_change, -20, 20)
        reasons.append(f"Volume: {volume_score:.0f}")

        # ── 4. Social sentiment ──────────────────────────────────────────
        social = features.get("social_sentiment", 0.0)
        social_score = self._map_to_index(social * 100, -50, 50)
        reasons.append(f"Social: {social_score:.0f}")

        # ── 5. Market dominance (from features) ──────────────────────────
        dominance = features.get("market_dominance", 0.0)
        dom_score = 50.0  # neutral
        if dominance > 0:
            dom_score = self._map_to_index(dominance, -2, 2)

        # ── 6. Exchange flows ────────────────────────────────────────────
        flows = features.get("exchange_flows", 0.0)  # positive = inflows
        flow_score = self._map_to_index(-flows, -1, 1)  # inverted: inflows = fear

        # ── Composite index ──────────────────────────────────────────────
        composite = (
            w["momentum"] * momentum_score
            + w["volatility"] * vol_score
            + w["volume"] * volume_score
            + w["social"] * social_score
            + w["dominance"] * dom_score
            + w["flows"] * flow_score
        )
        composite = max(0.0, min(100.0, composite))

        # ── Recent trend of the index (is fear/greed accelerating?) ──────
        # Approximate by checking if price momentum is diverging from vol
        fear_change = pct_change(prices, 5)

        # ── Signal logic ────────────────────────────────────────────────
        extreme_fear = self._extreme_fear[sym]
        extreme_greed = self._extreme_greed[sym]

        if composite <= extreme_fear:
            # Extreme fear = contrarian bullish
            direction = "bullish"
            confidence = min(0.82, 0.5 + (extreme_fear - composite) / extreme_fear * 0.3)
            reasons.append(f"EXTREME FEAR ({composite:.0f}/100) — contrarian buy signal")
        elif composite >= extreme_greed:
            # Extreme greed = contrarian bearish
            direction = "bearish"
            confidence = min(0.82, 0.5 + (composite - extreme_greed) / (100 - extreme_greed) * 0.3)
            reasons.append(f"EXTREME GREED ({composite:.0f}/100) — contrarian sell signal")
        elif composite > 60:
            direction = "bullish"
            confidence = min(0.65, 0.4 + (composite - 50) * 0.01)
            reasons.append(f"Greed zone ({composite:.0f}/100)")
        elif composite < 40:
            direction = "bearish"
            confidence = min(0.65, 0.4 + (50 - composite) * 0.01)
            reasons.append(f"Fear zone ({composite:.0f}/100)")
        else:
            direction = "neutral"
            confidence = 0.35
            reasons.append(f"Neutral sentiment ({composite:.0f}/100)")

        # Volatility-momentum divergence check
        if momentum_score > 70 and vol_score < 30:
            confidence = max(confidence, 0.55)
            reasons.append("High momentum + low vol = greed building")

        confidence = self.calibrate_confidence(confidence)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons[:5]),
            supporting_features=[
                f"composite={composite:.1f}", f"momentum={momentum_score:.1f}",
                f"volatility={vol_score:.1f}", f"volume={volume_score:.1f}",
                f"social={social_score:.1f}",
            ],
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _map_to_index(value: float, min_val: float, max_val: float) -> float:
        """Map a raw value to 0-100 index. Values outside range are clamped."""
        clamped = max(min_val, min(max_val, value))
        if max_val == min_val:
            return 50.0
        return (clamped - min_val) / (max_val - min_val) * 100

    @staticmethod
    def _compute_vol(prices: list[float]) -> float:
        """Annualized volatility %."""
        if len(prices) < 5:
            return 0.0
        rets = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                rets.append(math.log(prices[i] / prices[i - 1]) * 100)
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var) * math.sqrt(365)

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            sym = o.prediction.symbol
            for feat in o.prediction.supporting_features:
                if feat.startswith("composite="):
                    idx_val = float(feat.split("=")[1])
                    if o.was_correct:
                        # Successful contrarian signals — thresholds are right
                        if o.prediction.direction == "bullish":
                            self._extreme_fear[sym] = min(35, self._extreme_fear[sym] + 1)
                        elif o.prediction.direction == "bearish":
                            self._extreme_greed[sym] = max(65, self._extreme_greed[sym] - 1)
                    else:
                        # Failed — thresholds need widening
                        if o.prediction.direction == "bullish":
                            self._extreme_fear[sym] = max(10, self._extreme_fear[sym] - 2)
                        elif o.prediction.direction == "bearish":
                            self._extreme_greed[sym] = min(90, self._extreme_greed[sym] + 2)
                impact = 0.05 if o.was_correct else -0.03
                self.update_feature_importance(feat, impact)
