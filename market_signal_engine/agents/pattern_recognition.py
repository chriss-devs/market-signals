"""Pattern Recognition Agent — chart patterns, candlestick patterns, harmonic patterns.

Self-improvement: tracks which patterns have highest hit rate per asset.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import last_valid, pct_change, sma


class PatternRecognitionAgent(BaseAgent):
    name = "Pattern Recognition"
    agent_id = 6
    tier = 1
    category = "Technical"
    data_sources = ["price_history"]

    # Candlestick patterns that indicate reversal
    BULLISH_CANDLES = {"hammer", "morning_star", "bullish_engulfing", "piercing", "dragonfly_doji"}
    BEARISH_CANDLES = {"shooting_star", "evening_star", "bearish_engulfing", "dark_cloud", "gravestone_doji"}

    def __init__(self) -> None:
        super().__init__()
        # Per-asset pattern hit tracking
        self._pattern_hits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._pattern_misses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def analyze(self, context: AnalysisContext) -> Prediction:
        prices = context.price_history
        if len(prices) < 60:
            return Prediction(
                agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
                direction="neutral", confidence=0.3,
                reasoning="Insufficient price history (<60 points)",
            )

        sym = context.symbol
        highs = [p * 1.005 for p in prices]
        lows = [p * 0.995 for p in prices]

        reasons: list[str] = []
        score = 0.0

        # ── Swing point detection ───────────────────────────────────────
        swings_high, swings_low = self._find_swings(prices, lookback=5)

        # ── Chart patterns ──────────────────────────────────────────────
        # Double top / double bottom
        dt_score, dt_reason = self._detect_double_top_bottom(prices, swings_high, swings_low)
        score += dt_score
        if dt_reason:
            reasons.append(dt_reason)

        # Head and shoulders / inverse H&S
        hs_score, hs_reason = self._detect_head_shoulders(swings_high, swings_low, prices)
        score += hs_score
        if hs_reason:
            reasons.append(hs_reason)

        # Triangle / wedge detection via range contraction
        tri_score, tri_reason = self._detect_triangle(prices, highs, lows)
        score += tri_score
        if tri_reason:
            reasons.append(tri_reason)

        # Support / resistance from swing levels
        sr_score, sr_reason = self._evaluate_support_resistance(prices, swings_low, swings_high)
        score += sr_score
        if sr_reason:
            reasons.append(sr_reason)

        # ── Candlestick patterns ────────────────────────────────────────
        candle_score, candle_reason = self._detect_candlestick(prices)
        score += candle_score
        if candle_reason:
            reasons.append(candle_reason)

        # ── Trend context ───────────────────────────────────────────────
        sma_50 = sma(prices, 50)
        sma_200 = sma(prices, 200)
        s50 = last_valid(sma_50)
        s200 = last_valid(sma_200)
        price_now = prices[-1]

        trend_bonus = 0.0
        if s50 > 0 and s200 > 0:
            if price_now > s50 > s200:
                trend_bonus = 0.05
                reasons.append("Uptrend supports bullish patterns")
            elif price_now < s50 < s200:
                trend_bonus = -0.05
                reasons.append("Downtrend supports bearish patterns")

        score += trend_bonus

        # ── Direction & confidence ──────────────────────────────────────
        score = max(-1.0, min(1.0, score))

        if score > 0.10:
            direction = "bullish"
            confidence = min(0.80, 0.45 + abs(score) * 0.5)
        elif score < -0.10:
            direction = "bearish"
            confidence = min(0.80, 0.45 + abs(score) * 0.5)
        else:
            direction = "neutral"
            confidence = 0.35

        confidence = self.calibrate_confidence(confidence)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons[:5]) if reasons else "No significant patterns detected",
            supporting_features=[
                f"dt_score={dt_score:.2f}", f"hs_score={hs_score:.2f}",
                f"tri_score={tri_score:.2f}", f"candle_score={candle_score:.2f}",
            ],
        )

    # ── Pattern detection helpers ────────────────────────────────────────

    def _find_swings(self, prices: list[float], lookback: int = 5) -> tuple[list[int], list[int]]:
        """Find swing high and swing low indices using local extrema."""
        swing_highs: list[int] = []
        swing_lows: list[int] = []
        for i in range(lookback, len(prices) - lookback):
            window = prices[i - lookback : i + lookback + 1]
            mid = prices[i]
            if mid == max(window) and all(mid > p for j, p in enumerate(window) if j != lookback):
                swing_highs.append(i)
            if mid == min(window) and all(mid < p for j, p in enumerate(window) if j != lookback):
                swing_lows.append(i)
        return swing_highs, swing_lows

    def _detect_double_top_bottom(
        self, prices: list[float], swings_high: list[int], swings_low: list[int]
    ) -> tuple[float, str]:
        """Detect double top (bearish) and double bottom (bullish)."""
        # Double top: two similar swing highs
        if len(swings_high) >= 2:
            h1_idx, h2_idx = swings_high[-2], swings_high[-1]
            h1, h2 = prices[h1_idx], prices[h2_idx]
            mid_trough = min(prices[h1_idx:h2_idx + 1]) if h2_idx > h1_idx else prices[h1_idx]
            if abs(h1 - h2) / max(h1, 1) < 0.03 and h2_idx - h1_idx > 10:
                # Confirmed if price broke below mid trough
                if prices[-1] < mid_trough:
                    return -0.18, f"Double top confirmed (breakdown below {mid_trough:.2f})"
                return -0.10, f"Double top forming ({h1:.2f} / {h2:.2f})"

        # Double bottom: two similar swing lows
        if len(swings_low) >= 2:
            l1_idx, l2_idx = swings_low[-2], swings_low[-1]
            l1, l2 = prices[l1_idx], prices[l2_idx]
            mid_peak = max(prices[l1_idx:l2_idx + 1]) if l2_idx > l1_idx else prices[l1_idx]
            if abs(l1 - l2) / max(l1, 1) < 0.03 and l2_idx - l1_idx > 10:
                if prices[-1] > mid_peak:
                    return 0.18, f"Double bottom confirmed (breakout above {mid_peak:.2f})"
                return 0.10, f"Double bottom forming ({l1:.2f} / {l2:.2f})"

        return 0.0, ""

    def _detect_head_shoulders(
        self, swings_high: list[int], swings_low: list[int], prices: list[float]
    ) -> tuple[float, str]:
        """Detect head & shoulders (bearish) or inverse H&S (bullish)."""
        if len(swings_high) >= 3:
            s1, s2, s3 = swings_high[-3], swings_high[-2], swings_high[-1]
            if s1 < s2 < s3 and len(prices) > s3:
                p1, p2, p3 = prices[s1], prices[s2], prices[s3]
                # Head (p2) should be highest, shoulders (p1, p3) similar
                if p2 > p1 and p2 > p3 and abs(p1 - p3) / max(p1, 1) < 0.05:
                    # Neckline: lowest point between shoulders
                    neckline = min(prices[s1:s3 + 1])
                    if prices[-1] < neckline:
                        return -0.20, "Head & Shoulders breakdown confirmed"
                    return -0.12, "Head & Shoulders pattern detected"

        if len(swings_low) >= 3:
            s1, s2, s3 = swings_low[-3], swings_low[-2], swings_low[-1]
            if s1 < s2 < s3 and len(prices) > s3:
                p1, p2, p3 = prices[s1], prices[s2], prices[s3]
                # Head (p2) should be lowest, shoulders (p1, p3) similar
                if p2 < p1 and p2 < p3 and abs(p1 - p3) / max(p1, 1) < 0.05:
                    neckline = max(prices[s1:s3 + 1])
                    if prices[-1] > neckline:
                        return 0.20, "Inverse Head & Shoulders breakout confirmed"
                    return 0.12, "Inverse Head & Shoulders pattern detected"

        return 0.0, ""

    def _detect_triangle(
        self, prices: list[float], highs: list[float], lows: list[float]
    ) -> tuple[float, str]:
        """Detect contracting triangle/flag/wedge via narrowing price range."""
        if len(prices) < 20:
            return 0.0, ""
        # Compare range of last 10 days vs 10 days prior
        recent_range = max(prices[-10:]) - min(prices[-10:])
        prior_range = max(prices[-20:-10]) - min(prices[-20:-10])
        if prior_range > 0 and recent_range / prior_range < 0.6:
            change_10d = pct_change(prices, 10)
            if change_10d > 0:
                return 0.10, f"Bull flag / ascending triangle (range contracted {recent_range/prior_range:.0%})"
            else:
                return -0.10, f"Bear flag / descending triangle (range contracted {recent_range/prior_range:.0%})"
        return 0.0, ""

    def _evaluate_support_resistance(
        self, prices: list[float], swings_low: list[int], swings_high: list[int]
    ) -> tuple[float, str]:
        """Check if price is near support or resistance levels."""
        price_now = prices[-1]
        # Find nearest swing low (support) and swing high (resistance)
        support_levels = sorted([prices[i] for i in swings_low if prices[i] < price_now])
        resistance_levels = sorted([prices[i] for i in swings_high if prices[i] > price_now])

        # Bounce off support
        if support_levels and (price_now - support_levels[-1]) / price_now < 0.02:
            return 0.10, f"Price near support ({support_levels[-1]:.2f}) — potential bounce"

        # Rejection at resistance
        if resistance_levels and (resistance_levels[0] - price_now) / price_now < 0.02:
            return -0.10, f"Price near resistance ({resistance_levels[0]:.2f}) — potential rejection"

        return 0.0, ""

    def _detect_candlestick(self, prices: list[float]) -> tuple[float, str]:
        """Detect single and multi-candle reversal patterns using last 5 candles."""
        if len(prices) < 5:
            return 0.0, ""
        p = prices[-5:]
        o1, c1 = p[-2], p[-1]  # open/close approximation

        # Bullish engulfing: yesterday down, today up with larger range
        if c1 > o1 and p[-3] > p[-2] and (c1 - o1) > (p[-3] - p[-2]) * 1.2:
            return 0.12, "Bullish engulfing candle"

        # Bearish engulfing
        if c1 < o1 and p[-3] < p[-2] and (o1 - c1) > (p[-2] - p[-3]) * 1.2:
            return -0.12, "Bearish engulfing candle"

        # Hammer: small body at top, long lower wick
        body = abs(c1 - o1)
        range_3d = max(p[-3:]) - min(p[-3:])
        if range_3d > 0 and body / range_3d < 0.3 and c1 > p[-3]:
            return 0.08, "Hammer / bullish reversal candle"

        # Shooting star: small body at bottom, long upper wick
        if range_3d > 0 and body / range_3d < 0.3 and c1 < p[-3]:
            return -0.08, "Shooting star / bearish reversal candle"

        # Consecutive direction (momentum confirmation)
        ups = sum(1 for i in range(1, 5) if p[i] > p[i - 1])
        if ups >= 4:
            return 0.06, f"Strong bullish candle sequence ({ups}/4 up)"
        if ups <= 1:
            return -0.06, f"Strong bearish candle sequence ({4-ups}/4 down)"

        return 0.0, ""

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            for feat in o.prediction.supporting_features:
                pattern_key = feat.split("=")[0]
                if o.was_correct:
                    self._pattern_hits[o.prediction.symbol][pattern_key] += 1
                else:
                    self._pattern_misses[o.prediction.symbol][pattern_key] += 1
                impact = 0.05 if o.was_correct else -0.03
                self.update_feature_importance(feat, impact)
