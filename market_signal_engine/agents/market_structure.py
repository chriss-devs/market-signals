"""Market Structure Agent — HH/HL, market phases, Wyckoff schematics, fair value gaps.

Self-improvement: learns phase transition signals per market type.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import last_valid, pct_change, sma


class MarketStructureAgent(BaseAgent):
    name = "Market Structure"
    agent_id = 8
    tier = 1
    category = "Technical"
    data_sources = ["price_history"]

    PHASE_ACCUMULATION = "accumulation"
    PHASE_MARKUP = "markup"
    PHASE_DISTRIBUTION = "distribution"
    PHASE_MARKDOWN = "markdown"
    PHASE_UNKNOWN = "unknown"

    def __init__(self) -> None:
        super().__init__()
        self._phase_duration: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def analyze(self, context: AnalysisContext) -> Prediction:
        prices = context.price_history
        if len(prices) < 90:
            return Prediction(
                agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
                direction="neutral", confidence=0.3,
                reasoning="Insufficient price history (<90 points)",
            )

        sym = context.symbol
        reasons: list[str] = []
        score = 0.0

        # ── Swing structure: higher highs / lower lows ──────────────────
        swings_high, swings_low = self._find_swings(prices, lookback=5)
        hh, hl, lh, ll = self._analyze_swing_structure(prices, swings_high, swings_low)

        # Structure score
        if hh and hl:
            score += 0.18
            reasons.append("Bullish structure: higher highs + higher lows")
        elif hh:
            score += 0.08
            reasons.append("New higher high — potential trend change")
        elif ll and lh:
            score -= 0.18
            reasons.append("Bearish structure: lower lows + lower highs")
        elif ll:
            score -= 0.08
            reasons.append("New lower low — potential trend change")
        else:
            score += 0.02
            reasons.append("Choppy structure — no clear direction")

        # ── Market phase detection (Wyckoff-style) ──────────────────────
        phase, phase_conf = self._detect_market_phase(prices)

        if phase == self.PHASE_MARKUP:
            score += 0.15
            reasons.append(f"Market in Markup phase (conf={phase_conf:.0%})")
        elif phase == self.PHASE_ACCUMULATION:
            score += 0.08
            reasons.append(f"Accumulation phase — potential breakout")
        elif phase == self.PHASE_MARKDOWN:
            score -= 0.15
            reasons.append(f"Market in Markdown phase (conf={phase_conf:.0%})")
        elif phase == self.PHASE_DISTRIBUTION:
            score -= 0.08
            reasons.append(f"Distribution phase — potential breakdown")

        # ── Fair Value Gaps ─────────────────────────────────────────────
        fvg_score, fvg_reason = self._detect_fvg(prices)
        score += fvg_score
        if fvg_reason:
            reasons.append(fvg_reason)

        # ── Order blocks / key levels ───────────────────────────────────
        ob_score, ob_reason = self._detect_order_blocks(prices, swings_high, swings_low)
        score += ob_score
        if ob_reason:
            reasons.append(ob_reason)

        # ── Trend quality ───────────────────────────────────────────────
        sma_50 = sma(prices, 50)
        sma_200 = sma(prices, 200)
        s50 = last_valid(sma_50)
        s200 = last_valid(sma_200)

        if s50 > 0:
            price_now = prices[-1]
            if price_now > s50 * 1.05:
                score += 0.05
                reasons.append("Price well above SMA50 — strong trend")
            elif price_now < s50 * 0.95:
                score -= 0.05
                reasons.append("Price well below SMA50 — weak structure")

        if s200 > 0 and s50 > s200:
            score += 0.05
            reasons.append("SMA50 > SMA200 — structural uptrend")
        elif s200 > 0 and s50 < s200:
            score -= 0.05
            reasons.append("SMA50 < SMA200 — structural downtrend")

        # ── Direction & confidence ──────────────────────────────────────
        score = max(-1.0, min(1.0, score))

        if score > 0.10:
            direction = "bullish"
            confidence = min(0.82, 0.45 + abs(score) * 0.55)
        elif score < -0.10:
            direction = "bearish"
            confidence = min(0.82, 0.45 + abs(score) * 0.55)
        else:
            direction = "neutral"
            confidence = 0.35

        confidence = self.calibrate_confidence(confidence)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons[:5]) if reasons else "Market structure indeterminate",
            supporting_features=[
                f"phase={phase}", f"hh={hh}", f"hl={hl}", f"lh={lh}", f"ll={ll}",
            ],
        )

    # ── Structure analysis helpers ──────────────────────────────────────

    @staticmethod
    def _find_swings(prices: list[float], lookback: int = 5) -> tuple[list[int], list[int]]:
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

    def _analyze_swing_structure(
        self, prices: list[float], swings_high: list[int], swings_low: list[int]
    ) -> tuple[bool, bool, bool, bool]:
        """Check for higher highs, higher lows, lower highs, lower lows."""
        hh = len(swings_high) >= 2 and prices[swings_high[-1]] > prices[swings_high[-2]]
        hl = len(swings_low) >= 2 and prices[swings_low[-1]] > prices[swings_low[-2]]
        lh = len(swings_high) >= 2 and prices[swings_high[-1]] < prices[swings_high[-2]]
        ll = len(swings_low) >= 2 and prices[swings_low[-1]] < prices[swings_low[-2]]
        return hh, hl, lh, ll

    def _detect_market_phase(self, prices: list[float]) -> tuple[str, float]:
        """Classify market into Wyckoff-style phases."""
        if len(prices) < 60:
            return self.PHASE_UNKNOWN, 0.0

        # Split into 3 segments
        n = len(prices)
        seg1 = prices[:n // 3]
        seg2 = prices[n // 3 : 2 * n // 3]
        seg3 = prices[2 * n // 3:]

        s1_avg = sum(seg1) / len(seg1)
        s2_avg = sum(seg2) / len(seg2)
        s3_avg = sum(seg3) / len(seg3)

        # Volatility per segment
        def volatility(seg: list[float]) -> float:
            avg = sum(seg) / len(seg)
            return math.sqrt(sum((p - avg) ** 2 for p in seg) / len(seg)) / avg * 100 if avg else 0

        v1 = volatility(seg1)
        v2 = volatility(seg2)
        v3 = volatility(seg3)

        price_change = (s3_avg - s1_avg) / s1_avg * 100 if s1_avg else 0

        # Markup: rising prices, moderate/increasing vol
        if price_change > 5 and s3_avg > s2_avg > s1_avg:
            if v3 > v1 * 0.8:
                conf = min(0.9, 0.5 + price_change * 0.03)
                return self.PHASE_MARKUP, conf

        # Markdown: falling prices, increasing vol
        if price_change < -5 and s3_avg < s2_avg < s1_avg:
            conf = min(0.9, 0.5 + abs(price_change) * 0.03)
            return self.PHASE_MARKDOWN, conf

        # Accumulation: flat/slightly down, decreasing vol
        if abs(price_change) < 5 and v3 < v1 * 0.8 and s3_avg >= s2_avg:
            return self.PHASE_ACCUMULATION, min(0.8, 0.4 + (v1 - v3) / v1 if v1 else 0.5)

        # Distribution: flat/slightly up, increasing vol
        if abs(price_change) < 5 and v3 > v1 * 1.2:
            return self.PHASE_DISTRIBUTION, min(0.8, 0.4 + (v3 - v1) / v3 if v3 else 0.5)

        return self.PHASE_UNKNOWN, 0.0

    def _detect_fvg(self, prices: list[float]) -> tuple[float, str]:
        """Detect Fair Value Gaps — unfilled price jumps."""
        if len(prices) < 5:
            return 0.0, ""
        # Check last 5 candles for gaps
        recent = prices[-5:]
        gaps_bull = 0
        gaps_bear = 0
        for i in range(1, len(recent)):
            if recent[i] > recent[i - 1] * 1.015:  # 1.5%+ gap up
                gaps_bull += 1
            elif recent[i] < recent[i - 1] * 0.985:  # 1.5%+ gap down
                gaps_bear += 1

        if gaps_bear > 0 and prices[-1] > prices[-2]:
            # Price retracing into bearish FVG — bearish
            return -0.08, f"Filling bearish FVG ({gaps_bear} gaps detected)"
        elif gaps_bull > 0 and prices[-1] < prices[-2]:
            return 0.08, f"Filling bullish FVG ({gaps_bull} gaps detected)"
        return 0.0, ""

    def _detect_order_blocks(
        self, prices: list[float], swings_high: list[int], swings_low: list[int]
    ) -> tuple[float, str]:
        """Detect order blocks — last candle before strong impulse moves."""
        if len(prices) < 10:
            return 0.0, ""
        # Look for recent strong move (>2%) and check for OB near origin
        change_3d = pct_change(prices, 3)
        if abs(change_3d) > 2 and len(swings_low) > 0:
            nearest_swing_low = max(swings_low)
            if nearest_swing_low > len(prices) - 10:
                if change_3d > 0:
                    return 0.08, "Bullish order block respected — demand zone holding"
        if abs(change_3d) > 2 and len(swings_high) > 0:
            nearest_swing_high = max(swings_high)
            if nearest_swing_high > len(prices) - 10:
                if change_3d < 0:
                    return -0.08, "Bearish order block respected — supply zone holding"
        return 0.0, ""

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            sym = o.prediction.symbol
            for feat in o.prediction.supporting_features:
                if feat.startswith("phase="):
                    phase = feat.split("=")[1]
                    if o.was_correct:
                        self._phase_duration[sym][phase] += 1
                impact = 0.04 if o.was_correct else -0.03
                self.update_feature_importance(feat, impact)
