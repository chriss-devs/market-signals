"""Technical Analysis Agent — RSI, MACD, SMA cross, Bollinger, ATR, divergences.

Self-improvement: tunes indicator thresholds per asset based on outcome history.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import (
    adx, atr, bollinger_bands, last_valid, macd, pct_change, rsi, slope, sma,
)


class TechnicalAnalysisAgent(BaseAgent):
    name = "Technical Analysis"
    agent_id = 1
    tier = 1
    category = "Technical"
    data_sources = ["price_history", "volume_history"]

    def __init__(self) -> None:
        super().__init__()
        # Tunable thresholds per asset — keyed by symbol
        self._rsi_oversold: dict[str, float] = defaultdict(lambda: 30.0)
        self._rsi_overbought: dict[str, float] = defaultdict(lambda: 70.0)
        self._bb_squeeze_factor: dict[str, float] = defaultdict(lambda: 0.05)
        self._adx_trend_threshold: dict[str, float] = defaultdict(lambda: 25.0)

    def analyze(self, context: AnalysisContext) -> Prediction:
        prices = context.price_history
        if len(prices) < 50:
            return self._neutral(context, "Insufficient price history (<50 points)")

        sym = context.symbol
        highs = [p * 1.005 for p in prices]  # approximate
        lows = [p * 0.995 for p in prices]

        # ── Compute indicators ──────────────────────────────────────────
        rsi_vals = rsi(prices, 14)
        macd_data = macd(prices)
        sma_7 = sma(prices, 7)
        sma_25 = sma(prices, 25)
        sma_50 = sma(prices, 50)
        sma_200 = sma(prices, 200)
        bb = bollinger_bands(prices, 20, 2.0)
        atr_vals = atr(highs, lows, prices, 14)
        adx_vals = adx(highs, lows, prices, 14)

        rsi_now = last_valid(rsi_vals)
        macd_hist = last_valid(macd_data["histogram"])
        macd_slope = slope(macd_data["macd"], 5)
        s7 = last_valid(sma_7)
        s25 = last_valid(sma_25)
        s50 = last_valid(sma_50)
        s200 = last_valid(sma_200)
        bb_upper = last_valid(bb["upper"])
        bb_lower = last_valid(bb["lower"])
        bb_mid = last_valid(bb["middle"])
        atr_now = last_valid(atr_vals)
        adx_now = last_valid(adx_vals)
        price_now = prices[-1]
        change_1d = pct_change(prices, 1)
        change_7d = pct_change(prices, 7)

        # ── Scoring ─────────────────────────────────────────────────────
        score = 0.0
        reasons: list[str] = []

        # RSI
        rsi_os = self._rsi_oversold[sym]
        rsi_ob = self._rsi_overbought[sym]
        if rsi_now < rsi_os:
            score += 0.20
            reasons.append(f"RSI oversold ({rsi_now:.1f} < {rsi_os:.0f})")
        elif rsi_now > rsi_ob:
            score -= 0.20
            reasons.append(f"RSI overbought ({rsi_now:.1f} > {rsi_ob:.0f})")

        # MACD
        if macd_hist > 0 and macd_slope > 0:
            score += 0.15
            reasons.append("MACD bullish (positive histogram, rising)")
        elif macd_hist < 0 and macd_slope < 0:
            score -= 0.15
            reasons.append("MACD bearish (negative histogram, falling)")

        # SMA crossovers
        if s7 > s25 > s50:
            score += 0.12
            reasons.append("SMAs aligned bullish (7>25>50)")
        elif s7 < s25 < s50:
            score -= 0.12
            reasons.append("SMAs aligned bearish (7<25<50)")

        # SMA 50/200 golden cross
        if not math.isnan(s50) and not math.isnan(s200):
            if s50 > s200:
                score += 0.10
                reasons.append("Golden cross: SMA50 > SMA200")
            else:
                score -= 0.10
                reasons.append("Death cross: SMA50 < SMA200")

        # Bollinger Bands
        if not math.isnan(bb_upper) and not math.isnan(bb_lower):
            bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid else 1
            squeeze = self._bb_squeeze_factor[sym]
            if bb_width < squeeze and adx_now > self._adx_trend_threshold[sym]:
                if price_now > bb_mid:
                    score += 0.15
                    reasons.append(f"BB squeeze breakout upward (width={bb_width:.3f})")
                else:
                    score -= 0.15
                    reasons.append(f"BB squeeze breakdown (width={bb_width:.3f})")
            elif price_now <= bb_lower:
                score += 0.10
                reasons.append("Price at lower Bollinger Band")
            elif price_now >= bb_upper:
                score -= 0.10
                reasons.append("Price at upper Bollinger Band")

        # ADX trend strength
        adx_thresh = self._adx_trend_threshold[sym]
        if adx_now > adx_thresh:
            if change_7d > 0:
                score += 0.08
                reasons.append(f"Strong uptrend (ADX={adx_now:.1f})")
            else:
                score -= 0.08
                reasons.append(f"Strong downtrend (ADX={adx_now:.1f})")

        # ── Direction & confidence ──────────────────────────────────────
        score = max(-1.0, min(1.0, score))
        if score > 0.12:
            direction = "bullish"
            confidence = min(0.9, 0.5 + abs(score) * 0.6)
        elif score < -0.12:
            direction = "bearish"
            confidence = min(0.9, 0.5 + abs(score) * 0.6)
        else:
            direction = "neutral"
            confidence = 0.35 + abs(score) * 1.0

        confidence = self.calibrate_confidence(confidence)

        return Prediction(
            agent_name=self.name,
            agent_id=self.agent_id,
            symbol=context.symbol,
            direction=direction,
            confidence=round(confidence, 4),
            reasoning="; ".join(reasons[:5]) if reasons else "No strong technical signals",
            supporting_features=[f"RSI={rsi_now:.1f}", f"MACD_hist={macd_hist:.4f}", f"ADX={adx_now:.1f}"],
        )

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            sym = o.prediction.symbol
            # Tune RSI thresholds: if we were bullish and wrong, tighten oversold
            if o.prediction.direction == "bullish" and not o.was_correct:
                self._rsi_oversold[sym] = max(15, self._rsi_oversold[sym] - 2)
            elif o.prediction.direction == "bearish" and not o.was_correct:
                self._rsi_overbought[sym] = min(85, self._rsi_overbought[sym] + 2)
            # Update feature importance
            for feat in o.prediction.supporting_features:
                impact = 0.05 if o.was_correct else -0.03
                self.update_feature_importance(feat, impact)

    def _neutral(self, context: AnalysisContext, reason: str) -> Prediction:
        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction="neutral", confidence=0.3, reasoning=reason,
        )
