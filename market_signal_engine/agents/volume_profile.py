"""Volume Profile Agent — VWAP, volume nodes, POC, accumulation/distribution zones.

Self-improvement: adjusts zone sensitivity per market volatility regime.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import last_valid, pct_change, sma


class VolumeProfileAgent(BaseAgent):
    name = "Volume Profile"
    agent_id = 7
    tier = 1
    category = "Technical"
    data_sources = ["price_history", "volume_history"]

    def __init__(self) -> None:
        super().__init__()
        self._zone_sensitivity: dict[str, float] = defaultdict(lambda: 0.05)
        self._climax_threshold: dict[str, float] = defaultdict(lambda: 2.5)

    def analyze(self, context: AnalysisContext) -> Prediction:
        prices = context.price_history
        volumes = context.volume_history
        if len(prices) < 50 or len(volumes) < 50:
            return Prediction(
                agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
                direction="neutral", confidence=0.3,
                reasoning="Insufficient price/volume history (<50 points)",
            )

        sym = context.symbol
        reasons: list[str] = []
        score = 0.0

        # ── VWAP ────────────────────────────────────────────────────────
        vwap_now = self._compute_vwap(prices[-20:], volumes[-20:])
        price_now = prices[-1]
        if vwap_now > 0:
            vwap_dev = (price_now - vwap_now) / vwap_now * 100
            if vwap_dev > 2:
                score -= 0.08
                reasons.append(f"Price above VWAP ({vwap_dev:+.1f}%) — overextended")
            elif vwap_dev < -2:
                score += 0.08
                reasons.append(f"Price below VWAP ({vwap_dev:+.1f}%) — discount")
            elif vwap_dev > 0:
                score += 0.04
                reasons.append(f"Price above VWAP — bullish intraday bias")

        # ── Volume Nodes / POC ──────────────────────────────────────────
        poc_price, poc_volume = self._find_poc(prices[-30:], volumes[-30:])
        if poc_price > 0:
            poc_dev = (price_now - poc_price) / poc_price * 100
            # POC acts as magnet — price tends to return to high-volume node
            if abs(poc_dev) < 1:
                score += 0.03
                reasons.append(f"Price at POC ({poc_price:.2f}) — equilibrium")
            elif poc_dev > 3:
                reasons.append(f"Above POC ({poc_price:.2f}, +{poc_dev:.1f}%)")

        # ── Accumulation / Distribution zones ───────────────────────────
        zone_sens = self._zone_sensitivity[sym]
        acc_score, acc_reason = self._detect_accumulation(prices, volumes, zone_sens)
        score += acc_score
        if acc_reason:
            reasons.append(acc_reason)

        dist_score, dist_reason = self._detect_distribution(prices, volumes, zone_sens)
        score += dist_score
        if dist_reason:
            reasons.append(dist_reason)

        # ── Volume climax detection ─────────────────────────────────────
        climax_score, climax_reason = self._detect_volume_climax(prices, volumes, sym)
        score += climax_score
        if climax_reason:
            reasons.append(climax_reason)

        # ── Volume trend ────────────────────────────────────────────────
        vol_change = pct_change(volumes, 10) if len(volumes) > 10 else 0
        change_7d = pct_change(prices, 7) if len(prices) > 7 else 0

        # Rising volume with rising price = bullish confirmation
        if vol_change > 10 and change_7d > 0:
            score += 0.08
            reasons.append(f"Volume confirming uptrend (vol +{vol_change:.0f}%)")
        elif vol_change > 10 and change_7d < 0:
            score -= 0.08
            reasons.append(f"Volume confirming downtrend (vol +{vol_change:.0f}%)")

        # Drying up volume in uptrend = weakening
        if vol_change < -20 and change_7d > 0:
            score -= 0.06
            reasons.append(f"Volume drying up in uptrend — weakening")
        elif vol_change < -20 and change_7d < 0:
            score += 0.06
            reasons.append(f"Volume drying up in downtrend — selling exhausted")

        # ── Direction & confidence ──────────────────────────────────────
        score = max(-1.0, min(1.0, score))

        if score > 0.08:
            direction = "bullish"
            confidence = min(0.78, 0.45 + abs(score) * 0.5)
        elif score < -0.08:
            direction = "bearish"
            confidence = min(0.78, 0.45 + abs(score) * 0.5)
        else:
            direction = "neutral"
            confidence = 0.35

        confidence = self.calibrate_confidence(confidence)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons[:5]) if reasons else "Volume profile neutral",
            supporting_features=[
                f"vwap_dev={vwap_dev if vwap_now>0 else 0:.2f}",
                f"poc_dev={poc_dev if poc_price>0 else 0:.2f}",
                f"vol_change={vol_change:.1f}", f"climax={climax_score:.2f}",
            ],
        )

    # ── Volume analysis helpers ─────────────────────────────────────────

    @staticmethod
    def _compute_vwap(prices: list[float], volumes: list[float]) -> float:
        n = min(len(prices), len(volumes))
        if n == 0:
            return 0.0
        pv_sum = sum(prices[-n + i] * max(volumes[-n + i], 0) for i in range(n))
        v_sum = sum(max(v, 0) for v in volumes[-n:])
        return pv_sum / v_sum if v_sum > 0 else 0.0

    @staticmethod
    def _find_poc(prices: list[float], volumes: list[float]) -> tuple[float, float]:
        """Find Point of Control — price level with highest volume."""
        if len(prices) < 5:
            return 0.0, 0.0
        # Create price bins and sum volume in each bin
        mn, mx = min(prices), max(prices)
        if mx == mn:
            return mn, sum(volumes)
        bin_count = max(5, min(20, len(prices) // 2))
        bin_size = (mx - mn) / bin_count
        bins: dict[int, float] = defaultdict(float)
        for p, v in zip(prices, volumes):
            if p and v:
                idx = int((p - mn) / bin_size)
                idx = min(bin_count - 1, max(0, idx))
                bins[idx] += v
        if not bins:
            return 0.0, 0.0
        best_bin = max(bins, key=bins.get)
        poc = mn + (best_bin + 0.5) * bin_size
        return poc, bins[best_bin]

    def _detect_accumulation(
        self, prices: list[float], volumes: list[float], sensitivity: float
    ) -> tuple[float, str]:
        """Detect accumulation: flat/slightly down price + rising volume."""
        if len(prices) < 20:
            return 0.0, ""
        recent = prices[-20:]
        price_range = (max(recent) - min(recent)) / max(recent) * 100 if max(recent) > 0 else 0
        vol_early = sum(max(v, 0) for v in volumes[-20:-10])
        vol_late = sum(max(v, 0) for v in volumes[-10:])
        if vol_early > 0 and vol_late / vol_early > 1.3 and price_range < sensitivity * 100:
            change = pct_change(prices, 20)
            if change > -1:
                return 0.12, f"Accumulation zone detected (range={price_range:.1f}%, vol rising {vol_late/vol_early:.1f}x)"
        return 0.0, ""

    def _detect_distribution(
        self, prices: list[float], volumes: list[float], sensitivity: float
    ) -> tuple[float, str]:
        """Detect distribution: flat/slightly up price + rising volume."""
        if len(prices) < 20:
            return 0.0, ""
        recent = prices[-20:]
        price_range = (max(recent) - min(recent)) / max(recent) * 100 if max(recent) > 0 else 0
        vol_early = sum(max(v, 0) for v in volumes[-20:-10])
        vol_late = sum(max(v, 0) for v in volumes[-10:])
        if vol_early > 0 and vol_late / vol_early > 1.3 and price_range < sensitivity * 100:
            change = pct_change(prices, 20)
            if change < 1:
                return -0.12, f"Distribution zone detected (range={price_range:.1f}%, vol rising {vol_late/vol_early:.1f}x)"
        return 0.0, ""

    def _detect_volume_climax(
        self, prices: list[float], volumes: list[float], sym: str
    ) -> tuple[float, str]:
        """Detect volume climax — extreme volume relative to average."""
        if len(volumes) < 20:
            return 0.0, ""
        avg_vol = sum(max(v, 0) for v in volumes[-20:]) / 20
        recent_vol = max(v for v in volumes[-3:])
        if avg_vol > 0:
            z_score = (recent_vol - avg_vol) / (math.sqrt(sum((max(v,0)-avg_vol)**2 for v in volumes[-20:])/20) or 1)
            threshold = self._climax_threshold[sym]
            if z_score > threshold:
                change_3d = pct_change(prices, 3)
                if change_3d > 2:
                    return -0.15, f"Volume climax — buying exhaustion (z={z_score:.1f})"
                elif change_3d < -2:
                    return 0.15, f"Volume climax — selling exhaustion (z={z_score:.1f})"
                else:
                    # Huge volume, no price movement — potential reversal signal
                    if z_score > threshold + 0.5:
                        return 0.08, f"Volume climax — potential bottom (z={z_score:.1f})"
        return 0.0, ""

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            sym = o.prediction.symbol
            if o.was_correct:
                self._zone_sensitivity[sym] = max(0.02, self._zone_sensitivity[sym] - 0.002)
            else:
                self._zone_sensitivity[sym] = min(0.10, self._zone_sensitivity[sym] + 0.003)
            for feat in o.prediction.supporting_features:
                impact = 0.04 if o.was_correct else -0.03
                self.update_feature_importance(feat, impact)
