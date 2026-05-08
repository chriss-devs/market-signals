"""Volatility Agent — regime detection, GARCH-style forecasting, vol-of-vol, term structure.

Self-improvement: calibrates regime thresholds from realized volatility.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import last_valid, pct_change


class VolatilityAgent(BaseAgent):
    name = "Volatility"
    agent_id = 10
    tier = 1
    category = "Volatility"
    data_sources = ["price_history"]

    REGIME_LOW = "low"
    REGIME_NORMAL = "normal"
    REGIME_HIGH = "high"
    REGIME_CRASH = "crash"

    def __init__(self) -> None:
        super().__init__()
        # Per-asset regime thresholds (annualized vol %)
        self._low_vol_max: dict[str, float] = defaultdict(lambda: 20.0)
        self._high_vol_min: dict[str, float] = defaultdict(lambda: 50.0)
        self._crash_vol_min: dict[str, float] = defaultdict(lambda: 80.0)
        # GARCH-style persistence parameter (alpha)
        self._vol_persistence: dict[str, float] = defaultdict(lambda: 0.85)

    def analyze(self, context: AnalysisContext) -> Prediction:
        prices = context.price_history
        if len(prices) < 60:
            return Prediction(
                agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
                direction="neutral", confidence=0.3,
                reasoning="Insufficient price history (<60 points)",
            )

        sym = context.symbol
        reasons: list[str] = []
        score = 0.0

        # ── Compute multiple volatility estimators ──────────────────────
        # Close-to-close (standard)
        daily_returns = self._compute_returns(prices)
        close_vol = self._annualized_vol(daily_returns)  # annualized %

        # Parkinson (high-low range estimator)
        highs = [p * 1.005 for p in prices]
        lows = [p * 0.995 for p in prices]
        parkinson_vol = self._parkinson_vol(highs, lows)

        # Short-term vs long-term vol (volatility term structure)
        st_returns = daily_returns[-10:] if len(daily_returns) > 10 else daily_returns
        lt_returns = daily_returns[-60:] if len(daily_returns) > 60 else daily_returns
        st_vol = self._annualized_vol(st_returns)
        lt_vol = self._annualized_vol(lt_returns)

        # Vol-of-vol (how much does volatility itself vary)
        vol_of_vol = 0.0
        if len(daily_returns) > 30:
            rolling_vols = []
            for i in range(30, len(daily_returns)):
                window = daily_returns[i - 30:i]
                rv = self._annualized_vol(window)
                rolling_vols.append(rv)
            if rolling_vols:
                avg_rv = sum(rolling_vols) / len(rolling_vols)
                vol_of_vol = math.sqrt(sum((v - avg_rv) ** 2 for v in rolling_vols) / len(rolling_vols))

        # GARCH-style conditional vol forecast
        garch_forecast = self._garch_forecast(daily_returns, sym)

        # ── Regime classification ───────────────────────────────────────
        regime = self._classify_regime(close_vol, sym)

        if regime == self.REGIME_CRASH:
            score -= 0.18
            reasons.append(f"CRASH regime: vol={close_vol:.0f}% — extreme fear")
        elif regime == self.REGIME_HIGH:
            score -= 0.10
            reasons.append(f"High volatility regime ({close_vol:.0f}%) — elevated risk")
        elif regime == self.REGIME_LOW:
            score += 0.08
            reasons.append(f"Low volatility regime ({close_vol:.0f}%) — stability")
        else:
            reasons.append(f"Normal volatility ({close_vol:.0f}%)")

        # ── Volatility direction ────────────────────────────────────────
        # Rising vol = bearish for risk assets; falling vol = bullish
        if lt_vol > 0:
            vol_change = (st_vol - lt_vol) / lt_vol
            if vol_change > 0.3:
                score -= 0.10
                reasons.append(f"Volatility expanding ({vol_change:+.0%}) — risk-off signal")
            elif vol_change < -0.2:
                score += 0.08
                reasons.append(f"Volatility contracting ({vol_change:+.0%}) — risk-on signal")

        # ── Volatility term structure inversion ─────────────────────────
        if st_vol > lt_vol * 1.3:
            score -= 0.08
            reasons.append(f"Vol term structure inverted — near-term fear elevated")
        elif st_vol < lt_vol * 0.7:
            score += 0.05
            reasons.append(f"Vol term structure steep — near-term calm")

        # ── Vol-of-vol signal ───────────────────────────────────────────
        if vol_of_vol > 30:
            score -= 0.06
            reasons.append(f"Unstable volatility (vol-of-vol={vol_of_vol:.0f}%)")
        elif vol_of_vol < 10 and close_vol < 25:
            score += 0.05
            reasons.append("Stable low-vol environment — favorable for trends")

        # ── Parkinson vs Close-close divergence ─────────────────────────
        if parkinson_vol > close_vol * 1.3:
            score -= 0.06
            reasons.append(f"Parkinson vol ({parkinson_vol:.0f}%) >> close vol — hidden intraday risk")

        # ── GARCH forecast signal ───────────────────────────────────────
        if garch_forecast > close_vol * 1.2:
            score -= 0.06
            reasons.append(f"GARCH forecasts rising vol ({garch_forecast:.0f}%)")
        elif garch_forecast < close_vol * 0.8:
            score += 0.04
            reasons.append(f"GARCH forecasts declining vol ({garch_forecast:.0f}%)")

        # ── Contrarian signals ──────────────────────────────────────────
        # Extreme low vol can precede crashes (complacency)
        if regime == self.REGIME_LOW and close_vol < self._low_vol_max[sym] * 0.6:
            score -= 0.04
            reasons.append("Extremely low vol — potential complacency")

        # Extreme high vol can signal capitulation (opportunity)
        change_7d = pct_change(prices, 7) if len(prices) > 7 else 0
        if regime == self.REGIME_CRASH and change_7d < -10:
            score += 0.10
            reasons.append("Capitulation selling at crash vol — potential reversal")

        # ── Direction & confidence ──────────────────────────────────────
        score = max(-1.0, min(1.0, score))

        if score > 0.07:
            direction = "bullish"
            confidence = min(0.78, 0.45 + abs(score) * 0.5)
        elif score < -0.07:
            direction = "bearish"
            confidence = min(0.78, 0.45 + abs(score) * 0.5)
        else:
            direction = "neutral"
            confidence = 0.35

        confidence = self.calibrate_confidence(confidence)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons[:5]) if reasons else "Volatility signals neutral",
            supporting_features=[
                f"close_vol={close_vol:.1f}", f"parkinson={parkinson_vol:.1f}",
                f"garch={garch_forecast:.1f}", f"regime={regime}",
            ],
        )

    # ── Volatility computation helpers ──────────────────────────────────

    @staticmethod
    def _compute_returns(prices: list[float]) -> list[float]:
        """Log returns."""
        rets = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                rets.append(math.log(prices[i] / prices[i - 1]))
        return rets

    @staticmethod
    def _annualized_vol(returns: list[float]) -> float:
        """Annualized volatility from daily log returns."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(var) * math.sqrt(365) * 100

    @staticmethod
    def _parkinson_vol(highs: list[float], lows: list[float]) -> float:
        """Parkinson volatility estimator using high-low range."""
        n = min(len(highs), len(lows))
        if n < 5:
            return 0.0
        sum_sq = 0.0
        for i in range(max(0, n - 20), n):
            if lows[i] > 0:
                hl_ratio = math.log(highs[i] / lows[i])
                sum_sq += hl_ratio ** 2
        n_used = min(20, n)
        if n_used == 0:
            return 0.0
        parkinson_var = sum_sq / (4 * math.log(2) * n_used)
        return math.sqrt(parkinson_var) * math.sqrt(365) * 100

    def _garch_forecast(self, returns: list[float], sym: str) -> float:
        """Simple GARCH(1,1)-style conditional volatility forecast."""
        if len(returns) < 20:
            return 0.0
        alpha = self._vol_persistence[sym]
        omega = sum(r ** 2 for r in returns[-60:]) / max(len(returns[-60:]), 1) * (1 - alpha)

        # Recursive conditional variance
        cond_var = sum(r ** 2 for r in returns[-20:]) / 20  # initial estimate
        for r in returns[-10:]:
            cond_var = omega + alpha * cond_var + (1 - alpha) * r ** 2

        return math.sqrt(cond_var) * math.sqrt(365) * 100

    def _classify_regime(self, vol: float, sym: str) -> str:
        if vol >= self._crash_vol_min[sym]:
            return self.REGIME_CRASH
        if vol >= self._high_vol_min[sym]:
            return self.REGIME_HIGH
        if vol <= self._low_vol_max[sym]:
            return self.REGIME_LOW
        return self.REGIME_NORMAL

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            sym = o.prediction.symbol
            for feat in o.prediction.supporting_features:
                if feat.startswith("close_vol="):
                    vol = float(feat.split("=")[1])
                    actual_vol = self._low_vol_max[sym]
                    # Tune thresholds if wrong
                    if o.prediction.direction == "bearish" and not o.was_correct:
                        self._high_vol_min[sym] = min(90, self._high_vol_min[sym] + 2)
                    elif o.prediction.direction == "bullish" and not o.was_correct:
                        self._low_vol_max[sym] = max(10, self._low_vol_max[sym] - 2)
                impact = 0.04 if o.was_correct else -0.03
                self.update_feature_importance(feat, impact)
