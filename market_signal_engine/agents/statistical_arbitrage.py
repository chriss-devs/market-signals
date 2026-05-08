"""Statistical Arbitrage Agent — mean reversion and pairs trading signals.

Detects cointegration, mean reversion opportunities, statistical mispricing,
volatility arbitrage, and dispersion trades across correlated assets.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class StatisticalArbitrageAgent(BaseAgent):
    name = "Statistical Arbitrage"
    agent_id = 24
    tier = 3
    category = "Quant"
    data_sources = ["yfinance", "binance"]

    # Z-score thresholds for mean reversion
    Z_ENTRY = 2.0
    Z_EXIT = 0.5

    def __init__(self) -> None:
        super().__init__()
        self._zscore_history: dict[str, list[float]] = defaultdict(list)
        self._spread_history: dict[str, list[float]] = defaultdict(list)
        self._cointegration_pairs: dict[str, list[str]] = defaultdict(list)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. Mean reversion z-score
        z_score = features.get("mean_reversion_z", 0.0)
        if z_score > self.Z_ENTRY:
            score -= 0.08
            reasons.append(f"Extended above mean (z={z_score:.1f}) — mean reversion short")
        elif z_score < -self.Z_ENTRY:
            score += 0.08
            reasons.append(f"Extended below mean (z={z_score:.1f}) — mean reversion long")
        elif abs(z_score) < self.Z_EXIT:
            reasons.append(f"Near mean (z={z_score:.1f}) — no edge")

        # 2. Half-life of mean reversion
        half_life = features.get("mean_reversion_half_life", 20)
        if half_life < 5:
            score += (0.04 if z_score < -1 else -0.04 if z_score > 1 else 0)
            reasons.append(f"Fast reversion (t½={half_life:.0f} bars) — short-lived opportunity")
        elif half_life > 50:
            reasons.append(f"Slow reversion (t½={half_life:.0f} bars) — patience required")

        # 3. Pairs/cointegration signal
        coint_strength = features.get("cointegration_strength", 0.0)
        pair_z = features.get("pair_spread_z", 0.0)
        if coint_strength > 0.9 and abs(pair_z) > 2.0:
            score += (0.06 if pair_z < 0 else -0.06)
            direction = "long spread" if pair_z < 0 else "short spread"
            reasons.append(f"Cointegrated pair divergence (z={pair_z:.1f}) — {direction}")

        # 4. Volatility mean reversion
        vol_z = features.get("volatility_z", 0.0)
        vol_half_life = features.get("vol_half_life", 15)
        if vol_z > 2.0 and vol_half_life < 10:
            score += 0.04
            reasons.append(f"Vol spike mean-reverting (z={vol_z:.1f}, t½={vol_half_life:.0f}) — sell vol")

        # 5. Hurst exponent (trending vs mean-reverting)
        hurst = features.get("hurst_exponent", 0.5)
        if hurst < 0.4:
            reasons.append(f"Mean-reverting regime (H={hurst:.2f}) — fade moves")
        elif hurst > 0.6:
            reasons.append(f"Trending regime (H={hurst:.2f}) — follow, don't fade")

        # 6. Dispersion trading
        dispersion = features.get("index_dispersion", 0.0)
        disp_z = features.get("dispersion_z", 0.0)
        if disp_z > 2.0:
            score -= 0.03
            reasons.append(f"High dispersion (z={disp_z:.1f}) — correlation breakdown")
        elif disp_z < -2.0:
            score += 0.03
            reasons.append(f"Low dispersion (z={disp_z:.1f}) — compression, potential breakout")

        # 7. Statistical edge
        edge_score = features.get("stat_edge_score", 0.0)
        if edge_score > 0.6:
            score += 0.04
            reasons.append(f"Statistical edge present ({edge_score:.2f})")
        elif edge_score < 0.3:
            score -= 0.02
            reasons.append(f"No statistical edge ({edge_score:.2f})")

        # 8. Transaction cost adjustment
        spread_cost = features.get("bid_ask_spread_pct", 0.1)
        expected_return = features.get("expected_return_pct", 0.5)
        net_return = expected_return - spread_cost * 2
        if net_return < 0:
            score -= 0.03
            reasons.append(f"Costs exceed edge (net={net_return:+.2f}%) — untradeable")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.04 else "bearish" if score < -0.04 else "neutral"
        confidence = min(0.72, 0.32 + abs(score) * 0.45)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "No statistical edge detected",
            supporting_features=[
                f"z_score={z_score:.2f}",
                f"hurst={hurst:.3f}",
                f"coint={coint_strength:.3f}",
                f"half_life={half_life:.0f}",
            ],
        )

    def compute_hurst(self, prices: list[float], max_lag: int = 50) -> float:
        """Estimate Hurst exponent using rescaled range (R/S) analysis."""
        n = len(prices)
        if n < max_lag * 2:
            return 0.5

        returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, n)]
        lags = list(range(10, min(max_lag, n // 2), 5))
        rs_values: list[float] = []

        for lag in lags:
            segments = n // lag
            rs_seg: list[float] = []
            for s in range(segments):
                chunk = returns[s * lag:(s + 1) * lag]
                if len(chunk) < 5:
                    continue
                mean = sum(chunk) / len(chunk)
                deviations = [c - mean for c in chunk]
                cum_dev = []
                running = 0.0
                for d in deviations:
                    running += d
                    cum_dev.append(running)
                r = max(cum_dev) - min(cum_dev)
                stdev = math.sqrt(sum(d * d for d in deviations) / len(deviations))
                if stdev > 0:
                    rs_seg.append(r / stdev)
            if rs_seg:
                rs_values.append(sum(rs_seg) / len(rs_seg))

        if len(rs_values) < 3 or len(lags) < 3:
            return 0.5

        log_lags = [math.log(l) for l in lags[:len(rs_values)]]
        log_rs = [math.log(rs) for rs in rs_values]

        n_pts = len(log_lags)
        slope = (n_pts * sum(l * r for l, r in zip(log_lags, log_rs)) -
                 sum(log_lags) * sum(log_rs)) / \
                (n_pts * sum(l * l for l in log_lags) - sum(log_lags) ** 2)

        return max(0.1, min(0.9, slope))

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            self._zscore_history.setdefault(o.prediction.symbol, []).append(
                o.prediction.confidence if o.was_correct else -o.prediction.confidence
            )
            if len(self._zscore_history[o.prediction.symbol]) > 100:
                self._zscore_history[o.prediction.symbol] = \
                    self._zscore_history[o.prediction.symbol][-100:]
