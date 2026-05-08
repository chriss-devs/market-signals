"""Intermarket Agent — cross-asset relationship analysis.

Analyzes classical intermarket relationships: USD vs crypto, bonds vs equities,
gold vs dollar, commodity prices vs currencies. Detects regime shifts and
divergences that signal turning points.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class IntermarketAgent(BaseAgent):
    name = "Intermarket"
    agent_id = 17
    tier = 2
    category = "Cross-Market"
    data_sources = ["yfinance", "finnhub"]

    # DXY thresholds for risk-on/off classification
    DXY_STRONG = 105.0
    DXY_WEAK = 100.0

    def __init__(self) -> None:
        super().__init__()
        self._regime_scores: dict[str, list[float]] = defaultdict(list)
        self._divergence_history: list[dict] = []

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. DXY (USD strength) — inverse to crypto/commodities
        dxy = features.get("dxy", 100.0)
        dxy_change = features.get("dxy_change_30d", 0.0)
        if dxy > self.DXY_STRONG and dxy_change > 0:
            score -= 0.07
            reasons.append(f"DXY strong ({dxy:.1f}) and rising — USD headwind")
        elif dxy < self.DXY_WEAK and dxy_change < 0:
            score += 0.08
            reasons.append(f"DXY weak ({dxy:.1f}) and falling — USD tailwind")
        elif dxy_change > 2.0:
            score -= 0.05
            reasons.append(f"DXY surging ({dxy_change:+.1f}%) — risk-off pressure")
        elif dxy_change < -2.0:
            score += 0.05
            reasons.append(f"DXY dropping ({dxy_change:+.1f}%) — risk-on catalyst")

        # 2. Treasury yields (10Y)
        t10y = features.get("us10y_yield", 4.0)
        t10y_change = features.get("us10y_change_30d", 0.0)
        if t10y_change < -0.3:
            score += 0.04
            reasons.append(f"10Y yield falling ({t10y_change:+.1f}%) — dovish sentiment")
        elif t10y_change > 0.3:
            score -= 0.04
            reasons.append(f"10Y yield rising ({t10y_change:+.1f}%) — hawkish pressure")

        # 3. Yield curve (2s10s spread)
        yc_spread = features.get("yield_curve_spread", 0.0)
        if yc_spread < -0.5:
            score -= 0.06
            reasons.append(f"Yield curve deeply inverted ({yc_spread:.2f}%) — recession signal")
        elif yc_spread > 1.0:
            score += 0.04
            reasons.append(f"Yield curve steepening ({yc_spread:.2f}%) — growth optimism")

        # 4. Gold (safe haven flow)
        gold_change = features.get("gold_change_30d", 0.0)
        if gold_change > 5.0:
            score -= 0.04
            reasons.append(f"Gold surging ({gold_change:+.1f}%) — safe haven demand")
        elif gold_change < -3.0:
            score += 0.03
            reasons.append(f"Gold declining ({gold_change:+.1f}%) — risk appetite")

        # 5. VIX (fear gauge)
        vix = features.get("vix", 18.0)
        vix_change = features.get("vix_change_5d", 0.0)
        if vix > 30:
            score -= 0.08
            reasons.append(f"VIX elevated ({vix:.0f}) — fear regime")
        elif vix < 15:
            score += 0.05
            reasons.append(f"VIX low ({vix:.0f}) — complacency/stable")
        if vix_change > 20:
            score -= 0.06
            reasons.append(f"VIX spiking ({vix_change:+.0f}%) — panic event")

        # 6. Oil (inflation proxy)
        oil_change = features.get("oil_change_30d", 0.0)
        if oil_change > 10.0:
            score -= 0.05
            reasons.append(f"Oil surging ({oil_change:+.1f}%) — inflation risk")
        elif oil_change < -10.0:
            score += 0.04
            reasons.append(f"Oil plunging ({oil_change:+.1f}%) — disinflation tailwind")

        # 7. Risk-on/risk-off composite
        roro = features.get("risk_on_off_score", 0.0)
        if roro > 0.3:
            score += 0.06
            reasons.append(f"Risk-on regime ({roro:.2f}) — carry works")
        elif roro < -0.3:
            score -= 0.06
            reasons.append(f"Risk-off regime ({roro:.2f}) — capital preservation")

        # 8. Intermarket divergence
        divergence = features.get("intermarket_divergence", 0.0)
        if abs(divergence) > 0.4:
            score += 0.04 if divergence > 0 else -0.04
            reasons.append(f"Intermarket divergence ({divergence:+.2f}) — potential reversal")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.05 else "bearish" if score < -0.05 else "neutral"
        confidence = min(0.75, 0.35 + abs(score) * 0.45)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "Intermarket signals balanced",
            supporting_features=[
                f"dxy={dxy:.1f}",
                f"us10y={t10y:.2f}%",
                f"vix={vix:.0f}",
                f"roro={roro:.2f}",
            ],
        )

    def classify_regime(self, features: dict) -> str:
        """Classify the current intermarket regime."""
        dxy = features.get("dxy", 100)
        vix = features.get("vix", 18)
        yc = features.get("yield_curve_spread", 0)

        if vix > 30:
            return "crisis"
        if dxy > 105 and yc < 0:
            return "deflationary"
        if dxy < 100 and yc > 0.5:
            return "reflation"
        if vix < 15:
            return "complacency"
        return "transitional"

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            conf = o.prediction.confidence
            self._regime_scores.setdefault(o.prediction.direction, []).append(
                1.0 if o.was_correct else 0.0
            )
            if len(self._regime_scores[o.prediction.direction]) > 100:
                self._regime_scores[o.prediction.direction] = \
                    self._regime_scores[o.prediction.direction][-100:]
