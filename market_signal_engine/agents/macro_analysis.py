"""Macro Analysis Agent — yield curve, DXY, VIX, gold, inflation expectations.

Self-improvement: weighs macro factors differently per asset class.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class MacroAnalysisAgent(BaseAgent):
    name = "Macro Analysis"
    agent_id = 4
    tier = 1
    category = "Macro"
    data_sources = ["fred", "stooq", "exchangerate"]

    # Asset class mapping for factor relevance
    CRYPTO = {"BTC", "ETH", "SOL", "DOGE", "crypto"}
    STOCKS = {"AAPL", "NVDA", "TSLA", "MSFT", "SPX", "stock"}

    def __init__(self) -> None:
        super().__init__()
        # Per-factor weights per asset class — tuned over time
        self._factor_weights: dict[str, dict[str, float]] = {
            "crypto": {
                "dxy": 0.25, "yield_curve": 0.10, "vix": 0.12,
                "gold": 0.05, "inflation": 0.15, "fed_funds": 0.08,
                "global_pmi": 0.05, "usd_flows": 0.20,
            },
            "stock": {
                "dxy": 0.10, "yield_curve": 0.25, "vix": 0.18,
                "gold": 0.02, "inflation": 0.15, "fed_funds": 0.15,
                "global_pmi": 0.10, "usd_flows": 0.05,
            },
        }
        self._default_weights = self._factor_weights["crypto"]

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        # Extract macro indicators from features
        dxy = features.get("dxy_change", 0.0)           # DXY % change (negative = risk-on)
        t10y2y = features.get("t10y2y_spread", 0.0)     # Yield curve (negative = inversion)
        t10y3m = features.get("t10y3m_spread", 0.0)     # Short-end spread
        vix = features.get("vix", 20.0)                  # VIX level
        vix_change = features.get("vix_change", 0.0)     # VIX % change
        gold = features.get("gold_change", 0.0)          # Gold % change
        inflation_exp = features.get("inflation_expectation", 0.0)  # 5Y5Y breakeven
        fed_rate = features.get("fed_funds_rate", 5.25)  # Fed funds rate
        global_pmi = features.get("global_pmi", 50.0)    # PMI > 50 = expansion
        usd_flows = features.get("usd_flows", 0.0)       # Net USD inflow/outflow

        # Determine asset class
        asset_class = "crypto" if self._is_crypto(sym) else "stock"
        weights = self._factor_weights.get(asset_class, self._default_weights)

        reasons: list[str] = []

        # ── Score each factor ───────────────────────────────────────────
        # DXY (inverse relationship with risk assets)
        dxy_score = 0.0
        if dxy < -0.3:
            dxy_score = 0.15
            reasons.append(f"DXY weakening ({dxy:+.1f}%) — risk-on")
        elif dxy > 0.3:
            dxy_score = -0.15
            reasons.append(f"DXY strengthening ({dxy:+.1f}%) — risk-off")

        # Yield curve
        curve_score = 0.0
        if t10y2y > 0.1 and t10y3m > 0:  # Steepening
            curve_score = -0.10
            reasons.append(f"Yield curve steepening (2Y={t10y2y:.2f}%) — tightening risk")
        elif t10y2y < -0.1:  # Still inverted
            curve_score = -0.12
            reasons.append(f"Yield curve inverted ({t10y2y:.2f}%) — recession signal")
        elif 0 < t10y2y <= 0.1:
            curve_score = 0.05
            reasons.append("Yield curve normalizing")

        # VIX
        vix_score = 0.0
        if vix < 15:
            vix_score = 0.10
            reasons.append(f"VIX low ({vix:.0f}) — complacency / risk-on")
        elif vix > 25:
            vix_score = -0.18
            reasons.append(f"VIX elevated ({vix:.0f}) — fear / risk-off")
        elif vix > 20:
            vix_score = -0.08
            reasons.append(f"VIX above average ({vix:.0f})")

        # Gold (safe haven)
        gold_score = 0.0
        if gold > 1.0:
            gold_score = -0.06
            reasons.append(f"Gold rising ({gold:+.1f}%) — risk-off flows")
        elif gold < -0.5:
            gold_score = 0.05
            reasons.append(f"Gold declining ({gold:+.1f}%) — risk appetite")

        # Inflation
        infl_score = 0.0
        if inflation_exp < 2.2:
            infl_score = 0.08
            reasons.append(f"Inflation expectations anchored ({inflation_exp:.1f}%)")
        elif inflation_exp > 2.8:
            infl_score = -0.10
            reasons.append(f"Inflation expectations elevated ({inflation_exp:.1f}%)")

        # Global PMI
        pmi_score = 0.0
        if global_pmi > 52:
            pmi_score = 0.10
            reasons.append(f"Global PMI expanding ({global_pmi:.1f})")
        elif global_pmi < 48:
            pmi_score = -0.10
            reasons.append(f"Global PMI contracting ({global_pmi:.1f})")

        # USD flows
        flow_score = 0.0
        if usd_flows > 0.5:
            flow_score = -0.08
            reasons.append("USD inflows — dollar strength")
        elif usd_flows < -0.5:
            flow_score = 0.08
            reasons.append("USD outflows — risk asset positive")

        # ── Weighted aggregate ──────────────────────────────────────────
        raw_score = (
            weights["dxy"] * dxy_score
            + weights["yield_curve"] * curve_score
            + weights["vix"] * vix_score
            + weights["gold"] * gold_score
            + weights["inflation"] * infl_score
            + weights["fed_funds"] * (0.03 if fed_rate < 4.5 else -0.03)
            + weights["global_pmi"] * pmi_score
            + weights["usd_flows"] * flow_score
        )

        score = max(-1.0, min(1.0, raw_score))

        if score > 0.06:
            direction = "bullish"
            confidence = min(0.75, 0.45 + abs(score) * 0.5)
        elif score < -0.06:
            direction = "bearish"
            confidence = min(0.75, 0.45 + abs(score) * 0.5)
        else:
            direction = "neutral"
            confidence = 0.35

        confidence = self.calibrate_confidence(confidence)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons[:5]) if reasons else "Mixed macro signals",
            supporting_features=[
                f"dxy={dxy:.2f}", f"vix={vix:.1f}", f"yield_curve={t10y2y:.2f}",
                f"pmi={global_pmi:.1f}", f"asset_class={asset_class}",
            ],
        )

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            sym = o.prediction.symbol
            asset_class = "crypto" if self._is_crypto(sym) else "stock"
            weights = self._factor_weights.get(asset_class, self._default_weights)
            for feat in o.prediction.supporting_features:
                factor = feat.split("=")[0]
                if factor in weights:
                    delta = 0.02 if o.was_correct else -0.03
                    weights[factor] = max(0.02, min(0.35, weights[factor] + delta))
                impact = 0.04 if o.was_correct else -0.02
                self.update_feature_importance(feat, impact)

    def _is_crypto(self, symbol: str) -> bool:
        return any(c in symbol.upper() for c in self.CRYPTO) or "USDT" in symbol.upper()
