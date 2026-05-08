"""Geopolitical Agent — geopolitical risk assessment for markets.

Scores geopolitical events and tensions for market impact: wars, sanctions,
trade disputes, elections, regulatory actions, and energy security events.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class GeopoliticalAgent(BaseAgent):
    name = "Geopolitical"
    agent_id = 25
    tier = 3
    category = "Macro"
    data_sources = ["news_api", "finnhub"]

    # Event type impact weights
    EVENT_WEIGHTS = {
        "war": 1.0,
        "sanctions": 0.85,
        "trade_war": 0.8,
        "election": 0.7,
        "coup": 0.9,
        "terrorism": 0.75,
        "regulatory_crackdown": 0.7,
        "central_bank_emergency": 0.85,
        "energy_crisis": 0.8,
        "debt_default": 0.9,
        "diplomatic_break": 0.65,
        "border_conflict": 0.75,
    }

    # Region risk baselines (0-1)
    REGION_BASELINE = {
        "north_america": 0.15,
        "europe": 0.25,
        "east_asia": 0.35,
        "middle_east": 0.50,
        "south_asia": 0.40,
        "latin_america": 0.35,
        "africa": 0.45,
        "oceania": 0.15,
    }

    def __init__(self) -> None:
        super().__init__()
        self._event_cache: list[dict] = []
        self._risk_history: dict[str, list[float]] = defaultdict(list)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. Geopolitical risk index (GPR)
        gpr = features.get("geopolitical_risk_index", 0.25)
        gpr_change = features.get("gpr_change_30d", 0.0)
        if gpr > 0.5:
            score -= 0.06
            reasons.append(f"GPR elevated ({gpr:.2f}) — geopolitical uncertainty")
        elif gpr < 0.2:
            score += 0.03
            reasons.append(f"GPR low ({gpr:.2f}) — stable backdrop")

        if gpr_change > 0.1:
            score -= 0.04
            reasons.append(f"GPR rising ({gpr_change:+.2f}) — tension escalating")

        # 2. Active conflict impact
        conflict_score = features.get("conflict_impact_score", 0.0)
        if conflict_score > 0.6:
            score -= 0.07
            reasons.append(f"Active conflict impact high ({conflict_score:.2f})")

        # 3. Trade war / sanctions risk
        trade_tension = features.get("trade_tension_score", 0.0)
        if trade_tension > 0.5:
            score -= 0.05
            reasons.append(f"Trade tension elevated ({trade_tension:.2f}) — supply chain risk")
        elif trade_tension < 0.15:
            score += 0.02
            reasons.append(f"Trade environment calm ({trade_tension:.2f})")

        # 4. Energy security
        energy_risk = features.get("energy_security_risk", 0.0)
        if energy_risk > 0.6:
            score -= 0.05
            reasons.append(f"Energy security risk ({energy_risk:.2f}) — commodity shock potential")
        elif energy_risk < 0.2:
            score += 0.02

        # 5. Regulatory risk (crypto-specific)
        crypto_reg_risk = features.get("crypto_regulatory_risk", 0.3)
        reg_change = features.get("reg_risk_change", 0.0)
        if crypto_reg_risk > 0.5:
            score -= 0.06
            reasons.append(f"Crypto regulatory risk high ({crypto_reg_risk:.2f})")
        elif crypto_reg_risk < 0.2:
            score += 0.03
            reasons.append(f"Regulatory clarity improving ({crypto_reg_risk:.2f})")

        if reg_change > 0.05:
            score -= 0.03
            reasons.append(f"Regulatory tightening ({reg_change:+.2f})")

        # 6. Election / political uncertainty
        election_proximity = features.get("election_proximity_days", 365)
        election_uncertainty = features.get("election_uncertainty", 0.0)
        if election_proximity < 30 and election_uncertainty > 0.5:
            score -= 0.04
            reasons.append(f"Election in {election_proximity:.0f}d — policy uncertainty")

        # 7. Currency crisis risk
        currency_risk = features.get("currency_crisis_risk", 0.0)
        if currency_risk > 0.4:
            score -= 0.04
            reasons.append(f"Currency instability risk ({currency_risk:.2f}) — capital controls possible")

        # 8. Cyber security events
        cyber_threat = features.get("cyber_threat_level", 0.0)
        if cyber_threat > 0.6:
            score -= 0.03
            reasons.append(f"Cyber threat elevated ({cyber_threat:.2f}) — infrastructure risk")

        score = max(-1.0, min(1.0, score))

        # Geopolitical agent is typically risk-off biased; confidence is for bearish calls
        direction = "bearish" if score < -0.03 else "bullish" if score > 0.06 else "neutral"
        confidence = min(0.70, 0.33 + abs(score) * 0.42)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "Geopolitical environment stable",
            supporting_features=[
                f"gpr={gpr:.3f}",
                f"trade_tension={trade_tension:.2f}",
                f"crypto_reg={crypto_reg_risk:.2f}",
                f"energy_risk={energy_risk:.2f}",
            ],
        )

    def compute_gpr(self, events: list[dict]) -> float:
        """Compute a geopolitical risk index from event list."""
        if not events:
            return 0.15
        total_weight = 0.0
        for ev in events:
            event_type = ev.get("type", "")
            proximity = ev.get("proximity_days", 30)
            severity = ev.get("severity", 0.5)

            weight = self.EVENT_WEIGHTS.get(event_type, 0.5)
            decay = max(0.05, 1.0 - proximity / 90)
            total_weight += weight * severity * decay

        return round(min(1.0, total_weight / max(len(events), 1)), 3)

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            key = o.prediction.symbol
            self._risk_history.setdefault(key, []).append(
                1.0 if o.was_correct else 0.0
            )
            if len(self._risk_history[key]) > 100:
                self._risk_history[key] = self._risk_history[key][-100:]
