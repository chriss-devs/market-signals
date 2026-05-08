"""Alert/Recommendation Agent — structured trade recommendations.

Synthesizes all agent outputs into actionable recommendations with entry/exit
levels, position sizing guidance, stop-loss placement, and risk parameters.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class AlertRecommendationAgent(BaseAgent):
    name = "Alert/Recommendation"
    agent_id = 22
    tier = 2
    category = "Signal"
    data_sources = ["all"]

    # Risk thresholds
    MAX_POSITION_PCT = 0.25  # Max 25% of portfolio in one position
    BASE_POSITION_PCT = 0.05  # Base position size
    MAX_LEVERAGE = 3.0

    def __init__(self) -> None:
        super().__init__()
        self._recommendation_history: list[dict] = []
        self._alert_counts: dict[str, int] = defaultdict(int)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. Consensus strength
        consensus_strength = features.get("consensus_strength", 0.0)
        agent_agreement = features.get("agent_agreement_ratio", 0.5)
        if consensus_strength > 0.7 and agent_agreement > 0.7:
            score += 0.10
            reasons.append(f"Strong consensus ({consensus_strength:.0%}) with high agreement ({agent_agreement:.0%})")
        elif consensus_strength > 0.5:
            score += 0.05
            reasons.append(f"Moderate consensus ({consensus_strength:.0%})")
        elif agent_agreement < 0.4:
            score -= 0.05
            reasons.append(f"Low agent agreement ({agent_agreement:.0%}) — divided")

        # 2. Signal confidence tier
        signal_tier = features.get("signal_tier", 2)
        if signal_tier == 1:
            score += 0.08
            reasons.append("Tier-1 signal (high confidence)")
        elif signal_tier == 4:
            score -= 0.04
            reasons.append("Tier-4 signal (low confidence)")

        # 3. Risk/reward profile
        rr_ratio = features.get("risk_reward_ratio", 2.0)
        if rr_ratio > 3.0:
            score += 0.06
            reasons.append(f"Excellent R:R ({rr_ratio:.1f}:1)")
        elif rr_ratio > 2.0:
            score += 0.03
            reasons.append(f"Good R:R ({rr_ratio:.1f}:1)")
        elif rr_ratio < 1.0:
            score -= 0.05
            reasons.append(f"Poor R:R ({rr_ratio:.1f}:1) — skip")

        # 4. Position sizing recommendation
        position_pct = features.get("recommended_position_pct", self.BASE_POSITION_PCT)
        if position_pct > 0.15:
            reasons.append(f"Large position recommended ({position_pct:.0%}) — high conviction")
        elif position_pct < 0.03:
            reasons.append(f"Small position only ({position_pct:.0%}) — low conviction")

        # 5. Stop-loss distance
        stop_distance = features.get("stop_loss_pct", 5.0)
        if stop_distance < 2.0:
            score += 0.02
            reasons.append(f"Tight stop ({stop_distance:.1f}%) — well-defined risk")
        elif stop_distance > 10.0:
            score -= 0.03
            reasons.append(f"Wide stop ({stop_distance:.1f}%) — imprecise entry")

        # 6. Multi-timeframe alignment
        tf_alignment = features.get("timeframe_alignment", 0.0)
        if tf_alignment > 0.6:
            score += 0.05
            reasons.append(f"Multi-TF aligned ({tf_alignment:.0%})")
        elif tf_alignment < -0.3:
            score -= 0.04
            reasons.append(f"Multi-TF conflict ({tf_alignment:.0%})")

        # 7. Correlation to existing positions (portfolio context)
        portfolio_corr = features.get("portfolio_correlation", 0.0)
        if portfolio_corr > 0.8:
            score -= 0.03
            reasons.append(f"High portfolio correlation ({portfolio_corr:.2f}) — concentration risk")
        elif portfolio_corr < 0.3:
            score += 0.02
            reasons.append(f"Low portfolio correlation ({portfolio_corr:.2f}) — diversification")

        # 8. Recent alert frequency
        recent_alerts = features.get("recent_alerts_24h", 0)
        if recent_alerts > 5:
            score -= 0.04
            reasons.append(f"Alert fatigue ({recent_alerts} in 24h) — reduce sensitivity")

        # 9. Volatility-adjusted position
        vol_adjusted = features.get("vol_adjusted_position", 0.0)
        if vol_adjusted > 0.7:
            score += 0.03
            reasons.append(f"Vol-adjusted sizing favorable ({vol_adjusted:.2f})")
        elif vol_adjusted < 0.3:
            score -= 0.03
            reasons.append(f"Vol-adjusted sizing unfavorable ({vol_adjusted:.2f})")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.04 else "bearish" if score < -0.04 else "neutral"
        confidence = min(0.78, 0.35 + abs(score) * 0.50)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "No actionable recommendation",
            supporting_features=[
                f"consensus={consensus_strength:.2f}",
                f"rr={rr_ratio:.1f}",
                f"pos={position_pct:.0%}",
                f"tf_align={tf_alignment:.2f}",
            ],
        )

    def generate_recommendation(self, features: dict) -> dict:
        """Generate a structured trade recommendation."""
        rr = features.get("risk_reward_ratio", 2.0)
        consensus = features.get("consensus_strength", 0.5)
        vol = features.get("annualized_volatility", 0.3)

        # Position size: Kelly-inspired with vol adjustment
        base_pct = self.BASE_POSITION_PCT
        kelly_adjustment = max(0.2, min(2.0, rr / 2.0))
        vol_adjustment = max(0.3, min(1.5, 0.3 / vol))
        position_pct = round(base_pct * kelly_adjustment * vol_adjustment * consensus * 2, 3)

        return {
            "position_pct": min(position_pct, self.MAX_POSITION_PCT),
            "stop_loss_pct": round(vol * 1.5 * 100, 1),
            "take_profit_pct": round(vol * rr * 100, 1),
            "max_leverage": min(self.MAX_LEVERAGE, max(1.0, round(1.0 / vol, 1))),
            "confidence": round(consensus, 3),
        }

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            key = o.prediction.direction
            self._alert_counts[key] += 1
            self._recommendation_history.append({
                "direction": o.prediction.direction,
                "was_correct": o.was_correct,
                "confidence": o.prediction.confidence,
            })
            if len(self._recommendation_history) > 500:
                self._recommendation_history = self._recommendation_history[-500:]
