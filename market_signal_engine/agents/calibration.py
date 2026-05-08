"""Calibration Agent — ensures predicted confidence matches realized accuracy.

Self-improvement: adjusts each agent's confidence scaling factor.
Implements reliability scoring: "70% confidence = 70% correct".
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class CalibrationAgent(BaseAgent):
    name = "Calibration"
    agent_id = 15
    tier = 1
    category = "Validation"
    data_sources = ["prediction_outcomes"]

    # Confidence bucket edges for reliability analysis
    BUCKETS = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

    def __init__(self) -> None:
        super().__init__()
        # Per-agent calibration data
        self._buckets: dict[str, dict[tuple, dict]] = defaultdict(
            lambda: {b: {"total": 0, "correct": 0, "accuracy": 0.0} for b in self.BUCKETS}
        )
        # Per-agent scaling factor (multiplier applied to raw confidence)
        self._scaling_factors: dict[str, float] = defaultdict(lambda: 1.0)
        # Global reliability score per agent
        self._reliability: dict[str, float] = defaultdict(lambda: 0.5)

    def analyze(self, context: AnalysisContext) -> Prediction:
        """Calibration doesn't predict direction — it evaluates prediction quality."""
        features = context.features.features
        sym = context.symbol

        calibration_error = features.get("avg_calibration_error", 0.0)
        total_predictions = int(features.get("total_predictions", 0))

        reasons: list[str] = []
        score = 0.0

        if total_predictions < 20:
            return Prediction(
                agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
                direction="neutral", confidence=0.3,
                reasoning="Insufficient data for calibration (<20 predictions)",
            )

        # Calibration error: |confidence - realized_accuracy|
        # Low error = well-calibrated
        if calibration_error < 0.05:
            score += 0.15
            reasons.append(f"System well-calibrated (error={calibration_error:.3f})")
        elif calibration_error < 0.10:
            score += 0.05
            reasons.append(f"System reasonably calibrated (error={calibration_error:.3f})")
        elif calibration_error > 0.20:
            score -= 0.12
            reasons.append(f"System poorly calibrated (error={calibration_error:.3f}) — overconfident")
        elif calibration_error > 0.15:
            score -= 0.06
            reasons.append(f"Calibration degrading (error={calibration_error:.3f})")

        # Detected bias: are we systematically overconfident or underconfident?
        overconfidence = features.get("overconfidence_ratio", 0.0)
        if overconfidence > 0.1:
            score -= 0.08
            reasons.append(f"Overconfident: predicted conf > realized accuracy ({overconfidence:+.1%})")
        elif overconfidence < -0.1:
            score += 0.06
            reasons.append(f"Underconfident: predicted conf < realized accuracy ({overconfidence:+.1%})")

        # Reliability score
        reliability = features.get("reliability", 0.5)
        if reliability > 0.7:
            score += 0.08
            reasons.append(f"High reliability ({reliability:.0%})")
        elif reliability < 0.4:
            score -= 0.06
            reasons.append(f"Low reliability ({reliability:.0%})")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.03 else "bearish" if score < -0.03 else "neutral"
        confidence = min(0.70, 0.35 + abs(score) * 0.4)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "Calibration data neutral",
            supporting_features=[
                f"cal_error={calibration_error:.4f}",
                f"overconfidence={overconfidence:.3f}",
                f"reliability={reliability:.3f}",
            ],
        )

    # ── Calibration computation methods ─────────────────────────────────

    def update_reliability(
        self, agent_name: str, outcomes: list[PredictionOutcome]
    ) -> dict:
        """Compute reliability diagram data and scaling factor for an agent."""
        buckets = self._buckets[agent_name]

        # Reset bucket stats
        for b in buckets:
            buckets[b] = {"total": 0, "correct": 0, "accuracy": 0.0}

        for o in outcomes:
            if o.prediction.agent_name != agent_name:
                continue
            conf = o.prediction.confidence
            for (lo, hi) in self.BUCKETS:
                if lo <= conf < hi or (hi == 1.0 and lo <= conf <= hi):
                    buckets[(lo, hi)]["total"] += 1
                    if o.was_correct:
                        buckets[(lo, hi)]["correct"] += 1
                    break

        # Compute accuracy per bucket
        for (lo, hi), b in buckets.items():
            if b["total"] > 0:
                b["accuracy"] = b["correct"] / b["total"]

        # Compute calibration error
        total_error = 0.0
        total_weight = 0
        for (lo, hi), b in buckets.items():
            if b["total"] > 0:
                mid = (lo + hi) / 2
                error = abs(mid - b["accuracy"])
                total_error += error * b["total"]
                total_weight += b["total"]

        avg_error = total_error / max(total_weight, 1)

        # Compute scaling factor using simple Platt-style adjustment
        # If predicted confidence > realized accuracy, scale down
        total_pred = sum(b["total"] for b in buckets.values())
        total_correct = sum(b["correct"] for b in buckets.values())
        overall_acc = total_correct / max(total_pred, 1)

        avg_pred_conf = sum(
            (lo + hi) / 2 * b["total"] for (lo, hi), b in buckets.items()
        ) / max(total_pred, 1) if total_pred > 0 else 0.5

        if avg_pred_conf > 0:
            # Scale to align predictions with accuracy
            self._scaling_factors[agent_name] = max(
                0.5, min(1.5, overall_acc / avg_pred_conf)
            )

        self._reliability[agent_name] = max(0.0, min(1.0, 1.0 - avg_error))

        return {
            "agent": agent_name,
            "calibration_error": round(avg_error, 4),
            "reliability": round(self._reliability[agent_name], 4),
            "scaling_factor": round(self._scaling_factors[agent_name], 4),
            "overall_accuracy": round(overall_acc, 4),
            "avg_predicted_confidence": round(avg_pred_conf, 4),
            "buckets": {
                f"{lo:.1f}-{hi:.1f}": {
                    "total": b["total"],
                    "accuracy": round(b["accuracy"], 3),
                }
                for (lo, hi), b in buckets.items()
                if b["total"] > 0
            },
        }

    def get_scaling_factor(self, agent_name: str) -> float:
        """Get the confidence scaling factor for an agent."""
        return self._scaling_factors.get(agent_name, 1.0)

    def get_reliability(self, agent_name: str) -> float:
        """Get the reliability score for an agent (0-1)."""
        return self._reliability.get(agent_name, 0.5)

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        # Group outcomes by agent and update calibration
        by_agent: dict[str, list[PredictionOutcome]] = defaultdict(list)
        for o in outcomes:
            by_agent[o.prediction.agent_name].append(o)
        for agent_name, agent_outcomes in by_agent.items():
            if len(agent_outcomes) >= 20:
                self.update_reliability(agent_name, agent_outcomes)
