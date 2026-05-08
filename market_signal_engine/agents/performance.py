"""Agent performance tracking with EMA decay and dynamic weight calculation.

Weight formula (per market category, from the plan):
    weight_i = accuracy_i² / sum(all accuracy_j²)

Dispersion penalty:
    dispersion = abs(up_ratio - 0.5) * 2   # 0=agree, 1=split
    penalty = dispersion * 0.2              # max 20% confidence reduction
    final_confidence = weighted_confidence * (1 - penalty)

EMA accuracy:
    accuracy_ema = prev_accuracy * 0.9 + (was_correct ? 1 : 0) * 0.1
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from market_signal_engine.agents.base import AgentPerformance, Prediction, PredictionOutcome


@dataclass
class ConsensusResult:
    direction: str  # bullish / bearish / neutral
    confidence: float
    dispersion: float
    penalty: float
    agent_weights: dict[str, float]
    vote_tally: dict[str, int]


class AgentPerformanceTracker:
    """Tracks per-agent accuracy EMA and computes dynamic consensus weights.

    Key behaviors:
    - Accuracy is tracked as EMA with decay=0.9 per agent
    - Weights are accuracy² / sum(accuracy²) per category
    - Dispersion penalty reduces confidence when agents disagree
    """

    def __init__(self, ema_decay: float = 0.9) -> None:
        self._decay = ema_decay
        self._accuracy_ema: dict[str, float] = {}  # agent_name → EMA accuracy
        self._correct_counts: dict[str, int] = {}
        self._total_counts: dict[str, int] = {}
        self._weights: dict[str, float] = {}  # agent_name → current weight
        # Per-category tracking
        self._category_accuracy: dict[str, dict[str, float]] = defaultdict(dict)

    # ── Accuracy tracking ────────────────────────────────────────────────

    def record_outcome(self, agent_name: str, was_correct: bool) -> None:
        ema = self._accuracy_ema.get(agent_name, 0.5)
        ema = ema * self._decay + (1.0 if was_correct else 0.0) * (1.0 - self._decay)
        self._accuracy_ema[agent_name] = ema

        self._total_counts[agent_name] = self._total_counts.get(agent_name, 0) + 1
        if was_correct:
            self._correct_counts[agent_name] = self._correct_counts.get(agent_name, 0) + 1

    def batch_record(self, outcomes: list[PredictionOutcome]) -> None:
        for outcome in outcomes:
            self.record_outcome(outcome.prediction.agent_name, outcome.was_correct)

    def get_accuracy(self, agent_name: str) -> float:
        return self._accuracy_ema.get(agent_name, 0.5)

    def get_raw_accuracy(self, agent_name: str) -> float:
        total = self._total_counts.get(agent_name, 0)
        if total == 0:
            return 0.5
        return self._correct_counts.get(agent_name, 0) / total

    # ── Weight calculation ───────────────────────────────────────────────

    def recalculate_weights(self, agent_names: list[str], category: str | None = None) -> dict[str, float]:
        """Compute weights as accuracy_i² / sum(all accuracy_j²).

        If category is provided, weights are computed within that category only.
        """
        if not agent_names:
            return {}

        accuracies: dict[str, float] = {}
        for name in agent_names:
            if category and name in self._category_accuracy.get(category, {}):
                acc = self._category_accuracy[category][name]
            else:
                acc = self._accuracy_ema.get(name, 0.5)
            accuracies[name] = acc

        # accuracy²
        squared = {name: acc**2 for name, acc in accuracies.items()}
        total_sq = sum(squared.values())

        if total_sq == 0:
            # Uniform fallback
            w = 1.0 / len(agent_names)
            weights = {name: w for name in agent_names}
        else:
            weights = {name: sq / total_sq for name, sq in squared.items()}

        self._weights.update(weights)
        return weights

    def get_weight(self, agent_name: str) -> float:
        return self._weights.get(agent_name, 0.0)

    def get_all_weights(self) -> dict[str, float]:
        return dict(self._weights)

    # ── Consensus computation ────────────────────────────────────────────

    def compute_consensus(self, predictions: list[Prediction]) -> ConsensusResult:
        """Compute weighted consensus from a set of agent predictions.

        Uses the plan's formula:
            dispersion = abs(up_ratio - 0.5) * 2
            penalty = dispersion * 0.2
            final_confidence = weighted_confidence * (1 - penalty)
        """
        if not predictions:
            return ConsensusResult(
                direction="neutral",
                confidence=0.0,
                dispersion=0.0,
                penalty=0.0,
                agent_weights={},
                vote_tally={},
            )

        # Tally votes by direction
        votes: dict[str, float] = defaultdict(float)  # direction → total weight
        tally: dict[str, int] = defaultdict(int)

        for pred in predictions:
            weight = self._weights.get(pred.agent_name, 1.0 / max(len(predictions), 1))
            votes[pred.direction] += weight * pred.confidence
            tally[pred.direction] += 1

        # Weighted direction
        if not votes:
            direction = "neutral"
            weighted_confidence = 0.5
        else:
            direction = max(votes, key=votes.get)
            weighted_confidence = votes[direction] / sum(votes.values())
            weighted_confidence = max(0.0, min(1.0, weighted_confidence))

        # Dispersion penalty
        bullish_count = tally.get("bullish", 0)
        up_ratio = bullish_count / len(predictions) if predictions else 0.5
        dispersion = abs(up_ratio - 0.5) * 2
        penalty = dispersion * 0.2
        final_confidence = weighted_confidence * (1.0 - penalty)

        return ConsensusResult(
            direction=direction,
            confidence=round(final_confidence, 4),
            dispersion=round(dispersion, 4),
            penalty=round(penalty, 4),
            agent_weights=dict(self._weights),
            vote_tally=dict(tally),
        )

    # ── Category tracking ────────────────────────────────────────────────

    def record_category_outcome(
        self, agent_name: str, category: str, was_correct: bool
    ) -> None:
        ema = self._category_accuracy[category].get(agent_name, 0.5)
        ema = ema * self._decay + (1.0 if was_correct else 0.0) * (1.0 - self._decay)
        self._category_accuracy[category][agent_name] = ema

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self, performances: list[AgentPerformance]) -> list[AgentPerformance]:
        """Update AgentPerformance records with current EMA and weights."""
        for perf in performances:
            perf.accuracy_ema = round(self._accuracy_ema.get(perf.agent_name, perf.accuracy_ema), 4)
            perf.weight = round(self._weights.get(perf.agent_name, 0.0), 4)
        return performances
