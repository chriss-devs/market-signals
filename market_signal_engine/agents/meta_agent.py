"""Meta-Agent — weighted consensus engine.

Consumes predictions from all other agents, applies dynamic weights from the
PerformanceTracker, computes dispersion penalty, and emits the final signal.

From the plan:
    weight_i = accuracy_i² / sum(all accuracy_j²)
    dispersion = abs(up_ratio - 0.5) * 2
    penalty = dispersion * 0.2
    final_confidence = weighted_confidence * (1 - penalty)
"""

from __future__ import annotations

from datetime import datetime, timezone

from market_signal_engine.agents.base import (
    AgentPerformance,
    AnalysisContext,
    BaseAgent,
    Prediction,
    PredictionOutcome,
)
from market_signal_engine.agents.performance import AgentPerformanceTracker, ConsensusResult


class MetaAgent(BaseAgent):
    """Orchestrates all agents, computes weighted consensus, emits final signal.

    Not a traditional analysis agent — it consumes predictions from the other
    25 agents and produces a meta-consensus with confidence calibration.
    """

    name = "Meta-Agent"
    agent_id = 13
    tier = 1
    category = "Meta"
    data_sources = ["all_agent_outputs"]

    def __init__(self, tracker: AgentPerformanceTracker | None = None) -> None:
        super().__init__()
        self._tracker = tracker or AgentPerformanceTracker()
        self._sub_agents: list[BaseAgent] = []
        self._version = "0.2.1"

    def register_sub_agent(self, agent: BaseAgent) -> None:
        self._sub_agents.append(agent)

    def register_sub_agents(self, agents: list[BaseAgent]) -> None:
        self._sub_agents.extend(agents)

    def analyze(self, context: AnalysisContext) -> Prediction:
        """Run all sub-agents on the context and produce weighted consensus."""
        if not self._sub_agents:
            return Prediction(
                agent_name=self.name,
                agent_id=self.agent_id,
                symbol=context.symbol,
                direction="neutral",
                confidence=0.0,
                reasoning="No sub-agents registered.",
            )

        # Collect predictions from all sub-agents
        predictions: list[Prediction] = []
        for agent in self._sub_agents:
            if agent.name == self.name:
                continue  # skip self
            try:
                pred = agent.analyze(context)
                predictions.append(pred)
            except Exception:
                # Agent failed — skip it, don't crash the pipeline
                continue

        # Update weights from tracker state
        agent_names = [p.agent_name for p in predictions]
        self._tracker.recalculate_weights(agent_names)

        # Compute consensus
        consensus = self._tracker.compute_consensus(predictions)

        # Build reasoning
        top_agents = sorted(
            predictions, key=lambda p: self._tracker.get_weight(p.agent_name), reverse=True
        )[:3]
        reasons = "; ".join(
            f"{a.agent_name}: {a.direction} ({a.confidence:.0%})" for a in top_agents
        )

        return Prediction(
            agent_name=self.name,
            agent_id=self.agent_id,
            symbol=context.symbol,
            direction=consensus.direction,
            confidence=consensus.confidence,
            reasoning=(
                f"Consensus ({consensus.direction}, {consensus.confidence:.0%} conf, "
                f"disp={consensus.dispersion:.2f}, penalty={consensus.penalty:.0%}). "
                f"Top agents: {reasons}"
            ),
            supporting_features=[f"dispersion={consensus.dispersion:.3f}"],
        )

    def compute_consensus_from_predictions(self, predictions: list[Prediction]) -> ConsensusResult:
        """Direct access to consensus computation (for API/dashboard use)."""
        agent_names = [p.agent_name for p in predictions]
        self._tracker.recalculate_weights(agent_names)
        return self._tracker.compute_consensus(predictions)

    def self_tune(self, outcomes: list[PredictionOutcome]) -> None:
        """Feed resolved outcomes back into the performance tracker."""
        self._tracker.batch_record(outcomes)

    def get_agent_weights(self) -> dict[str, float]:
        return self._tracker.get_all_weights()

    def get_tracker(self) -> AgentPerformanceTracker:
        return self._tracker

    @property
    def version(self) -> str:
        return self._version

    @property
    def sub_agent_count(self) -> int:
        return len(self._sub_agents)
