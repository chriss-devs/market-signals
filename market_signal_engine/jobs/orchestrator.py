"""AgentOrchestrator — connects collectors → agents → meta-agent → signals → alerts.

The central pipeline: run collectors, build features, execute agents, compute consensus, emit signals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from market_signal_engine.agents.base import AnalysisContext, Prediction
from market_signal_engine.agents.registry import AgentRegistry, get_registry
from market_signal_engine.agents.performance import AgentPerformanceTracker
from market_signal_engine.agents.meta_agent import MetaAgent
from market_signal_engine.collectors.base import CollectorResult
from market_signal_engine.jobs.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates the full analysis pipeline for a set of assets."""

    def __init__(self, registry: AgentRegistry | None = None) -> None:
        self._registry = registry or get_registry()
        self._meta_agent = MetaAgent()
        self._feature_builder = FeatureBuilder()
        self._tracker = AgentPerformanceTracker()
        self._cycle_count = 0
        self._last_cycle_at: str | None = None

    # ── Pipeline ─────────────────────────────────────────────────────────

    def run_cycle(
        self, symbol: str, collector_results: list[CollectorResult]
    ) -> dict[str, Any]:
        """Run one full analysis cycle for an asset.

        Returns a dict with predictions, consensus, and signal details.
        """
        self._cycle_count += 1
        self._last_cycle_at = datetime.now(timezone.utc).isoformat()

        # 1. Build FeatureSet
        feature_set = self._feature_builder.build(collector_results, symbol)

        # 2. Build AnalysisContext
        prices = feature_set.metadata.get("prices", [])
        volumes = feature_set.metadata.get("volumes", [])
        context = AnalysisContext(
            symbol=symbol,
            price_history=prices if isinstance(prices, list) else [],
            volume_history=volumes if isinstance(volumes, list) else [],
            features=feature_set,
        )

        # 3. Run all registered agents
        predictions: list[Prediction] = []
        agent_results: list[dict] = []

        for name in self._registry.list_by_tier(1):
            agent = self._registry.get(name)
            if agent is None:
                continue
            try:
                pred = agent.analyze(context)
                predictions.append(pred)
                agent_results.append({
                    "agent_name": pred.agent_name,
                    "agent_id": pred.agent_id,
                    "direction": pred.direction,
                    "confidence": pred.confidence,
                    "reasoning": pred.reasoning[:120],
                })
            except Exception as e:
                logger.error(f"Agent {name} failed: {e}")

        if not predictions:
            return {
                "symbol": symbol,
                "timestamp": self._last_cycle_at,
                "predictions": 0,
                "error": "No agents produced predictions",
            }

        # 4. Register sub-agents with meta-agent and compute consensus
        self._meta_agent.register_sub_agents(
            [self._registry.get(p.agent_name) for p in predictions if self._registry.get(p.agent_name)]
        )
        consensus = self._meta_agent.compute_consensus_from_predictions(predictions)

        # 5. Snapshot performances (the tracker computes weights internally)
        agent_perfs = []
        for p in predictions:
            agent = self._registry.get(p.agent_name)
            if agent:
                agent_perfs.append(agent.get_performance())
        if agent_perfs:
            self._tracker.snapshot(agent_perfs)

        # 6. Determine alert importance for Telegram
        from market_signal_engine.telegram.formatter import SignalAlert, importance_level

        alert_data = SignalAlert(
            symbol=symbol,
            direction=consensus.direction,
            confidence=consensus.confidence,
            consensus_count=consensus.vote_tally.get(consensus.direction, 0),
            total_agents=len(predictions),
            reasons=[p.reasoning.split(";")[0] for p in predictions[:5] if p.reasoning],
            dispersion=consensus.dispersion,
            recommendation=self._generate_recommendation(consensus),
        )
        tag = importance_level(alert_data)

        return {
            "symbol": symbol,
            "timestamp": self._last_cycle_at,
            "cycle": self._cycle_count,
            "predictions": len(predictions),
            "agent_results": agent_results[:8],  # top 8 for display
            "consensus": {
                "direction": consensus.direction,
                "confidence": round(consensus.confidence, 4),
                "dispersion": round(consensus.dispersion, 4),
                "vote_tally": consensus.vote_tally,
                "agent_weights": {
                    k: round(v, 4) for k, v in list(consensus.agent_weights.items())[:5]
                },
            },
            "alert_level": tag,
            "should_alert": tag in ("CRITICAL", "IMPORTANT"),
            "alert_summary": (
                f"{consensus.direction.upper()} {symbol} "
                f"({consensus.confidence*100:.0f}%, "
                f"{consensus.vote_tally.get(consensus.direction, 0)}/{len(predictions)} agents)"
            ),
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _generate_recommendation(consensus) -> str:
        """Generate a brief recommendation from consensus."""
        c = consensus.confidence
        d = consensus.direction
        if d == "neutral":
            return "Wait for clearer signal — consensus is divided"
        if c >= 0.85:
            return f"Strong {d} signal — review position sizing carefully"
        if c >= 0.75:
            return f"Consider {d} entry — confirm with price action"
        return f"Weak {d} bias — do not act on this alone"

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_cycle_at(self) -> str | None:
        return self._last_cycle_at
