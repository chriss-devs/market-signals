"""L1/L2 Analysis Agent — blockchain network comparison and health assessment.

Compares transaction costs, throughput, developer activity, TVL distribution,
sequencer economics, and bridge flows across L1 and L2 networks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class L1L2AnalysisAgent(BaseAgent):
    name = "L1/L2 Analysis"
    agent_id = 19
    tier = 2
    category = "Chain"
    data_sources = ["defillama", "blockchain.com"]

    # Gas price thresholds (gwei)
    GAS_HIGH = 100
    GAS_LOW = 15

    # Chain scoring weights
    CHAIN_WEIGHTS = {
        "tvl_growth": 0.25,
        "tx_cost_efficiency": 0.15,
        "developer_activity": 0.20,
        "user_growth": 0.20,
        "sequencer_decentralization": 0.10,
        "bridge_security": 0.10,
    }

    def __init__(self) -> None:
        super().__init__()
        self._chain_rankings: list[dict] = []
        self._gas_history: dict[str, list[float]] = defaultdict(list)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. Gas price environment
        gas_price = features.get("avg_gas_price", 30)
        gas_trend = features.get("gas_trend_30d", 0.0)
        if gas_price > self.GAS_HIGH:
            score -= 0.05
            reasons.append(f"Gas high ({gas_price:.0f} gwei) — network congestion")
        elif gas_price < self.GAS_LOW:
            score += 0.04
            reasons.append(f"Gas low ({gas_price:.0f} gwei) — cheap execution")

        if gas_trend > 20:
            score -= 0.03
            reasons.append(f"Gas trending up ({gas_trend:+.0f}%) — demand increasing")

        # 2. Transaction throughput (TPS)
        tps = features.get("tps", 30)
        tps_change = features.get("tps_change_30d", 0.0)
        if tps_change > 20:
            score += 0.06
            reasons.append(f"TPS growing +{tps_change:.0f}% — network adoption")
        elif tps_change < -15:
            score -= 0.04
            reasons.append(f"TPS declining {tps_change:.0f}% — usage drop")

        # 3. Active addresses
        active_addrs = features.get("active_addresses", 0)
        addr_change = features.get("addr_change_30d", 0.0)
        if addr_change > 15:
            score += 0.05
            reasons.append(f"Active addresses +{addr_change:.0f}% — user growth")
        elif addr_change < -10:
            score -= 0.04
            reasons.append(f"Active addresses {addr_change:.0f}% — user decline")

        # 4. TVL comparison (L1 vs L2)
        chain_tvl_rank = features.get("chain_tvl_rank", 5)
        tvl_retention = features.get("tvl_retention_30d", 0.5)
        if chain_tvl_rank <= 3 and tvl_retention > 0.9:
            score += 0.05
            reasons.append(f"Top-3 chain with high TVL retention ({tvl_retention:.0%})")
        elif tvl_retention < 0.7:
            score -= 0.04
            reasons.append(f"Low TVL retention ({tvl_retention:.0%}) — capital rotation")

        # 5. L2 adoption metrics
        l2_tvl_share = features.get("l2_tvl_share", 0.2)
        l2_tvl_trend = features.get("l2_tvl_trend_30d", 0.0)
        if l2_tvl_share > 0.3 and l2_tvl_trend > 0:
            score += 0.04
            reasons.append(f"L2 TVL share growing ({l2_tvl_share:.0%}) — scaling adoption")
        elif l2_tvl_trend < -0.03:
            score -= 0.02
            reasons.append(f"L2 momentum fading")

        # 6. Developer activity proxy
        dev_activity = features.get("dev_activity_score", 50)
        dev_trend = features.get("dev_activity_trend", 0.0)
        if dev_trend > 10:
            score += 0.04
            reasons.append(f"Dev activity rising ({dev_activity:.0f}) — ecosystem growth")
        elif dev_trend < -10:
            score -= 0.03
            reasons.append(f"Dev activity declining — talent drain risk")

        # 7. Sequencer/validator decentralization
        nakamoto_coef = features.get("nakamoto_coefficient", 5)
        if nakamoto_coef < 3:
            score -= 0.04
            reasons.append(f"Low Nakamoto coefficient ({nakamoto_coef}) — centralization risk")
        elif nakamoto_coef > 20:
            score += 0.03
            reasons.append(f"High Nakamoto coefficient ({nakamoto_coef}) — decentralized")

        # 8. Fee revenue and burn
        fee_revenue = features.get("fee_revenue_30d", 0)
        fee_trend = features.get("fee_revenue_trend", 0.0)
        if fee_trend > 10:
            score += 0.03
            reasons.append(f"Fee revenue growing — economic sustainability")
        elif fee_trend < -20:
            score -= 0.03
            reasons.append(f"Fee revenue collapsing — economic stress")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.04 else "bearish" if score < -0.04 else "neutral"
        confidence = min(0.72, 0.33 + abs(score) * 0.42)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "L1/L2 metrics balanced",
            supporting_features=[
                f"gas={gas_price:.0f}gwei",
                f"tps={tps:.0f}",
                f"addr_change={addr_change:+.0f}%",
                f"l2_share={l2_tvl_share:.0%}",
            ],
        )

    def rank_chain(self, features: dict) -> float:
        """Compute a composite chain health score (0-100)."""
        w = self.CHAIN_WEIGHTS
        score = 0.0

        tvl_growth = features.get("tvl_change_7d", 0)
        score += w["tvl_growth"] * min(100, max(0, 50 + tvl_growth * 2))

        gas = features.get("avg_gas_price", 50)
        gas_score = max(0, 100 - gas)
        score += w["tx_cost_efficiency"] * gas_score

        dev = features.get("dev_activity_score", 50)
        score += w["developer_activity"] * dev

        addr_change = features.get("addr_change_30d", 0)
        score += w["user_growth"] * min(100, max(0, 50 + addr_change * 2))

        nakamoto = features.get("nakamoto_coefficient", 5)
        score += w["sequencer_decentralization"] * min(100, nakamoto * 5)

        return round(score, 1)

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            conf = o.prediction.confidence
            self._gas_history.setdefault(o.prediction.symbol, []).append(conf)
            if len(self._gas_history[o.prediction.symbol]) > 100:
                self._gas_history[o.prediction.symbol] = \
                    self._gas_history[o.prediction.symbol][-100:]
