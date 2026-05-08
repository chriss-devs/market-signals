"""DeFi Agent — decentralized finance ecosystem analysis.

Monitors TVL trends, protocol dominance, yield dynamics, liquidation cascades,
stablecoin flows, and bridge activity. Surfaces DeFi-native alpha signals.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class DeFiAgent(BaseAgent):
    name = "DeFi"
    agent_id = 18
    tier = 2
    category = "DeFi"
    data_sources = ["defillama", "dexscreener", "blockchain.com"]

    # TVL thresholds (in billions)
    TVL_HIGH = 100
    TVL_LOW = 10

    def __init__(self) -> None:
        super().__init__()
        self._protocol_alerts: dict[str, int] = defaultdict(int)
        self._liquidation_history: list[dict] = []
        self._yield_history: dict[str, list[float]] = defaultdict(list)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. TVL momentum
        tvl = features.get("tvl", 0)
        tvl_change_1d = features.get("tvl_change_1d", 0)
        tvl_change_7d = features.get("tvl_change_7d", 0)

        if tvl_change_7d > 10:
            score += 0.08
            reasons.append(f"TVL surging +{tvl_change_7d:.0f}% WoW — capital inflow")
        elif tvl_change_7d > 5:
            score += 0.04
            reasons.append(f"TVL growing +{tvl_change_7d:.0f}% WoW — healthy inflow")
        elif tvl_change_7d < -10:
            score -= 0.07
            reasons.append(f"TVL declining {tvl_change_7d:.0f}% WoW — capital flight")
        elif tvl_change_7d < -5:
            score -= 0.03
            reasons.append(f"TVL slipping {tvl_change_7d:.0f}% WoW")

        # 2. Stablecoin market cap
        stablecoin_mcap = features.get("stablecoin_mcap", 0)
        stablecoin_change = features.get("stablecoin_change_30d", 0)
        if stablecoin_change > 5:
            score += 0.06
            reasons.append(f"Stablecoin supply expanding +{stablecoin_change:.0f}% — dry powder")
        elif stablecoin_change < -5:
            score -= 0.05
            reasons.append(f"Stablecoin supply contracting {stablecoin_change:.0f}% — risk-off")

        # 3. DEX vs CEX volume ratio
        dex_cex_ratio = features.get("dex_cex_ratio", 0.15)
        dex_cex_trend = features.get("dex_cex_trend", 0.0)
        if dex_cex_ratio > 0.25 and dex_cex_trend > 0:
            score += 0.05
            reasons.append(f"DEX share growing ({dex_cex_ratio:.0%}) — DeFi dominance")
        elif dex_cex_trend < -0.05:
            score -= 0.03
            reasons.append(f"DEX share declining — CEX migration")

        # 4. Liquidation risk
        liq_volume = features.get("liquidation_volume_24h", 0)
        liq_spike = features.get("liq_spike_ratio", 1.0)
        if liq_spike > 3.0:
            score -= 0.08
            reasons.append(f"Liquidation cascade ({liq_spike:.0f}x normal) — forced selling")
        elif liq_spike > 1.5:
            score -= 0.04
            reasons.append(f"Elevated liquidations ({liq_spike:.1f}x) — leverage flush")

        # 5. Yield environment
        avg_yield = features.get("avg_defi_yield", 5.0)
        yield_trend = features.get("yield_trend_30d", 0.0)
        if yield_trend > 1.0:
            score += 0.03
            reasons.append(f"Yields rising ({avg_yield:.1f}% avg) — demand for leverage")
        elif yield_trend < -1.0:
            score -= 0.02
            reasons.append(f"Yields compressing — risk appetite fading")

        # 6. Bridge flows (L1↔L2)
        bridge_inflow = features.get("bridge_net_inflow_7d", 0)
        if bridge_inflow > 50_000_000:
            score += 0.04
            reasons.append(f"Strong bridge inflows (${bridge_inflow/1e6:.0f}M) — L2 migration")
        elif bridge_inflow < -50_000_000:
            score -= 0.03
            reasons.append(f"Bridge outflows (${abs(bridge_inflow)/1e6:.0f}M) — L1 flight")

        # 7. Protocol concentration risk
        top_protocol_share = features.get("top_protocol_tvl_share", 0.5)
        if top_protocol_share > 0.7:
            score -= 0.03
            reasons.append(f"High protocol concentration ({top_protocol_share:.0%}) — systemic risk")

        # 8. DeFi dominance
        defi_dominance = features.get("defi_dominance", 0.1)
        if defi_dominance > 0.15:
            score += 0.03
            reasons.append(f"DeFi dominance high ({defi_dominance:.0%}) — sector strength")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.04 else "bearish" if score < -0.04 else "neutral"
        confidence = min(0.75, 0.35 + abs(score) * 0.45)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "DeFi metrics neutral",
            supporting_features=[
                f"tvl_change_7d={tvl_change_7d:.1f}%",
                f"stablecoin_change={stablecoin_change:.1f}%",
                f"liq_spike={liq_spike:.1f}x",
                f"dex_cex_ratio={dex_cex_ratio:.2f}",
            ],
        )

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
