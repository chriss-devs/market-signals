"""NFT/Gaming Agent — NFT market health and gaming token analysis.

Monitors NFT trading volumes, floor prices, gaming token metrics,
metaverse activity, and Web3 gaming adoption trends.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class NFTGamingAgent(BaseAgent):
    name = "NFT/Gaming"
    agent_id = 26
    tier = 3
    category = "Alternative"
    data_sources = ["dexscreener", "defillama"]

    # Leading NFT/gaming tokens to track
    GAMING_TOKENS = {"IMX", "GALA", "SAND", "MANA", "AXS", "APE", "ILV"}
    NFT_MARKETPLACES = {"Blur", "OpenSea", "MagicEden", "Tensor"}

    def __init__(self) -> None:
        super().__init__()
        self._nft_volume_history: dict[str, list[float]] = defaultdict(list)
        self._gaming_metrics: dict[str, list[dict]] = defaultdict(list)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. NFT market volume
        nft_volume = features.get("nft_volume_24h", 0)
        nft_volume_change = features.get("nft_volume_change_7d", 0.0)
        if nft_volume_change > 30:
            score += 0.04
            reasons.append(f"NFT volume surging +{nft_volume_change:.0f}% — interest returning")
        elif nft_volume_change < -30:
            score -= 0.03
            reasons.append(f"NFT volume crashing {nft_volume_change:.0f}% — sector cooling")

        # 2. NFT floor price trends (blue chips)
        floor_price_index = features.get("blue_chip_floor_index", 100)
        floor_change = features.get("floor_price_change_30d", 0.0)
        if floor_change > 15:
            score += 0.04
            reasons.append(f"Blue chip floors rising +{floor_change:.0f}%")
        elif floor_change < -15:
            score -= 0.04
            reasons.append(f"Blue chip floors declining {floor_change:.0f}%")

        # 3. Gaming token sector momentum
        gaming_momentum = features.get("gaming_sector_momentum", 0.0)
        if gaming_momentum > 5:
            score += 0.05
            reasons.append(f"Gaming sector strong ({gaming_momentum:+.1f}%)")
        elif gaming_momentum < -5:
            score -= 0.03
            reasons.append(f"Gaming sector weak ({gaming_momentum:+.1f}%)")

        # 4. Web3 gaming users
        gaming_users = features.get("web3_gaming_dau", 0)
        user_growth = features.get("gaming_user_growth_30d", 0.0)
        if user_growth > 20:
            score += 0.05
            reasons.append(f"Gaming DAU growing +{user_growth:.0f}% — adoption")
        elif user_growth < -10:
            score -= 0.03
            reasons.append(f"Gaming DAU declining {user_growth:.0f}%")

        # 5. Marketplace dominance
        blur_dominance = features.get("blur_volume_dominance", 0.0)
        if blur_dominance > 0.6:
            reasons.append(f"Blur dominance ({blur_dominance:.0%}) — pro trader activity")

        # 6. NFT wash trading filter
        wash_trade_pct = features.get("wash_trade_estimate", 0.0)
        if wash_trade_pct > 0.3:
            score -= 0.04
            reasons.append(f"High wash trading ({wash_trade_pct:.0%}) — inflated metrics")

        # 7. Metaverse land prices
        land_price_index = features.get("metaverse_land_index", 100)
        land_change = features.get("land_price_change_90d", 0.0)
        if land_change > 20:
            score += 0.03
            reasons.append(f"Metaverse land appreciating +{land_change:.0f}%")
        elif land_change < -20:
            score -= 0.02
            reasons.append(f"Metaverse land declining {land_change:.0f}%")

        # 8. Royalty revenue (creator economy)
        royalty_volume = features.get("nft_royalty_volume_30d", 0)
        royalty_change = features.get("royalty_change_30d", 0.0)
        if royalty_change > 20:
            score += 0.03
            reasons.append(f"Creator royalties growing +{royalty_change:.0f}%")
        elif royalty_change < -20:
            score -= 0.02
            reasons.append(f"Creator royalties declining — ecosystem stress")

        # 9. Gaming token unlocks
        upcoming_unlocks = features.get("upcoming_unlocks_value", 0)
        if upcoming_unlocks > 10_000_000:
            score -= 0.04
            reasons.append(f"Large token unlocks ahead (${upcoming_unlocks/1e6:.0f}M) — supply pressure")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.03 else "bearish" if score < -0.03 else "neutral"
        confidence = min(0.68, 0.30 + abs(score) * 0.40)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "NFT/Gaming metrics neutral",
            supporting_features=[
                f"nft_vol_change={nft_volume_change:+.0f}%",
                f"gaming_momentum={gaming_momentum:+.1f}%",
                f"user_growth={user_growth:+.0f}%",
                f"wash_trade={wash_trade_pct:.0%}",
            ],
        )

    def nft_health_score(self, features: dict) -> float:
        """Composite NFT ecosystem health score (0-100)."""
        components = [
            (features.get("nft_volume_change_7d", 0), 0.25),
            (features.get("floor_price_change_30d", 0), 0.25),
            (features.get("gaming_user_growth_30d", 0), 0.20),
            (-features.get("wash_trade_estimate", 0) * 100, 0.15),
            (features.get("royalty_change_30d", 0), 0.15),
        ]
        score = sum(max(-100, min(100, val)) * weight for val, weight in components)
        return round(max(0, min(100, 50 + score / 2)), 1)

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            self._nft_volume_history.setdefault(o.prediction.symbol, []).append(
                1.0 if o.was_correct else 0.0
            )
