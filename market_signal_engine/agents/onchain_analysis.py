"""On-Chain Analysis Agent — DEX volume/liquidity, TVL flows, stablecoins, exchange netflows.

Self-improvement: learns which on-chain metrics predict moves per token.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import pct_change


class OnChainAnalysisAgent(BaseAgent):
    name = "On-Chain Analysis"
    agent_id = 3
    tier = 1
    category = "On-Chain"
    data_sources = ["dexscreener", "defillama", "blockchain", "coingecko"]

    # Metric categories
    DEX_METRICS = {"dex_volume", "dex_liquidity", "buy_sell_ratio", "dex_volume_change"}
    TVL_METRICS = {"tvl", "tvl_change_1d", "tvl_change_7d", "protocol_count"}
    NETWORK_METRICS = {"active_addresses", "tx_count", "hash_rate", "mempool_size", "fees"}
    FLOW_METRICS = {"exchange_netflow", "stablecoin_mcap_change", "exchange_reserves"}
    HOLDER_METRICS = {"new_wallets", "active_wallets", "concentration"}

    def __init__(self) -> None:
        super().__init__()
        # Per-token metric weights — tuned over time
        self._metric_weights: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "dex_volume_change": 0.15,
                "dex_liquidity": 0.10,
                "buy_sell_ratio": 0.12,
                "tvl_change_7d": 0.15,
                "exchange_netflow": 0.18,
                "active_addresses": 0.12,
                "stablecoin_mcap_change": 0.08,
                "tx_count_change": 0.10,
            }
        )
        # Per-token TVL sensitivity thresholds
        self._tvl_alert_threshold: dict[str, float] = defaultdict(lambda: 10.0)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol
        prices = context.price_history

        reasons: list[str] = []
        score = 0.0

        # ── Extract on-chain signals ─────────────────────────────────────
        # DEX metrics
        dex_volume = features.get("dex_volume_24h", 0)
        dex_liquidity = features.get("dex_liquidity", 0)
        dex_buys = features.get("dex_buys_24h", 0)
        dex_sells = features.get("dex_sells_24h", 0)
        dex_vol_change = features.get("dex_volume_change", 0)  # % change

        # TVL
        tvl = features.get("tvl", 0)
        tvl_change_1d = features.get("tvl_change_1d", 0)
        tvl_change_7d = features.get("tvl_change_7d", 0)

        # Exchange flows
        exchange_netflow = features.get("exchange_netflow", 0)  # negative = outflow
        exchange_reserves = features.get("exchange_reserves", 0)

        # Network
        active_addresses = features.get("active_addresses", 0)
        addr_change = features.get("address_change", 0)
        tx_count = features.get("tx_count", 0)
        tx_change = features.get("tx_change", 0)
        hash_rate = features.get("hash_rate", 0)
        hash_change = features.get("hash_change", 0)
        mempool_size = features.get("mempool_size", 0)
        fees = features.get("avg_fee", 0)

        # Stablecoins
        stablecoin_mcap = features.get("stablecoin_mcap", 0)
        stablecoin_change = features.get("stablecoin_mcap_change", 0)

        # Holders
        new_wallets = features.get("new_wallets", 0)
        holder_concentration = features.get("holder_concentration", 0)  # top 100 / total

        # ── Scoring ───────────────────────────────────────────────────────
        w = self._metric_weights[sym]

        # DEX volume & liquidity
        if dex_vol_change > 50 and dex_buys > dex_sells * 1.3:
            score += 0.12
            reasons.append(f"DEX volume surging +{dex_vol_change:.0f}% — bullish flow")
        elif dex_vol_change > 30:
            score += 0.06
            reasons.append(f"DEX volume rising +{dex_vol_change:.0f}%")
        elif dex_vol_change < -30:
            score -= 0.06
            reasons.append(f"DEX volume declining {dex_vol_change:.0f}%")

        # Buy/sell ratio
        if dex_buys + dex_sells > 0:
            buy_ratio = dex_buys / (dex_buys + dex_sells)
            if buy_ratio > 0.65:
                score += 0.10
                reasons.append(f"Strong buy pressure ({buy_ratio:.0%} buys)")
            elif buy_ratio < 0.35:
                score -= 0.10
                reasons.append(f"Strong sell pressure ({1-buy_ratio:.0%} sells)")

        # DEX liquidity — rising liquidity = confidence
        if dex_liquidity > 0:
            liq_to_vol = dex_liquidity / max(dex_volume, 1)
            if liq_to_vol > 5:
                score += 0.04
                reasons.append(f"Deep liquidity (liq/vol={liq_to_vol:.1f}x)")
            elif liq_to_vol < 0.5 and dex_volume > 100_000:
                score -= 0.06
                reasons.append(f"Thin liquidity ({liq_to_vol:.1f}x) — risky")

        # TVL flows
        tvl_alert = self._tvl_alert_threshold[sym]
        if tvl_change_7d > tvl_alert:
            score += 0.12
            reasons.append(f"TVL surging +{tvl_change_7d:.1f}% in 7d — capital inflow")
        elif tvl_change_7d > 5:
            score += 0.06
            reasons.append(f"TVL growing +{tvl_change_7d:.1f}%")
        elif tvl_change_7d < -tvl_alert:
            score -= 0.12
            reasons.append(f"TVL collapsing {tvl_change_7d:.1f}% in 7d — capital flight")
        elif tvl_change_7d < -5:
            score -= 0.06
            reasons.append(f"TVL declining {tvl_change_7d:.1f}%")

        # Exchange netflows (most predictive on-chain metric)
        if abs(exchange_netflow) > 1:
            if exchange_netflow < -1.0:
                score += 0.15
                reasons.append("Exchange reserves draining — supply shock signal")
            elif exchange_netflow < -0.5:
                score += 0.08
                reasons.append("Moderate exchange outflows — accumulation")
            elif exchange_netflow > 1.0:
                score -= 0.15
                reasons.append("Exchange reserves surging — selling pressure")
            elif exchange_netflow > 0.5:
                score -= 0.08
                reasons.append("Moderate exchange inflows — distribution")

        # Active addresses & network usage
        if addr_change > 20:
            score += 0.10
            reasons.append(f"Active addresses surging +{addr_change:.0f}% — network growth")
        elif addr_change > 10:
            score += 0.05
            reasons.append(f"Active addresses rising +{addr_change:.0f}%")
        elif addr_change < -15:
            score -= 0.08
            reasons.append(f"Active addresses declining {addr_change:.0f}% — waning interest")

        # Transaction count
        if tx_change > 25:
            score += 0.08
            reasons.append(f"Transaction volume surging +{tx_change:.0f}%")
        elif tx_change < -20:
            score -= 0.06
            reasons.append(f"Transaction volume falling {tx_change:.0f}%")

        # Hash rate (security)
        if hash_change > 10:
            score += 0.05
            reasons.append(f"Hash rate growing +{hash_change:.0f}% — network secure")

        # Mempool / fees — high fees = high demand
        if fees > 0:
            fee_change = features.get("fee_change", 0)
            if fee_change > 40:
                score += 0.06
                reasons.append(f"Fees spiking +{fee_change:.0f}% — high network demand")

        # Stablecoin market cap — growing = buying power
        if stablecoin_change > 5:
            score += 0.08
            reasons.append(f"Stablecoin mcap growing +{stablecoin_change:.1f}% — buying power expanding")
        elif stablecoin_change < -5:
            score -= 0.06
            reasons.append(f"Stablecoin mcap shrinking {stablecoin_change:.1f}%")

        # Holder concentration
        if holder_concentration > 0.6:
            score -= 0.06
            reasons.append(f"High holder concentration ({holder_concentration:.0%} top 100)")
        elif holder_concentration < 0.3:
            score += 0.04
            reasons.append("Well distributed holders")

        # New wallet creation
        if new_wallets > 0:
            wallet_growth = features.get("wallet_growth_rate", 0)
            if wallet_growth > 5:
                score += 0.06
                reasons.append(f"New wallets surging {wallet_growth:+.1f}% daily")
            elif wallet_growth < -3:
                score -= 0.04
                reasons.append(f"Wallet creation declining {wallet_growth:+.1f}%")

        # ── Divergence detection ──────────────────────────────────────────
        change_7d = pct_change(prices, 7) if len(prices) > 7 else 0
        tvl_alert_val = self._tvl_alert_threshold[sym]

        if tvl_change_7d > tvl_alert_val and change_7d < 0:
            score += 0.10
            reasons.append("Bullish divergence: TVL surging despite price decline")
        elif tvl_change_7d < -tvl_alert_val and change_7d > 0:
            score -= 0.10
            reasons.append("Bearish divergence: TVL falling despite price rise")

        # ── Weighted scoring ──────────────────────────────────────────────
        raw_score = (
            w["dex_volume_change"] * (score * 0.3)  # distribute score proportionally
            + w["dex_liquidity"] * (0.04 if dex_liquidity > 1_000_000 else 0)
            + w["buy_sell_ratio"] * (0.10 if dex_buys > dex_sells else -0.10 if dex_sells > dex_buys else 0)
            + w["tvl_change_7d"] * (0.12 if tvl_change_7d > 5 else -0.12 if tvl_change_7d < -5 else 0)
            + w["exchange_netflow"] * (0.15 if exchange_netflow < -0.5 else -0.15 if exchange_netflow > 0.5 else 0)
            + w["active_addresses"] * (0.10 if addr_change > 10 else -0.08 if addr_change < -15 else 0)
            + w["stablecoin_mcap_change"] * (0.08 if stablecoin_change > 3 else -0.06 if stablecoin_change < -3 else 0)
            + w["tx_count_change"] * (0.08 if tx_change > 20 else -0.06 if tx_change < -20 else 0)
        )

        # Blend raw weighted score with the individually computed score
        score = score * 0.4 + raw_score * 0.6
        score = max(-1.0, min(1.0, score))

        if score > 0.07:
            direction = "bullish"
            confidence = min(0.80, 0.45 + abs(score) * 0.55)
        elif score < -0.07:
            direction = "bearish"
            confidence = min(0.80, 0.45 + abs(score) * 0.55)
        else:
            direction = "neutral"
            confidence = 0.35

        confidence = self.calibrate_confidence(confidence)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons[:5]) if reasons else "On-chain metrics neutral",
            supporting_features=[
                f"tvl_change={tvl_change_7d:.1f}", f"netflow={exchange_netflow:.2f}",
                f"addr_change={addr_change:.1f}", f"dex_vol_change={dex_vol_change:.1f}",
            ],
        )

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            sym = o.prediction.symbol
            w = self._metric_weights[sym]
            for feat in o.prediction.supporting_features:
                key = feat.split("=")[0]
                # Map feature names to weight keys
                weight_map = {
                    "tvl_change": "tvl_change_7d",
                    "netflow": "exchange_netflow",
                    "addr_change": "active_addresses",
                    "dex_vol_change": "dex_volume_change",
                }
                metric_key = weight_map.get(key, key)
                if metric_key in w:
                    delta = 0.02 if o.was_correct else -0.03
                    w[metric_key] = max(0.03, min(0.30, w[metric_key] + delta))
                    total = sum(w.values())
                    if total > 0:
                        for k in w:
                            w[k] /= total
                impact = 0.04 if o.was_correct else -0.02
                self.update_feature_importance(feat, impact)
            # Tune TVL alert threshold
            if not o.was_correct:
                if o.prediction.direction == "bullish":
                    self._tvl_alert_threshold[sym] = min(25, self._tvl_alert_threshold[sym] + 1)
                elif o.prediction.direction == "bearish":
                    self._tvl_alert_threshold[sym] = max(5, self._tvl_alert_threshold[sym] - 1)
