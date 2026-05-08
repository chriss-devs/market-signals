"""Whale Tracking Agent — large transactions, wallet accumulation, exchange flows.

Self-improvement: tracks which whale behaviors precede moves.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import pct_change


class WhaleTrackingAgent(BaseAgent):
    name = "Whale Tracking"
    agent_id = 12
    tier = 1
    category = "On-Chain"
    data_sources = ["blockchain", "whale_alert"]

    # Whale behavior categories
    BEHAVIOR_ACCUMULATION = "accumulation"
    BEHAVIOR_DISTRIBUTION = "distribution"
    BEHAVIOR_ACCUMULATION_SPIKE = "accumulation_spike"
    BEHAVIOR_NETWORK_GROWTH = "network_growth"

    def __init__(self) -> None:
        super().__init__()
        # Per-asset whale behavior hit rate
        self._behavior_hits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._behavior_total: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Per-asset thresholds
        self._large_tx_threshold: dict[str, float] = defaultdict(lambda: 2.0)  # z-score

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol
        prices = context.price_history

        reasons: list[str] = []
        score = 0.0

        # ── Extract whale signals from features ──────────────────────────
        large_tx_count = features.get("large_tx_count", 0)         # >$100k txns in 24h
        large_tx_volume = features.get("large_tx_volume", 0)       # total $ volume
        whale_netflow = features.get("whale_netflow", 0)           # net exchange flow (+ = deposits)
        whale_ratio = features.get("whale_ratio", 0)              # whale tx / total tx
        new_wallets = features.get("new_wallets", 0)              # new wallet creation
        active_addresses = features.get("active_addresses", 0)     # daily active addresses
        exchange_inflow = features.get("exchange_inflow", 0)       # $ to exchanges
        exchange_outflow = features.get("exchange_outflow", 0)     # $ from exchanges
        whale_wallets = features.get("whale_wallets", 0)          # wallets with >$1M

        # ── 1. Large transaction spikes ──────────────────────────────────
        tx_thresh = self._large_tx_threshold[sym]
        if large_tx_count > 50 and large_tx_volume > 1_000_000_000:
            score += 0.10
            reasons.append(f"Whale activity spike ({large_tx_count} txns, ${large_tx_volume/1e9:.1f}B)")
        elif large_tx_count > 20:
            reasons.append(f"Elevated whale activity ({large_tx_count} txns)")

        # ── 2. Exchange flows ────────────────────────────────────────────
        net_exchange_flow = exchange_inflow - exchange_outflow
        if net_exchange_flow < -500_000_000:
            # Large outflow from exchanges = whales accumulating
            score += 0.15
            reasons.append(f"Massive exchange outflow (${abs(net_exchange_flow)/1e9:.1f}B) — whales accumulating")
        elif net_exchange_flow < -100_000_000:
            score += 0.08
            reasons.append(f"Exchange outflow (${abs(net_exchange_flow)/1e6:.0f}M) — bullish")
        elif net_exchange_flow > 500_000_000:
            # Large inflow to exchanges = whales preparing to sell
            score -= 0.15
            reasons.append(f"Massive exchange inflow (${net_exchange_flow/1e9:.1f}B) — whales distributing")
        elif net_exchange_flow > 100_000_000:
            score -= 0.08
            reasons.append(f"Exchange inflow (${net_exchange_flow/1e6:.0f}M) — bearish")

        # ── 3. Whale wallet accumulation ─────────────────────────────────
        if whale_wallets > 0:
            wallet_change = features.get("whale_wallet_change", 0)  # % change in whale count
            if wallet_change > 5:
                score += 0.12
                reasons.append(f"Whale wallet growth ({wallet_change:+.0f}%) — accumulation")
            elif wallet_change < -5:
                score -= 0.12
                reasons.append(f"Whale wallet decline ({wallet_change:+.0f}%) — distribution")

        # ── 4. Whale ratio ───────────────────────────────────────────────
        if whale_ratio > 0.6:
            # Whales dominate volume
            if whale_netflow < 0:
                score += 0.10
                reasons.append(f"Whales dominating + accumulating (ratio={whale_ratio:.0%})")
            elif whale_netflow > 0:
                score -= 0.10
                reasons.append(f"Whales dominating + distributing (ratio={whale_ratio:.0%})")

        # ── 5. New wallet creation ───────────────────────────────────────
        if new_wallets > 0:
            wallet_growth = features.get("wallet_growth_rate", 0)  # % daily growth
            if wallet_growth > 3:
                score += 0.08
                reasons.append(f"New wallet surge ({wallet_growth:+.1f}% daily) — retail entering")
            elif wallet_growth < -2:
                score -= 0.06
                reasons.append(f"Wallet abandonment ({wallet_growth:+.1f}%)")

        # ── 6. Smart money tracking ──────────────────────────────────────
        smart_money = features.get("smart_money_flow", 0)  # net flow of known smart wallets
        if smart_money > 0.5:
            score += 0.10
            reasons.append("Smart money accumulating — high conviction")
        elif smart_money < -0.5:
            score -= 0.10
            reasons.append("Smart money distributing")

        # ── 7. Active addresses trend ────────────────────────────────────
        addr_trend = features.get("address_trend", 0)  # % change in active addresses
        if addr_trend > 10:
            score += 0.06
            reasons.append(f"Active addresses surging ({addr_trend:+.0f}%)")

        # ── 8. Whale behavior divergence ─────────────────────────────────
        change_7d = pct_change(prices, 7) if len(prices) > 7 else 0
        if change_7d < -5 and net_exchange_flow < -200_000_000:
            # Price down but whales buying = bullish divergence
            score += 0.12
            reasons.append("Bullish divergence: price down, whales accumulating")
        elif change_7d > 5 and net_exchange_flow > 200_000_000:
            # Price up but whales selling = bearish divergence
            score -= 0.12
            reasons.append("Bearish divergence: price up, whales distributing")

        # ── Direction & confidence ──────────────────────────────────────
        score = max(-1.0, min(1.0, score))

        if score > 0.08:
            direction = "bullish"
            confidence = min(0.80, 0.45 + abs(score) * 0.5)
        elif score < -0.08:
            direction = "bearish"
            confidence = min(0.80, 0.45 + abs(score) * 0.5)
        else:
            direction = "neutral"
            confidence = 0.35

        confidence = self.calibrate_confidence(confidence)

        # Track behavior patterns
        behavior = self._classify_behavior(features)
        self._behavior_total[sym][behavior] += 1

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons[:5]) if reasons else "No significant whale activity",
            supporting_features=[
                f"netflow={net_exchange_flow/1e6 if net_exchange_flow else 0:.0f}M",
                f"whale_ratio={whale_ratio:.2f}",
                f"smart_money={smart_money:.2f}",
                f"behavior={behavior}",
            ],
        )

    def _classify_behavior(self, features: dict[str, float]) -> str:
        """Classify current whale behavior pattern."""
        netflow = features.get("whale_netflow", 0)
        smart = features.get("smart_money_flow", 0)
        new_wallets = features.get("wallet_growth_rate", 0)

        if new_wallets > 3:
            return self.BEHAVIOR_NETWORK_GROWTH
        if netflow < -0.3 and smart > 0.3:
            return self.BEHAVIOR_ACCUMULATION_SPIKE
        if netflow < 0 and smart > 0:
            return self.BEHAVIOR_ACCUMULATION
        return self.BEHAVIOR_DISTRIBUTION

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            sym = o.prediction.symbol
            for feat in o.prediction.supporting_features:
                if feat.startswith("behavior="):
                    behavior = feat.split("=")[1]
                    if o.was_correct:
                        self._behavior_hits[sym][behavior] += 1
                impact = 0.05 if o.was_correct else -0.03
                self.update_feature_importance(feat, impact)
