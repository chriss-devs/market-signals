"""Sentiment Analysis Agent — news aggregate, volume spikes, sentiment-price divergence.

Self-improvement: adjusts source weighting by correlation with outcomes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import pct_change


class SentimentAnalysisAgent(BaseAgent):
    name = "Sentiment Analysis"
    agent_id = 2
    tier = 1
    category = "Sentiment"
    data_sources = ["news_sentiment", "social_sentiment", "volume_data"]

    def __init__(self) -> None:
        super().__init__()
        # Source weights — tuned by outcome correlation
        self._source_weights: dict[str, float] = defaultdict(lambda: 0.33)
        self._divergence_threshold: dict[str, float] = defaultdict(lambda: 0.15)
        self._extreme_sentiment: dict[str, float] = defaultdict(lambda: 0.75)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol
        prices = context.price_history

        # ── Extract sentiment signals ───────────────────────────────────
        news_score = features.get("news_sentiment", 0.0)      # -1 to +1
        social_score = features.get("social_sentiment", 0.0)  # -1 to +1
        volume_score = features.get("volume_sentiment", 0.0)  # -1 to +1 (volume = conviction)
        keyword_trend = features.get("keyword_trend", 0.0)

        # Weighted sentiment aggregate
        w = self._source_weights
        w_news = w.get("news", 0.33)
        w_social = w.get("social", 0.33)
        w_volume = w.get("volume", 0.34)

        aggregate = news_score * w_news + social_score * w_social + volume_score * w_volume + keyword_trend * 0.1

        # Volume spike detection
        volume_spike = features.get("volume_spike", 0.0)  # 0-1, how many std above avg

        # Sentiment-price divergence
        change_3d = pct_change(prices, 3) if len(prices) > 3 else 0
        divergence = False
        div_reason = ""
        if abs(change_3d) > 2 and abs(aggregate) > 0.1:
            if change_3d > 0 and aggregate < 0:
                divergence = True
                div_reason = "Price up but sentiment bearish (bearish divergence)"
            elif change_3d < 0 and aggregate > 0:
                divergence = True
                div_reason = "Price down but sentiment bullish (bullish divergence)"

        # ── Scoring ─────────────────────────────────────────────────────
        reasons: list[str] = []
        extreme_thresh = self._extreme_sentiment[sym]

        if divergence:
            # Divergence is contrarian: sentiment disagrees with price
            if aggregate > 0:
                direction = "bullish"
                confidence = min(0.75, 0.45 + abs(aggregate) * 0.4)
            else:
                direction = "bearish"
                confidence = min(0.75, 0.45 + abs(aggregate) * 0.4)
            reasons.append(div_reason)
        elif abs(aggregate) > extreme_thresh:
            direction = "bullish" if aggregate > 0 else "bearish"
            confidence = min(0.85, 0.5 + abs(aggregate) * 0.4)
            reasons.append(f"Extreme sentiment ({aggregate:+.2f})")
        elif abs(aggregate) > 0.2:
            direction = "bullish" if aggregate > 0 else "bearish"
            confidence = min(0.70, 0.5 + abs(aggregate) * 0.3)
            reasons.append(f"Moderate sentiment ({aggregate:+.2f})")
        else:
            direction = "neutral"
            confidence = 0.35
            reasons.append(f"Neutral sentiment ({aggregate:+.2f})")

        if volume_spike > 0.7:
            reasons.append(f"Volume spike ({volume_spike:.0%}) — conviction signal")
            if direction != "neutral":
                confidence = min(0.90, confidence * 1.15)
        elif volume_spike > 0.4:
            reasons.append(f"Elevated volume ({volume_spike:.0%})")

        confidence = self.calibrate_confidence(confidence)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons),
            supporting_features=[
                f"aggregate={aggregate:.3f}", f"news={news_score:.2f}",
                f"social={social_score:.2f}", f"volume_spike={volume_spike:.2f}",
            ],
        )

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            for feat in o.prediction.supporting_features:
                if feat.startswith("news="):
                    delta = 0.03 if o.was_correct else -0.05
                    self._source_weights["news"] = max(0.1, min(0.6, w["news"] + delta))
                elif feat.startswith("social="):
                    delta = 0.03 if o.was_correct else -0.05
                    self._source_weights["social"] = max(0.1, min(0.6, w["social"] + delta))
                impact = 0.05 if o.was_correct else -0.03
                self.update_feature_importance(feat, impact)
            # Renormalize source weights
            w = self._source_weights
            total = w["news"] + w["social"] + w.get("volume", 0.34)
            if total > 0:
                for k in ["news", "social"]:
                    w[k] /= total
