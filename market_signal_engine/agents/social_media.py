"""Social Media Agent — sentiment extraction from social platforms.

Analyzes Twitter/X, Reddit, 4chan, and crypto-native platforms for
sentiment extremes, trending narratives, and influencer signals.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class SocialMediaAgent(BaseAgent):
    name = "Social Media"
    agent_id = 21
    tier = 2
    category = "Sentiment"
    data_sources = ["social_scraper"]

    # Z-score thresholds for sentiment extremes
    EXTREME_BULLISH = 2.5
    EXTREME_BEARISH = -2.5

    def __init__(self) -> None:
        super().__init__()
        self._sentiment_history: dict[str, list[float]] = defaultdict(list)
        self._mention_volume: dict[str, list[int]] = defaultdict(list)
        self._narrative_memory: dict[str, int] = defaultdict(int)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. Composite social sentiment z-score
        social_z = features.get("social_sentiment_z", 0.0)
        if social_z > self.EXTREME_BULLISH:
            score -= 0.05  # Contrarian: extreme bullish = warning
            reasons.append(f"Extreme bullish sentiment (z={social_z:.1f}) — contrarian caution")
        elif social_z > 1.5:
            score += 0.04
            reasons.append(f"Strong bullish sentiment (z={social_z:.1f})")
        elif social_z < self.EXTREME_BEARISH:
            score += 0.06  # Contrarian: extreme bearish = opportunity
            reasons.append(f"Extreme bearish sentiment (z={social_z:.1f}) — contrarian buy signal")
        elif social_z < -1.5:
            score -= 0.04
            reasons.append(f"Strong bearish sentiment (z={social_z:.1f})")

        # 2. Mention volume trend
        mention_volume = features.get("mention_volume_24h", 0)
        volume_change = features.get("mention_volume_change", 0.0)
        if volume_change > 50 and social_z > 0:
            score += 0.05
            reasons.append(f"Mentions surging +{volume_change:.0f}% — attention spike")
        elif volume_change > 50 and social_z < 0:
            score -= 0.04
            reasons.append(f"Mentions surging bearish — panic attention")

        # 3. Twitter/X specific
        twitter_sentiment = features.get("twitter_sentiment", 0.0)
        twitter_volume = features.get("twitter_volume_24h", 0)
        if twitter_volume > 1000 and abs(twitter_sentiment) > 0.6:
            direction_word = "bullish" if twitter_sentiment > 0 else "bearish"
            score += (0.03 if twitter_sentiment > 0 else -0.03)
            reasons.append(f"Twitter {direction_word} ({twitter_sentiment:.2f}, {twitter_volume} tweets)")

        # 4. Reddit specific
        reddit_sentiment = features.get("reddit_sentiment", 0.0)
        reddit_posts = features.get("reddit_posts_24h", 0)
        if reddit_posts > 50 and abs(reddit_sentiment) > 0.5:
            direction_word = "bullish" if reddit_sentiment > 0 else "bearish"
            score += (0.03 if reddit_sentiment > 0 else -0.03)
            reasons.append(f"Reddit {direction_word} ({reddit_sentiment:.2f}, {reddit_posts} posts)")

        # 5. 4chan /biz/ (crypto-native)
        biz_sentiment = features.get("biz_sentiment", 0.0)
        biz_threads = features.get("biz_threads_24h", 0)
        if biz_threads > 20 and abs(biz_sentiment) > 0.4:
            direction_word = "bullish" if biz_sentiment > 0 else "bearish"
            score += (0.02 if biz_sentiment > 0 else -0.02)
            reasons.append(f"/biz/ {direction_word} ({biz_sentiment:.2f})")

        # 6. Influencer activity
        influencer_score = features.get("influencer_score", 0.0)
        if abs(influencer_score) > 0.3:
            direction_word = "bullish" if influencer_score > 0 else "bearish"
            score += (0.03 if influencer_score > 0 else -0.03)
            reasons.append(f"Influencers turning {direction_word} ({influencer_score:.2f})")

        # 7. Narrative tracking
        narrative_count = features.get("distinct_narratives", 1)
        narrative_coherence = features.get("narrative_coherence", 0.5)
        if narrative_coherence > 0.7 and narrative_count <= 2:
            score += 0.03
            reasons.append(f"Coherent narrative ({narrative_count} themes, coh={narrative_coherence:.2f})")
        elif narrative_count > 5:
            score -= 0.02
            reasons.append(f"Narrative fragmentation ({narrative_count} themes) — no consensus")

        # 8. Sentiment-momentum divergence
        sentiment_momentum_div = features.get("sentiment_momentum_divergence", 0.0)
        if abs(sentiment_momentum_div) > 0.5:
            score += 0.04 if sentiment_momentum_div > 0 else -0.04
            reasons.append(f"Sentiment-price divergence ({sentiment_momentum_div:+.2f}) — reversal signal")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.04 else "bearish" if score < -0.04 else "neutral"
        confidence = min(0.70, 0.32 + abs(score) * 0.42)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "Social sentiment neutral",
            supporting_features=[
                f"social_z={social_z:.2f}",
                f"mention_vol={mention_volume}",
                f"narratives={narrative_count}",
                f"sent_div={sentiment_momentum_div:.2f}",
            ],
        )

    def compute_sentiment_z(self, sentiment: float, history: list[float]) -> float:
        """Compute z-score of current sentiment vs history."""
        if len(history) < 10:
            return 0.0
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance) if variance > 0 else 1.0
        return (sentiment - mean) / std

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            sym = o.prediction.symbol
            self._sentiment_history.setdefault(sym, []).append(
                1.0 if o.was_correct else 0.0
            )
            if len(self._sentiment_history[sym]) > 100:
                self._sentiment_history[sym] = self._sentiment_history[sym][-100:]
