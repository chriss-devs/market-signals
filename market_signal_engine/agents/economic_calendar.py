"""Economic Calendar Agent — event-driven risk assessment.

Evaluates upcoming economic events (FOMC, CPI, NFP, GDP) for market impact.
Tracks surprise vs consensus and adjusts risk posture accordingly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class EconomicCalendarAgent(BaseAgent):
    name = "Economic Calendar"
    agent_id = 20
    tier = 2
    category = "Macro"
    data_sources = ["finnhub"]

    # Event importance multipliers (higher = bigger market impact)
    EVENT_IMPACT = {
        "fomc": 1.0,
        "cpi": 0.9,
        "ppi": 0.7,
        "nfp": 0.85,
        "gdp": 0.75,
        "retail_sales": 0.6,
        "ism_manufacturing": 0.55,
        "ism_services": 0.5,
        "consumer_confidence": 0.4,
        "initial_claims": 0.35,
        "housing_starts": 0.3,
    }

    # Proximity decay: how event impact decays with distance (days)
    PROXIMITY_WEIGHTS = {0: 1.0, 1: 0.9, 2: 0.7, 3: 0.5, 4: 0.35, 5: 0.2, 6: 0.1, 7: 0.05}

    def __init__(self) -> None:
        super().__init__()
        self._event_history: list[dict] = []
        self._surprise_memory: dict[str, list[float]] = defaultdict(list)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. Upcoming event density
        event_density = features.get("event_density_7d", 0)
        if event_density > 5:
            reasons.append(f"High event density ({event_density:.0f} in 7d) — volatile week ahead")
            score -= 0.02
        elif event_density < 2:
            score += 0.02
            reasons.append(f"Light calendar ({event_density:.0f} events) — technicals dominate")

        # 2. Proximity to high-impact events
        next_high_impact = features.get("next_high_impact_days", 7)
        if next_high_impact <= 1:
            reasons.append(f"High-impact event within {next_high_impact:.0f}d — elevated uncertainty")
            score -= 0.04
        elif next_high_impact > 5:
            score += 0.02

        # 3. FOMC posture
        fomc_days = features.get("days_to_fomc", 30)
        fomc_hawkish = features.get("fomc_hawkish_score", 0.0)
        if fomc_days <= 7 and fomc_hawkish > 0.3:
            score -= 0.06
            reasons.append(f"FOMC in {fomc_days:.0f}d — hawkish bias ({fomc_hawkish:.2f})")
        elif fomc_days <= 7 and fomc_hawkish < -0.3:
            score += 0.05
            reasons.append(f"FOMC in {fomc_days:.0f}d — dovish bias ({fomc_hawkish:.2f})")

        # 4. Economic surprise index
        surprise_index = features.get("economic_surprise_index", 0.0)
        surprise_trend = features.get("surprise_trend", 0.0)
        if surprise_index > 0.2 and surprise_trend > 0:
            score += 0.05
            reasons.append(f"Positive econ surprises ({surprise_index:.2f}) — growth beats")
        elif surprise_index < -0.2 and surprise_trend < 0:
            score -= 0.05
            reasons.append(f"Negative econ surprises ({surprise_index:.2f}) — growth misses")

        # 5. Inflation trajectory
        cpi_surprise = features.get("cpi_surprise_last", 0.0)
        core_pce = features.get("core_pce_yoy", 3.0)
        if cpi_surprise < -0.2:
            score += 0.04
            reasons.append(f"CPI below consensus ({cpi_surprise:+.1f}%) — disinflation")
        elif cpi_surprise > 0.2:
            score -= 0.04
            reasons.append(f"CPI above consensus ({cpi_surprise:+.1f}%) — inflation sticky")

        if core_pce < 2.5:
            score += 0.03
            reasons.append(f"Core PCE near target ({core_pce:.1f}%) — Fed pivot possible")

        # 6. Employment health
        nfp_surprise = features.get("nfp_surprise_last", 0)
        unemp_rate = features.get("unemployment_rate", 4.0)
        if nfp_surprise > 100_000:
            score += 0.03
            reasons.append(f"NFP beat +{nfp_surprise/1000:.0f}K — labor strong")
        elif nfp_surprise < -100_000:
            score -= 0.03
            reasons.append(f"NFP miss {nfp_surprise/1000:.0f}K — labor weakening")

        if unemp_rate > 4.5:
            score -= 0.03
            reasons.append(f"Unemployment elevated ({unemp_rate:.1f}%) — recession risk")

        # 7. GDP growth
        gdp_growth = features.get("gdp_growth_qoq", 2.0)
        gdp_surprise = features.get("gdp_surprise_last", 0.0)
        if gdp_growth > 3.0 and gdp_surprise > 0:
            score += 0.04
            reasons.append(f"GDP strong +{gdp_growth:.1f}% — economic expansion")
        elif gdp_growth < 1.0:
            score -= 0.04
            reasons.append(f"GDP sluggish ({gdp_growth:.1f}%) — growth concern")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.03 else "bearish" if score < -0.03 else "neutral"
        confidence = min(0.70, 0.33 + abs(score) * 0.40)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "Macro calendar quiet",
            supporting_features=[
                f"events_7d={event_density:.0f}",
                f"surprise_idx={surprise_index:.2f}",
                f"cpi_surprise={cpi_surprise:.2f}",
                f"gdp={gdp_growth:.1f}%",
            ],
        )

    def event_risk_score(self, features: dict) -> float:
        """Compute a forward-looking event risk score (0-1, higher = riskier)."""
        event_density = features.get("event_density_7d", 0)
        next_impact = features.get("next_high_impact_days", 7)
        surprise_vol = features.get("surprise_volatility", 0.1)

        density_score = min(1.0, event_density / 8)
        proximity_score = max(0.0, 1.0 - next_impact / 7)
        surprise_score = min(1.0, surprise_vol / 0.5)

        return round((density_score * 0.4 + proximity_score * 0.35 + surprise_score * 0.25), 3)

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            self._surprise_memory.setdefault(o.prediction.direction, []).append(
                1.0 if o.was_correct else 0.0
            )
