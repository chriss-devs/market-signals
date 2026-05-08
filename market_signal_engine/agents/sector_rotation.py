"""Sector Rotation Agent — detects capital flows between market sectors.

Identifies rotation patterns: growth↔value, cyclical↔defensive,
tech↔energy, large-cap↔small-cap. Scores momentum of sector flows
and predicts which sectors will outperform.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome


class SectorRotationAgent(BaseAgent):
    name = "Sector Rotation"
    agent_id = 23
    tier = 3
    category = "Cross-Market"
    data_sources = ["yfinance", "finnhub"]

    # Sector definitions
    SECTORS = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
        "XLB": "Materials",
        "XLC": "Communication Services",
    }

    CYCLICAL = {"XLK", "XLY", "XLI", "XLB", "XLE"}
    DEFENSIVE = {"XLP", "XLU", "XLV", "XLRE"}
    GROWTH = {"XLK", "XLC", "XLY"}
    VALUE = {"XLF", "XLE", "XLP", "XLU"}

    def __init__(self) -> None:
        super().__init__()
        self._sector_momentum: dict[str, list[float]] = defaultdict(list)
        self._rotation_signals: list[dict] = []

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol

        reasons: list[str] = []
        score = 0.0

        # 1. Cyclical vs defensive rotation
        cyclical_momentum = features.get("cyclical_momentum", 0.0)
        defensive_momentum = features.get("defensive_momentum", 0.0)
        cycl_def_spread = cyclical_momentum - defensive_momentum

        if cycl_def_spread > 5:
            score += 0.06
            reasons.append(f"Cyclicals leading ({cycl_def_spread:+.1f}%) — risk-on rotation")
        elif cycl_def_spread < -5:
            score -= 0.06
            reasons.append(f"Defensives leading ({cycl_def_spread:+.1f}%) — risk-off rotation")

        # 2. Growth vs value rotation
        growth_momentum = features.get("growth_momentum", 0.0)
        value_momentum = features.get("value_momentum", 0.0)
        gv_spread = growth_momentum - value_momentum

        if gv_spread > 5:
            score += 0.04
            reasons.append(f"Growth outperforming ({gv_spread:+.1f}%)")
        elif gv_spread < -5:
            score -= 0.03
            reasons.append(f"Value rotation ({gv_spread:+.1f}%) — defensive tilt")

        # 3. Large cap vs small cap
        large_cap = features.get("large_cap_momentum", 0.0)
        small_cap = features.get("small_cap_momentum", 0.0)
        cap_spread = small_cap - large_cap

        if cap_spread > 3:
            score += 0.04
            reasons.append(f"Small caps leading ({cap_spread:+.1f}%) — breadth healthy")
        elif cap_spread < -5:
            score -= 0.03
            reasons.append(f"Large caps only ({cap_spread:+.1f}%) — narrow market")

        # 4. Sector breadth
        breadth = features.get("sector_breadth", 0.5)
        if breadth > 0.7:
            score += 0.04
            reasons.append(f"Broad participation ({breadth:.0%} sectors rising)")
        elif breadth < 0.3:
            score -= 0.04
            reasons.append(f"Narrow breadth ({breadth:.0%} sectors) — weak internals")

        # 5. Sector correlation
        sector_correlation = features.get("sector_correlation", 0.6)
        if sector_correlation > 0.85:
            score -= 0.03
            reasons.append(f"High sector correlation ({sector_correlation:.2f}) — macro-driven, no alpha")
        elif sector_correlation < 0.4:
            score += 0.03
            reasons.append(f"Low sector correlation ({sector_correlation:.2f}) — dispersion, stock-picking works")

        # 6. Rotation velocity
        rotation_velocity = features.get("rotation_velocity", 0.0)
        if rotation_velocity > 0.3:
            reasons.append(f"High rotation velocity ({rotation_velocity:.2f}) — regime change")
            score += 0.03 if cycl_def_spread > 0 else -0.03

        # 7. Technology sector specifically
        tech_momentum = features.get("tech_sector_momentum", 0.0)
        if sym in ("AAPL", "NVDA", "MSFT") and tech_momentum > 3:
            score += 0.04
            reasons.append(f"Tech sector strong ({tech_momentum:+.1f}%)")
        elif sym in ("AAPL", "NVDA", "MSFT") and tech_momentum < -3:
            score -= 0.04
            reasons.append(f"Tech sector weak ({tech_momentum:+.1f}%)")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.04 else "bearish" if score < -0.04 else "neutral"
        confidence = min(0.70, 0.33 + abs(score) * 0.40)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "No clear sector rotation signal",
            supporting_features=[
                f"cycl_def={cycl_def_spread:+.1f}%",
                f"growth_value={gv_spread:+.1f}%",
                f"breadth={breadth:.0%}",
                f"rotation_vel={rotation_velocity:.2f}",
            ],
        )

    def detect_rotation(self, sector_returns: dict[str, float]) -> str:
        """Detect the active rotation regime from sector returns."""
        cycl_ret = sum(sector_returns.get(s, 0) for s in self.CYCLICAL) / len(self.CYCLICAL)
        def_ret = sum(sector_returns.get(s, 0) for s in self.DEFENSIVE) / len(self.DEFENSIVE)
        spread = cycl_ret - def_ret

        if spread > 3:
            return "risk_on"
        elif spread < -3:
            return "risk_off"
        return "neutral"

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            self._sector_momentum.setdefault(o.prediction.direction, []).append(
                1.0 if o.was_correct else 0.0
            )
