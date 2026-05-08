"""Fundamental Analysis Agent — PE, EPS, PEG, D/E, FCF yield, analyst consensus.

Self-improvement: learns which fundamentals matter per sector.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import pct_change


class FundamentalAnalysisAgent(BaseAgent):
    name = "Fundamental Analysis"
    agent_id = 5
    tier = 1
    category = "Fundamental"
    data_sources = ["yfinance", "finnhub"]

    # Sector taxonomy
    TECH = {"technology", "software", "semiconductors", "hardware", "electronics"}
    FINANCE = {"financial", "bank", "insurance", "capital markets", "reit"}
    HEALTHCARE = {"healthcare", "biotechnology", "pharmaceuticals", "medical"}
    ENERGY = {"energy", "oil", "gas", "renewable", "utilities"}
    CONSUMER = {"consumer", "retail", "auto", "entertainment", "hospitality"}
    INDUSTRIAL = {"industrial", "manufacturing", "aerospace", "defense", "construction"}

    def __init__(self) -> None:
        super().__init__()
        # Per-sector factor weights — tuned over time
        self._sector_weights: dict[str, dict[str, float]] = {
            "tech": {
                "pe": 0.15, "peg": 0.20, "eps": 0.10, "fcf_yield": 0.18,
                "debt_equity": 0.05, "beta": 0.05, "price_52w": 0.05,
                "analyst": 0.12, "earnings_surprise": 0.10,
            },
            "finance": {
                "pe": 0.20, "peg": 0.10, "eps": 0.15, "fcf_yield": 0.05,
                "debt_equity": 0.18, "beta": 0.10, "price_52w": 0.07,
                "analyst": 0.10, "earnings_surprise": 0.05,
            },
            "healthcare": {
                "pe": 0.12, "peg": 0.22, "eps": 0.15, "fcf_yield": 0.15,
                "debt_equity": 0.08, "beta": 0.05, "price_52w": 0.05,
                "analyst": 0.10, "earnings_surprise": 0.08,
            },
            "energy": {
                "pe": 0.15, "peg": 0.08, "eps": 0.20, "fcf_yield": 0.10,
                "debt_equity": 0.15, "beta": 0.08, "price_52w": 0.07,
                "analyst": 0.10, "earnings_surprise": 0.07,
            },
            "consumer": {
                "pe": 0.18, "peg": 0.15, "eps": 0.15, "fcf_yield": 0.10,
                "debt_equity": 0.10, "beta": 0.07, "price_52w": 0.07,
                "analyst": 0.10, "earnings_surprise": 0.08,
            },
            "industrial": {
                "pe": 0.18, "peg": 0.15, "eps": 0.15, "fcf_yield": 0.12,
                "debt_equity": 0.12, "beta": 0.08, "price_52w": 0.05,
                "analyst": 0.08, "earnings_surprise": 0.07,
            },
        }
        self._default_weights = self._sector_weights["tech"]
        # Per-sector valuation thresholds
        self._pe_fair: dict[str, float] = defaultdict(lambda: 20.0)
        self._peg_fair: dict[str, float] = defaultdict(lambda: 1.5)

    def analyze(self, context: AnalysisContext) -> Prediction:
        features = context.features.features
        sym = context.symbol
        prices = context.price_history

        # ── Extract fundamental signals ───────────────────────────────────
        pe = features.get("pe_ratio", 0.0)
        forward_pe = features.get("forward_pe", 0.0)
        eps = features.get("eps", 0.0)
        div_yield = features.get("dividend_yield", 0.0)
        beta = features.get("beta", 1.0)
        peg = features.get("peg_ratio", 0.0)
        debt_equity = features.get("debt_to_equity", 0.0)
        fcf = features.get("free_cashflow", 0.0)
        market_cap = features.get("market_cap", 0.0)
        price = features.get("price", 0.0)
        w52_high = features.get("52w_high", 0.0)
        w52_low = features.get("52w_low", 0.0)
        sector = features.get("sector", "")
        sector_str = str(sector) if sector else ""

        # Derived metrics
        fcf_yield = (fcf / market_cap * 100) if market_cap and fcf else 0.0
        price_52w = 0.0
        if w52_high and w52_low and price:
            rng = w52_high - w52_low
            price_52w = ((price - w52_low) / rng * 100) if rng > 0 else 50.0

        # Analyst consensus from Finnhub
        analyst_score = features.get("analyst_score", 0.0)      # -1 to +1
        earnings_surprise = features.get("earnings_surprise", 0.0)  # % surprise
        insider_score = features.get("insider_score", 0.0)      # -1 to +1

        # ── Determine sector ──────────────────────────────────────────────
        sector_key = self._classify_sector(sector_str)
        weights = self._sector_weights.get(sector_key, self._default_weights)

        # ── Scoring ───────────────────────────────────────────────────────
        reasons: list[str] = []

        # PE ratio — lower is better (value), but sector-dependent
        pe_score = 0.0
        pe_use = forward_pe if forward_pe and forward_pe > 0 else pe
        pe_fair = self._pe_fair[sector_key]
        if pe_use > pe_fair * 2:
            pe_score = -0.12
            reasons.append(f"PE elevated ({pe_use:.1f}x vs fair {pe_fair:.0f}x)")
        elif pe_use > pe_fair * 1.3:
            pe_score = -0.06
            reasons.append(f"PE above average ({pe_use:.1f}x)")
        elif 0 < pe_use < pe_fair * 0.7:
            pe_score = 0.12
            reasons.append(f"PE attractive ({pe_use:.1f}x — value)")
        elif 0 < pe_use <= pe_fair * 1.3:
            pe_score = 0.05
            reasons.append(f"PE fair ({pe_use:.1f}x)")
        elif pe_use <= 0:
            pe_score = -0.06
            reasons.append(f"Negative earnings — PE N/A")
        else:
            pe_score = -0.03
            reasons.append(f"PE unknown")

        # PEG ratio — growth at reasonable price
        peg_score = 0.0
        peg_fair = self._peg_fair[sector_key]
        if 0 < peg < 0.8:
            peg_score = 0.15
            reasons.append(f"PEG attractive ({peg:.2f} — undervalued growth)")
        elif 0 < peg <= peg_fair:
            peg_score = 0.08
            reasons.append(f"PEG reasonable ({peg:.2f})")
        elif peg > 3.0:
            peg_score = -0.10
            reasons.append(f"PEG extreme ({peg:.2f} — overvalued)")
        elif peg > peg_fair:
            peg_score = -0.05
            reasons.append(f"PEG elevated ({peg:.2f})")
        elif peg <= 0:
            peg_score = -0.04
            reasons.append("Negative PEG")

        # EPS — earnings power
        eps_score = 0.0
        if eps > 0:
            eps_score = 0.03
            if eps > 10:
                eps_score = 0.08
                reasons.append(f"Strong EPS (${eps:.2f})")
            else:
                reasons.append(f"Positive EPS (${eps:.2f})")
        else:
            eps_score = -0.06
            reasons.append(f"Negative EPS (${eps:.2f})")

        # Free cash flow yield
        fcf_score = 0.0
        if fcf_yield > 8:
            fcf_score = 0.15
            reasons.append(f"High FCF yield ({fcf_yield:.1f}%)")
        elif fcf_yield > 4:
            fcf_score = 0.08
            reasons.append(f"Good FCF yield ({fcf_yield:.1f}%)")
        elif fcf_yield > 0:
            fcf_score = 0.02
        elif fcf < 0 and market_cap > 0:
            fcf_score = -0.08
            reasons.append(f"Negative free cash flow")

        # Debt/Equity — lower is healthier
        de_score = 0.0
        if 0 < debt_equity < 50:
            de_score = 0.06
            reasons.append(f"Low D/E ({debt_equity:.1f})")
        elif debt_equity > 200:
            de_score = -0.10
            reasons.append(f"High D/E ({debt_equity:.1f})")
        elif debt_equity > 100:
            de_score = -0.05
            reasons.append(f"Elevated D/E ({debt_equity:.1f})")

        # Beta
        beta_score = 0.0
        if 0 < beta < 0.7:
            beta_score = 0.03
            reasons.append(f"Low beta ({beta:.2f}) — defensive")
        elif beta > 2.0:
            beta_score = -0.06
            reasons.append(f"High beta ({beta:.2f}) — volatile")
        elif beta > 1.5:
            beta_score = -0.03
            reasons.append(f"Above-market beta ({beta:.2f})")

        # 52-week position
        pos52_score = 0.0
        if price_52w < 20:
            pos52_score = 0.08
            reasons.append(f"Near 52w low ({price_52w:.0f}%) — potential value")
        elif price_52w > 90:
            pos52_score = -0.06
            reasons.append(f"Near 52w high ({price_52w:.0f}%) — extended")
        elif price_52w > 70:
            pos52_score = -0.03

        # Analyst consensus
        analyst_sc = 0.0
        if analyst_score > 0.3:
            analyst_sc = 0.10
            reasons.append(f"Strong analyst consensus ({analyst_score:+.2f})")
        elif analyst_score > 0.1:
            analyst_sc = 0.05
            reasons.append(f"Positive analyst tilt ({analyst_score:+.2f})")
        elif analyst_score < -0.3:
            analyst_sc = -0.10
            reasons.append(f"Bearish analyst consensus ({analyst_score:+.2f})")
        elif analyst_score < -0.1:
            analyst_sc = -0.05
            reasons.append(f"Negative analyst tilt ({analyst_score:+.2f})")

        # Earnings surprise
        earn_sc = 0.0
        if earnings_surprise > 10:
            earn_sc = 0.12
            reasons.append(f"Strong earnings beat ({earnings_surprise:+.1f}%)")
        elif earnings_surprise > 3:
            earn_sc = 0.06
            reasons.append(f"Earnings beat ({earnings_surprise:+.1f}%)")
        elif earnings_surprise < -10:
            earn_sc = -0.12
            reasons.append(f"Major earnings miss ({earnings_surprise:+.1f}%)")
        elif earnings_surprise < -3:
            earn_sc = -0.06
            reasons.append(f"Earnings miss ({earnings_surprise:+.1f}%)")

        # Insider sentiment
        insider_sc = 0.0
        if insider_score > 0.2:
            insider_sc = 0.06
            reasons.append("Net insider buying")
        elif insider_score < -0.2:
            insider_sc = -0.06
            reasons.append("Net insider selling")

        # Dividend yield — income factor
        div_score = 0.0
        if div_yield is not None and div_yield > 4:
            div_score = 0.06
            reasons.append(f"High dividend yield ({div_yield:.1f}%)")
        elif div_yield is not None and div_yield > 2:
            div_score = 0.03
            reasons.append(f"Moderate dividend ({div_yield:.1f}%)")

        # ── Weighted aggregate ──────────────────────────────────────────
        raw_score = (
            weights["pe"] * pe_score
            + weights["peg"] * peg_score
            + weights["eps"] * eps_score
            + weights["fcf_yield"] * fcf_score
            + weights["debt_equity"] * de_score
            + weights["beta"] * beta_score
            + weights["price_52w"] * pos52_score
            + weights["analyst"] * analyst_sc
            + weights["earnings_surprise"] * earn_sc
            + 0.05 * insider_sc  # small fixed weight
            + 0.03 * div_score   # small fixed weight
        )

        score = max(-1.0, min(1.0, raw_score))

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

        # Price momentum overlay — fundamentals + positive momentum = stronger conviction
        change_30d = pct_change(prices, 30) if len(prices) > 30 else 0
        if direction == "bullish" and change_30d > 0:
            confidence = min(0.85, confidence * 1.1)
            reasons.append("Price trend confirms bullish fundamentals")
        elif direction == "bearish" and change_30d < 0:
            confidence = min(0.85, confidence * 1.1)
            reasons.append("Price trend confirms bearish fundamentals")
        elif direction != "neutral" and change_30d * (1 if direction == "bullish" else -1) < 0:
            confidence = max(0.35, confidence * 0.9)
            reasons.append("Price trend diverges from fundamentals")

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons[:5]) if reasons else "Mixed fundamental signals",
            supporting_features=[
                f"pe={pe:.1f}", f"peg={peg:.2f}", f"fcf_yield={fcf_yield:.1f}",
                f"de={debt_equity:.1f}", f"analyst={analyst_score:.2f}",
                f"sector={sector_key}",
            ],
        )

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        for o in outcomes:
            if o.prediction.agent_name != self.name:
                continue
            for feat in o.prediction.supporting_features:
                if feat.startswith("sector="):
                    continue
                factor = feat.split("=")[0]
                # Find which weight key this factor maps to
                sector_key = "tech"
                for sf in o.prediction.supporting_features:
                    if sf.startswith("sector="):
                        sector_key = sf.split("=")[1]
                        break
                weights = self._sector_weights.get(sector_key, self._default_weights)
                if factor in weights:
                    delta = 0.02 if o.was_correct else -0.03
                    weights[factor] = max(0.03, min(0.30, weights[factor] + delta))
                    # Renormalize
                    total = sum(weights.values())
                    if total > 0:
                        for k in weights:
                            weights[k] /= total
                impact = 0.04 if o.was_correct else -0.02
                self.update_feature_importance(feat, impact)
            # Tune sector PE fair value
            if not o.was_correct:
                sector_key = "tech"
                for sf in o.prediction.supporting_features:
                    if sf.startswith("sector="):
                        sector_key = sf.split("=")[1]
                        break
                if o.prediction.direction == "bullish":
                    self._pe_fair[sector_key] = max(10.0, self._pe_fair[sector_key] - 0.5)
                elif o.prediction.direction == "bearish":
                    self._pe_fair[sector_key] = min(35.0, self._pe_fair[sector_key] + 0.5)

    def _classify_sector(self, sector: str) -> str:
        s = sector.lower()
        for kw in self.TECH:
            if kw in s:
                return "tech"
        for kw in self.FINANCE:
            if kw in s:
                return "finance"
        for kw in self.HEALTHCARE:
            if kw in s:
                return "healthcare"
        for kw in self.ENERGY:
            if kw in s:
                return "energy"
        for kw in self.CONSUMER:
            if kw in s:
                return "consumer"
        for kw in self.INDUSTRIAL:
            if kw in s:
                return "industrial"
        return "tech"
