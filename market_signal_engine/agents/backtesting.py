"""Backtesting Agent — validates agent predictions against historical data.

Self-improvement: discovers which agent combinations had best historical performance.
Computes Sharpe, Calmar, win rate, profit factor from prediction outcomes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Sequence

from market_signal_engine.agents.base import AnalysisContext, BaseAgent, Prediction, PredictionOutcome
from market_signal_engine.agents.indicators import pct_change


class BacktestingAgent(BaseAgent):
    name = "Backtesting"
    agent_id = 14
    tier = 1
    category = "Validation"
    data_sources = ["prediction_history", "price_history"]

    def __init__(self) -> None:
        super().__init__()
        # Historical backtest results per agent
        self._backtests: dict[str, dict] = defaultdict(lambda: {
            "signals": [],
            "win_rate": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
            "avg_return": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
        })
        # Per-agent-combination performance tracking
        self._combo_performance: dict[str, dict] = defaultdict(lambda: {
            "signals": 0, "wins": 0, "avg_confidence": 0.0,
        })

    def analyze(self, context: AnalysisContext) -> Prediction:
        """Backtesting doesn't produce directional predictions — it validates others.

        This analyze() returns a meta-prediction about the prediction system's health.
        """
        prices = context.price_history
        features = context.features.features

        # Extract past prediction outcomes for evaluation
        prediction_count = int(features.get("prediction_count", 0))
        recent_win_rate = features.get("recent_win_rate", 0.0)

        reasons: list[str] = []
        score = 0.0

        if prediction_count < 10:
            return Prediction(
                agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
                direction="neutral", confidence=0.3,
                reasoning="Insufficient predictions for backtest (<10)",
            )

        # System health assessment based on backtest results
        if recent_win_rate > 0.60:
            score += 0.15
            reasons.append(f"System performing well (win rate={recent_win_rate:.0%})")
        elif recent_win_rate > 0.50:
            score += 0.05
            reasons.append(f"System above random (win rate={recent_win_rate:.0%})")
        elif recent_win_rate < 0.40:
            score -= 0.10
            reasons.append(f"System underperforming (win rate={recent_win_rate:.0%})")

        # Sharpe ratio check
        sharpe = features.get("sharpe", 0.0)
        if sharpe > 1.5:
            score += 0.10
            reasons.append(f"Strong risk-adjusted returns (Sharpe={sharpe:.2f})")
        elif sharpe < 0:
            score -= 0.08
            reasons.append(f"Negative Sharpe ratio ({sharpe:.2f})")

        # Profit factor check
        profit_factor = features.get("profit_factor", 0.0)
        if profit_factor > 2.0:
            score += 0.08
            reasons.append(f"Excellent profit factor ({profit_factor:.2f})")
        elif profit_factor < 1.0:
            score -= 0.06
            reasons.append(f"Losing profit factor ({profit_factor:.2f})")

        score = max(-1.0, min(1.0, score))

        direction = "bullish" if score > 0.03 else "bearish" if score < -0.03 else "neutral"
        confidence = min(0.70, 0.35 + abs(score) * 0.4)

        return Prediction(
            agent_name=self.name, agent_id=self.agent_id, symbol=context.symbol,
            direction=direction, confidence=round(confidence, 4),
            reasoning="; ".join(reasons) if reasons else "Backtest results neutral",
            supporting_features=[
                f"win_rate={recent_win_rate:.3f}", f"sharpe={sharpe:.2f}",
                f"profit_factor={profit_factor:.2f}", f"trades={prediction_count}",
            ],
        )

    # ── Backtesting computation methods ──────────────────────────────────

    def run_backtest(
        self,
        outcomes: list[PredictionOutcome],
        prices: list[float],
        risk_free_rate: float = 0.04,
    ) -> dict:
        """Compute comprehensive backtest metrics from a series of outcomes."""
        if len(outcomes) < 5:
            return {"error": "Need at least 5 predictions for meaningful backtest"}

        # Simulate returns: for each bullish prediction, go long 7d; bearish, short 7d
        returns: list[float] = []
        equity_curve: list[float] = [1.0]
        wins = 0
        total = 0

        for outcome in outcomes:
            total += 1
            pred = outcome.prediction
            # Use confidence as position size (0.5-1.0x)
            position = 0.5 + pred.confidence * 0.5

            if outcome.was_correct:
                wins += 1
                ret = position * 0.02  # assume ~2% move per correct signal
            else:
                ret = -position * 0.02

            returns.append(ret)
            equity_curve.append(equity_curve[-1] * (1 + ret))

        win_rate = wins / max(total, 1)
        avg_ret = sum(returns) / max(len(returns), 1)
        std_ret = math.sqrt(sum((r - avg_ret) ** 2 for r in returns) / max(len(returns), 1)) if len(returns) > 1 else 0

        # Sharpe ratio (annualized, assume daily signals)
        sharpe = ((avg_ret - risk_free_rate / 365) / std_ret * math.sqrt(252)) if std_ret > 0 else 0

        # Calmar ratio
        max_dd = self._max_drawdown(equity_curve)
        calmar = (avg_ret * 252) / max_dd if max_dd > 0 else 0

        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "win_rate": round(win_rate, 4),
            "sharpe": round(sharpe, 4),
            "calmar": round(calmar, 4),
            "profit_factor": round(profit_factor, 4),
            "avg_return": round(avg_ret, 6),
            "max_drawdown": round(max_dd, 4),
            "total_trades": total,
            "winning_trades": wins,
            "total_return": round(equity_curve[-1] - 1, 4),
        }

    def backtest_agent(
        self, agent_name: str, outcomes: list[PredictionOutcome], prices: list[float]
    ) -> dict:
        """Run backtest for a specific agent's predictions."""
        agent_outcomes = [o for o in outcomes if o.prediction.agent_name == agent_name]
        result = self.run_backtest(agent_outcomes, prices)
        self._backtests[agent_name] = result
        return result

    def backtest_combination(
        self, agent_names: list[str], outcomes: list[PredictionOutcome], prices: list[float]
    ) -> dict:
        """Test a combination of agents — how well do they perform together?"""
        combo_key = "+".join(sorted(agent_names))
        filtered = [o for o in outcomes if o.prediction.agent_name in agent_names]
        result = self.run_backtest(filtered, prices)
        self._combo_performance[combo_key] = {
            "signals": result.get("total_trades", 0),
            "wins": result.get("winning_trades", 0),
            "avg_confidence": sum(o.prediction.confidence for o in filtered) / max(len(filtered), 1),
        }
        return result

    def get_best_agents(self, min_trades: int = 10) -> list[tuple[str, dict]]:
        """Return agents ranked by Sharpe ratio."""
        ranked = [
            (name, bt) for name, bt in self._backtests.items()
            if bt.get("total_trades", 0) >= min_trades
        ]
        ranked.sort(key=lambda x: x[1].get("sharpe", -999), reverse=True)
        return ranked

    @staticmethod
    def _max_drawdown(equity: list[float]) -> float:
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        self.record_outcomes(outcomes)
        # Backtesting agent tunes itself by updating stored metrics
        for o in outcomes:
            impact = 0.03 if o.was_correct else -0.02
            self.update_feature_importance(o.prediction.agent_name, impact)
