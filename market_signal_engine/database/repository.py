from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from market_signal_engine.database.connection import get_session
from market_signal_engine.database.models import AgentPerformance, AgentPrediction, MarketAsset, Signal


def _session() -> Session:
    return get_session()


# ── Market Assets ───────────────────────────────────────────────────────────

def get_or_create_asset(symbol: str, asset_type: str, name: str = "", exchange: str | None = None) -> MarketAsset:
    s = _session()
    try:
        asset = s.execute(select(MarketAsset).where(MarketAsset.symbol == symbol)).scalar_one_or_none()
        if not asset:
            asset = MarketAsset(symbol=symbol, asset_type=asset_type, name=name or symbol, exchange=exchange)
            s.add(asset)
            s.commit()
            s.refresh(asset)
        return asset
    finally:
        s.close()


def get_assets(active_only: bool = True) -> Sequence[MarketAsset]:
    s = _session()
    try:
        q = select(MarketAsset)
        if active_only:
            q = q.where(MarketAsset.is_active == True)
        return s.execute(q).scalars().all()
    finally:
        s.close()


# ── Signals ────────────────────────────────────────────────────────────────

def create_signal(
    asset_id: str,
    direction: str,
    confidence: float,
    agent_weights: dict | None = None,
    consensus_dispersion: float | None = None,
    meta_agent_version: str = "0.2.1",
    price: float | None = None,
    entry_price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> Signal:
    s = _session()
    try:
        signal = Signal(
            asset_id=asset_id,
            direction=direction,
            confidence=confidence,
            agent_weights=agent_weights,
            consensus_dispersion=consensus_dispersion,
            meta_agent_version=meta_agent_version,
            price=price,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        s.add(signal)
        s.commit()
        s.refresh(signal)
        return signal
    finally:
        s.close()


def get_signals(limit: int = 20, status: str | None = None) -> Sequence[Signal]:
    s = _session()
    try:
        q = select(Signal).order_by(Signal.created_at.desc()).limit(limit)
        if status:
            q = q.where(Signal.status == status)
        return s.execute(q).scalars().all()
    finally:
        s.close()


def get_signal(signal_id: int) -> Signal | None:
    s = _session()
    try:
        return s.execute(select(Signal).where(Signal.id == signal_id)).scalar_one_or_none()
    finally:
        s.close()


def resolve_signal(signal_id: int, actual_direction: str) -> Signal | None:
    """Mark signal resolved after outcome is known."""
    s = _session()
    try:
        signal = s.execute(select(Signal).where(Signal.id == signal_id)).scalar_one_or_none()
        if signal:
            signal.status = "resolved"
            signal.resolved_at = datetime.now(timezone.utc)
            s.commit()
            s.refresh(signal)
        return signal
    finally:
        s.close()


# ── Agent Predictions ──────────────────────────────────────────────────────

def create_prediction(
    signal_id: int,
    agent_name: str,
    agent_id: int,
    tier: int,
    category: str,
    vote: str,
    confidence: float,
    reasoning: str | None = None,
) -> AgentPrediction:
    s = _session()
    try:
        pred = AgentPrediction(
            signal_id=signal_id,
            agent_name=agent_name,
            agent_id=agent_id,
            tier=tier,
            category=category,
            vote=vote,
            confidence=confidence,
            reasoning=reasoning,
        )
        s.add(pred)
        s.commit()
        s.refresh(pred)
        return pred
    finally:
        s.close()


def get_predictions_for_signal(signal_id: int) -> Sequence[AgentPrediction]:
    s = _session()
    try:
        return s.execute(
            select(AgentPrediction).where(AgentPrediction.signal_id == signal_id)
        ).scalars().all()
    finally:
        s.close()


def get_prediction_history(agent_name: str, limit: int = 100) -> Sequence[AgentPrediction]:
    s = _session()
    try:
        return s.execute(
            select(AgentPrediction)
            .where(AgentPrediction.agent_name == agent_name)
            .order_by(AgentPrediction.created_at.desc())
            .limit(limit)
        ).scalars().all()
    finally:
        s.close()


# ── Agent Performance ──────────────────────────────────────────────────────

def upsert_agent_performance(
    agent_name: str,
    agent_id: int,
    tier: int,
    category: str,
    accuracy_ema: float,
    total_predictions: int,
    correct_predictions: int,
    weight: float,
) -> AgentPerformance:
    s = _session()
    try:
        perf = s.execute(
            select(AgentPerformance).where(AgentPerformance.agent_id == agent_id)
        ).scalar_one_or_none()
        if perf:
            perf.accuracy_ema = accuracy_ema
            perf.total_predictions = total_predictions
            perf.correct_predictions = correct_predictions
            perf.weight = weight
            perf.last_updated = datetime.now(timezone.utc)
        else:
            perf = AgentPerformance(
                agent_name=agent_name,
                agent_id=agent_id,
                tier=tier,
                category=category,
                accuracy_ema=accuracy_ema,
                total_predictions=total_predictions,
                correct_predictions=correct_predictions,
                weight=weight,
            )
            s.add(perf)
        s.commit()
        s.refresh(perf)
        return perf
    finally:
        s.close()


def get_all_agent_performance() -> Sequence[AgentPerformance]:
    s = _session()
    try:
        return s.execute(select(AgentPerformance).order_by(AgentPerformance.weight.desc())).scalars().all()
    finally:
        s.close()


def get_agent_performance(agent_name: str) -> AgentPerformance | None:
    s = _session()
    try:
        return s.execute(
            select(AgentPerformance).where(AgentPerformance.agent_name == agent_name)
        ).scalar_one_or_none()
    finally:
        s.close()


def update_agent_weight(agent_id: int, weight: float) -> None:
    s = _session()
    try:
        s.execute(update(AgentPerformance).where(AgentPerformance.agent_id == agent_id).values(weight=weight))
        s.commit()
    finally:
        s.close()


def get_accuracy_stats() -> dict:
    s = _session()
    try:
        result = s.execute(
            select(
                func.avg(AgentPerformance.accuracy_ema).label("avg_accuracy"),
                func.count(AgentPerformance.id).label("total"),
            )
        ).one()
        return {
            "avg_accuracy": round(result.avg_accuracy or 0, 3),
            "total_agents": result.total,
        }
    finally:
        s.close()


def get_signals_today_count() -> int:
    s = _session()
    try:
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return s.execute(
            select(func.count(Signal.id)).where(Signal.created_at >= today)
        ).scalar_one()
    finally:
        s.close()


def get_top_agent() -> AgentPerformance | None:
    s = _session()
    try:
        return s.execute(
            select(AgentPerformance).order_by(AgentPerformance.accuracy_ema.desc()).limit(1)
        ).scalar_one_or_none()
    finally:
        s.close()


def get_active_agent_count() -> int:
    """Agents that have made at least one prediction."""
    s = _session()
    try:
        return s.execute(
            select(func.count(AgentPerformance.id)).where(AgentPerformance.total_predictions > 0)
        ).scalar_one()
    finally:
        s.close()


# ── Seed data ────────────────────────────────────────────────────────────────

AGENT_SEED_DATA = [
    {"agent_name": "Technical Analysis",   "agent_id": 1,  "tier": 1, "category": "Technical"},
    {"agent_name": "Sentiment Analysis",   "agent_id": 2,  "tier": 1, "category": "Sentiment"},
    {"agent_name": "On-Chain Analysis",    "agent_id": 3,  "tier": 1, "category": "On-Chain"},
    {"agent_name": "Macro Analysis",       "agent_id": 4,  "tier": 1, "category": "Macro"},
    {"agent_name": "Fundamental Analysis",  "agent_id": 5,  "tier": 1, "category": "Fundamental"},
    {"agent_name": "Pattern Recognition",  "agent_id": 6,  "tier": 1, "category": "Technical"},
    {"agent_name": "Volume Profile",       "agent_id": 7,  "tier": 1, "category": "Technical"},
    {"agent_name": "Market Structure",     "agent_id": 8,  "tier": 1, "category": "Technical"},
    {"agent_name": "Momentum",             "agent_id": 9,  "tier": 1, "category": "Momentum"},
    {"agent_name": "Volatility",           "agent_id": 10, "tier": 1, "category": "Volatility"},
    {"agent_name": "Fear & Greed",         "agent_id": 11, "tier": 1, "category": "Sentiment"},
    {"agent_name": "Whale Tracking",       "agent_id": 12, "tier": 1, "category": "On-Chain"},
    {"agent_name": "Meta-Agent",           "agent_id": 13, "tier": 1, "category": "Meta"},
    {"agent_name": "Backtesting",          "agent_id": 14, "tier": 1, "category": "Validation"},
    {"agent_name": "Calibration",          "agent_id": 15, "tier": 1, "category": "Validation"},
    {"agent_name": "Correlation",          "agent_id": 16, "tier": 2, "category": "Cross-Market"},
    {"agent_name": "Intermarket",          "agent_id": 17, "tier": 2, "category": "Cross-Market"},
    {"agent_name": "DeFi",                 "agent_id": 18, "tier": 2, "category": "DeFi"},
    {"agent_name": "L1/L2 Analysis",       "agent_id": 19, "tier": 2, "category": "Chain"},
    {"agent_name": "Economic Calendar",    "agent_id": 20, "tier": 2, "category": "Macro"},
    {"agent_name": "Social Media",         "agent_id": 21, "tier": 2, "category": "Sentiment"},
    {"agent_name": "Alert/Recommendation", "agent_id": 22, "tier": 2, "category": "Signal"},
    {"agent_name": "Sector Rotation",      "agent_id": 23, "tier": 3, "category": "Cross-Market"},
    {"agent_name": "Statistical Arbitrage","agent_id": 24, "tier": 3, "category": "Quant"},
    {"agent_name": "Geopolitical",         "agent_id": 25, "tier": 3, "category": "Macro"},
    {"agent_name": "NFT/Gaming",           "agent_id": 26, "tier": 3, "category": "Alternative"},
]


def seed_agent_performance() -> None:
    """Ensure all 26 agents have a row in agent_performance. Safe to call repeatedly."""
    s = _session()
    try:
        for entry in AGENT_SEED_DATA:
            existing = s.execute(
                select(AgentPerformance).where(AgentPerformance.agent_id == entry["agent_id"])
            ).scalar_one_or_none()
            if not existing:
                perf = AgentPerformance(
                    agent_name=entry["agent_name"],
                    agent_id=entry["agent_id"],
                    tier=entry["tier"],
                    category=entry["category"],
                    accuracy_ema=0.5,
                    total_predictions=0,
                    correct_predictions=0,
                    weight=0.0,
                )
                s.add(perf)
        s.commit()
    finally:
        s.close()
