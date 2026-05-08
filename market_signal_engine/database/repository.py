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
