import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class MarketAsset(Base):
    __tablename__ = "market_assets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    asset_type: Mapped[str] = mapped_column(String(10), nullable=False)  # crypto / stock
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    exchange: Mapped[str | None] = mapped_column(String(50))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    signals: Mapped[list["Signal"]] = relationship(back_populates="asset", lazy="selectin")


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset_id: Mapped[str] = mapped_column(String(32), ForeignKey("market_assets.id"), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # bullish / bearish / neutral
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    meta_agent_version: Mapped[str] = mapped_column(String(20), default="0.2.1")
    agent_weights: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    consensus_dispersion: Mapped[float | None] = mapped_column(Float, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")  # pending / active / resolved / expired
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    asset: Mapped["MarketAsset"] = relationship(back_populates="signals")
    predictions: Mapped[list["AgentPrediction"]] = relationship(back_populates="signal", lazy="selectin")


class AgentPrediction(Base):
    __tablename__ = "agent_predictions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    signal_id: Mapped[int] = mapped_column(Integer, ForeignKey("signals.id"), nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    tier: Mapped[int] = mapped_column(Integer, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    vote: Mapped[str] = mapped_column(String(10), nullable=False)  # bullish / bearish / neutral
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    signal: Mapped["Signal"] = relationship(back_populates="predictions")


class AgentPerformance(Base):
    __tablename__ = "agent_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    agent_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    tier: Mapped[int] = mapped_column(Integer, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    accuracy_ema: Mapped[float] = mapped_column(Float, default=0.5)
    total_predictions: Mapped[int] = mapped_column(Integer, default=0)
    correct_predictions: Mapped[int] = mapped_column(Integer, default=0)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
