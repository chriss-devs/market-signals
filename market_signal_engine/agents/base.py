from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence


# ── Data Classes ────────────────────────────────────────────────────────────


@dataclass
class MarketSnapshot:
    """Price and volume data at a point in time for a single asset."""

    symbol: str
    price: float
    volume_24h: float | None = None
    change_24h_pct: float | None = None
    high_24h: float | None = None
    low_24h: float | None = None
    market_cap: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class FeatureSet:
    """Key-value feature map produced by FeatureBuilder or collectors."""

    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisContext:
    """All data available to an agent for a single analysis cycle."""

    symbol: str
    price_history: list[float] = field(default_factory=list)
    volume_history: list[float] = field(default_factory=list)
    features: FeatureSet = field(default_factory=FeatureSet)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Prediction:
    """Single agent prediction output."""

    agent_name: str
    agent_id: int
    symbol: str
    direction: str  # bullish / bearish / neutral
    confidence: float  # 0.0–1.0
    reasoning: str = ""
    supporting_features: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PredictionOutcome:
    """Resolved outcome for a past prediction — used for self-tuning."""

    prediction: Prediction
    actual_direction: str  # bullish / bearish / neutral
    was_correct: bool
    resolved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AgentPerformance:
    """Current performance snapshot for an agent."""

    agent_name: str
    agent_id: int
    tier: int
    category: str
    accuracy_ema: float  # EMA with 0.9 decay
    total_predictions: int
    correct_predictions: int
    weight: float  # current weight in meta-consensus
    calibration_error: float = 0.0  # |confidence - accuracy| gap


# ── Abstract Base Agent ─────────────────────────────────────────────────────


class BaseAgent(ABC):
    """Abstract base for all 26 market analysis agents.

    Each agent receives an AnalysisContext, produces a Prediction, and can
    self-tune its internal parameters from historical outcomes.
    """

    name: str
    agent_id: int
    tier: int
    category: str
    data_sources: list[str] = []

    def __init__(self) -> None:
        self._prediction_count: int = 0
        self._correct_count: int = 0
        self._accuracy_ema: float = 0.5
        self._ema_decay: float = 0.9
        self._calibration_error: float = 0.0
        self._feature_importance: dict[str, float] = {}

    # ── Subclass contract ────────────────────────────────────────────────

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> Prediction:
        """Produce a directional prediction from the analysis context."""
        ...

    @abstractmethod
    def self_tune(self, outcomes: Sequence[PredictionOutcome]) -> None:
        """Adjust internal parameters based on resolved prediction outcomes."""
        ...

    # ── Common behavior ──────────────────────────────────────────────────

    def update_accuracy(self, was_correct: bool) -> None:
        self._prediction_count += 1
        if was_correct:
            self._correct_count += 1
        self._accuracy_ema = (
            self._accuracy_ema * self._ema_decay
            + (1.0 if was_correct else 0.0) * (1.0 - self._ema_decay)
        )

    def calibrate_confidence(self, raw_confidence: float) -> float:
        """Apply calibration scaling so predicted confidence matches realized accuracy."""
        if self._calibration_error > 0.1:
            return raw_confidence * (1.0 - self._calibration_error * 0.5)
        return raw_confidence

    def record_outcomes(self, outcomes: Sequence[PredictionOutcome]) -> None:
        """Update accuracy EMA and calibration error from resolved outcomes."""
        for outcome in outcomes:
            if outcome.prediction.agent_name == self.name:
                self.update_accuracy(outcome.was_correct)
        if self._prediction_count > 0:
            realized_accuracy = self._correct_count / max(self._prediction_count, 1)
            self._calibration_error = abs(
                realized_accuracy - self._accuracy_ema
            )

    def update_feature_importance(self, feature: str, impact: float) -> None:
        """Track which features correlate with correct predictions (EMA)."""
        current = self._feature_importance.get(feature, 0.0)
        self._feature_importance[feature] = current * 0.9 + impact * 0.1

    def get_performance(self) -> AgentPerformance:
        return AgentPerformance(
            agent_name=self.name,
            agent_id=self.agent_id,
            tier=self.tier,
            category=self.category,
            accuracy_ema=round(self._accuracy_ema, 4),
            total_predictions=self._prediction_count,
            correct_predictions=self._correct_count,
            weight=0.0,  # set by PerformanceTracker
            calibration_error=round(self._calibration_error, 4),
        )

    def clamp_confidence(self, confidence: float) -> float:
        return max(0.0, min(1.0, confidence))

    def __repr__(self) -> str:
        return f"<{self.name} tier={self.tier} acc_ema={self._accuracy_ema:.3f}>"
