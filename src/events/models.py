"""
models.py — Pydantic-модели событий для внутренней шины (Event Bus).

Каждое событие — типизированный, сериализуемый объект.
Контракт строгий: модули общаются только через эти модели.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════════════
# Enum типов событий
# ═════════════════════════════════════════════════════════════════════════════

class EventType(str, Enum):
    """Все типы событий в системе."""
    # Ingestion
    NEW_CANDLE = "new_candle"
    NEW_TRADE = "new_trade"
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    CLUSTER_UPDATED = "cluster_updated"
    LIQUIDITY_MAP_UPDATED = "liquidity_map_updated"

    # Wave Engine
    EXTREMUM_DETECTED = "extremum_detected"
    HYPOTHESIS_CREATED = "hypothesis_created"
    HYPOTHESIS_UPDATED = "hypothesis_updated"
    HYPOTHESIS_INVALIDATED = "hypothesis_invalidated"

    # Confluence / ML
    SCORING_COMPLETED = "scoring_completed"
    SIGNAL_GENERATED = "signal_generated"

    # Execution
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    POSITION_CLOSED = "position_closed"

    # HITL
    LABEL_SUBMITTED = "label_submitted"


# ═════════════════════════════════════════════════════════════════════════════
# Базовый класс события
# ═════════════════════════════════════════════════════════════════════════════

class BaseEvent(BaseModel):
    """Базовый класс для всех событий."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str = Field(..., description="Торговая пара, напр. BTCUSDT")


# ═════════════════════════════════════════════════════════════════════════════
# Ingestion Events
# ═════════════════════════════════════════════════════════════════════════════

class NewCandleEvent(BaseEvent):
    """Новая закрытая свеча с биржи."""
    event_type: EventType = EventType.NEW_CANDLE
    timeframe: str = Field(..., description="Таймфрейм: 1m, 5m, 15m, 1h, 4h, 1d, 1w")
    ts: int = Field(..., description="Unix timestamp открытия свечи (ms)")
    open: float
    high: float
    low: float
    close: float
    volume: float


class NewTradeEvent(BaseEvent):
    """Агрегированный трейд (AggTrade) с ленты принтов."""
    event_type: EventType = EventType.NEW_TRADE
    trade_id: int
    price: float
    quantity: float
    is_buyer_maker: bool = Field(..., description="True = продажа (рыночный sell)")
    trade_time: int = Field(..., description="Unix timestamp (ms)")


class OrderBookSnapshotEvent(BaseEvent):
    """Снимок стакана."""
    event_type: EventType = EventType.ORDERBOOK_SNAPSHOT
    bids: List[List[float]] = Field(..., description="[[price, qty], ...]")
    asks: List[List[float]] = Field(..., description="[[price, qty], ...]")
    mid_price: Optional[float] = None


class ClusterUpdatedEvent(BaseEvent):
    """Обновление кластерного профиля (Market Profile) для свечи."""
    event_type: EventType = EventType.CLUSTER_UPDATED
    timeframe: str
    candle_ts: int = Field(..., description="Timestamp свечи, к которой относится кластер")
    poc_price: float = Field(..., description="Point of Control (макс. объём)")
    vah_price: float = Field(..., description="Value Area High")
    val_price: float = Field(..., description="Value Area Low")
    total_volume: float
    delta: float = Field(..., description="Buy volume - Sell volume")
    levels: Dict[str, float] = Field(
        default_factory=dict,
        description="Карта цена→объём для каждого уровня"
    )


class LiquidityMapUpdatedEvent(BaseEvent):
    """Обновление карты ликвидности."""
    event_type: EventType = EventType.LIQUIDITY_MAP_UPDATED
    liquidity_zones: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Зоны ликвидности: [{price, estimated_volume, type: 'stop_cluster'}]"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Wave Engine Events
# ═════════════════════════════════════════════════════════════════════════════

class ExtremumDetectedEvent(BaseEvent):
    """Обнаружен новый экстремум (пик/впадина)."""
    event_type: EventType = EventType.EXTREMUM_DETECTED
    timeframe: str
    price: float
    is_high: bool = Field(..., description="True = локальный максимум")
    detection_method: str = Field(..., description="atr_fractal | find_peaks | cwt")
    confidence: float = Field(default=1.0, description="Уверенность в экстремуме 0..1")


class HypothesisCreatedEvent(BaseEvent):
    """Создана новая волновая гипотеза."""
    event_type: EventType = EventType.HYPOTHESIS_CREATED
    hypothesis_id: str
    pattern_type: str = Field(..., description="IMPULSE, ZIGZAG, FLAT, TRIANGLE, DIAGONAL, WXY")
    degree: str = Field(..., description="Степень волны: Primary, Intermediate, Minor, Minute...")
    direction: str = Field(..., description="BULLISH | BEARISH")
    points: List[Dict[str, Any]] = Field(..., description="Текущие точки гипотезы")
    confidence: float


class HypothesisUpdatedEvent(BaseEvent):
    """Гипотеза обновлена (добавлена новая точка или пересчитан скоринг)."""
    event_type: EventType = EventType.HYPOTHESIS_UPDATED
    hypothesis_id: str
    points: List[Dict[str, Any]]
    confidence: float
    update_reason: str


class HypothesisInvalidatedEvent(BaseEvent):
    """Гипотеза инвалидирована (нарушено абсолютное правило)."""
    event_type: EventType = EventType.HYPOTHESIS_INVALIDATED
    hypothesis_id: str
    invalidation_reason: str
    violated_rule: str = Field(..., description="Какое правило нарушено: W2_BEYOND_W1, W3_SHORTEST, etc.")


# ═════════════════════════════════════════════════════════════════════════════
# Confluence / ML Events
# ═════════════════════════════════════════════════════════════════════════════

class ScoringCompletedEvent(BaseEvent):
    """Завершен ML-скоринг для гипотезы."""
    event_type: EventType = EventType.SCORING_COMPLETED
    hypothesis_id: str
    probability_score: float = Field(..., description="P(setup) от 0 до 1")
    feature_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Значения фич: {fibo_dist: 0.02, vol_zscore: 2.3, ...}"
    )
    passed_threshold: bool


class SignalGeneratedEvent(BaseEvent):
    """Сгенерирован торговый сигнал (прошёл все фильтры)."""
    event_type: EventType = EventType.SIGNAL_GENERATED
    hypothesis_id: str
    direction: str = Field(..., description="LONG | SHORT")
    probability_score: float

    # Execution params
    entry_zone: List[float] = Field(..., description="[min_entry, max_entry]")
    invalidation_stop: float
    take_profit_targets: List[float]

    # Context
    trend_degree: str
    current_wave_hypothesis: str
    confluence_triggers: Dict[str, Any] = Field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# HITL Events
# ═════════════════════════════════════════════════════════════════════════════

class LabelSubmittedEvent(BaseEvent):
    """Эксперт отправил разметку (Accept/Reject/Correct)."""
    event_type: EventType = EventType.LABEL_SUBMITTED
    hypothesis_id: str
    label: int = Field(..., description="1 = Accept, 0 = Reject")
    source: str = Field(..., description="'algorithm' | 'human_corrected'")
    corrected_points: Optional[List[Dict[str, Any]]] = None
    notes: Optional[str] = None
