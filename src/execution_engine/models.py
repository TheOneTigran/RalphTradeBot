"""
models.py — Машина состояний жизненного цикла ордера.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.wave_strategy.models import DtwTradePlan

class OrderStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FILLED_TP = "FILLED_TP"
    FILLED_SL = "FILLED_SL"
    CANCELLED_TIME = "CANCELLED_TIME"
    CANCELLED_INVALID = "CANCELLED_INVALID"
    CANCELLED_POOR_RR = "CANCELLED_POOR_RR"
    REJECTED_MARGIN = "REJECTED_MARGIN"
    CLOSED_TIME_TTL = "CLOSED_TIME_TTL"
    WAITING_PULLBACK = "WAITING_PULLBACK"

@dataclass
class SimulatedOrder:
    """Обёртка над DtwTradePlan, отслеживающая жизнь ордера в бэктестере."""
    plan: DtwTradePlan
    status: OrderStatus = OrderStatus.PENDING
    
    # Детали исполнения
    entry_time: Optional[int] = None
    exit_time: Optional[int] = None
    fill_price: Optional[float] = None
    close_price: Optional[float] = None
    
    # Трекинг
    bars_held: int = 0
    pnl_usd: float = 0.0
    drawdown_pct: float = 0.0
    max_favorable_price: float = 0.0
    pullback_limit: Optional[float] = None
    
    # Для управления деньгами
    position_size: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.plan.direction == "LONG"
