"""
base_strategy.py — Абстрактный базовый класс для торговых стратегий в strategies_lab.

Каждая стратегия инкапсулирует:
1. filter_and_trigger: проверку условий входа после формирования импульса Эллиотта.
2. manage_position: сопровождение открытой позиции (проверка ТП, СЛ, БУ или трейлинга).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class TradeSignal:
    """Исходный обнаруженный импульс Эллиотта."""
    symbol: str
    interval: str
    direction: str  # "LONG" или "SHORT"
    w5_bar: int
    w5_time: Any
    w5_price: float
    w3_bar: int
    w3_price: float
    w0_bar: int
    w0_price: float
    algo_score: float
    wave_points: Dict[str, float]
    raw_df: pd.DataFrame  # Срез свечей основного ТФ (15m)
    df_1m: Optional[pd.DataFrame] = None  # Минутные котировки для Multi-TF анализа


@dataclass
class EntryDecision:
    """Решение стратегии о входе в сделку."""
    should_enter: bool
    entry_bar: int = 0
    entry_time: Any = None
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_levels: List[Dict[str, Any]] = field(default_factory=list)  # [{"price": ..., "share": ...}]
    reason: str = ""
    slippage_tax_pct: float = 0.0  # % проскальзывания от дистанции до TP1
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeResult:
    """Результат выполнения сделки с учетом реалистичных комиссий и банкролла."""
    signal_id: str
    symbol: str
    interval: str
    direction: str
    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    initial_sl: float
    final_pnl_pct: float  # Сырой PnL сделки в %
    pnl_r: float          # Результат в единицах риска R (net of fees)
    exit_reason: str      # 'sl', 'be', 'tp_full', 'trailing', 'fail_fast', 'timeout'
    bars_held: int
    mfe_pct: float        # Maximum Favorable Excursion (% в сторону профита)
    mae_pct: float        # Maximum Adverse Excursion (% просадки)
    mfe_r: float = 0.0    # MFE в единицах R
    mae_r: float = 0.0    # MAE в единицах R
    position_size_usd: float = 0.0
    fee_usd: float = 0.0
    pnl_usd: float = 0.0
    net_pnl_usd: float = 0.0
    slippage_tax_pct: float = 0.0
    tp_hits: List[str] = field(default_factory=list)


class BaseStrategy(ABC):
    """Базовый абстрактный класс стратегии."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def evaluate_entry(
        self,
        signal: TradeSignal,
        future_candles: pd.DataFrame,
    ) -> EntryDecision:
        """
        Проверяет, даёт ли стратегия подтверждение на вход в сделку.
        future_candles: последующие свечи начиная с бара W5.
        """
        pass

    @abstractmethod
    def simulate_execution(
        self,
        decision: EntryDecision,
        direction: str,
        execution_candles: pd.DataFrame,
    ) -> TradeResult:
        """
        Симулирует сопровождение и закрытие позиции на исторических свечах.
        """
        pass
