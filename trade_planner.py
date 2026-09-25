"""
trade_planner.py — Модуль расчёта профессионального торгового плана на импульсе Эллиотта.

Логика:
1. Вход: на Open следующей свечи после завершения Волны 5 (W5).
2. Стоп-Лосс: структурный, за экстремумом Волны 5 с буфером 0.15% (защита от шпильки).
3. Тейк-Профиты: 4 уровня частичной фиксации по сетке Фибоначчи от длины импульса (W0 -> W5):
   - TP1 (23.6% отката) — 25% позиции (минимальный откат, волна A)
   - TP2 (38.2% отката) — 35% позиции (стандартная коррекция волны A)
   - TP3 (50.0% отката) — 25% позиции (глубокая коррекция)
   - TP4 (61.8% отката) — 15% позиции (золотое сечение, завершение ABC)
4. Перенос в безубыток: при достижении TP1 стоп переносится в точку входа.
5. Расчёт соотношения Риск/Прибыль (R:R) и фильтрация сигналов с R:R < 1.5.
6. Длина импульса рассчитывается как в валюте, так и в процентах (|W5 - W0| / W0 * 100%).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradePlan:
    """Структурированный торговый план для завершённого импульса Эллиотта."""
    direction: str                  # "LONG" или "SHORT"
    entry_price: float              # Цена входа
    
    # Стоп-лосс
    sl_price: float                 # Абсолютная цена SL
    sl_pct: float                   # Риск в % от точки входа
    sl_distance: float              # |Entry - SL| в валюте
    
    # Тейк-профиты по Фибоначчи (от длины импульса |W5 - W0|)
    tp1_price: float                # 23.6% Фибо
    tp1_pct: float                  # % прибыли от входа
    tp1_share: float                # Доля позиции (25%)
    
    tp2_price: float                # 38.2% Фибо
    tp2_pct: float                  # % прибыли от входа
    tp2_share: float                # Доля позиции (35%)
    
    tp3_price: float                # 50.0% Фибо
    tp3_pct: float                  # % прибыли от входа
    tp3_share: float                # Доля позиции (25%)
    
    tp4_price: float                # 61.8% Фибо
    tp4_pct: float                  # % прибыли от входа
    tp4_share: float                # Доля позиции (15%)
    
    # Метрики риск/прибыль
    weighted_reward_pct: float      # Средневзвешенный % профита
    rr_ratio: float                 # Средневзвешенный R:R
    is_viable: bool                 # R:R >= min_rr_ratio
    
    # Параметры импульса
    impulse_range: float            # |W5 - W0| в валюте
    impulse_pct: float              # |W5 - W0| / W0 * 100% (длина импульса в %)
    
    # Волновые координаты
    w0_price: float = 0.0
    w1_price: float = 0.0
    w2_price: float = 0.0
    w3_price: float = 0.0
    w4_price: float = 0.0
    w5_price: float = 0.0
    
    invalidation_note: str = ""     # Пояснение по зоне отмены


@dataclass
class TradeSimulationResult:
    """Результат исполнения торгового плана на исторических барах."""
    trade_result: str               # "win", "loss"
    final_pnl_pct: float            # Взвешенный итоговый PnL в %
    exit_bar: int                   # Индекс последнего бара сделки
    exit_price: float               # Цена финального выхода
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    tp4_hit: bool = False
    sl_hit: bool = False
    be_hit: bool = False            # Выход по безубытку после TP1
    bars_held: int = 0


def calculate_trade_plan(
    wave_points: Dict[str, float],
    direction: str,
    entry_price: float,
    buffer_pct: float = 0.15,
    min_rr_ratio: float = 1.5,
) -> TradePlan:
    """
    Рассчитывает профессиональный торговый план на основе волновой структуры.
    
    direction: "SHORT" (после бычьего импульса) или "LONG" (после медвежьего).
    entry_price: цена входа (Open бара после W5).
    buffer_pct: защитный буфер за экстремумом W5 в процентах (по умолчанию 0.15%).
    min_rr_ratio: минимально допустимый R:R для подтверждения сигнала.
    """
    w0 = float(wave_points.get("W0", entry_price))
    w1 = float(wave_points.get("W1", entry_price))
    w2 = float(wave_points.get("W2", entry_price))
    w3 = float(wave_points.get("W3", entry_price))
    w4 = float(wave_points.get("W4", entry_price))
    w5 = float(wave_points.get("W5", entry_price))

    impulse_range = abs(w5 - w0)
    w0_base = max(w0, 1e-9)
    impulse_pct = (impulse_range / w0_base) * 100.0

    tp1_share = 0.25
    tp2_share = 0.35
    tp3_share = 0.25
    tp4_share = 0.15

    if direction == "SHORT":
        # Бычий импульс завершился на W5 (хай). Входим в SHORT.
        # Стоп-лосс выше хая W5
        sl_price = w5 * (1.0 + buffer_pct / 100.0)
        if sl_price <= entry_price:
            sl_price = entry_price * (1.0 + buffer_pct / 100.0)

        sl_distance = abs(sl_price - entry_price)
        sl_pct = (sl_distance / entry_price) * 100.0

        # Коррекция вниз от W5
        tp1_price = w5 - impulse_range * 0.236
        tp2_price = w5 - impulse_range * 0.382
        tp3_price = w5 - impulse_range * 0.500
        tp4_price = w5 - impulse_range * 0.618

        tp1_pct = (entry_price - tp1_price) / entry_price * 100.0
        tp2_pct = (entry_price - tp2_price) / entry_price * 100.0
        tp3_pct = (entry_price - tp3_price) / entry_price * 100.0
        tp4_pct = (entry_price - tp4_price) / entry_price * 100.0

        invalidation_note = f"Пробой W5 (${w5:,.2f}) вверх отменяет разворот"
    else:
        # Медвежий импульс завершился на W5 (лоу). Входим в LONG.
        # Стоп-лосс ниже лоу W5
        sl_price = w5 * (1.0 - buffer_pct / 100.0)
        if sl_price >= entry_price:
            sl_price = entry_price * (1.0 - buffer_pct / 100.0)

        sl_distance = abs(entry_price - sl_price)
        sl_pct = (sl_distance / entry_price) * 100.0

        # Коррекция вверх от W5
        tp1_price = w5 + impulse_range * 0.236
        tp2_price = w5 + impulse_range * 0.382
        tp3_price = w5 + impulse_range * 0.500
        tp4_price = w5 + impulse_range * 0.618

        tp1_pct = (tp1_price - entry_price) / entry_price * 100.0
        tp2_pct = (tp2_price - entry_price) / entry_price * 100.0
        tp3_pct = (tp3_price - entry_price) / entry_price * 100.0
        tp4_pct = (tp4_price - entry_price) / entry_price * 100.0

        invalidation_note = f"Пробой W5 (${w5:,.2f}) вниз отменяет разворот"

    # Коррекция отрицательных процентов если точка входа уже зашла за TP1
    tp1_pct = max(0.01, tp1_pct)
    tp2_pct = max(tp1_pct + 0.05, tp2_pct)
    tp3_pct = max(tp2_pct + 0.05, tp3_pct)
    tp4_pct = max(tp3_pct + 0.05, tp4_pct)

    weighted_reward_pct = (
        tp1_share * tp1_pct +
        tp2_share * tp2_pct +
        tp3_share * tp3_pct +
        tp4_share * tp4_pct
    )

    rr_ratio = (weighted_reward_pct / sl_pct) if sl_pct > 1e-6 else 0.0
    is_viable = (rr_ratio >= min_rr_ratio) and (sl_pct > 0.0)

    return TradePlan(
        direction=direction,
        entry_price=entry_price,
        sl_price=sl_price,
        sl_pct=sl_pct,
        sl_distance=sl_distance,
        tp1_price=tp1_price,
        tp1_pct=tp1_pct,
        tp1_share=tp1_share,
        tp2_price=tp2_price,
        tp2_pct=tp2_pct,
        tp2_share=tp2_share,
        tp3_price=tp3_price,
        tp3_pct=tp3_pct,
        tp3_share=tp3_share,
        tp4_price=tp4_price,
        tp4_pct=tp4_pct,
        tp4_share=tp4_share,
        weighted_reward_pct=weighted_reward_pct,
        rr_ratio=rr_ratio,
        is_viable=is_viable,
        impulse_range=impulse_range,
        impulse_pct=impulse_pct,
        w0_price=w0,
        w1_price=w1,
        w2_price=w2,
        w3_price=w3,
        w4_price=w4,
        w5_price=w5,
        invalidation_note=invalidation_note,
    )


def simulate_trade_multi_tp(
    df: pd.DataFrame,
    entry_bar: int,
    plan: TradePlan,
    breakeven_after_tp2: bool = True,
    breakeven_offset_pct: float = 0.1,
) -> TradeSimulationResult:
    """
    Симулирует исполнение торгового плана с частичной фиксацией на 4 уровнях TP
    и автоматическим переносом стопа в безубыток (+0.1%) при достижении TP2.
    """
    n = len(df)
    if entry_bar >= n:
        return TradeSimulationResult(
            trade_result="loss",
            final_pnl_pct=0.0,
            exit_bar=n - 1,
            exit_price=plan.entry_price,
            bars_held=0,
        )

    open_p = df['open'].values.astype(np.float64)
    high_p = df['high'].values.astype(np.float64)
    low_p = df['low'].values.astype(np.float64)
    close_p = df['close'].values.astype(np.float64)

    rem_share = 1.0
    accumulated_pnl = 0.0
    current_sl = plan.sl_price
    
    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    tp4_hit = False
    sl_hit = False
    be_hit = False
    
    last_exit_price = plan.entry_price
    exit_bar = n - 1

    # Уровень безубытка (вход + 0.1% в сторону профита)
    if plan.direction == "SHORT":
        be_sl_price = plan.entry_price * (1.0 - breakeven_offset_pct / 100.0)
    else:
        be_sl_price = plan.entry_price * (1.0 + breakeven_offset_pct / 100.0)

    for b in range(entry_bar, n):
        o = open_p[b]
        h = high_p[b]
        l = low_p[b]
        c = close_p[b]

        if plan.direction == "SHORT":
            # 1. Проверка стоп-лосса
            if h >= current_sl:
                actual_exit = max(current_sl, o)
                last_exit_price = actual_exit
                exit_bar = b
                
                if tp2_hit and breakeven_after_tp2:
                    # Стоп был перенесен в зону безубытка (вход +0.1% профита)
                    be_hit = True
                    be_pnl = (plan.entry_price - actual_exit) / plan.entry_price * 100.0
                    accumulated_pnl += rem_share * be_pnl
                else:
                    sl_hit = True
                    loss_pnl = (plan.entry_price - actual_exit) / plan.entry_price * 100.0
                    accumulated_pnl += rem_share * loss_pnl

                rem_share = 0.0
                break

            # 2. Проверка тейк-профитов
            if not tp1_hit and l <= plan.tp1_price:
                tp1_hit = True
                gain1 = (plan.entry_price - plan.tp1_price) / plan.entry_price * 100.0
                accumulated_pnl += plan.tp1_share * gain1
                rem_share -= plan.tp1_share
                last_exit_price = plan.tp1_price

            if not tp2_hit and l <= plan.tp2_price:
                tp2_hit = True
                gain2 = (plan.entry_price - plan.tp2_price) / plan.entry_price * 100.0
                accumulated_pnl += plan.tp2_share * gain2
                rem_share -= plan.tp2_share
                last_exit_price = plan.tp2_price
                # После касания TP2 SL переносится в безубыток (вход +0.1%)
                if breakeven_after_tp2:
                    current_sl = be_sl_price

            if not tp3_hit and l <= plan.tp3_price:
                tp3_hit = True
                gain3 = (plan.entry_price - plan.tp3_price) / plan.entry_price * 100.0
                accumulated_pnl += plan.tp3_share * gain3
                rem_share -= plan.tp3_share
                last_exit_price = plan.tp3_price

            if not tp4_hit and l <= plan.tp4_price:
                tp4_hit = True
                gain4 = (plan.entry_price - plan.tp4_price) / plan.entry_price * 100.0
                accumulated_pnl += plan.tp4_share * gain4
                rem_share -= plan.tp4_share
                last_exit_price = plan.tp4_price
                exit_bar = b
                break

        else:  # LONG
            # 1. Проверка стоп-лосса
            if l <= current_sl:
                actual_exit = min(current_sl, o)
                last_exit_price = actual_exit
                exit_bar = b
                
                if tp2_hit and breakeven_after_tp2:
                    # Стоп был перенесен в зону безубытка (вход +0.1% профита)
                    be_hit = True
                    be_pnl = (actual_exit - plan.entry_price) / plan.entry_price * 100.0
                    accumulated_pnl += rem_share * be_pnl
                else:
                    sl_hit = True
                    loss_pnl = (actual_exit - plan.entry_price) / plan.entry_price * 100.0
                    accumulated_pnl += rem_share * loss_pnl

                rem_share = 0.0
                break

            # 2. Проверка тейк-профитов
            if not tp1_hit and h >= plan.tp1_price:
                tp1_hit = True
                gain1 = (plan.tp1_price - plan.entry_price) / plan.entry_price * 100.0
                accumulated_pnl += plan.tp1_share * gain1
                rem_share -= plan.tp1_share
                last_exit_price = plan.tp1_price

            if not tp2_hit and h >= plan.tp2_price:
                tp2_hit = True
                gain2 = (plan.tp2_price - plan.entry_price) / plan.entry_price * 100.0
                accumulated_pnl += plan.tp2_share * gain2
                rem_share -= plan.tp2_share
                last_exit_price = plan.tp2_price
                # После касания TP2 SL переносится в безубыток (вход +0.1%)
                if breakeven_after_tp2:
                    current_sl = be_sl_price

            if not tp3_hit and h >= plan.tp3_price:
                tp3_hit = True
                gain3 = (plan.tp3_price - plan.entry_price) / plan.entry_price * 100.0
                accumulated_pnl += plan.tp3_share * gain3
                rem_share -= plan.tp3_share
                last_exit_price = plan.tp3_price

            if not tp4_hit and h >= plan.tp4_price:
                tp4_hit = True
                gain4 = (plan.tp4_price - plan.entry_price) / plan.entry_price * 100.0
                accumulated_pnl += plan.tp4_share * gain4
                rem_share -= plan.tp4_share
                last_exit_price = plan.tp4_price
                exit_bar = b
                break

    # Если позиция осталась открыта к концу датасета
    if rem_share > 0:
        c_last = close_p[-1]
        last_exit_price = c_last
        exit_bar = n - 1
        if plan.direction == "SHORT":
            tail_pnl = (plan.entry_price - c_last) / plan.entry_price * 100.0
        else:
            tail_pnl = (c_last - plan.entry_price) / plan.entry_price * 100.0
        accumulated_pnl += rem_share * tail_pnl

    res_str = "win" if accumulated_pnl > 0 else "loss"

    return TradeSimulationResult(
        trade_result=res_str,
        final_pnl_pct=accumulated_pnl,
        exit_bar=exit_bar,
        exit_price=last_exit_price,
        tp1_hit=tp1_hit,
        tp2_hit=tp2_hit,
        tp3_hit=tp3_hit,
        tp4_hit=tp4_hit,
        sl_hit=sl_hit,
        be_hit=be_hit,
        bars_held=exit_bar - entry_bar + 1,
    )
