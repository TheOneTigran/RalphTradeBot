"""
matcher.py — Ядро сведения ордеров (Matching Engine).

Применяет правила шпилек (пессимистичный сценарий), 
обработку ценовых гэпов, ребалансировку RR и ловлю микро-откатов (Волна 2).
"""

from typing import Dict

from .models import SimulatedOrder, OrderStatus
from .config import SLIPPAGE_PCT, MAX_TTL_BARS


def process_candle(order: SimulatedOrder, candle: Dict):
    """
    Обрабатывает движение свечи относительно выставленного (или активного) ордера.
    """
    c_open = candle["open"]
    c_high = candle["high"]
    c_low = candle["low"]
    c_close = candle["close"]
    c_time = candle.get("timestamp", candle.get("ts", 0))

    p = order.plan

    if order.status in [OrderStatus.PENDING, OrderStatus.WAITING_PULLBACK]:
        order.bars_held += 1

        if order.is_long:
            # 1. Инвалидация: задели стоп до входа
            if c_low <= p.sl_conservative:
                order.status = OrderStatus.CANCELLED_INVALID
                order.exit_time = c_time
                return

            if order.status == OrderStatus.PENDING:
                # Вход на первой свече (Limit / Market Zone / Pullback Trap)
                if order.bars_held == 1:
                    if c_open <= p.entry_tolerance:
                        fill_price = c_open * (1 + SLIPPAGE_PCT / 100.0)
                        _activate(order, fill_price, c_time)
                    else:
                        # Улетела в космос -> ловушка для Волны 2
                        order.status = OrderStatus.WAITING_PULLBACK
                        height = c_high - p.point_c_price
                        order.pullback_limit = p.point_c_price + (height * 0.5)
                else:
                    if c_low <= p.entry_price:
                        fill_price = min(p.entry_price, c_open) * (1 + SLIPPAGE_PCT / 100.0)
                        _activate(order, fill_price, c_time)
                        
            elif order.status == OrderStatus.WAITING_PULLBACK:
                if c_low <= order.pullback_limit:
                    fill_price = min(order.pullback_limit, c_open) * (1 + SLIPPAGE_PCT / 100.0)
                    _activate(order, fill_price, c_time)

        else: # SHORT
            if c_high >= p.sl_conservative:
                order.status = OrderStatus.CANCELLED_INVALID
                order.exit_time = c_time
                return

            if order.status == OrderStatus.PENDING:
                if order.bars_held == 1:
                    if c_open >= p.entry_tolerance:
                        fill_price = c_open * (1 - SLIPPAGE_PCT / 100.0)
                        _activate(order, fill_price, c_time)
                    else:
                        order.status = OrderStatus.WAITING_PULLBACK
                        height = p.point_c_price - c_low
                        order.pullback_limit = p.point_c_price - (height * 0.5)
                else:
                    if c_high >= p.entry_price:
                        fill_price = max(p.entry_price, c_open) * (1 - SLIPPAGE_PCT / 100.0)
                        _activate(order, fill_price, c_time)
                        
            elif order.status == OrderStatus.WAITING_PULLBACK:
                if c_high >= order.pullback_limit:
                    fill_price = max(order.pullback_limit, c_open) * (1 - SLIPPAGE_PCT / 100.0)
                    _activate(order, fill_price, c_time)

        # 3. Отмена по таймауту
        if order.status in [OrderStatus.PENDING, OrderStatus.WAITING_PULLBACK] and order.bars_held > p.ttl_minutes:
            order.status = OrderStatus.CANCELLED_TIME
            order.exit_time = c_time
            return
            
    # Если ордер ТОЛЬКО ЧТО стал ACTIVE в этой же свече, мы даем ему шанс задеть стоп прямо сейчас
    if order.status == OrderStatus.ACTIVE:
        order.bars_held += 1
        
        # Обновляем max_favorable_price
        if order.is_long:
            if c_high > order.max_favorable_price:
                order.max_favorable_price = c_high
        else:
            if order.max_favorable_price == 0.0 or c_low < order.max_favorable_price:
                order.max_favorable_price = c_low
                
        # Time-Trailing
        if order.bars_held >= 20: 
            half_dist = abs(order.plan.take_profit - order.plan.entry_price) * 0.5
            if order.is_long:
                if order.max_favorable_price < order.plan.entry_price + half_dist:
                    _exit(order, c_close, OrderStatus.CLOSED_TIME_TTL, c_time)
                    return
            else:
                if order.max_favorable_price > order.plan.entry_price - half_dist:
                    _exit(order, c_close, OrderStatus.CLOSED_TIME_TTL, c_time)
                    return
        
        sl = p.sl_aggressive
        tp = p.take_profit

        if order.is_long:
            hit_tp_gap = c_open >= tp
            hit_sl_gap = c_open <= sl

            hit_tp = c_high >= tp
            hit_sl = c_low <= sl

            if hit_sl_gap:
                close_price = c_open * (1 - SLIPPAGE_PCT / 100.0)
                _exit(order, close_price, OrderStatus.FILLED_SL, c_time)
                return
            elif hit_tp_gap:
                close_price = c_open * (1 - SLIPPAGE_PCT / 100.0) 
                _exit(order, close_price, OrderStatus.FILLED_TP, c_time)
                return

            if hit_tp and hit_sl:
                close_price = sl * (1 - SLIPPAGE_PCT / 100.0)
                _exit(order, close_price, OrderStatus.FILLED_SL, c_time)
            elif hit_sl:
                close_price = sl * (1 - SLIPPAGE_PCT / 100.0)
                _exit(order, close_price, OrderStatus.FILLED_SL, c_time)
            elif hit_tp:
                close_price = tp * (1 - SLIPPAGE_PCT / 100.0) 
                _exit(order, close_price, OrderStatus.FILLED_TP, c_time)

        else: # SHORT
            hit_tp_gap = c_open <= tp
            hit_sl_gap = c_open >= sl

            hit_tp = c_low <= tp
            hit_sl = c_high >= sl

            if hit_sl_gap:
                close_price = c_open * (1 + SLIPPAGE_PCT / 100.0)
                _exit(order, close_price, OrderStatus.FILLED_SL, c_time)
                return
            elif hit_tp_gap:
                close_price = c_open * (1 + SLIPPAGE_PCT / 100.0)
                _exit(order, close_price, OrderStatus.FILLED_TP, c_time)
                return

            if hit_tp and hit_sl:
                close_price = sl * (1 + SLIPPAGE_PCT / 100.0)
                _exit(order, close_price, OrderStatus.FILLED_SL, c_time)
            elif hit_sl:
                close_price = sl * (1 + SLIPPAGE_PCT / 100.0)
                _exit(order, close_price, OrderStatus.FILLED_SL, c_time)
            elif hit_tp:
                close_price = tp * (1 + SLIPPAGE_PCT / 100.0)
                _exit(order, close_price, OrderStatus.FILLED_TP, c_time)


def _activate(order: SimulatedOrder, fill_price: float, c_time: int):
    # Minimal RR Gatekeeper & Dynamic Sizing Recalculation
    p = order.plan
    
    risk_diff = abs(fill_price - p.sl_aggressive)
    if risk_diff == 0:
        order.status = OrderStatus.CANCELLED_POOR_RR
        order.exit_time = c_time
        return
        
    reward_diff = abs(p.take_profit - fill_price)
    actual_rr = reward_diff / risk_diff
    
    if actual_rr < 1.2:
        order.status = OrderStatus.CANCELLED_POOR_RR
        order.exit_time = c_time
        return
        
    # Resize Logic to cap $Risk
    planned_diff = abs(p.entry_price - p.sl_aggressive)
    new_size = order.position_size * (planned_diff / risk_diff)
    order.position_size = new_size

    order.status = OrderStatus.ACTIVE
    order.fill_price = fill_price
    order.entry_time = c_time


def _exit(order: SimulatedOrder, close_price: float, reason: OrderStatus, c_time: int):
    order.status = reason
    order.close_price = close_price
    order.exit_time = c_time
