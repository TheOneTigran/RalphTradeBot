"""
evaluator.py — Логика оценки исполнения торгового плана на исторических данных.
"""
from typing import List, Dict, Optional
from src.core.models import TradePlan
import re
import logging

logger = logging.getLogger(__name__)

class BacktestResult:
    def __init__(self, status: str, pnl_pct: float = 0.0, entry_price: float = 0.0, exit_price: float = 0.0, exit_ts: int = 0):
        self.status = status # "PROFIT", "LOSS", "CANCELLED", "EXPIRED"
        self.pnl_pct = pnl_pct
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.exit_ts = exit_ts

def _extract_price(text) -> Optional[float]:
    if text is None: return None
    if isinstance(text, (int, float)): return float(text)
    import re
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    m = re.search(pattern, str(text).replace(',', ''))
    return float(m.group(0)) if m else None

def evaluate_plan(plan: TradePlan, future_candles: List[Dict]) -> BacktestResult:
    """
    Проверяет, как отработал план на будущих свечах.
    """
    params = plan.trade_params or {}
    trig = plan.trigger_prices or {}
    
    direction = str(params.get("direction", "")).upper()
    if "WAIT" in direction or not direction:
        return BacktestResult("NO_TRADE")

    # Извлекаем уровни
    entry_price = _extract_price(trig.get("confirmation_level")) or _extract_price(trig.get("entry_zone"))
    inv_price = _extract_price(trig.get("invalid_level"))
    sl_price = _extract_price(params.get("stop_loss"))
    
    tps = params.get("take_profit_levels", [])
    tp_price = _extract_price(tps[0]) if tps else None

    if not entry_price or not sl_price or not tp_price:
        return BacktestResult("INVALID_PLAN")

    is_long = "LONG" in direction
    
    # --- Валидация уровней ---
    if is_long:
        if tp_price <= entry_price:
            return BacktestResult("INVALID_PLAN_TP_BELOW_ENTRY")
        if sl_price >= entry_price:
            return BacktestResult("INVALID_PLAN_SL_ABOVE_ENTRY")
    else: # SHORT
        if tp_price >= entry_price:
            return BacktestResult("INVALID_PLAN_TP_ABOVE_ENTRY")
        if sl_price <= entry_price:
            return BacktestResult("INVALID_PLAN_SL_BELOW_ENTRY")

    entered = False
    entry_ts = 0

    for candle in future_candles:
        high = candle['high']
        low = candle['low']
        ts = candle['ts']

        if not entered:
            # Проверка входа
            if is_long:
                if low <= entry_price <= high:
                    entered = True
                    entry_ts = ts
            else:
                if low <= entry_price <= high:
                    entered = True
                    entry_ts = ts
            
            # Если до входа коснулись инвалидации — отмена
            if inv_price:
                if (is_long and low <= inv_price) or (not is_long and high >= inv_price):
                    return BacktestResult("CANCELLED", exit_ts=ts)
        else:
            # Мы в сделке, проверяем SL/TP
            if is_long:
                # Сначала проверяем худший вариант — SL
                if low <= sl_price:
                    pnl = ((sl_price - entry_price) / entry_price) * 100
                    return BacktestResult("LOSS", pnl, entry_price, sl_price, ts)
                # Затем TP
                if high >= tp_price:
                    pnl = ((tp_price - entry_price) / entry_price) * 100
                    return BacktestResult("PROFIT", pnl, entry_price, tp_price, ts)
            else: # SHORT
                if high >= sl_price:
                    pnl = ((entry_price - sl_price) / entry_price) * 100
                    return BacktestResult("LOSS", pnl, entry_price, sl_price, ts)
                if low <= tp_price:
                    pnl = ((entry_price - tp_price) / entry_price) * 100
                    return BacktestResult("PROFIT", pnl, entry_price, tp_price, ts)

    last_ts = future_candles[-1]['ts'] if future_candles else 0
    return BacktestResult("EXPIRED", exit_ts=last_ts) # Сделка не закрылась за отведенное время
