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
    params = plan.trade_params or {}
    trig = plan.trigger_prices or {}
    
    direction = str(params.get("direction", "")).upper()
    if "WAIT" in direction or not direction:
        return BacktestResult("NO_TRADE")

    entry_price = _extract_price(trig.get("confirmation_level")) or _extract_price(trig.get("entry_zone"))
    inv_price = _extract_price(trig.get("invalid_level"))
    sl_price = _extract_price(params.get("stop_loss"))
    
    tps = params.get("take_profit_levels", [])
    tp_price = _extract_price(tps[0]) if tps else None

    if not entry_price or not sl_price or not tp_price:
        return BacktestResult("INVALID_PLAN")

    is_long = "LONG" in direction
    
    # --- КОНСТАНТЫ УПРАВЛЕНИЯ (v5) ---
    REAL_FEE_ROUNDTRIP = 0.15 
    # Дистанция Стоп-Лосса для активации БУ
    STOP_DIST = abs(entry_price - sl_price)
    BE_PRICE_OFFSET = entry_price * ((REAL_FEE_ROUNDTRIP + 0.05) / 100) # Минимальный профит в БУ
    
    INV_BUFFER = entry_price * 0.0015
    
    time_diff = 0
    if len(future_candles) > 1:
        time_diff = future_candles[1]['ts'] - future_candles[0]['ts']
    
    if time_diff <= 300000: # 5m
        TIME_EXIT_BARS = 100 
    elif time_diff <= 3600000: # 1h
        TIME_EXIT_BARS = 300 # Ждем 12 дней
    else: 
        TIME_EXIT_BARS = 500
        
    entered = False
    bar_counter = 0
    is_trailing = False
    is_protected = False
    current_sl = sl_price
    
    for candle in future_candles:
        high = candle['high']
        low = candle['low']
        ts = candle['ts']

        if not entered:
            if low <= entry_price <= high:
                entered = True
                bar_counter = 0
            
            if inv_price:
                if is_long and low < (inv_price - INV_BUFFER):
                    return BacktestResult("CANCELLED", exit_ts=ts)
                if not is_long and high > (inv_price + INV_BUFFER):
                    return BacktestResult("CANCELLED", exit_ts=ts)
        else:
            bar_counter += 1
            
            if is_long:
                if bar_counter >= TIME_EXIT_BARS:
                    pnl = ((low - entry_price) / entry_price) * 100
                    return BacktestResult("EXPIRED", pnl, entry_price, low, ts)

                # FRACTAL TRAILING
                if bar_counter % 20 == 0 and low > entry_price:
                    new_sl = low * 0.98
                    if new_sl > current_sl: current_sl = new_sl

                # BREAKEVEN: Когда прошли расстояние Стоп-Лосса (1:1)
                if not is_protected and high >= (entry_price + STOP_DIST):
                    is_protected = True
                    current_sl = max(current_sl, entry_price + BE_PRICE_OFFSET)

                # TRAILING: на TP1
                if not is_trailing and high >= tp_price:
                    is_trailing = True
                    current_sl = max(current_sl, tp_price * 0.975)

                if is_trailing:
                    if high * 0.975 > current_sl: current_sl = high * 0.975

                if low <= current_sl:
                    pnl = ((current_sl - entry_price) / entry_price) * 100
                    return BacktestResult("PROFIT" if pnl > REAL_FEE_ROUNDTRIP else "LOSS", pnl, entry_price, current_sl, ts)
            
            else: # SHORT
                if bar_counter >= TIME_EXIT_BARS:
                    pnl = ((entry_price - high) / entry_price) * 100
                    return BacktestResult("EXPIRED", pnl, entry_price, high, ts)

                if bar_counter % 20 == 0 and high < entry_price:
                    new_sl = high * 1.02
                    if new_sl < current_sl: current_sl = new_sl

                # BREAKEVEN: Когда прошли расстояние Стоп-Лосса (1:1)
                if not is_protected and low <= (entry_price - STOP_DIST):
                    is_protected = True
                    current_sl = min(current_sl, entry_price - BE_PRICE_OFFSET)

                if not is_trailing and low <= tp_price:
                    is_trailing = True
                    current_sl = min(current_sl, tp_price * 1.025)

                if is_trailing:
                    if low * 1.025 < current_sl: current_sl = low * 1.025

                if high >= current_sl:
                    pnl = ((entry_price - current_sl) / entry_price) * 100
                    return BacktestResult("PROFIT" if pnl > REAL_FEE_ROUNDTRIP else "LOSS", pnl, entry_price, current_sl, ts)

    last_ts = future_candles[-1]['ts'] if future_candles else 0
    return BacktestResult("EXPIRED", exit_ts=last_ts)




 # Сделка не закрылась за отведенное время
