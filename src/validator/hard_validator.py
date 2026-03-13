"""
hard_validator.py — Программный валидатор торгового плана (Hard Critic).

Проверяет waves_breakdown и trade_params на соответствие математическим правилам:
  1. Связность цен и корректность timestamps.
  2. Правила импульса (W3 не короче всех, W4 не в зоне W1, W5>W3).
  3. Правила коррекции (C должна пробивать A).
  4. Логика уровней (SL < Entry < TP) и R:R >= MIN_RR_RATIO.
  5. Последовательность тейк-профитов (TP1 < TP2 < TP3).

В отличие от LLM-Критика, этот валидатор работает мгновенно и не ошибается в расчетах.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import List, Optional

from src.core.models import TradePlan, WaveCoordinate
from src.core.config import MIN_RR_RATIO

logger = logging.getLogger(__name__)


# Допустимые метки волн для каждого ТФ
VALID_WAVE_NAMES: dict[str, set[str]] = {
    "1w": {"I", "II", "III", "IV", "V", "A", "B", "C", "W", "X", "Y", "D", "E"},
    "1d": {"(1)", "(2)", "(3)", "(4)", "(5)", "(A)", "(B)", "(C)", "(W)", "(X)", "(Y)", "(D)", "(E)"},
    "4h": {"1", "2", "3", "4", "5", "A", "B", "C", "W", "X", "Y", "D", "E"},
    "1h": {"[i]", "[ii]", "[iii]", "[iv]", "[v]", "[a]", "[b]", "[c]", "[w]", "[x]", "[y]", "[d]", "[e]"},
    "15m": {"(i)", "(ii)", "(iii)", "(iv)", "(v)", "(a)", "(b)", "(c)", "(w)", "(x)", "(y)", "(d)", "(e)"},
    "5m": {"i", "ii", "iii", "iv", "v", "a", "b", "c", "w", "x", "y", "d", "e"},
}

# Метки импульса и коррекции
IMPULSE_LABELS = ["1", "2", "3", "4", "5"]
ZIGZAG_LABELS = ["A", "B", "C"]

def validate_plan(plan: TradePlan) -> list[str]:
    """
    Программная валидация торгового плана. Возвращает список ошибок.
    """
    errors: list[str] = []
    waves = plan.waves_breakdown
    
    if not waves:
        return errors # Если ИИ не расписал волны, пропускаем эту часть
    
    # Группируем волны по ТФ
    by_tf: dict[str, list[WaveCoordinate]] = defaultdict(list)
    for w in waves:
        by_tf[w.timeframe.lower()].append(w)
    
    for tf, tf_waves in by_tf.items():
        # 1. Проверка timestamps
        for w in tf_waves:
            if w.start_time > 0 and w.end_time > 0:
                if w.start_time >= w.end_time:
                    errors.append(f"[{tf}] {w.wave_name}: время начала ({w.start_time}) должно быть меньше времени конца ({w.end_time})")
        
        # 2. Связность цен (допуск 1%)
        for i in range(len(tf_waves) - 1):
            curr, nxt = tf_waves[i], tf_waves[i + 1]
            if curr.end_price > 0 and nxt.start_price > 0:
                tolerance = 0.01 * max(curr.end_price, nxt.start_price)
                if abs(curr.end_price - nxt.start_price) > tolerance:
                    errors.append(f"[{tf}] Разрыв цен: {curr.wave_name} закончена на {curr.end_price:.2f}, а {nxt.wave_name} начата на {nxt.start_price:.2f}")

        # 3. Правила ВА (Импульсы и Зигзаги)
        _validate_wave_rules(tf_waves, tf, errors)
    
    # 4. Проверка торговых уровней
    _validate_trade_levels(plan, errors)
    
    return errors

def _validate_wave_rules(tf_waves: list[WaveCoordinate], tf: str, errors: list[str]):
    # Упрощенный поиск паттернов по именам (игнорируем скобки для проверки сути)
    clean_map = {re.sub(r'[()\[\]]', '', w.wave_name).upper(): w for w in tf_waves}
    
    # А) Импульс (1-2-3-4-5)
    if all(k in clean_map for k in ["1", "2", "3", "4", "5"]):
        w1, w2, w3, w4, w5 = [clean_map[k] for k in ["1", "2", "3", "4", "5"]]
        len1, len3, len5 = [abs(w.end_price - w.start_price) for w in [w1, w3, w5]]
        bullish = w1.end_price > w1.start_price
        
        if len3 > 0 and len3 < len1 and len3 < len5:
            errors.append(f"[{tf}] Нарушение импульса: Волна 3 не может быть самой короткой.")
        
        if bullish:
            if w2.end_price <= w1.start_price: errors.append(f"[{tf}] W2 зашла за начало W1.")
            if w4.end_price <= w1.end_price: errors.append(f"[{tf}] W4 зашла в зону W1.")
            if w5.end_price <= w3.end_price: errors.append(f"[{tf}] W5 не пробила пик W3 (усечение/ошибка).")
        else:
            if w2.end_price >= w1.start_price: errors.append(f"[{tf}] W2 зашла за начало W1.")
            if w4.end_price >= w1.end_price: errors.append(f"[{tf}] W4 зашла в зону W1.")
            if w5.end_price >= w3.end_price: errors.append(f"[{tf}] W5 не пробила дно W3.")

    # Б) Зигзаг (A-B-C)
    if all(k in clean_map for k in ["A", "B", "C"]):
        wa, wb, wc = [clean_map[k] for k in ["A", "B", "C"]]
        bullish = wa.end_price > wa.start_price # BULLISH A means correction is BEARISH overall or part of it
        # Actually in EW, if A is down, C must be below A.
        if (wa.end_price < wa.start_price and wc.end_price >= wa.end_price) or \
           (wa.end_price > wa.start_price and wc.end_price <= wa.end_price):
            errors.append(f"[{tf}] Нарушение зигзага: Волна C должна выходить за пределы Волны A.")

def _validate_trade_levels(plan: TradePlan, errors: list[str]):
    params = plan.trade_params or {}
    trig = plan.trigger_prices or {}
    
    # Извлекаем цену входа (аналогично runner.py)
    entry_str = str(trig.get("confirmation_level", "") or trig.get("entry_zone", ""))
    entry = _try_parse_number(entry_str)
    
    sl = _try_parse_number(str(params.get("stop_loss", "")))
    tps_raw = params.get("take_profit_levels", [])
    tps = [_try_parse_number(str(tp)) for tp in tps_raw if _try_parse_number(str(tp))]
    direction = str(params.get("direction", "")).upper()
    inv = _try_parse_number(str(trig.get("invalid_level", "")))

    if "WAIT" in direction: return

    if not entry or not sl or not tps:
        errors.append("Неполные торговые уровни (отсутствует Entry, SL или TP).")
        return

    tp1 = tps[0]
    
    # 1. Логика направления
    if "LONG" in direction:
        if sl >= entry: 
            errors.append(f"LONG: Стоп ({sl:.2f}) должен быть ниже входа ({entry:.2f})")
        if tp1 <= entry: 
            errors.append(f"LONG: Тейк ({tp1:.2f}) должен быть выше входа ({entry:.2f})")
        if inv and sl > inv: 
            errors.append(f"LONG: Стоп ({sl:.2f}) должен быть ниже инвалидации ({inv:.2f}).")
            
        # Последовательность ТП
        for i in range(len(tps) - 1):
            if tps[i] >= tps[i+1]:
                errors.append(f"LONG: Тейк-профиты должны идти по возрастанию (TP{i+1}: {tps[i]:.2f} >= TP{i+2}: {tps[i+1]:.2f})")
                
    elif "SHORT" in direction:
        if sl <= entry: 
            errors.append(f"SHORT: Стоп ({sl:.2f}) должен быть выше входа ({entry:.2f})")
        if tp1 >= entry: 
            errors.append(f"SHORT: Тейк ({tp1:.2f}) должен быть ниже входа ({entry:.2f})")
        if inv and sl < inv: 
            errors.append(f"SHORT: Стоп ({sl:.2f}) должен быть выше инвалидации ({inv:.2f}).")
            
        # Последовательность ТП
        for i in range(len(tps) - 1):
            if tps[i] <= tps[i+1]:
                errors.append(f"SHORT: Тейк-профиты должны идти по убыванию (TP{i+1}: {tps[i]:.2f} <= TP{i+2}: {tps[i+1]:.2f})")

    # 2. R:R check
    risk = abs(entry - sl)
    reward = abs(tp1 - entry)
    if risk > 1e-6:
        rr = reward / risk
        if rr < MIN_RR_RATIO:
            errors.append(f"Низкий R:R ({rr:.2f}). Требуется минимум {MIN_RR_RATIO}. Вход: {entry:.2f}, СЛ: {sl:.2f}, ТП1: {tp1:.2f}")
    else:
        errors.append("Риск равен нулю (Entry == SL). Проверь уровни.")

def _try_parse_number(s: str) -> Optional[float]:
    if not s or s == "None": return None
    pattern = r'\b[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\b'
    match = re.search(pattern, s.replace(',', ''))
    return float(match.group()) if match else None
