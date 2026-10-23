"""
outcome_tracker.py — Фоновый трекер отработки торговых планов и целей импульсов Эллиотта.

Функционал:
1. Загружает активные сделки (status='active') из SQLite базы `ralph_analytics.db`.
2. Запрашивает высокоточные 1-минутные свечи (1m) с Binance Futures REST API (с автоматическим фоллбэком на Bybit)
   для исключения внутрисвечной неоднозначности (intrabar ambiguity).
3. Строгий 3-фазный конечный автомат сопровождения сделки (Model 2: Sweep & Engulfing):
   - Фаза 0 (Ожидание входа Limit 50%): контролирует реальный возврат цены к уровню 50% сигнальной свечи S
     после её закрытия. Если цена уходит сразу к TP1 без ретеста 50% или не доходит до входа за 8 свечей ТФ —
     лимитный ордер отменяется (`cancelled_unfilled`, PnL = $0.00).
   - Фаза 1 (В позиции до TP1): отслеживает достижение TP1 (+1.5R Net, фиксация 30%) или исходного SL (-1.0R).
   - Фаза 2 (После TP1 — сопровождение остатка 70%): стоп-лосс перенесён в Безубыток (`be_price`).
     Проверяются строго 1m свечи ПОСЛЕ минуты взятия TP1 (`c_open_ms > tp1_hit_ts`) до достижения
     финального TP2 (+3.0R Net, фиксация 70%) или возврата к Безубытку (`closed_be`).
4. ОТПРАВЛЯЕТ УВЕДОМЛЕНИЯ В TELEGRAM СТРОГО В ВИДЕ ОТВЕТОВ (reply) на оригинальный сигнал:
   - При активации лимитного входа 50% ➔ подтверждение открытия позиции
   - При отмене неисполненной лимитки ➔ команда снять лимитный ордер
   - При взятии TP1 ➔ пуш с чистой прибылью и указанием перенести SL в безубыток
   - При взятии TP2 ➔ финальная фиксация 100% позиции с расчётом итогового чистого PnL
   - При SL / BE ➔ фиксация итогового PnL
5. Рассчитывает чистый PnL с вычетом комиссий биржи (0.05% + 0.05% = 0.10% roundtrip).
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from analytics_db import get_active_signals, update_signal_outcome, init_db
from config import (
    ANALYTICS_DB_PATH,
    BREAKEVEN_AFTER_TP1,
    BREAKEVEN_AFTER_TP2,
    BREAKEVEN_OFFSET_PCT,
    COMMISSION_RATE,
    ENTRY_MAX_WAIT_BARS,
    TP_SHARES,
    OUTCOME_MAX_BARS_TTL,
    OUTCOME_TRACKER_INTERVAL_SEC,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)
from telegram_notifier import send_trade_update_reply, _fmt_p, _fmt_pc, _fmt_vol

logger = logging.getLogger("RalphOutcomeTracker")

TF_SECONDS_MAP = {
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def fetch_recent_klines(symbol: str, interval: str, limit: int = 150) -> Optional[List[Dict[str, Any]]]:
    """Загружает последние свечи заданного ТФ через Binance Futures REST API с фоллбэком на Bybit."""
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    headers = {"User-Agent": "RalphTradeBot-Tracker/2.2"}
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            raw = resp.json()
            if raw and isinstance(raw, list):
                candles = []
                for row in raw:
                    candles.append({
                        "open_time": int(row[0]),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "close_time": int(row[6]),
                    })
                return candles
    except Exception as e:
        logger.debug(f"Binance klines error for {symbol} {interval}: {e}")

    try:
        bybit_tf_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
        bybit_interval = bybit_tf_map.get(interval, "15")
        bybit_url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={bybit_interval}&limit={limit}"
        resp_b = requests.get(bybit_url, headers=headers, timeout=8)
        if resp_b.status_code == 200:
            data = resp_b.json()
            if data.get("retCode") == 0 and "result" in data and "list" in data["result"]:
                rows = data["result"]["list"]
                if rows:
                    rows.reverse()
                    tf_ms = TF_SECONDS_MAP.get(interval, 60) * 1000
                    candles = []
                    for row in rows:
                        candles.append({
                            "open_time": int(row[0]),
                            "open": float(row[1]),
                            "high": float(row[2]),
                            "low": float(row[3]),
                            "close": float(row[4]),
                            "close_time": int(row[0]) + tf_ms,
                        })
                    return candles
    except Exception as e:
        logger.debug(f"Bybit klines fallback error for {symbol} {interval}: {e}")

    return None


def fetch_recent_1m_klines(
    symbol: str,
    start_time_ms: int,
    limit: int = 1000,
) -> Optional[List[Dict[str, Any]]]:
    """
    Загружает 1-минутные свечи (1m), начиная с `start_time_ms`, через Binance Futures REST API
    с автоматическим фоллбэком на Bybit Linear.
    Использование 1m свечей позволяет с точностью до минуты определить:
    1) момент фактического касания лимитного ордера 50% (entry_filled_ts),
    2) момент взятия TP1 (tp1_hit_ts),
    3) движение к TP2 или возврат к безубытку строго ПОСЛЕ tp1_hit_ts.
    """
    headers = {"User-Agent": "RalphTradeBot-Tracker/2.2"}
    now_ms = int(time.time() * 1000)
    # Ограничиваем окно последними 950 минутами, если start_time_ms слишком старый
    effective_start_ms = max(start_time_ms, now_ms - 950 * 60 * 1000) if start_time_ms > 0 else (now_ms - 240 * 60 * 1000)

    url = (
        f"https://fapi.binance.com/fapi/v1/klines"
        f"?symbol={symbol}&interval=1m&startTime={effective_start_ms}&limit={limit}"
    )
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            raw = resp.json()
            if raw and isinstance(raw, list):
                candles = []
                for row in raw:
                    candles.append({
                        "open_time": int(row[0]),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "close_time": int(row[6]),
                    })
                return candles
    except Exception as e:
        logger.debug(f"Binance 1m klines error for {symbol}: {e}")

    # Фоллбэк: Bybit Linear 1m
    try:
        bybit_url = (
            f"https://api.bybit.com/v5/market/kline"
            f"?category=linear&symbol={symbol}&interval=1&start={effective_start_ms}&limit={min(limit, 1000)}"
        )
        resp_b = requests.get(bybit_url, headers=headers, timeout=8)
        if resp_b.status_code == 200:
            data = resp_b.json()
            if data.get("retCode") == 0 and "result" in data and "list" in data["result"]:
                rows = data["result"]["list"]
                if rows:
                    rows.reverse()
                    candles = []
                    for row in rows:
                        candles.append({
                            "open_time": int(row[0]),
                            "open": float(row[1]),
                            "high": float(row[2]),
                            "low": float(row[3]),
                            "close": float(row[4]),
                            "close_time": int(row[0]) + 60000,
                        })
                    return candles
    except Exception as e:
        logger.debug(f"Bybit 1m klines fallback error for {symbol}: {e}")

    return None


def _iso_to_ms(iso_str: Optional[str]) -> int:
    """Конвертирует ISO-8601 строку времени в миллисекунды Unix (0 при ошибке)."""
    if not iso_str:
        return 0
    try:
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


def evaluate_signal_progress(signal: Dict[str, Any], candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Пошагово и строго хронологически оценивает развитие цены по 1-минутным свечам (или свечам ТФ):
    - Фаза 0: Ожидание исполнения лимитного входа на коррекции 50% сигнальной свечи S
              (только после закрытия сигнальной свечи S: open_time >= candle_s_close_ms).
    - Фаза 1: Сопровождение открытой позиции от момента входа (entry_filled_ts) до TP1 или SL.
    - Фаза 2: Сопровождение остатка 70% позиции строго ПОСЛЕ минуты взятия TP1 (open_time > tp1_hit_ts)
              до финального TP2 или закрытия по безубытку (be_price).
    """
    direction = signal["direction"]
    entry = float(signal["entry_price"])
    sl = float(signal["sl_price"])
    tp1 = float(signal["tp1_price"])
    tp2 = float(signal["tp2_price"])
    tp3 = float(signal["tp3_price"])
    tp4 = float(signal.get("tp4_price", tp3))
    interval = signal.get("interval", "15m")
    tf_sec = TF_SECONDS_MAP.get(interval, 900)

    bar_ts = int(signal.get("bar_timestamp") or 0)
    detection_ts = int(signal.get("detection_ts") or signal.get("created_at_ts") or 0)

    # Время закрытия сигнальной свечи S (в мс), начиная с которого лимитный ордер 50% может исполниться
    if bar_ts > 0:
        candle_s_close_ms = (bar_ts + tf_sec) * 1000
    elif detection_ts > 0:
        candle_s_close_ms = detection_ts * 1000
    else:
        candle_s_close_ms = 0

    entry_filled = bool(signal.get("entry_filled", 0))
    entry_filled_at = signal.get("entry_filled_at")
    entry_filled_ts = int(signal.get("entry_filled_ts") or 0)

    tp1_hit = bool(signal.get("tp1_hit", 0))
    tp1_hit_at = signal.get("tp1_hit_at")
    tp1_hit_ts = int(signal.get("tp1_hit_ts") or 0)

    tp2_hit = bool(signal.get("tp2_hit", 0))
    tp2_hit_at = signal.get("tp2_hit_at")
    tp3_hit = bool(signal.get("tp3_hit", 0))
    tp3_hit_at = signal.get("tp3_hit_at")
    tp4_hit = bool(signal.get("tp4_hit", 0))
    tp4_hit_at = signal.get("tp4_hit_at")
    sl_hit = bool(signal.get("sl_hit", 0))
    sl_hit_at = signal.get("sl_hit_at")
    be_triggered = bool(signal.get("be_triggered", 0))
    be_hit_at = signal.get("be_hit_at")
    fail_fast_triggered = bool(signal.get("fail_fast_triggered", 0))
    fail_fast_hit_at = signal.get("fail_fast_hit_at")

    # Обратная совместимость для сделок, которые уже взяли TP1 до миграции схемы
    if tp1_hit and not entry_filled:
        entry_filled = True
        if not entry_filled_at:
            entry_filled_at = tp1_hit_at
    if entry_filled and entry_filled_ts == 0:
        entry_filled_ts = _iso_to_ms(entry_filled_at) or candle_s_close_ms
    if tp1_hit and tp1_hit_ts == 0:
        tp1_hit_ts = _iso_to_ms(tp1_hit_at) or entry_filled_ts

    status = signal.get("status", "active")
    final_pnl = float(signal.get("final_pnl_pct", 0.0))
    bars_to_first_tp = int(signal.get("bars_to_first_tp", 0))
    bars_to_close = int(signal.get("bars_to_close", 0))
    max_favorable = float(signal.get("max_favorable", 0.0))
    max_adverse = float(signal.get("max_adverse", 0.0))
    cancel_reason = ""

    if not candles:
        return {}

    current_price = float(candles[-1]["close"])

    # Оцениваем только свечи, открывшиеся ПОСЛЕ закрытия сигнальной свечи S
    eval_candles = [c for c in candles if int(c["open_time"]) >= candle_s_close_ms]
    if not eval_candles:
        # Сигнальная свеча S ещё закрывается (пре-клоуз за 1 минуту до закрытия)
        return {
            "current_price": current_price,
            "entry_filled": 1 if entry_filled else 0,
            "entry_filled_at": entry_filled_at,
            "entry_filled_ts": entry_filled_ts,
            "tp1_hit": 1 if tp1_hit else 0,
            "tp1_hit_at": tp1_hit_at,
            "tp1_hit_ts": tp1_hit_ts,
            "tp2_hit": 1 if tp2_hit else 0,
            "tp2_hit_at": tp2_hit_at,
            "tp3_hit": 1 if tp3_hit else 0,
            "tp3_hit_at": tp3_hit_at,
            "tp4_hit": 1 if tp4_hit else 0,
            "tp4_hit_at": tp4_hit_at,
            "sl_hit": 1 if sl_hit else 0,
            "sl_hit_at": sl_hit_at,
            "be_triggered": 1 if be_triggered else 0,
            "be_hit_at": be_hit_at,
            "fail_fast_triggered": 1 if fail_fast_triggered else 0,
            "fail_fast_hit_at": fail_fast_hit_at,
            "status": status,
            "final_pnl_pct": final_pnl,
            "bars_to_first_tp": bars_to_first_tp,
            "bars_to_close": bars_to_close,
            "max_favorable": max_favorable,
            "max_adverse": max_adverse,
        }

    be_price = (
        entry * (1.0 + BREAKEVEN_OFFSET_PCT / 100.0)
        if direction == "LONG"
        else entry * (1.0 - BREAKEVEN_OFFSET_PCT / 100.0)
    )

    tf_ms = tf_sec * 1000

    for c in eval_candles:
        c_open_ms = int(c["open_time"])
        high_p = float(c["high"])
        low_p = float(c["low"])
        close_p = float(c["close"])
        c_time_iso = datetime.fromtimestamp(c_open_ms / 1000.0, tz=timezone.utc).isoformat()
        tf_bars_since_s_close = max(1, int((c_open_ms - candle_s_close_ms) // tf_ms) + 1)

        # ══════════════════════════════════════════════════════════════════════
        # ФАЗА 0: Ожидание исполнения лимитного ордера на 50% коррекции свечи S
        # ══════════════════════════════════════════════════════════════════════
        if not entry_filled:
            if direction == "LONG":
                # Если цена ушла сразу к TP1 без коррекции к уровню входа 50% -> отмена лимитки
                if low_p > entry and high_p >= tp1:
                    status = "cancelled_unfilled"
                    final_pnl = 0.0
                    bars_to_close = tf_bars_since_s_close
                    cancel_reason = "tp1_without_entry"
                    break

                # Реальное касание уровня 50% коррекции сверху вниз -> ВХОД АКТИВИРОВАН!
                if low_p <= entry:
                    entry_filled = True
                    entry_filled_at = c_time_iso
                    entry_filled_ts = c_open_ms

                    fav = max(0.0, (high_p - entry) / entry * 100.0)
                    adv = max(0.0, (entry - low_p) / entry * 100.0)
                    max_favorable = max(max_favorable, round(fav, 2))
                    max_adverse = max(max_adverse, round(adv, 2))

                    # Проверка пробоя стоп-лосса на той же минутной свече входа
                    if low_p <= sl:
                        sl_hit = True
                        sl_hit_at = c_time_iso
                        status = "closed_sl"
                        bars_to_close = 1
                        final_pnl = round((sl - entry) / entry * 100.0, 2)
                        break

                    # Если на минуте входа свеча закрылась выше TP1 (мощный отскок от 50%)
                    if close_p >= tp1:
                        tp1_hit = True
                        tp1_hit_at = c_time_iso
                        tp1_hit_ts = c_open_ms
                        bars_to_first_tp = 1
                        if close_p >= tp2:
                            tp2_hit = True
                            tp2_hit_at = c_time_iso
                            status = "closed_tp2"
                            bars_to_close = 1
                            tp1_g = (tp1 - entry) / entry * 100.0
                            tp2_g = (tp2 - entry) / entry * 100.0
                            final_pnl = round(0.30 * tp1_g + 0.70 * tp2_g, 2)
                            break
                    continue

                # Превышен лимит ожидания коррекции 50% (по умолч. 8 баров ТФ)
                if tf_bars_since_s_close > ENTRY_MAX_WAIT_BARS:
                    status = "cancelled_unfilled"
                    final_pnl = 0.0
                    bars_to_close = tf_bars_since_s_close
                    cancel_reason = "entry_timeout"
                    break
                continue

            else:  # SHORT
                # Если цена обвалилась сразу к TP1 без ретеста 50% свечи S -> отмена лимитки
                if high_p < entry and low_p <= tp1:
                    status = "cancelled_unfilled"
                    final_pnl = 0.0
                    bars_to_close = tf_bars_since_s_close
                    cancel_reason = "tp1_without_entry"
                    break

                # Реальное касание уровня 50% коррекции снизу вверх -> ВХОД АКТИВИРОВАН!
                if high_p >= entry:
                    entry_filled = True
                    entry_filled_at = c_time_iso
                    entry_filled_ts = c_open_ms

                    fav = max(0.0, (entry - low_p) / entry * 100.0)
                    adv = max(0.0, (high_p - entry) / entry * 100.0)
                    max_favorable = max(max_favorable, round(fav, 2))
                    max_adverse = max(max_adverse, round(adv, 2))

                    # Проверка пробоя стоп-лосса на той же минутной свече входа
                    if high_p >= sl:
                        sl_hit = True
                        sl_hit_at = c_time_iso
                        status = "closed_sl"
                        bars_to_close = 1
                        final_pnl = round((entry - sl) / entry * 100.0, 2)
                        break

                    # Если на минуте входа свеча закрылась ниже TP1 (мощный отбой от 50%)
                    if close_p <= tp1:
                        tp1_hit = True
                        tp1_hit_at = c_time_iso
                        tp1_hit_ts = c_open_ms
                        bars_to_first_tp = 1
                        if close_p <= tp2:
                            tp2_hit = True
                            tp2_hit_at = c_time_iso
                            status = "closed_tp2"
                            bars_to_close = 1
                            tp1_g = (entry - tp1) / entry * 100.0
                            tp2_g = (entry - tp2) / entry * 100.0
                            final_pnl = round(0.30 * tp1_g + 0.70 * tp2_g, 2)
                            break
                    continue

                if tf_bars_since_s_close > ENTRY_MAX_WAIT_BARS:
                    status = "cancelled_unfilled"
                    final_pnl = 0.0
                    bars_to_close = tf_bars_since_s_close
                    cancel_reason = "entry_timeout"
                    break
                continue

        # Игнорируем исторические свечи до или в момент входа (они уже обработаны при заполнении входа)
        if c_open_ms <= entry_filled_ts:
            continue

        bars_since_entry = max(1, int((c_open_ms - entry_filled_ts) // tf_ms) + 1)

        if direction == "LONG":
            fav = (high_p - entry) / entry * 100.0
            adv = (entry - low_p) / entry * 100.0
        else:
            fav = (entry - low_p) / entry * 100.0
            adv = (high_p - entry) / entry * 100.0

        if fav > max_favorable:
            max_favorable = round(fav, 2)
        if adv > max_adverse:
            max_adverse = round(adv, 2)

        # ══════════════════════════════════════════════════════════════════════
        # ФАЗА 1: Позиция открыта, TP1 ещё не взят (проверяем SL и TP1)
        # ══════════════════════════════════════════════════════════════════════
        if not tp1_hit:
            if direction == "LONG":
                if low_p <= sl:
                    sl_hit = True
                    sl_hit_at = c_time_iso
                    status = "closed_sl"
                    bars_to_close = bars_since_entry
                    final_pnl = round((sl - entry) / entry * 100.0, 2)
                    break

                if high_p >= tp1:
                    tp1_hit = True
                    tp1_hit_at = c_time_iso
                    tp1_hit_ts = c_open_ms
                    if bars_to_first_tp == 0:
                        bars_to_first_tp = bars_since_entry

                    if high_p >= tp2:
                        tp2_hit = True
                        tp2_hit_at = c_time_iso
                        status = "closed_tp2"
                        bars_to_close = bars_since_entry
                        tp1_g = (tp1 - entry) / entry * 100.0
                        tp2_g = (tp2 - entry) / entry * 100.0
                        final_pnl = round(0.30 * tp1_g + 0.70 * tp2_g, 2)
                        break

                    # ВАЖНО: На этой же минутной свече (c_open_ms == tp1_hit_ts) НЕ проверяем безубыток!
                    # Перенос SL в безубыток действует только со следующей минуты после фиксации TP1.
                    continue

            else:  # SHORT
                if high_p >= sl:
                    sl_hit = True
                    sl_hit_at = c_time_iso
                    status = "closed_sl"
                    bars_to_close = bars_since_entry
                    final_pnl = round((entry - sl) / entry * 100.0, 2)
                    break

                if low_p <= tp1:
                    tp1_hit = True
                    tp1_hit_at = c_time_iso
                    tp1_hit_ts = c_open_ms
                    if bars_to_first_tp == 0:
                        bars_to_first_tp = bars_since_entry

                    if low_p <= tp2:
                        tp2_hit = True
                        tp2_hit_at = c_time_iso
                        status = "closed_tp2"
                        bars_to_close = bars_since_entry
                        tp1_g = (entry - tp1) / entry * 100.0
                        tp2_g = (entry - tp2) / entry * 100.0
                        final_pnl = round(0.30 * tp1_g + 0.70 * tp2_g, 2)
                        break

                    # ВАЖНО: На этой же минутной свече (c_open_ms == tp1_hit_ts) НЕ проверяем безубыток!
                    continue

        # ══════════════════════════════════════════════════════════════════════
        # ФАЗА 2: TP1 уже взят (30% зафиксировано), SL перенесён в Безубыток.
        # Проверяем ТОЛЬКО свечи строго после минуты взятия TP1 (c_open_ms > tp1_hit_ts)!
        # ══════════════════════════════════════════════════════════════════════
        else:
            if c_open_ms <= tp1_hit_ts:
                continue

            if direction == "LONG":
                # TP2 (70%) -> Финальный тейк! Закрываем позицию на 100%
                if high_p >= tp2:
                    tp2_hit = True
                    tp2_hit_at = c_time_iso
                    status = "closed_tp2"
                    bars_to_close = bars_since_entry
                    tp1_g = (tp1 - entry) / entry * 100.0
                    tp2_g = (tp2 - entry) / entry * 100.0
                    final_pnl = round(0.30 * tp1_g + 0.70 * tp2_g, 2)
                    break

                # Возврат цены вниз к уровню Безубытка после взятия TP1
                if low_p <= be_price:
                    be_triggered = True
                    be_hit_at = c_time_iso
                    status = "closed_be"
                    bars_to_close = bars_since_entry
                    tp1_g = (tp1 - entry) / entry * 100.0
                    final_pnl = round(0.30 * tp1_g, 2)
                    break

            else:  # SHORT
                # TP2 (70%) -> Финальный тейк! Закрываем позицию на 100%
                if low_p <= tp2:
                    tp2_hit = True
                    tp2_hit_at = c_time_iso
                    status = "closed_tp2"
                    bars_to_close = bars_since_entry
                    tp1_g = (entry - tp1) / entry * 100.0
                    tp2_g = (entry - tp2) / entry * 100.0
                    final_pnl = round(0.30 * tp1_g + 0.70 * tp2_g, 2)
                    break

                # Возврат цены вверх к уровню Безубытка после взятия TP1
                if high_p >= be_price:
                    be_triggered = True
                    be_hit_at = c_time_iso
                    status = "closed_be"
                    bars_to_close = bars_since_entry
                    tp1_g = (entry - tp1) / entry * 100.0
                    final_pnl = round(0.30 * tp1_g, 2)
                    break

    # Экспирация по TTL (считаем в барах исходного ТФ с момента входа)
    if status == "active" and entry_filled and entry_filled_ts > 0:
        last_open_ms = int(eval_candles[-1]["open_time"])
        total_tf_bars_in_trade = max(1, int((last_open_ms - entry_filled_ts) // tf_ms) + 1)
        if total_tf_bars_in_trade >= OUTCOME_MAX_BARS_TTL:
            status = "expired"
            bars_to_close = total_tf_bars_in_trade
            cur_move = (
                ((current_price - entry) / entry * 100.0)
                if direction == "LONG"
                else ((entry - current_price) / entry * 100.0)
            )
            if tp1_hit:
                tp1_g = abs(tp1 - entry) / entry * 100.0
                final_pnl = round(0.30 * tp1_g + 0.70 * cur_move, 2)
            else:
                final_pnl = round(cur_move, 2)

    res = {
        "current_price": current_price,
        "entry_filled": 1 if entry_filled else 0,
        "entry_filled_at": entry_filled_at,
        "entry_filled_ts": entry_filled_ts,
        "tp1_hit": 1 if tp1_hit else 0,
        "tp1_hit_at": tp1_hit_at,
        "tp1_hit_ts": tp1_hit_ts,
        "tp2_hit": 1 if tp2_hit else 0,
        "tp2_hit_at": tp2_hit_at,
        "tp3_hit": 1 if tp3_hit else 0,
        "tp3_hit_at": tp3_hit_at,
        "tp4_hit": 1 if tp4_hit else 0,
        "tp4_hit_at": tp4_hit_at,
        "sl_hit": 1 if sl_hit else 0,
        "sl_hit_at": sl_hit_at,
        "be_triggered": 1 if be_triggered else 0,
        "be_hit_at": be_hit_at,
        "fail_fast_triggered": 1 if fail_fast_triggered else 0,
        "fail_fast_hit_at": fail_fast_hit_at,
        "status": status,
        "final_pnl_pct": final_pnl,
        "bars_to_first_tp": bars_to_first_tp,
        "bars_to_close": bars_to_close,
        "max_favorable": max_favorable,
        "max_adverse": max_adverse,
    }
    if cancel_reason:
        res["_cancel_reason"] = cancel_reason
    return res


def track_active_signals(db_path: Path | str = ANALYTICS_DB_PATH) -> int:
    """
    Проверяет все активные сигналы по 1-минутным свечам, обновляет их в БД и отправляет
    уведомления-сопровождения ответом на сигнал в Telegram.
    """
    active_signals = get_active_signals(db_path=db_path)
    if not active_signals:
        return 0

    checked_count = 0
    for sig in active_signals:
        sym = sig["symbol"]
        tf = sig["interval"]
        sig_id = sig["signal_id"]
        tg_msg_id = sig.get("telegram_msg_id")
        entry = float(sig["entry_price"])
        sl = float(sig["sl_price"])
        tp1 = float(sig["tp1_price"])
        tp2 = float(sig["tp2_price"])
        tp3 = float(sig["tp3_price"])
        direction = sig["direction"]
        pos_size_usd = float(sig.get("pos_size_usd") or 500.0)
        risk_usd = float(sig.get("risk_usd") or 10.0)

        tf_sec = TF_SECONDS_MAP.get(tf, 900)
        bar_ts = int(sig.get("bar_timestamp") or 0)
        detection_ts = int(sig.get("detection_ts") or sig.get("created_at_ts") or 0)
        candle_s_close_ms = (bar_ts + tf_sec) * 1000 if bar_ts > 0 else (detection_ts * 1000)

        # Выбираем стартовую метку времени для запроса 1m свечей с учётом уже пройденных этапов
        saved_tp1_ts = int(sig.get("tp1_hit_ts") or 0)
        saved_entry_ts = int(sig.get("entry_filled_ts") or 0)
        if sig.get("tp1_hit") and saved_tp1_ts > 0:
            query_start_ms = saved_tp1_ts
        elif sig.get("entry_filled") and saved_entry_ts > 0:
            query_start_ms = saved_entry_ts
        else:
            query_start_ms = max(0, candle_s_close_ms - 60000)

        candles = fetch_recent_1m_klines(sym, start_time_ms=query_start_ms, limit=1000)
        if not candles:
            candles = fetch_recent_klines(sym, tf, limit=120)
        if not candles:
            continue

        updates = evaluate_signal_progress(sig, candles)
        if not updates:
            continue

        cancel_reason = updates.pop("_cancel_reason", "")
        checked_count += 1
        new_status = updates.get("status", "active")

        # Доли Model 2: 30% TP1 (+1.5R Net), 70% TP2 (+3.0R Net)
        tp1_share = TP_SHARES[0] if len(TP_SHARES) > 0 else 0.30
        tp2_share = TP_SHARES[1] if len(TP_SHARES) > 1 else 0.70

        tp1_gain_pct = abs(entry - tp1) / entry * 100.0
        tp2_gain_pct = abs(entry - tp2) / entry * 100.0

        # Чистый PnL с вычетом комиссий roundtrip (0.10%)
        fee_roundtrip = COMMISSION_RATE * 2.0
        tp1_net_usd = (pos_size_usd * tp1_share * (tp1_gain_pct / 100.0)) - (pos_size_usd * tp1_share * fee_roundtrip)
        tp2_net_usd = (pos_size_usd * tp2_share * (tp2_gain_pct / 100.0)) - (pos_size_usd * tp2_share * fee_roundtrip)
        full_tp_profit_usd = tp1_net_usd + tp2_net_usd

        # ── 0. Уведомление: Лимитный ордер 50% исполнен (Вход в сделку активирован) ──
        if updates.get("entry_filled") and not sig.get("notified_entry"):
            updates["notified_entry"] = 1
            if tg_msg_id:
                dir_label = "LONG 🟢" if direction == "LONG" else "SHORT 🔴"
                entry_text = (
                    f"✅ <b>ВХОД В СДЕЛКУ АКТИВИРОВАН (Limit 50%): {_fmt_pc(entry)}</b>\n\n"
                    f"📌 <b>Инструмент:</b> #{sym} ({tf}) — {dir_label}\n"
                    f"🎯 <b>Цена вернулась на 50% сигнальной свечи и исполнила лимитный ордер!</b>\n"
                    f"├ 🛑 <b>Stop Loss:</b> {_fmt_pc(sl)} (Риск: -${risk_usd:.2f} / -1.0R)\n"
                    f"├ 🎯 <b>TP1 (30% объёма | +1.5R Net):</b> {_fmt_pc(tp1)} ➔ после взятия стоп в БУ\n"
                    f"└ 🏆 <b>TP2 (70% объёма | +3.0R Net):</b> {_fmt_pc(tp2)} (финальный выход)"
                )
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=entry_text)
                logger.info(f"📤 Отправлено уведомление о входе (Limit 50%) для #{sig_id} {sym}")

        # ── 0b. Уведомление: Лимитный ордер отменён (цена не дала коррекцию 50%) ──
        if new_status == "cancelled_unfilled" and not sig.get("notified_closed"):
            updates["notified_closed"] = 1
            updates["net_pnl_usd"] = 0.0
            if tg_msg_id:
                if cancel_reason == "tp1_without_entry":
                    reason_str = (
                        f"Цена импульсно дошла до уровня TP1 ({_fmt_pc(tp1)}) <b>без ретеста точки входа 50%</b> "
                        f"({_fmt_pc(entry)})."
                    )
                else:
                    reason_str = (
                        f"За отведённые {ENTRY_MAX_WAIT_BARS} свечей ({tf}) цена не скорректировалась "
                        f"до лимитного уровня входа 50% ({_fmt_pc(entry)})."
                    )
                cancel_text = (
                    f"⚪ <b>ЛИМИТНЫЙ ОРДЕР ОТМЕНЁН (СДЕЛКА НЕ АКТИВИРОВАНА)</b>\n\n"
                    f"ℹ️ {reason_str}\n"
                    f"👉 <b>ДЕЙСТВИЕ:</b> Снимите неисполненный лимитный ордер по цене {_fmt_pc(entry)}.\n"
                    f"💰 Финансовый результат: <b>$0.00 (0.0R — без входа в рынок)</b>."
                )
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=cancel_text)
                logger.info(f"📤 Отправлено уведомление об отмене неисполненной лимитки для #{sig_id} {sym} ({cancel_reason})")

        # ── 1. Уведомление: Сработал TP1 (+1.5R Net) -> Перенос остатка 70% в БУ ────
        if updates.get("tp1_hit") and not sig.get("notified_tp1"):
            updates["notified_tp1"] = 1
            if tg_msg_id:
                be_price_c = _fmt_pc(entry)
                tp1_text = (
                    f"🎯 <b>TP1 ВЗЯТ по цене {_fmt_pc(tp1)} (+{tp1_gain_pct:.2f}%)!</b>\n\n"
                    f"✅ Зафиксировано <b>30% позиции (+1.5R Net)</b>: <b>+${tp1_net_usd:.2f}</b> чистыми (с вычетом комиссий).\n"
                    f"🛡️ <b>СРОЧНОЕ ДЕЙСТВИЕ:</b> Перенесите Stop Loss оставшихся 70% в <b>БЕЗУБЫТОК</b> на цену входа {be_price_c}.\n\n"
                    f"👉 Держите лимитный ордер на финальный TP2: {_fmt_pc(tp2)} (+3.0R Net)."
                )
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=tp1_text)
                logger.info(f"📤 Отправлено уведомление TP1 для #{sig_id} {sym}")

        # ── 2. Уведомление: Сработал TP2 (+3.0R Net / 100% сделки закрыто) ───────
        if (updates.get("tp2_hit") or new_status == "closed_tp2") and not sig.get("notified_tp2"):
            updates["notified_tp2"] = 1
            updates["notified_closed"] = 1
            updates["net_pnl_usd"] = round(full_tp_profit_usd, 2)
            full_r = full_tp_profit_usd / risk_usd if risk_usd > 0 else 0.0
            if tg_msg_id:
                tp2_text = (
                    f"🏆 <b>TP2 (ФИНАЛЬНАЯ ЦЕЛЬ +3.0R Net) ВЗЯТ по цене {_fmt_pc(tp2)} (+{tp2_gain_pct:.2f}%)!</b>\n\n"
                    f"✅ Зафиксированы оставшиеся <b>70% позиции</b>: <b>+${tp2_net_usd:.2f}</b> чистыми.\n"
                    f"🎉 <b>СДЕЛКА ПОЛНОСТЬЮ ЗАКРЫТА С МАКСИМАЛЬНЫМ ПРОФИТОМ:</b>\n"
                    f"├ TP1 (30%): +${tp1_net_usd:.2f} (+1.5R Net)\n"
                    f"├ TP2 (70%): +${tp2_net_usd:.2f} (+3.0R Net)\n"
                    f"└ 💰 <b>Итоговый чистый PnL: +${full_tp_profit_usd:.2f} (+{full_r:+.2f}R с учётом комиссий)</b>\n\n"
                    f"🏁 <i>Обе цели по Стратегии Sweep & Engulfing филигранно взяты!</i>"
                )
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=tp2_text)
                logger.info(f"📤 Отправлено уведомление TP2 (Full Win) для #{sig_id} {sym} (Net: +${full_tp_profit_usd:.2f})")

        # ── 3. Уведомление: Сработал Безубыток (после взятия TP1) ───────────────
        if new_status == "closed_be" and not sig.get("notified_closed"):
            updates["notified_closed"] = 1
            total_be_net_usd = tp1_net_usd
            updates["net_pnl_usd"] = round(total_be_net_usd, 2)
            be_r = total_be_net_usd / risk_usd if risk_usd > 0 else 0.0
            if tg_msg_id:
                be_p_c = _fmt_pc(entry)
                be_text = (
                    f"🛡️ <b>БЕЗУБЫТОК СРАБОТАЛ по цене {be_p_c}!</b>\n\n"
                    f"✅ Оставшиеся 70% позиции закрыты в точке безубытка.\n"
                    f"💰 Итоговая чистая прибыль по сделке: <b>+${total_be_net_usd:.2f} (+{be_r:+.2f}R)</b> (благодаря фиксации 30% на TP1).\n"
                    f"Сделка защищена от разворота рынка и завершена в плюс!"
                )
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=be_text)
                logger.info(f"📤 Отправлено уведомление Безубыток для #{sig_id} {sym}")

        # ── 4. Уведомление: Сработал Стоп-Лосс ────────────────────────────────
        if new_status == "closed_sl" and not sig.get("notified_closed"):
            updates["notified_closed"] = 1
            updates["net_pnl_usd"] = -risk_usd
            sl_text = (
                f"🛑 <b>СТОП-ЛОСС СРАБОТАЛ по цене {_fmt_pc(sl)}!</b>\n\n"
                f"❌ Убыток по сделке: <b>-${risk_usd:.2f}</b> (-1.0R с учётом комиссий).\n"
                f"Позиция полностью закрыта."
            )
            if tg_msg_id:
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=sl_text)
                logger.info(f"📤 Отправлено уведомление SL для #{sig_id} {sym} (Net PnL: ${updates.get('net_pnl_usd')})")

        # Сохраняем обновленные поля в БД
        update_signal_outcome(sig_id, updates, db_path=db_path)

        # Небольшая пауза между запросами к бирже
        time.sleep(0.15)

    return checked_count


class OutcomeTrackerWorker:
    """Фоновый демон для периодического отслеживания отработки сигналов."""

    def __init__(self, interval_sec: int = OUTCOME_TRACKER_INTERVAL_SEC, db_path: Path | str = ANALYTICS_DB_PATH):
        self.interval_sec = interval_sec
        self.db_path = db_path
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Запускает фоновый поток отслеживания."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="RalphOutcomeTracker", daemon=True)
        self._thread.start()
        logger.info(f"Фоновый трекер отработки сигналов запущен (интервал: {self.interval_sec}с, 1m свечи)")

    def stop(self):
        """Останавливает фоновый поток."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Фоновый трекер отработки сигналов остановлен.")

    def _run_loop(self):
        while self._running:
            try:
                active_cnt = track_active_signals(db_path=self.db_path)
                if active_cnt > 0:
                    logger.debug(f"Трекер отработки проверил {active_cnt} активных сигналов.")
            except Exception as e:
                logger.warning(f"Ошибка в цикле отслеживания отработки: {e}")

            # Прерываемый сон
            for _ in range(self.interval_sec):
                if not self._running:
                    break
                time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="RalphTradeBot Outcome Tracker CLI")
    parser.add_argument("--interval", type=int, default=20, help="Интервал проверки в секундах")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    init_db()

    logger.info("Запуск автономного трекера исходов сигналов с Telegram-сопровождением...")
    worker = OutcomeTrackerWorker(interval_sec=args.interval)
    worker.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Остановка трекера...")
        worker.stop()


if __name__ == "__main__":
    main()
