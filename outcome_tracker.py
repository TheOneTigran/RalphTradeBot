"""
outcome_tracker.py — Фоновый трекер отработки торговых планов и целей импульсов Эллиотта.

Функционал:
1. Загружает активные сделки (status='active') из SQLite базы `ralph_analytics.db`.
2. Запрашивает свежие свечи с Binance Futures REST API (с автоматическим фоллбэком на Bybit).
3. Отслеживает достижение уровней:
   - TP1 (23.6% Фибо): фиксация 40% позиции
   - TP2 (38.2% Фибо): фиксация 30% позиции и перенос SL в безубыток (+0.1%)
   - Fail-Fast Time-Stop на баре 3 (45 мин для 15m, 3 часа для 1h)
   - Срабатывание безубытка (closed_be) или первоначального SL (closed_sl)
4. ОТПРАВЛЯЕТ УВЕДОМЛЕНИЯ В TELEGRAM СТРОГО В ВИДЕ ОТВЕТОВ (reply) на оригинальный сигнал:
   - При взятии TP1 ➔ пуш с чистой прибылью и указанием держать TP2
   - При взятии TP2 ➔ срочная инструкция перенести SL в безубыток
   - При Fail-Fast ➔ команда закрыть остаток по рынку
   - При SL / BE ➔ фиксация итогового PnL
5. Рассчитывает чистый PnL с вычетом комиссий биржи (0.1% вход + 0.1% выход).
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
    BREAKEVEN_AFTER_TP2,
    BREAKEVEN_OFFSET_PCT,
    FAIL_FAST_BARS,
    FAIL_FAST_MIN_R,
    COMMISSION_RATE,
    TP_SHARES,
    OUTCOME_MAX_BARS_TTL,
    OUTCOME_TRACKER_INTERVAL_SEC,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)
from telegram_notifier import send_trade_update_reply, _fmt_p, _fmt_pc, _fmt_vol

logger = logging.getLogger("RalphOutcomeTracker")


def fetch_recent_klines(symbol: str, interval: str, limit: int = 150) -> Optional[List[Dict[str, Any]]]:
    """Загружает последние свечи через Binance Futures REST API с фоллбэком на Bybit."""
    # 1. Binance Futures
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

    # 2. Фоллбэк: Bybit Linear
    try:
        bybit_tf_map = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
        bybit_interval = bybit_tf_map.get(interval, "15")
        bybit_url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={bybit_interval}&limit={limit}"
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
                            "close_time": int(row[0]) + 900000,
                        })
                    return candles
    except Exception as e:
        logger.debug(f"Bybit klines fallback error for {symbol} {interval}: {e}")

    return None


def evaluate_signal_progress(signal: Dict[str, Any], candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Пошагово оценивает развитие цены после точки входа сигнала.
    Возвращает словарь с обновлёнными полями для `signal_outcomes`.
    """
    direction = signal["direction"]
    entry = float(signal["entry_price"])
    sl = float(signal["sl_price"])
    tp1 = float(signal["tp1_price"])
    tp2 = float(signal["tp2_price"])
    tp3 = float(signal["tp3_price"])
    tp4 = float(signal.get("tp4_price", tp3))
    bar_ts = signal.get("bar_timestamp")
    sl_pct = float(signal.get("sl_pct", 1.5))
    interval = signal.get("interval", "15m")

    # Определяем динамический порог Fail-Fast в зависимости от ТФ
    # 5m=6 баров (30 мин), 15m=3 бара (45 мин), 1h=3 бара (3ч), 4h=2 бара (8ч)
    ff_bars_map = {"5m": 6, "15m": 3, "1h": 3, "4h": 2, "1d": 2}
    effective_ff_bars = ff_bars_map.get(interval, FAIL_FAST_BARS)

    # Определяем стартовую точку для отсчёта баров:
    # Используем detection_ts (когда сигнал был реально отправлен пользователю)
    # вместо bar_timestamp W5, чтобы пользователь успел увидеть сигнал
    detection_ts = signal.get("detection_ts") or signal.get("created_at_ts")

    tp1_hit = bool(signal.get("tp1_hit", 0))
    tp1_hit_at = signal.get("tp1_hit_at")
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

    status = signal.get("status", "active")
    final_pnl = float(signal.get("final_pnl_pct", 0.0))
    bars_to_first_tp = int(signal.get("bars_to_first_tp", 0))
    bars_to_close = int(signal.get("bars_to_close", 0))
    max_favorable = float(signal.get("max_favorable", 0.0))
    max_adverse = float(signal.get("max_adverse", 0.0))

    # Находим индекс свечи входа
    # Приоритет: detection_ts (когда сигнал реально отправлен), затем bar_ts (W5 бар)
    ref_ts = detection_ts or bar_ts
    entry_idx = -1
    if ref_ts:
        for idx, c in enumerate(candles):
            c_ts_sec = c["open_time"] // 1000
            if abs(c_ts_sec - ref_ts) <= 60 or abs(c["open_time"] - ref_ts) <= 60000:
                entry_idx = idx
                break
        # Если detection_ts не найден в свечах, фоллбэк на bar_ts
        if entry_idx == -1 and ref_ts != bar_ts and bar_ts:
            for idx, c in enumerate(candles):
                c_ts_sec = c["open_time"] // 1000
                if abs(c_ts_sec - bar_ts) <= 60 or abs(c["open_time"] - bar_ts) <= 60000:
                    entry_idx = idx
                    break

    eval_candles = candles[entry_idx + 1 :] if entry_idx != -1 else candles[-50:]
    if not eval_candles:
        return {}

    current_price = eval_candles[-1]["close"]
    be_price = (
        entry * (1.0 + BREAKEVEN_OFFSET_PCT / 100.0)
        if direction == "LONG"
        else entry * (1.0 - BREAKEVEN_OFFSET_PCT / 100.0)
    )

    bar_counter = 0
    tp2_bar_idx: Optional[int] = None

    for c in eval_candles:
        bar_counter += 1
        high_p = c["high"]
        low_p = c["low"]
        close_p = c["close"]
        c_time_iso = datetime.fromtimestamp(c["open_time"] / 1000, tz=timezone.utc).isoformat()

        if direction == "LONG":
            fav = (high_p - entry) / entry * 100.0
            adv = (entry - low_p) / entry * 100.0
            if fav > max_favorable:
                max_favorable = round(fav, 2)
            if adv > max_adverse:
                max_adverse = round(adv, 2)

            # TP1 (40%)
            if high_p >= tp1 and not tp1_hit:
                tp1_hit = True
                tp1_hit_at = c_time_iso
                if bars_to_first_tp == 0:
                    bars_to_first_tp = bar_counter

            # TP2 (30%) -> Активация безубытка СТРОГО на последующих барах
            if high_p >= tp2 and not tp2_hit:
                tp2_hit = True
                tp2_hit_at = c_time_iso
                tp2_bar_idx = bar_counter
            elif tp2_hit and tp2_bar_idx is None:
                if tp2_hit_at and c_time_iso == tp2_hit_at:
                    tp2_bar_idx = bar_counter
                elif high_p >= tp2:
                    tp2_bar_idx = bar_counter

            # TP3
            if high_p >= tp3 and not tp3_hit:
                tp3_hit = True
                tp3_hit_at = c_time_iso

            # Fail-Fast Time-Stop (динамический порог по ТФ)
            if bar_counter >= effective_ff_bars and not tp1_hit and not tp2_hit and not fail_fast_triggered:
                cur_profit_pct = (close_p - entry) / entry * 100.0
                cur_r = cur_profit_pct / sl_pct if sl_pct > 0 else 0.0
                if cur_r < FAIL_FAST_MIN_R:
                    fail_fast_triggered = True
                    fail_fast_hit_at = c_time_iso
                    status = "closed_fail_fast"
                    bars_to_close = bar_counter
                    final_pnl = round(cur_profit_pct, 2)
                    break

            # Проверка Стоп-лосса / Безубытка
            # Безубыток разрешён ТОЛЬКО если TP2 был взят на ПРЕДЫДУЩЕМ баре (bar_counter > tp2_bar_idx)
            is_be_eligible = (BREAKEVEN_AFTER_TP2 and tp2_bar_idx is not None and bar_counter > tp2_bar_idx) or be_triggered
            if is_be_eligible:
                if low_p <= be_price:
                    be_triggered = True
                    be_hit_at = c_time_iso
                    status = "closed_be"
                    bars_to_close = bar_counter
                    final_pnl = BREAKEVEN_OFFSET_PCT
                    break
            else:
                if low_p <= sl:
                    sl_hit = True
                    sl_hit_at = c_time_iso
                    status = "closed_sl"
                    bars_to_close = bar_counter
                    final_pnl = round((sl - entry) / entry * 100.0, 2)
                    break

        else:  # SHORT
            fav = (entry - low_p) / entry * 100.0
            adv = (high_p - entry) / entry * 100.0
            if fav > max_favorable:
                max_favorable = round(fav, 2)
            if adv > max_adverse:
                max_adverse = round(adv, 2)

            # TP1 (40%)
            if low_p <= tp1 and not tp1_hit:
                tp1_hit = True
                tp1_hit_at = c_time_iso
                if bars_to_first_tp == 0:
                    bars_to_first_tp = bar_counter

            # TP2 (30%) -> Активация безубытка СТРОГО на последующих барах
            if low_p <= tp2 and not tp2_hit:
                tp2_hit = True
                tp2_hit_at = c_time_iso
                tp2_bar_idx = bar_counter
            elif tp2_hit and tp2_bar_idx is None:
                if tp2_hit_at and c_time_iso == tp2_hit_at:
                    tp2_bar_idx = bar_counter
                elif low_p <= tp2:
                    tp2_bar_idx = bar_counter

            # TP3
            if low_p <= tp3 and not tp3_hit:
                tp3_hit = True
                tp3_hit_at = c_time_iso

            # Fail-Fast Time-Stop (динамический порог по ТФ)
            if bar_counter >= effective_ff_bars and not tp1_hit and not tp2_hit and not fail_fast_triggered:
                cur_profit_pct = (entry - close_p) / entry * 100.0
                cur_r = cur_profit_pct / sl_pct if sl_pct > 0 else 0.0
                if cur_r < FAIL_FAST_MIN_R:
                    fail_fast_triggered = True
                    fail_fast_hit_at = c_time_iso
                    status = "closed_fail_fast"
                    bars_to_close = bar_counter
                    final_pnl = round(cur_profit_pct, 2)
                    break

            # Проверка Стоп-лосса / Безубытка (SHORT: строго bar_counter > tp2_bar_idx)
            is_be_eligible = (BREAKEVEN_AFTER_TP2 and tp2_bar_idx is not None and bar_counter > tp2_bar_idx) or be_triggered
            if is_be_eligible:
                if high_p >= be_price:
                    be_triggered = True
                    be_hit_at = c_time_iso
                    status = "closed_be"
                    bars_to_close = bar_counter
                    final_pnl = BREAKEVEN_OFFSET_PCT
                    break
            else:
                if high_p >= sl:
                    sl_hit = True
                    sl_hit_at = c_time_iso
                    status = "closed_sl"
                    bars_to_close = bar_counter
                    final_pnl = round((entry - sl) / entry * 100.0, 2)
                    break

    # Экспирация по TTL
    if status == "active" and bar_counter >= OUTCOME_MAX_BARS_TTL:
        status = "expired"
        bars_to_close = bar_counter
        final_pnl = round(
            ((current_price - entry) / entry * 100.0)
            if direction == "LONG"
            else ((entry - current_price) / entry * 100.0),
            2,
        )

    return {
        "current_price": current_price,
        "tp1_hit": 1 if tp1_hit else 0,
        "tp1_hit_at": tp1_hit_at,
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


def track_active_signals(db_path: Path | str = ANALYTICS_DB_PATH) -> int:
    """
    Проверяет все активные сигналы, обновляет их в БД и отправляет
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

        candles = fetch_recent_klines(sym, tf, limit=120)
        if not candles:
            continue

        updates = evaluate_signal_progress(sig, candles)
        if not updates:
            continue

        checked_count += 1
        new_status = updates.get("status", "active")
        prev_status = sig.get("status", "active")

        # Доли Model E: 40% TP1, 30% TP2, 30% Runner
        tp1_share = TP_SHARES[0] if len(TP_SHARES) > 0 else 0.40
        tp2_share = TP_SHARES[1] if len(TP_SHARES) > 1 else 0.30

        tp1_gain_pct = abs(entry - tp1) / entry * 100.0
        tp2_gain_pct = abs(entry - tp2) / entry * 100.0

        # Чистый PnL с вычетом комиссий roundtrip (0.2%)
        fee_roundtrip = COMMISSION_RATE * 2.0
        tp1_net_usd = (pos_size_usd * tp1_share * (tp1_gain_pct / 100.0)) - (pos_size_usd * tp1_share * fee_roundtrip)
        tp2_net_usd = (pos_size_usd * tp2_share * (tp2_gain_pct / 100.0)) - (pos_size_usd * tp2_share * fee_roundtrip)
        cum_tp_profit_usd = tp1_net_usd + tp2_net_usd

        # ── 1. Уведомление: Сработал TP1 ───────────────────────────────────────
        if updates.get("tp1_hit") and not sig.get("notified_tp1"):
            updates["notified_tp1"] = 1
            if tg_msg_id:
                tp1_text = (
                    f"🎯 <b>TP1 ВЗЯТ по цене {_fmt_pc(tp1)} (+{tp1_gain_pct:.2f}%)!</b>\n\n"
                    f"✅ Зафиксировано <b>{int(tp1_share*100)}% позиции</b>: <b>+${tp1_net_usd:.2f}</b> чистыми (с вычетом комиссий).\n"
                    f"👉 <b>ДЕЙСТВИЕ:</b> Держи лимитный ордер на TP2 ({_fmt_pc(tp2)})."
                )
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=tp1_text)
                logger.info(f"📤 Отправлено уведомление TP1 для #{sig_id} {sym}")

        # ── 2. Уведомление: Сработал TP2 (перенос в БУ) ────────────────────────
        if updates.get("tp2_hit") and not sig.get("notified_tp2"):
            updates["notified_tp2"] = 1
            if tg_msg_id:
                be_price_c = _fmt_pc(entry * (1.001 if direction == "LONG" else 0.999))
                tp2_text = (
                    f"🎯 <b>TP2 ВЗЯТ по цене {_fmt_pc(tp2)} (+{tp2_gain_pct:.2f}%)!</b>\n\n"
                    f"✅ Зафиксировано ещё <b>{int(tp2_share*100)}% позиции</b> (суммарно взято <b>+${cum_tp_profit_usd:.2f}</b> чистыми).\n"
                    f"👉 <b>СРОЧНОЕ ДЕЙСТВИЕ:</b> Перенеси Stop Loss в <b>БЕЗУБЫТОК</b> на цену {be_price_c}\n\n"
                    f"🏃 Остаток 30% сопровождаем по трейлингу или ждём цель TP3 ({_fmt_pc(tp3)})."
                )
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=tp2_text)
                logger.info(f"📤 Отправлено уведомление TP2 + Безубыток для #{sig_id} {sym}")

        # ── 3. Уведомление: Fail-Fast Time-Stop на баре 3 (45 мин / 3 часа) ──
        if updates.get("fail_fast_triggered") and not sig.get("notified_fail_fast"):
            updates["notified_fail_fast"] = 1
            updates["notified_closed"] = 1
            cur_pnl_pct = updates.get("final_pnl_pct", 0.0)
            ff_net_usd = (pos_size_usd * (cur_pnl_pct / 100.0)) - (pos_size_usd * fee_roundtrip)
            updates["net_pnl_usd"] = round(ff_net_usd, 2)
            time_name_map = {"5m": "30 минут (6 свечей 5m)", "15m": "45 минут (3 свечи 15m)", "1h": "3 часа (3 свечи 1h)", "4h": "8 часов (2 свечи 4h)"}
            time_name = time_name_map.get(tf, f"{tf} свечи")

            if tg_msg_id:
                ff_text = (
                    f"⏱ <b>ВРЕМЕННОЙ СТОП (FAIL-FAST — {time_name}):</b>\n\n"
                    f"⚠️ Прошло 3 свечи ({time_name}). Импульс угас, цена топчется на месте (прибыль менее +0.4R).\n"
                    f"👉 <b>ДЕЙСТВИЕ:</b> Закрой остаток позиции <b>по рынку (Market Close)</b>!\n\n"
                    f"📊 Текущий PnL с вычетом комиссий: <b>{ff_net_usd:+.2f}$</b> ({cur_pnl_pct:+.2f}%).\n"
                    f"🛡️ <i>Правило Fail-Fast отсекает 80% затяжных стопов.</i>"
                )
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=ff_text)
                logger.info(f"📤 Отправлено уведомление Fail-Fast для #{sig_id} {sym}")

        # ── 4. Уведомление: Сработал Стоп-Лосс ────────────────────────────────
        if new_status == "closed_sl" and not sig.get("notified_closed"):
            updates["notified_closed"] = 1
            has_tp1 = bool(updates.get("tp1_hit") or sig.get("tp1_hit"))
            if has_tp1:
                # 40% позиции уже зафиксировано на TP1 в плюс!
                # Стоп-лосс сработал только по оставшимся 60% позиции
                rem_share = 1.0 - tp1_share
                rem_sl_usd = rem_share * risk_usd
                net_pnl_usd = round(tp1_net_usd - rem_sl_usd, 2)
                net_r = net_pnl_usd / risk_usd if risk_usd > 0 else 0.0
                updates["net_pnl_usd"] = net_pnl_usd

                sign_str = "+" if net_pnl_usd > 0 else ""
                icon = "🟡" if net_pnl_usd >= 0 else "❌"
                sl_text = (
                    f"🛑 <b>СТОП-ЛОСС СРАБОТАЛ по цене {_fmt_pc(sl)}!</b>\n\n"
                    f"ℹ️ Ранее на TP1 было зафиксировано <b>{int(tp1_share*100)}%</b>: <b>+${tp1_net_usd:.2f}</b> чистыми.\n"
                    f"Оставшаяся часть позиции (<b>{int(rem_share*100)}%</b>) закрыта по SL (-${rem_sl_usd:.2f}).\n\n"
                    f"{icon} <b>Итоговый чистый PnL по сделке: {sign_str}${net_pnl_usd:.2f} ({net_r:+.2f}R с учётом комиссий).</b>\n"
                    f"Позиция полностью закрыта."
                )
            else:
                updates["net_pnl_usd"] = -risk_usd
                sl_text = (
                    f"🛑 <b>СТОП-ЛОСС СРАБОТАЛ по цене {_fmt_pc(sl)}!</b>\n\n"
                    f"❌ Убыток по сделке: <b>-${risk_usd:.2f}</b> (-1.0R с учётом комиссий).\n"
                    f"Позиция полностью закрыта."
                )
            if tg_msg_id:
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=sl_text)
                logger.info(f"📤 Отправлено уведомление SL для #{sig_id} {sym} (Net PnL: ${updates.get('net_pnl_usd')})")

        # ── 5. Уведомление: Сработал Безубыток (после взятия TP2) ──────────────
        if new_status == "closed_be" and not sig.get("notified_closed"):
            updates["notified_closed"] = 1
            updates["net_pnl_usd"] = round(cum_tp_profit_usd, 2)
            if tg_msg_id:
                be_p_c = _fmt_pc(entry * (1.001 if direction == "LONG" else 0.999))
                be_text = (
                    f"🛡️ <b>БЕЗУБЫТОК СРАБОТАЛ по цене {be_p_c}!</b>\n\n"
                    f"✅ Оставшаяся часть позиции закрыта в точке безубытка (+0.1%).\n"
                    f"💰 Итоговая чистая прибыль по сделке: <b>+${cum_tp_profit_usd:.2f}</b> (с учётом зафиксированных TP1 и TP2).\n"
                    f"Сделка успешно завершена в плюс!"
                )
                send_trade_update_reply(reply_to_message_id=tg_msg_id, text=be_text)
                logger.info(f"📤 Отправлено уведомление Безубыток для #{sig_id} {sym}")

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
        logger.info(f"Фоновый трекер отработки сигналов запущен (интервал: {self.interval_sec}с)")

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
    parser.add_argument("--interval", type=int, default=30, help="Интервал проверки в секундах")
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
