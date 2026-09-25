"""
outcome_tracker.py — Фоновый трекер отработки торговых планов и целей импульсов Эллиотта.

Функционал:
1. Загружает активные сделки (status='active') из SQLite базы `ralph_analytics.db`.
2. Запрашивает свежие свечи с Binance Futures API.
3. Проверяет достижение целей:
   - TP1 (23.6%), TP2 (38.2%), TP3 (50.0%), TP4 (61.8%)
   - Активация безубытка (вход +0.1%) после взятия TP2
   - Срабатывание безубытка (closed_be) или первоначального SL (closed_sl)
   - Полное взятие импульса по TP4 (closed_tp)
   - Экспирация по таймауту баров (expired)
4. Рассчитывает MFE (Maximum Favorable Excursion) и MAE (Maximum Adverse Excursion).
5. Сохраняет обновлённые метрики в базу данных.
6. Может запускаться автономно или как фоновый поток (daemon) внутри `live_scanner.py`.
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
    OUTCOME_MAX_BARS_TTL,
    OUTCOME_TRACKER_INTERVAL_SEC,
)

logger = logging.getLogger("RalphOutcomeTracker")


def fetch_recent_klines(symbol: str, interval: str, limit: int = 150) -> Optional[List[Dict[str, Any]]]:
    """Загружает последние свечи через Binance Futures REST API."""
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    headers = {"User-Agent": "RalphTradeBot-Tracker/2.2"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            raw = resp.json()
            candles = []
            for row in raw:
                candles.append(
                    {
                        "open_time": int(row[0]),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "close_time": int(row[6]),
                    }
                )
            return candles
    except Exception as e:
        logger.debug(f"Ошибка загрузки свечей для {symbol} {interval}: {e}")
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
    tp4 = float(signal["tp4_price"])
    bar_ts = signal.get("bar_timestamp")

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

    status = signal.get("status", "active")
    final_pnl = float(signal.get("final_pnl_pct", 0.0))
    bars_to_first_tp = int(signal.get("bars_to_first_tp", 0))
    bars_to_close = int(signal.get("bars_to_close", 0))
    max_favorable = float(signal.get("max_favorable", 0.0))
    max_adverse = float(signal.get("max_adverse", 0.0))

    # Находим индекс свечи входа
    entry_idx = -1
    if bar_ts:
        for idx, c in enumerate(candles):
            # Проверяем совпадение по секундам или миллисекундам
            c_ts_sec = c["open_time"] // 1000
            if abs(c_ts_sec - bar_ts) <= 60 or abs(c["open_time"] - bar_ts) <= 60000:
                entry_idx = idx
                break

    # Если точная свеча не найдена, берём свечи за последние N баров
    eval_candles = candles[entry_idx + 1 :] if entry_idx != -1 else candles[-50:]

    if not eval_candles:
        return {}

    current_price = eval_candles[-1]["close"]
    be_active = tp2_hit and BREAKEVEN_AFTER_TP2
    be_price = (
        entry * (1.0 + BREAKEVEN_OFFSET_PCT / 100.0)
        if direction == "LONG"
        else entry * (1.0 - BREAKEVEN_OFFSET_PCT / 100.0)
    )

    bar_counter = 0

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

            # TP1
            if high_p >= tp1 and not tp1_hit:
                tp1_hit = True
                tp1_hit_at = c_time_iso
                if bars_to_first_tp == 0:
                    bars_to_first_tp = bar_counter

            # TP2 -> Активация безубытка
            if high_p >= tp2 and not tp2_hit:
                tp2_hit = True
                tp2_hit_at = c_time_iso
                be_active = BREAKEVEN_AFTER_TP2

            # TP3
            if high_p >= tp3 and not tp3_hit:
                tp3_hit = True
                tp3_hit_at = c_time_iso

            # TP4 -> Полный тейк-профит
            if high_p >= tp4 and not tp4_hit:
                tp4_hit = True
                tp4_hit_at = c_time_iso
                status = "closed_tp"
                bars_to_close = bar_counter
                final_pnl = round((tp4 - entry) / entry * 100.0, 2)
                break

            # Проверка Стоп-лосса / Безубытка
            if be_active:
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

            # TP1
            if low_p <= tp1 and not tp1_hit:
                tp1_hit = True
                tp1_hit_at = c_time_iso
                if bars_to_first_tp == 0:
                    bars_to_first_tp = bar_counter

            # TP2 -> Активация безубытка
            if low_p <= tp2 and not tp2_hit:
                tp2_hit = True
                tp2_hit_at = c_time_iso
                be_active = BREAKEVEN_AFTER_TP2

            # TP3
            if low_p <= tp3 and not tp3_hit:
                tp3_hit = True
                tp3_hit_at = c_time_iso

            # TP4 -> Полный тейк-профит
            if low_p <= tp4 and not tp4_hit:
                tp4_hit = True
                tp4_hit_at = c_time_iso
                status = "closed_tp"
                bars_to_close = bar_counter
                final_pnl = round((entry - tp4) / entry * 100.0, 2)
                break

            # Проверка Стоп-лосса / Безубытка
            if be_active:
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

    # Экспирация по TTL (превышено максимальное число баров)
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
        "status": status,
        "final_pnl_pct": final_pnl,
        "bars_to_first_tp": bars_to_first_tp,
        "bars_to_close": bars_to_close,
        "max_favorable": max_favorable,
        "max_adverse": max_adverse,
    }


def track_active_signals(db_path: Path | str = ANALYTICS_DB_PATH) -> int:
    """
    Проверяет все активные сигналы и обновляет их состояние в базе данных.
    Возвращает количество проверенных сигналов.
    """
    active_signals = get_active_signals(db_path=db_path)
    if not active_signals:
        return 0

    checked_count = 0
    for sig in active_signals:
        sym = sig["symbol"]
        tf = sig["interval"]
        sig_id = sig["signal_id"]

        candles = fetch_recent_klines(sym, tf, limit=120)
        if not candles:
            continue

        updates = evaluate_signal_progress(sig, candles)
        if updates:
            update_signal_outcome(sig_id, updates, db_path=db_path)
            checked_count += 1

            prev_status = sig.get("status")
            new_status = updates.get("status")
            if new_status != prev_status:
                pnl = updates.get("final_pnl_pct", 0.0)
                icon = "🎯" if new_status == "closed_tp" else "🛡️" if new_status == "closed_be" else "🛑"
                logger.info(
                    f"{icon} Сигнал #{sig_id} {sym} {tf} {sig['direction']} закрыт со статусом '{new_status}'! PnL: {pnl:+.2f}%"
                )
            elif updates.get("tp2_hit") and not sig.get("tp2_hit"):
                logger.info(
                    f"⭐ Сигнал #{sig_id} {sym} {tf} достиг TP2! Стоп-лосс переведён в безубыток (+{BREAKEVEN_OFFSET_PCT}%)."
                )

        # Небольшая пауза между запросами к API биржи
        time.sleep(0.2)

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
    parser.add_argument("--interval", type=int, default=60, help="Интервал проверки в секундах")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    init_db()

    logger.info("Запуск автономного трекера исходов сигналов...")
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
