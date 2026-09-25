"""
live_scanner.py — Автономный сканер 5-волновых импульсов Эллиотта в реальном времени.

Функционал:
1. Непрерывный мониторинг ТОП-30 криптовалютных активов на 5 таймфреймах:
   - 5m, 15m, 1h, 4h, 1d (всего 150 торговых пар).
2. Загрузка свечей в реальном времени через Binance Futures API (fapi.binance.com) с фоллбэком на Bybit.
3. Поиск регулярных RSI-дивергенций на закрывшихся свечах.
4. Математическая верификация 5-волнового импульса W0-W1-W2-W3-W4-W5 (Score >= 65).
5. Расчёт профессионального торгового плана:
   - Вход на открытии свечи
   - Структурный Stop Loss за экстремумом W5 (буфер 0.15%)
   - 4 Take Profit по Фибоначчи (23.6%, 38.2%, 50.0%, 61.8%)
   - Перенос в безубыток (+0.1%) после взятия TP2
   - Фильтрация по R:R >= 1.5
6. Рендер графиков TradingView Dark + аудит Vision AI.
7. Отправка подтверждённых эталонных сигналов (Score >= 85%) в Telegram-канал.
8. Защита от дубликатов (seen_signals.json).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd
import requests

# Корневой путь
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    TOP_30_SYMBOLS, LIVE_TIMEFRAMES,
    MIN_ALGO_ELLIOTT_SCORE, MIN_VISION_CONFIRM_SCORE, TELEGRAM_MIN_SCORE,
    LOOKBACK_CANDLES, SL_BUFFER_PCT, MIN_RR_RATIO,
    BREAKEVEN_AFTER_TP2, BREAKEVEN_OFFSET_PCT,
    CONFIRMED_DIR, REJECTED_DIR, RESULTS_DIR,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
)
from elliott_detector import detect_elliott_impulse
from chart_renderer import render_signal_chart, annotate_chart_with_analysis
from elliott_prompt import build_prompt
from vision_filter import evaluate_elliott_impulse
from signal_scanner import calculate_rsi_wilder
from scipy.signal import argrelextrema
from trade_planner import calculate_trade_plan, TradePlan
from telegram_notifier import send_signal_to_telegram
from analytics_db import init_db, save_signal, log_scanner_health, get_analytics_summary
from outcome_tracker import OutcomeTrackerWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RalphLiveScanner")

SEEN_SIGNALS_FILE = BASE_DIR / "seen_signals.json"


def load_seen_signals() -> Set[str]:
    """Загружает идентификаторы уже обработанных сигналов."""
    if SEEN_SIGNALS_FILE.exists():
        try:
            data = json.loads(SEEN_SIGNALS_FILE.read_text(encoding="utf-8"))
            return set(data)
        except Exception as e:
            logger.warning(f"Ошибка загрузки {SEEN_SIGNALS_FILE}: {e}")
    return set()


def save_seen_signals(seen: Set[str]):
    """Сохраняет идентификаторы сигналов в файл."""
    try:
        # Сохраняем последние 10,000 сигналов для предотвращения разрастания
        trimmed = list(seen)[-10000:]
        SEEN_SIGNALS_FILE.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Ошибка сохранения {SEEN_SIGNALS_FILE}: {e}")


def fetch_binance_klines(symbol: str, interval: str, limit: int = 150) -> Optional[pd.DataFrame]:
    """
    Загружает исторические свечи через Binance Futures REST API.
    Фоллбэк на Bybit при ошибке.
    """
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    headers = {"User-Agent": "RalphTradeBot/2.1"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            raw = resp.json()
            if not raw or not isinstance(raw, list):
                return None
            
            # Binance: [0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume, 6: close_time, ...]
            # Отбрасываем последнюю незакрытую свечу (она ещё формируется)
            closed_candles = raw[:-1] if len(raw) > 1 else raw
            
            df = pd.DataFrame(closed_candles, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
            ])
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df.index = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.debug(f"Binance fetch error for {symbol} {interval}: {e}")

    # Фоллбэк: Bybit linear klines
    try:
        # Конвертация интервала для Bybit: 5m -> 5, 15m -> 15, 1h -> 60, 4h -> 240, 1d -> D
        bybit_tf_map = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
        bybit_interval = bybit_tf_map.get(interval, "15")
        bybit_url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={bybit_interval}&limit={limit}"
        resp_b = requests.get(bybit_url, headers=headers, timeout=10)
        if resp_b.status_code == 200:
            data = resp_b.json()
            if data.get("retCode") == 0 and "result" in data and "list" in data["result"]:
                rows = data["result"]["list"]
                if rows:
                    # Bybit возвращает в обратном порядке (новейшие первые)
                    rows.reverse()
                    closed_candles = rows[:-1] if len(rows) > 1 else rows
                    # [0: startTime, 1: openPrice, 2: highPrice, 3: lowPrice, 4: closePrice, 5: volume, 6: turnover]
                    df = pd.DataFrame(closed_candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    df['open'] = df['open'].astype(float)
                    df['high'] = df['high'].astype(float)
                    df['low'] = df['low'].astype(float)
                    df['close'] = df['close'].astype(float)
                    df['volume'] = df['volume'].astype(float)
                    df.index = pd.to_datetime(df['open_time'].astype(int), unit='ms', utc=True)
                    return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.debug(f"Bybit fetch error for {symbol} {interval}: {e}")

    return None


def scan_live_pair(
    symbol: str,
    interval: str,
    seen_signals: Set[str],
    dry_run: bool = False,
    send_tg: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Сканирует одну связку (монета, таймфрейм) на появление свежей дивергенции
    и валидного 5-волнового импульса Эллиотта.
    """
    df = fetch_binance_klines(symbol, interval, limit=120)
    if df is None or len(df) < 50:
        return None

    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    open_p = df['open'].values.astype(np.float64)
    n = len(df)

    rsi_len = 7 if interval in ("5m", "15m") else 14
    rsi = calculate_rsi_wilder(close, period=rsi_len)

    rsi_ob = 70.0
    rsi_os = 30.0
    pivot_order = 3
    min_pivot_dist = 6
    max_pivot_dist = 45

    high_pivots = argrelextrema(high, np.greater_equal, order=pivot_order)[0]
    low_pivots = argrelextrema(low, np.less_equal, order=pivot_order)[0]

    # Ищем дивергенцию, триггер которой произошёл на последней или предпоследней закрытой свече
    # Чтобы сигнал был актуальным и своевременным!
    latest_candle_idx = n - 1

    candidates: List[Tuple[int, str, int, float, float, float, float]] = []

    # 1. Bearish Divergence (SHORT)
    if len(high_pivots) >= 2:
        for i in range(len(high_pivots) - 1, 0, -1):
            p2 = int(high_pivots[i])
            p1 = int(high_pivots[i - 1])
            # Сигнал должен быть свежим (p2 не старше 3 свечей назад)
            if (latest_candle_idx - p2) > 3:
                break
            
            dist = p2 - p1
            if dist < min_pivot_dist or dist > max_pivot_dist:
                continue
            if high[p2] <= high[p1]:
                continue
            
            r1, r2 = float(rsi[p1]), float(rsi[p2])
            if r2 >= r1 or (r1 - r2) < 2.5:
                continue
            if r1 < 64.0 or r2 < 64.0:
                continue

            # Проверка чистоты
            if len(high[p1 + 1 : p2]) > 0 and np.max(high[p1 + 1 : p2]) > high[p2]:
                continue
            if len(rsi[p1 + 1 : p2]) > 0 and np.min(rsi[p1 + 1 : p2]) < rsi_os:
                continue

            candidates.append((p2, "SHORT", p1, float(high[p1]), r1, r2, float(high[p2])))

    # 2. Bullish Divergence (LONG)
    if len(low_pivots) >= 2:
        for i in range(len(low_pivots) - 1, 0, -1):
            p2 = int(low_pivots[i])
            p1 = int(low_pivots[i - 1])
            if (latest_candle_idx - p2) > 3:
                break
            
            dist = p2 - p1
            if dist < min_pivot_dist or dist > max_pivot_dist:
                continue
            if low[p2] >= low[p1]:
                continue
            
            r1, r2 = float(rsi[p1]), float(rsi[p2])
            if r2 <= r1 or (r2 - r1) < 2.5:
                continue
            if r1 > 36.0 or r2 > 36.0:
                continue

            # Проверка чистоты
            if len(low[p1 + 1 : p2]) > 0 and np.min(low[p1 + 1 : p2]) < low[p2]:
                continue
            if len(rsi[p1 + 1 : p2]) > 0 and np.max(rsi[p1 + 1 : p2]) > rsi_ob:
                continue

            candidates.append((p2, "LONG", p1, float(low[p1]), r1, r2, float(low[p2])))

    if not candidates:
        return None

    # Берём самый свежий кандидат
    candidates.sort(key=lambda c: c[0], reverse=True)
    p2, direction, p1, p1_price, p1_rsi, p2_rsi, p2_price = candidates[0]

    # Проверка на дубликат по временной метке бара W5
    bar_ts = int(df.index[p2].timestamp()) if isinstance(df.index, pd.DatetimeIndex) else p2
    sig_id = f"{symbol}_{interval}_{direction}_{bar_ts}"
    if sig_id in seen_signals:
        return None

    # 1. Алгоритмический анализ импульса Эллиотта
    algo_res = detect_elliott_impulse(
        high=high, low=low, close=close, open_p=open_p,
        w3_bar=p1, w5_bar=p2,
        direction=direction, rsi=rsi,
    )

    if not algo_res.is_valid or algo_res.score < MIN_ALGO_ELLIOTT_SCORE:
        # Помечаем как просмотренный, чтобы не пересчитывать
        seen_signals.add(sig_id)
        return None

    # Точка входа: цена закрытия W5 (или open текущей свечи если есть)
    entry_price = float(close[p2]) if p2 + 1 >= n else float(open_p[p2 + 1])

    # 2. Расчёт профессионального торгового плана
    trade_plan = calculate_trade_plan(
        wave_points=algo_res.wave_points,
        direction=direction,
        entry_price=entry_price,
        buffer_pct=SL_BUFFER_PCT,
        min_rr_ratio=MIN_RR_RATIO,
    )

    if not trade_plan.is_viable:
        logger.info(f"⏭️ {symbol} {interval} {direction}: R:R {trade_plan.rr_ratio:.2f} < {MIN_RR_RATIO} — пропуск.")
        seen_signals.add(sig_id)
        return None

    # Длительность формирования
    w0_bar = algo_res.wave_indices.get("W0", p1 - 20)
    dur_bars = p2 - w0_bar
    unit_m = 5 if interval == "5m" else 15 if interval == "15m" else 60 if interval == "1h" else 240 if interval == "4h" else 1440
    dur_hours = (dur_bars * unit_m) / 60.0

    w0_rsi = algo_res.details.get("w0_rsi")
    w0_stat = algo_res.details.get("w0_status", "N/A")
    orig_div = algo_res.details.get("origin_div", False)
    orig_b = algo_res.details.get("orig_prev_bar")
    orig_p = algo_res.details.get("orig_prev_price")
    orig_r = algo_res.details.get("orig_prev_rsi")
    w3_mom = algo_res.details.get("w3_momentum_peak", False)

    logger.info(f"⚡ Кандидат найден: {symbol} {interval} {direction} (Algo Score: {algo_res.score:.0f}, R:R: {trade_plan.rr_ratio:.1f}:1)")

    # 3. Рендер графика TradingView
    chart_png = render_signal_chart(
        df=df,
        rsi_values=rsi,
        signal_idx=p2,
        direction=direction,
        symbol=symbol,
        interval=interval,
        rsi_ob=rsi_ob,
        rsi_os=rsi_os,
        swept_price=p1_price,
        lookback=min(n - 1, LOOKBACK_CANDLES),
        wave_points=algo_res.wave_points,
        wave_direction=algo_res.wave_direction,
        wave_indices=algo_res.wave_indices,
        algo_score=algo_res.score,
        swept_bar_idx=p1,
        swept_rsi=p1_rsi,
        signal_rsi=p2_rsi,
        origin_div=orig_div,
        orig_prev_bar=orig_b,
        orig_prev_price=orig_p,
        orig_prev_rsi=orig_r,
        trade_plan=trade_plan,
    )

    # 4. Аудит через Vision AI (Gemini / OpenRouter / Groq)
    if dry_run:
        score = int(algo_res.score)
        passed = True
        reason = f"LIVE DRY RUN: {algo_res.reason}"
        analysis = {
            "passed": True, "score": score, "reason": reason, "provider": "algo_only",
            "wave_direction": algo_res.wave_direction,
            "waves_identified": "; ".join(f"{k}: ${v:,.2f}" for k, v in sorted(algo_res.wave_points.items())),
            "rule_violations": "none",
        }
    else:
        prompt = build_prompt(
            symbol=symbol, interval=interval, direction=direction,
            signal_price=entry_price, rsi_value=p2_rsi,
            swept_level=p1_price, algo_score=algo_res.score,
            algo_summary=algo_res.reason, w0_rsi=w0_rsi,
            w0_status=w0_stat, origin_div=orig_div,
            w3_momentum_peak=w3_mom,
        )
        vision_res = evaluate_elliott_impulse(png_bytes=chart_png, prompt=prompt)
        score = vision_res["score"]
        passed = vision_res["passed"]
        reason = vision_res["reason"]

        # Если внешние Vision API временно лимитированы (429/offline), но импульс математически идеален
        if vision_res.get("provider") == "fallback" and algo_res.score >= TELEGRAM_MIN_SCORE:
            score = int(algo_res.score)
            passed = True
            reason = f"Математический эталон Эллиотта: {algo_res.reason}"
            logger.info(f"✨ Математический эталон подтверждён (Vision API offline/fallback): {symbol} {interval} {direction} (Score: {score})")

        analysis = {
            "passed": passed, "score": score, "reason": reason,
            "provider": vision_res.get("provider", "vision_ai") if vision_res.get("provider") != "fallback" else "algo_auditor",
            "wave_direction": algo_res.wave_direction,
            "waves_identified": "; ".join(f"{k}: ${v:,.2f}" for k, v in sorted(algo_res.wave_points.items())),
            "rule_violations": vision_res.get("rule_violations", "none"),
        }

    # 5. Сохранение аннотированного PNG
    target_dir = CONFIRMED_DIR if (passed and score >= MIN_VISION_CONFIRM_SCORE) else REJECTED_DIR
    fn = f"{symbol}_{interval}_{direction}_{p2}_s{score}.png"
    chart_path = target_dir / fn

    annotated_png = chart_png
    try:
        annotated_png = annotate_chart_with_analysis(
            chart_png=chart_png,
            analysis=analysis,
            signal_direction=direction,
            signal_price=entry_price,
            trade_plan=trade_plan,
        )
        chart_path.write_bytes(annotated_png)
    except Exception as e:
        logger.warning(f"Ошибка сохранения аннотированного графика {fn}: {e}")
        chart_path.write_bytes(chart_png)

    # Добавляем в список обработанных
    seen_signals.add(sig_id)
    save_seen_signals(seen_signals)

    # 6. Отправка подтверждённого сигнала в Telegram (Score >= 85)
    sent_to_tg_flag = False
    sig_dt = df.index[p2].to_pydatetime() if isinstance(df.index, pd.DatetimeIndex) else datetime.now(timezone.utc)
    if send_tg and passed and score >= TELEGRAM_MIN_SCORE:
        sent = send_signal_to_telegram(
            chart_png=annotated_png,
            symbol=symbol,
            interval=interval,
            direction=direction,
            signal_price=entry_price,
            swept_rsi=p1_rsi,
            signal_rsi=p2_rsi,
            current_rsi=float(rsi[-1]),
            reason=reason,
            score=score,
            detection_time=sig_dt,
            trade_plan=trade_plan,
            dur_bars=dur_bars,
            dur_hours=dur_hours,
            w0_rsi=w0_rsi,
            w0_status=w0_stat,
            origin_div=orig_div,
            w3_longest=algo_res.details.get("w3_longest", True),
        )
        if sent:
            sent_to_tg_flag = True
            logger.info(f"🚀 СИГНАЛ УСПЕШНО ОТПРАВЛЕН В ТЕЛЕГРАМ: {symbol} {interval} {direction} (Score: {score})")
        else:
            logger.error(f"❌ Ошибка отправки сигнала в Telegram: {symbol} {interval}")

    # 7. Сохранение сигнала в SQLite базу аналитики
    try:
        sig_data_db = {
            "created_at": sig_dt.isoformat(),
            "symbol": symbol,
            "interval": interval,
            "direction": direction,
            "algo_score": algo_res.score,
            "vision_score": score,
            "vision_provider": analysis.get("provider", "unknown"),
            "vision_reason": reason,
            "entry_price": entry_price,
            "sl_price": trade_plan.sl_price,
            "tp1_price": trade_plan.tp1_price,
            "tp2_price": trade_plan.tp2_price,
            "tp3_price": trade_plan.tp3_price,
            "tp4_price": trade_plan.tp4_price,
            "rr_ratio": trade_plan.rr_ratio,
            "impulse_pct": trade_plan.impulse_pct,
            "sl_pct": trade_plan.sl_pct,
            "sent_to_telegram": 1 if sent_to_tg_flag else 0,
            "w0_price": algo_res.wave_points.get("W0"),
            "w1_price": algo_res.wave_points.get("W1"),
            "w2_price": algo_res.wave_points.get("W2"),
            "w3_price": algo_res.wave_points.get("W3"),
            "w4_price": algo_res.wave_points.get("W4"),
            "w5_price": algo_res.wave_points.get("W5"),
            "wave_direction": algo_res.wave_direction,
            "origin_div": 1 if orig_div else 0,
            "w3_longest": 1 if algo_res.details.get("w3_longest", True) else 0,
            "dur_bars": dur_bars,
            "dur_hours": dur_hours,
            "bar_timestamp": bar_ts,
        }
        db_id = save_signal(sig_data_db)
        logger.info(f"💾 Сигнал #{db_id} сохранён в ralph_analytics.db ({symbol} {interval} {direction})")
    except Exception as e:
        logger.warning(f"Ошибка сохранения сигнала в БД: {e}")

    return {
        "symbol": symbol, "interval": interval, "direction": direction,
        "score": score, "passed": passed, "rr": trade_plan.rr_ratio,
        "impulse_pct": trade_plan.impulse_pct,
    }


def run_scanner_loop(
    symbols: List[str] = TOP_30_SYMBOLS,
    intervals: List[str] = LIVE_TIMEFRAMES,
    scan_interval_sec: int = 60,
    dry_run: bool = False,
    send_tg: bool = True,
    max_workers: int = 8,
):
    """
    Главный цикл сканирования всех инструментов в реальном времени.
    """
    logger.info("=" * 72)
    logger.info("🚀 Запуск RalphTradeBot Real-Time Scanner v2.2")
    logger.info(f"   Активов: {len(symbols)} | Таймфреймов: {len(intervals)} | Всего пар: {len(symbols) * len(intervals)}")
    logger.info(f"   Интервал сканирования: каждые {scan_interval_sec} сек | Поток: {max_workers}")
    logger.info(f"   Минимум для Telegram: Score ≥ {TELEGRAM_MIN_SCORE}% (Textbook)")
    logger.info(f"   Правило безубытка: вход +{BREAKEVEN_OFFSET_PCT}% после взятия TP2")
    logger.info("=" * 72)

    init_db()

    # Запуск фонового трекера отработки активных сигналов (проверяет TP1-TP4 / BE / SL)
    tracker_worker = OutcomeTrackerWorker(interval_sec=scan_interval_sec)
    tracker_worker.start()

    seen_signals = load_seen_signals()
    logger.info(f"Загружено ранее обработанных сигналов: {len(seen_signals)}")

    # Создаём список всех задач (символ, интервал)
    tasks = [(s, i) for s in symbols for i in intervals]

    running = True

    def _sig_handler(sig, frame):
        nonlocal running
        logger.info("\n🛑 Получен сигнал завершения. Остановка сканера...")
        running = False
        tracker_worker.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    scan_iteration = 0

    while running:
        scan_iteration += 1
        t_start = time.time()
        found_signals = []

        logger.info(f"\n🔄 [Итерация #{scan_iteration}] Сканирование {len(tasks)} пар...")

        # Параллельное сканирование пар через пул потоков
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(scan_live_pair, sym, tf, seen_signals, dry_run, send_tg): (sym, tf)
                for sym, tf in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                sym, tf = future_to_task[future]
                try:
                    res = future.result()
                    if res:
                        found_signals.append(res)
                except Exception as e:
                    logger.debug(f"Ошибка сканирования {sym} {tf}: {e}")

        elapsed = time.time() - t_start
        confirmed_cnt = len([s for s in found_signals if s.get("passed") and s.get("score", 0) >= TELEGRAM_MIN_SCORE])
        rejected_cnt = len(found_signals) - confirmed_cnt

        status_msg = f"⏱️ Итерация #{scan_iteration} завершена за {elapsed:.1f}с. Найдено новых сигналов: {len(found_signals)}"
        if found_signals:
            for s in found_signals:
                status_msg += f"\n   ★ {s['symbol']} {s['interval']} {s['direction']} (Score {s['score']}, R:R {s['rr']:.1f}:1)"
        logger.info(status_msg)

        # Запись метрики здоровья в БД
        try:
            log_scanner_health(
                iteration=scan_iteration,
                pairs_scanned=len(tasks),
                candidates_found=len(found_signals),
                signals_confirmed=confirmed_cnt,
                signals_rejected=rejected_cnt,
                scan_duration_sec=elapsed,
                vision_api_status="online" if not dry_run else "dry_run",
            )
        except Exception as e:
            logger.warning(f"Ошибка сохранения метрик здоровья: {e}")

        # Каждые 10 итераций выводим краткую сводку аналитики базы
        if scan_iteration % 10 == 0:
            try:
                stats = get_analytics_summary()
                logger.info(
                    f"📊 [Ralph Analytics] Сигналов: {stats['total_signals']} | "
                    f"Активных: {stats['active_signals']} | Закрытых: {stats['closed_signals']} | "
                    f"Win Rate: {stats['win_rate_pct']}% | TP2 hit: {stats['tp_hit_rates']['tp2_count']}"
                )
            except Exception as e:
                logger.debug(f"Ошибка получения сводки: {e}")

        # Ожидание до следующего цикла
        sleep_time = max(1.0, scan_interval_sec - elapsed)
        time.sleep(sleep_time)

    tracker_worker.stop()
    logger.info("Сканер успешно остановлен.")


def main():
    parser = argparse.ArgumentParser(description="RalphTradeBot Real-Time Elliott Wave Scanner")
    parser.add_argument("--dry-run", action="store_true", help="Запуск без запросов к платной Vision AI модели")
    parser.add_argument("--no-tg", action="store_true", help="Не отправлять сообщения в Telegram")
    parser.add_argument("--interval", type=int, default=60, help="Период сканирования в секундах (по умолч. 60)")
    parser.add_argument("--workers", type=int, default=8, help="Количество параллельных потоков загрузки (по умолч. 8)")
    parser.add_argument("--symbols", type=str, default=None, help="Список символов через запятую (по умолч. ТОП-30)")
    parser.add_argument("--timeframes", type=str, default=None, help="Список ТФ через запятую (5m,15m,1h,4h,1d)")

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else TOP_30_SYMBOLS
    timeframes = [t.strip().lower() for t in args.timeframes.split(",") if t.strip()] if args.timeframes else LIVE_TIMEFRAMES

    run_scanner_loop(
        symbols=symbols,
        intervals=timeframes,
        scan_interval_sec=args.interval,
        dry_run=args.dry_run,
        send_tg=not args.no_tg,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
