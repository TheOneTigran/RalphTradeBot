"""
backtest_runner.py — Мульти-инструментальный бэктест-раннер для RalphTradeBot.

1. Запускает сканирование и аудит на топ-15 инструментах на разных таймфреймах.
2. Оценивает структуру 5-волнового импульса через математический детектор v2.1 и Vision AI.
3. При score >= 85% отправляет живые сигналы в Telegram канал/чат (если указан флаг --send-tg).
4. Вычисляет длительность формирования импульса W0->W5 в барах и часах.
5. Сохраняет графики и генерирует автономный отчет 'RalphTradeBot backtest.html'.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, RESULTS_DIR, CONFIRMED_DIR, REJECTED_DIR,
    DEFAULT_PRESETS, TOP_15_SYMBOLS,
    MIN_ALGO_ELLIOTT_SCORE, MIN_VISION_CONFIRM_SCORE, TELEGRAM_MIN_SCORE,
    LOOKBACK_CANDLES,
)
from elliott_detector import detect_elliott_impulse
from chart_renderer import render_signal_chart, annotate_chart_with_analysis
from elliott_prompt import build_prompt
from vision_filter import evaluate_elliott_impulse
from signal_scanner import scan_signals, calculate_rsi_wilder
from telegram_notifier import send_signal_to_telegram
from html_reporter import generate_html_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RalphTradeBot")


def _interval_to_minutes(interval: str) -> int:
    unit = interval[-1].lower()
    val = int(interval[:-1])
    if unit == 'm':
        return val
    elif unit == 'h':
        return val * 60
    elif unit == 'd':
        return val * 1440
    return 15


def _resample_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    minutes = _interval_to_minutes(target_interval)
    if minutes <= 1:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime_utc' in df.columns:
            df.index = pd.to_datetime(df['datetime_utc'], utc=True)
        elif 'timestamp_ms' in df.columns:
            df.index = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
        elif 'ts_open' in df.columns:
            df.index = pd.to_datetime(df['ts_open'], unit='ms', utc=True)
        elif 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        else:
            try:
                df.index = pd.to_datetime(df.iloc[:, 0], unit='ms', utc=True)
            except Exception:
                return df

    rule = f"{minutes}min"
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }
    if 'volume' in df.columns:
        agg['volume'] = 'sum'

    return df.resample(rule).agg(agg).dropna()


def load_data(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Загружает исторические данные и ресемплирует их при необходимости."""
    data_dirs = [
        DATA_DIR,
        Path("c:/Users/user/Desktop/RalphTradeBot/data"),
        Path("c:/Users/user/Desktop/Hernya/RalphTradeBot/data"),
    ]
    for d_dir in data_dirs:
        patterns = [
            (d_dir / f"{symbol}_{interval}.parquet", False),
            (d_dir / f"{symbol}_1m.parquet", True),
        ]
        for path, needs_resample in patterns:
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    col_map = {c.lower(): c for c in df.columns}
                    for std_col in ['open', 'high', 'low', 'close', 'volume']:
                        if std_col in col_map and col_map[std_col] != std_col:
                            df.rename(columns={col_map[std_col]: std_col}, inplace=True)
                    
                    if needs_resample:
                        df = _resample_ohlcv(df, interval)
                    return df
                except Exception as e:
                    logger.warning(f"Ошибка загрузки {path.name}: {e}")
                    continue
    return None


def run_backtest(
    presets: Dict[str, Dict[str, Any]],
    send_telegram: bool = False,
    dry_run: bool = False,
    limit_signals_per_preset: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Основной цикл бэктестинга по списку пресетов."""
    all_signals_data: List[Dict[str, Any]] = []

    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Запуск RalphTradeBot Backtest")
    logger.info(f"   Пресетов: {len(presets)} | Send TG: {send_telegram} | Dry Run: {dry_run}")
    logger.info(f"{'='*70}\n")

    for p_name, p_cfg in presets.items():
        sym = p_cfg["symbol"]
        tf = p_cfg["interval"]
        logger.info(f"🔍 Анализ пресета: {p_name} ({sym} {tf})")

        df = load_data(sym, tf)
        if df is None or len(df) < 100:
            logger.warning(f"⚠️ Недостаточно данных для {sym} {tf}")
            continue

        rsi_len = p_cfg.get("rsi_len", 7)
        rsi_array = calculate_rsi_wilder(df['close'].values.astype(np.float64), rsi_len)

        signals = scan_signals(
            df=df,
            left_bars=p_cfg.get("left_bars", 8),
            right_bars=p_cfg.get("right_bars", 4),
            rsi_len=rsi_len,
            rsi_ob=p_cfg.get("rsi_ob", 70.0),
            rsi_os=p_cfg.get("rsi_os", 30.0),
            tp_pct=p_cfg.get("tp_pct", 1.8),
            sl_pct=p_cfg.get("sl_pct", 3.5),
        )

        logger.info(f"  Найдено первичных дивергенций: {len(signals)}")
        if limit_signals_per_preset and len(signals) > limit_signals_per_preset:
            signals = signals[:limit_signals_per_preset]

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_p = df['open'].values

        minutes = _interval_to_minutes(tf)

        for idx, sig in enumerate(signals):
            # 1. Алгоритмический детектор
            algo_res = detect_elliott_impulse(
                high=high, low=low, close=close, open_p=open_p,
                w3_bar=sig.swept_level_bar, w5_bar=sig.bar_index,
                direction=sig.direction, rsi=rsi_array,
            )

            algo_passed = algo_res.is_valid and (algo_res.score >= MIN_ALGO_ELLIOTT_SCORE)
            
            # Извлечение параметров W0 и W3
            w0_rsi = algo_res.details.get("w0_rsi")
            w0_stat = algo_res.details.get("w0_status", "N/A")
            orig_div = algo_res.details.get("origin_div", False)
            orig_b = algo_res.details.get("orig_prev_bar")
            orig_p = algo_res.details.get("orig_prev_price")
            orig_r = algo_res.details.get("orig_prev_rsi")
            w3_mom = algo_res.details.get("w3_momentum_peak", False)

            # Вычисление времени формирования импульса
            w0_bar = algo_res.wave_indices.get("W0", sig.swept_level_bar - 20) if algo_res.is_valid else (sig.bar_index - 30)
            dur_bars = sig.bar_index - w0_bar
            dur_hours = (dur_bars * minutes) / 60.0

            # 2. Рендер графика
            chart_png = render_signal_chart(
                df=df,
                rsi_values=rsi_array,
                signal_idx=sig.bar_index,
                direction=sig.direction,
                symbol=sym,
                interval=tf,
                rsi_ob=p_cfg.get("rsi_ob", 70.0),
                rsi_os=p_cfg.get("rsi_os", 30.0),
                swept_price=sig.swept_level_price,
                lookback=LOOKBACK_CANDLES,
                wave_points=algo_res.wave_points if algo_res.is_valid else None,
                wave_direction=algo_res.wave_direction,
                wave_indices=algo_res.wave_indices if algo_res.is_valid else None,
                algo_score=algo_res.score if algo_res.is_valid else None,
                swept_bar_idx=sig.swept_level_bar,
                swept_rsi=sig.swept_level_rsi,
                signal_rsi=sig.rsi_value,
                origin_div=orig_div,
                orig_prev_bar=orig_b,
                orig_prev_price=orig_p,
                orig_prev_rsi=orig_r,
            )

            # 3. Аудит Vision AI
            if not algo_passed:
                analysis = {
                    "passed": False,
                    "score": int(algo_res.score),
                    "reason": f"Отклонено алгоритмом: {algo_res.reason}",
                    "provider": "algo_detector",
                    "wave_direction": algo_res.wave_direction,
                    "waves_identified": "",
                    "rule_violations": "; ".join(algo_res.rule_violations),
                    "w0_rsi": w0_rsi, "w0_status": w0_stat,
                    "origin_div": orig_div, "w3_momentum_peak": w3_mom,
                }
            elif dry_run:
                analysis = {
                    "passed": True,
                    "score": int(algo_res.score),
                    "reason": f"DRY RUN: {algo_res.reason}",
                    "provider": "algo_only",
                    "wave_direction": algo_res.wave_direction,
                    "waves_identified": ", ".join(f"{k}: ${v:,.1f}" for k, v in sorted(algo_res.wave_points.items())),
                    "rule_violations": "none",
                    "w0_rsi": w0_rsi, "w0_status": w0_stat,
                    "origin_div": orig_div, "w3_momentum_peak": w3_mom,
                }
            else:
                prompt = build_prompt(
                    symbol=sym, interval=tf, direction=sig.direction,
                    signal_price=sig.signal_price, rsi_value=sig.rsi_value,
                    swept_level=sig.swept_level_price, algo_score=algo_res.score,
                    algo_summary=algo_res.reason, w0_rsi=w0_rsi,
                    w0_status=w0_stat, origin_div=orig_div,
                    w3_momentum_peak=w3_mom,
                )
                vision_analysis = evaluate_elliott_impulse(png_bytes=chart_png, prompt=prompt)
                analysis = {
                    "passed": vision_analysis["passed"],
                    "score": vision_analysis["score"],
                    "reason": vision_analysis["reason"],
                    "provider": vision_analysis["provider"],
                    "wave_direction": algo_res.wave_direction,
                    "waves_identified": ", ".join(f"{k}: ${v:,.1f}" for k, v in sorted(algo_res.wave_points.items())),
                    "rule_violations": vision_analysis.get("rule_violations", "none"),
                    "w0_rsi": w0_rsi, "w0_status": w0_stat,
                    "origin_div": orig_div, "w3_momentum_peak": w3_mom,
                }
                time.sleep(0.4)

            score = analysis["score"]
            passed = analysis["passed"]

            # 4. Сохранение аннотированного PNG
            target_dir = CONFIRMED_DIR if passed else REJECTED_DIR
            fn = f"{p_name}_{sig.direction}_{sig.bar_index}_s{score}.png"
            chart_path = target_dir / fn

            annotated_png = chart_png
            try:
                annotated_png = annotate_chart_with_analysis(
                    chart_png=chart_png,
                    analysis=analysis,
                    signal_direction=sig.direction,
                    signal_price=sig.signal_price,
                    trade_result=sig.trade_result,
                    trade_pnl=sig.trade_pnl_pct,
                )
                chart_path.write_bytes(annotated_png)
            except Exception as e:
                logger.warning(f"Ошибка аннотации графика {fn}: {e}")
                chart_path.write_bytes(chart_png)

            # Относительный путь для HTML
            rel_img = f"results/{'confirmed_signals' if passed else 'rejected_signals'}/{fn}"

            # 5. Отправка в Telegram при Score >= 85
            if send_telegram and passed and score >= TELEGRAM_MIN_SCORE:
                # Дата свечи сигнала если доступна
                sig_dt = None
                if isinstance(df.index, pd.DatetimeIndex):
                    sig_dt = df.index[sig.bar_index].to_pydatetime()

                send_signal_to_telegram(
                    chart_png=annotated_png,
                    symbol=sym,
                    interval=tf,
                    direction=sig.direction,
                    signal_price=sig.signal_price,
                    swept_rsi=sig.swept_level_rsi,
                    signal_rsi=sig.rsi_value,
                    current_rsi=sig.rsi_value,
                    reason=analysis["reason"],
                    score=score,
                    detection_time=sig_dt,
                )

            item = {
                "preset": p_name,
                "symbol": sym,
                "interval": tf,
                "direction": sig.direction,
                "signal_price": sig.signal_price,
                "swept_rsi": sig.swept_level_rsi,
                "signal_rsi": sig.rsi_value,
                "score": score,
                "passed": passed,
                "trade_result": sig.trade_result,
                "trade_pnl_pct": sig.trade_pnl_pct,
                "duration_bars": dur_bars,
                "duration_hours": dur_hours,
                "w0_rsi": w0_rsi,
                "w0_status": w0_stat,
                "origin_div": orig_div,
                "reason": analysis["reason"],
                "chart_img_rel": rel_img,
                "provider": analysis["provider"],
            }
            all_signals_data.append(item)

            status_str = "✅ TEXTBOOK" if score >= 85 else "✅ CONFIRMED" if passed else "❌ REJECTED"
            logger.info(f"    [{idx+1}/{len(signals)}] {status_str} | Score: {score} | PnL: {sig.trade_pnl_pct:+.2f}% | Длит.: {dur_hours:.1f}ч")

    # 6. Генерация HTML-отчета
    report_html = RESULTS_DIR.parent / "RalphTradeBot backtest.html"
    generate_html_report(all_signals_data, report_html, days_span=30.0)

    # 7. Сохранение JSON данных
    summary_json = RESULTS_DIR / "backtest_summary.json"
    clean_items = []
    for it in all_signals_data:
        c_it = {}
        for k, v in it.items():
            if isinstance(v, (np.integer,)):
                c_it[k] = int(v)
            elif isinstance(v, (np.floating,)):
                c_it[k] = float(v)
            else:
                c_it[k] = v
        clean_items.append(c_it)
    summary_json.write_text(json.dumps(clean_items, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"\n🎉 Бэктест завершен! Всего сигналов: {len(all_signals_data)}")
    logger.info(f"📄 Отчет доступен: {report_html}")
    return all_signals_data


def main():
    parser = argparse.ArgumentParser(description="RalphTradeBot Multi-Asset Backtester")
    parser.add_argument("--preset", type=str, default=None, help="Список пресетов через запятую")
    parser.add_argument("--top15", action="store_true", help="Запустить по всем 15 монетам")
    parser.add_argument("--dry-run", action="store_true", help="Быстрый прогон без Vision AI")
    parser.add_argument("--send-tg", action="store_true", help="Отправлять подтвержденные сигналы (Score >= 85) в Telegram")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить количество сигналов на пресет")

    args = parser.parse_args()

    if args.top15:
        presets = DEFAULT_PRESETS
    elif args.preset:
        p_names = [p.strip() for p in args.preset.split(",") if p.strip()]
        presets = {p: DEFAULT_PRESETS[p] for p in p_names if p in DEFAULT_PRESETS}
    else:
        # По умолчанию: топ-4 пресета
        sample_keys = ["BTCUSDT_5m", "ETHUSDT_5m", "BTCUSDT_15m", "ETHUSDT_15m"]
        presets = {k: DEFAULT_PRESETS[k] for k in sample_keys if k in DEFAULT_PRESETS}

    run_backtest(presets, send_telegram=args.send_tg, dry_run=args.dry_run, limit_signals_per_preset=args.limit)


if __name__ == "__main__":
    main()
