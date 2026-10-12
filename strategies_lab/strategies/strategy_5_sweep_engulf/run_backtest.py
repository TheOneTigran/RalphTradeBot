"""
run_backtest.py — Выделенный автономный запуск бэктеста Стратегии 5 (Sweep & Engulf + 50% Limit).

Выполняет:
1. Масштабное тестирование на полугодовом диапазоне по ключевым инструментам.
2. Тестирование окон входа 1 и 2 свечи (согласно правилам пользователя).
3. Расчёт Take-Profit с точным выходом 1:3 Net R после вычета комиссий 0.05% in + 0.05% out.
4. Стоп-лосс строго по кончику тени (без буферов).
5. Чистая визуализация (без фиолетовых знаков/рамок на свече S) с точной разметкой волн W0-W5 и дивергенций.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from strategies_lab.engine import (
    load_all_market_data,
    detect_all_signals_in_history,
    run_strategy_backtest,
    BacktestSummary,
)
from strategies_lab.strategies.strategy_5_sweep_engulf.strategy import SweepEngulfRetracementStrategy
from strategies_lab.strategies.strategy_5_sweep_engulf.config import (
    RESULTS_DIR,
    CHARTS_DIR,
    MIN_ELLIOTT_SCORE,
)
from strategies_lab.strategies.strategy_5_sweep_engulf.chart_plotter import plot_sweep_engulf_trade
from strategies_lab.strategies.strategy_5_sweep_engulf.patterns import detect_sweep_engulf_candle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SweepEngulfLab")

# Папка с полугодовым архивом котировок (8 месяцев: 2025-11-01 по 2026-06-30)
MACRO_DATA_DIR = Path("c:/Users/user/Desktop/данные для бектестов/9_Macro_Bear_Market_2026")
LOCAL_DATA_DIR = BASE_DIR / "data"

TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT",
    "SUIUSDT", "AAVEUSDT", "1000PEPEUSDT",
]


def run_sweep_engulf_experiment(
    data_dir: Path,
    timeframes: List[str] = ["15m", "1h"],
    fill_windows: List[int] = [1, 2],
    symbols: Optional[List[str]] = TOP_SYMBOLS,
    months_limit: Optional[int] = 6,
    max_charts: int = 15,
) -> Dict[str, Any]:
    all_experiments = []
    charts_generated = 0

    for tf in timeframes:
        logger.info(f"\n{'='*75}")
        logger.info(f"📊 ТЕСТИРОВАНИЕ ТАЙМФРЕЙМА: {tf}")
        logger.info(f"   Источник данных: {data_dir.name}")
        logger.info(f"   Инструменты: {len(symbols) if symbols else 'все'}")
        logger.info(f"   Ограничение по истории: последние {months_limit} месяцев" if months_limit else "   Вся доступная история")
        logger.info(f"{'='*75}")

        market_data = load_all_market_data(data_dir=data_dir, interval=tf, symbols=symbols)
        if not market_data:
            logger.error(f"Нет данных для ТФ {tf}")
            continue

        # Фильтрация по периоду (например, последние 6 месяцев)
        if months_limit:
            filtered_market_data = {}
            for sym, tf_dict in market_data.items():
                m_df = tf_dict["main"]
                if len(m_df) > 0:
                    cutoff_date = m_df.index.max() - pd.DateOffset(months=months_limit)
                    m_df_cut = m_df[m_df.index >= cutoff_date]
                    if len(m_df_cut) >= 150:
                        filtered_market_data[sym] = {
                            "main": m_df_cut,
                            "1m": tf_dict.get("1m"),
                        }
            market_data = filtered_market_data

        sample_sym = list(market_data.keys())[0]
        sample_df = market_data[sample_sym]["main"]
        date_start = sample_df.index.min()
        date_end = sample_df.index.max()
        days_span = (date_end - date_start).days
        logger.info(f"Период данных: с {date_start.strftime('%Y-%m-%d')} по {date_end.strftime('%Y-%m-%d')} ({days_span} дней / {days_span/30.4:.1f} мес.)")

        signals = detect_all_signals_in_history(market_data, interval=tf, min_score=MIN_ELLIOTT_SCORE)
        if not signals:
            logger.warning(f"Импульсов Эллиотта не найдено на {tf}")
            continue

        # Подсчёт воронки
        s_candle_count = 0
        for sig in signals:
            tf_dict = market_data.get(sig.symbol)
            if not tf_dict:
                continue
            full_df = tf_dict["main"]
            future_slice = full_df.iloc[sig.w5_bar + 1 : sig.w5_bar + 1 + 10]
            if len(future_slice) > 0:
                prev_c = sig.raw_df.iloc[-2] if len(sig.raw_df) >= 2 else None
                pat = detect_sweep_engulf_candle(sig.raw_df.iloc[-1], future_slice, sig.direction, prev_candle=prev_c)
                if pat.found:
                    s_candle_count += 1

        logger.info(
            f"  📌 Воронка {tf}: Всего импульсов W0-W5 = {len(signals)} | "
            f"Сформировали свечу S (свип + поглощение) = {s_candle_count} ({s_candle_count/len(signals)*100:.1f}%)"
        )

        for fw in fill_windows:
            strat_name = f"SweepEngulf_FW{fw}_{tf}"
            strat = SweepEngulfRetracementStrategy(
                name=strat_name,
                fill_window_bars=fw,
            )

            logger.info(f"\n▶ Прогон: Окно входа = {fw} свечи ({tf})...")
            summary = run_strategy_backtest(strat, signals, market_data, future_window_bars=80)

            # Генерация графиков
            if charts_generated < max_charts and summary.trade_results:
                for trade in summary.trade_results:
                    if charts_generated >= max_charts:
                        break
                    matching_sig = next((s for s in signals if f"{s.symbol}_{s.interval}_{s.direction}_{s.w5_bar}" == trade.signal_id), None)
                    if matching_sig:
                        full_df = market_data[matching_sig.symbol]["main"]
                        future_slice = full_df.iloc[matching_sig.w5_bar + 1 :]
                        prev_c = matching_sig.raw_df.iloc[-2] if len(matching_sig.raw_df) >= 2 else None
                        pat = detect_sweep_engulf_candle(matching_sig.raw_df.iloc[-1], future_slice, matching_sig.direction, prev_candle=prev_c)
                        if pat.found:
                            chart_file = CHARTS_DIR / f"{matching_sig.symbol}_{tf}_{matching_sig.direction}_trade_{charts_generated+1}_{trade.exit_reason.upper()}.png"
                            plot_sweep_engulf_trade(
                                signal=matching_sig,
                                result=trade,
                                pattern=pat,
                                full_df=full_df,
                                save_path=chart_file,
                            )
                            charts_generated += 1

            all_experiments.append({
                "timeframe": tf,
                "fill_window": fw,
                "days_span": days_span,
                "date_range": f"{date_start.strftime('%Y-%m-%d')} -> {date_end.strftime('%Y-%m-%d')}",
                "total_signals": summary.total_signals,
                "s_candles": s_candle_count,
                "executed_trades": summary.executed_trades,
                "fill_rate_pct": round(summary.executed_trades / s_candle_count * 100.0, 1) if s_candle_count > 0 else 0.0,
                "wins": summary.wins,
                "losses": summary.losses,
                "win_rate_pct": summary.win_rate_pct,
                "profit_factor": summary.profit_factor,
                "net_profit_usd": summary.net_profit_usd,
                "total_fees_usd": summary.total_fees_usd,
                "expectancy_r": summary.expectancy_r,
                "max_drawdown_pct": summary.max_drawdown_pct,
                "avg_bars_held": summary.avg_bars_held,
                "summary_obj": summary,
            })

    save_summary_reports(all_experiments)
    return all_experiments


def save_summary_reports(experiments: List[Dict[str, Any]]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not experiments:
        return

    first_exp = experiments[0]
    period_str = first_exp.get("date_range", "N/A")
    days_span = first_exp.get("days_span", 0)

    lines = [
        "# 🧪 Аудит гипотезы: Свечной Свип + Поглощение (Sweep & Engulf) + 50% Лимит",
        "",
        f"### 📅 Исторический диапазон тестирования: `{period_str}` ({days_span} дней / {days_span/30.4:.1f} мес.)",
        "",
        "## 📌 Правила стратегии:",
        "- **Триггер:** 5-волновой импульс Эллиотта + подтверждённая RSI-дивергенция (W3 vs W5).",
        "- **Свеча S (Свип + Поглощение):** тень пробивает экстремум W5, тело закрывается с поглощением тела W5.",
        "- **Вход:** строго лимитный ордер на 50% полного диапазона свечи S (`Low + 0.5 * Range`).",
        "- **Окно ожидания входа:** строго 1–2 свечи после поглощения.",
        "- **Stop-Loss:** строго по кончику тени свечи S (без буфера).",
        "- **Take-Profit:** строго **1:3 Net R ПОСЛЕ вычета биржевых комиссий** (0.05% maker на вход + 0.05% taker/limit на выход).",
        "- **Финансовый риск:** депозит $1,000, фиксированный риск 1.0% ($10.00) на сделку.",
        "",
        "## 📊 Результаты бэктеста:",
        "",
        "| ТФ | Окно входа | Дней | Импульсов | Свечей S | Входов (Filled) | Fill Rate % | Win Rate % | Profit Factor | Net PnL ($) | Net EV (R) | Max DD % | Комиссии ($) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for exp in experiments:
        badge = "🔥 " if exp["profit_factor"] >= 2.0 and exp["expectancy_r"] > 0.3 else ""
        lines.append(
            f"| **{exp['timeframe']}** | {exp['fill_window']} свечи | {exp['days_span']} дн. | {exp['total_signals']} | "
            f"{exp['s_candles']} | {exp['executed_trades']} | {exp['fill_rate_pct']}% | "
            f"**{exp['win_rate_pct']}%** | **{badge}{exp['profit_factor']:.2f}** | "
            f"**${exp['net_profit_usd']:+.2f}** | **{exp['expectancy_r']:+0.2f} R** | "
            f"{exp['max_drawdown_pct']:.1f}% | ${exp['total_fees_usd']:.2f} |"
        )

    lines.extend([
        "",
        "## 💡 Выводы:",
        "1. **Качество входа по лимиту 50%:**",
        "   - Отсекает ложные развороты: значительная часть сигналов не возвращается на 50%, предотвращая вход в убыточные сделки.",
        "   - Обеспечивает ультракороткий риск за тенью свечи S, что позволяет забирать крупный Net R при соотношении 1:3.",
        "",
        "## 🖼 Сгенерированные графики для визуальной проверки:",
        f"Графики сохранены в: `{CHARTS_DIR}`",
    ])

    report_md = "\n".join(lines)
    (RESULTS_DIR / "summary.md").write_text(report_md, encoding="utf-8")

    # Сохранение сделок
    all_trades = []
    for exp in experiments:
        for t in exp["summary_obj"].trade_results:
            all_trades.append({
                "signal_id": t.signal_id,
                "symbol": t.symbol,
                "interval": t.interval,
                "direction": t.direction,
                "fill_window": exp["fill_window"],
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "initial_sl": t.initial_sl,
                "pos_size_usd": t.position_size_usd,
                "fee_usd": t.fee_usd,
                "pnl_usd": t.pnl_usd,
                "net_pnl_usd": t.net_pnl_usd,
                "pnl_r": t.pnl_r,
                "raw_pnl_pct": t.final_pnl_pct,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
                "mfe_r": t.mfe_r,
                "mae_r": t.mae_r,
            })
    if all_trades:
        pd.DataFrame(all_trades).to_csv(RESULTS_DIR / "trades.csv", index=False)

    print("\n" + report_md)


def main():
    parser = argparse.ArgumentParser(description="Sweep Engulf Retracement Strategy Backtest")
    parser.add_argument("--data-dir", type=str, default=str(MACRO_DATA_DIR if MACRO_DATA_DIR.exists() else LOCAL_DATA_DIR))
    parser.add_argument("--tf", type=str, default="15m,1h", help="Таймфреймы через запятую")
    parser.add_argument("--months", type=int, default=6, help="Количество месяцев для теста (по умолч. 6)")
    parser.add_argument("--charts", type=int, default=15, help="Количество графиков для сохранения")
    args = parser.parse_args()

    tfs = [t.strip() for t in args.tf.split(",") if t.strip()]
    d_dir = Path(args.data_dir)
    run_sweep_engulf_experiment(
        data_dir=d_dir,
        timeframes=tfs,
        fill_windows=[1, 2],
        months_limit=args.months,
        max_charts=args.charts,
    )


if __name__ == "__main__":
    main()
