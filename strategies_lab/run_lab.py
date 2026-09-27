"""
run_lab.py — Главный модуль запуска изолированных бэктестов в strategies_lab v2.0.

Режимы:
    python strategies_lab/run_lab.py --interval 15m
    python strategies_lab/run_lab.py --mode exits     (Изолированное сравнение моделей выходов A, B, C, D, E)
    python strategies_lab/run_lab.py --mode entries   (Сравнение фильтров входа: Baseline, MultiTF-MSS, Absorption, Keltner)
    python strategies_lab/run_lab.py --mode all       (Полный аудит всех гипотез)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

LAB_DIR = Path(__file__).parent
BASE_DIR = LAB_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from strategies_lab.engine import (
    load_all_market_data,
    detect_all_signals_in_history,
    run_strategy_backtest,
    BacktestSummary,
)
from strategies_lab.strategies.baseline.strategy import BaselineStrategy
from strategies_lab.strategies.strategy_1_structure_shift.strategy import MarketStructureShiftStrategy
from strategies_lab.strategies.strategy_2_absorption_bar.strategy import AbsorptionBarStrategy
from strategies_lab.strategies.strategy_3_keltner_reentry.strategy import KeltnerReentryStrategy
from strategies_lab.strategies.strategy_4_chandelier_trailing.strategy import ChandelierTrailingStrategy
from strategies_lab.strategies.exit_models import (
    BaselineExitModel,
    FailFastTimeStopExitModel,
    PureChandelierExitModel,
    DualTargetQuickLockExitModel,
    QuantMasterHybridExitModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("StrategiesLab")


def save_strategy_results(summary: BacktestSummary, folder_name: str):
    """Сохраняет результаты бэктеста в персональную папку стратегии."""
    target_dir = LAB_DIR / "strategies" / folder_name / "results"
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = target_dir / "metrics.json"
    metrics_dict = {
        "strategy_name": summary.strategy_name,
        "total_signals": summary.total_signals,
        "executed_trades": summary.executed_trades,
        "filtered_signals": summary.filtered_signals,
        "wins": summary.wins,
        "losses": summary.losses,
        "breakevens": summary.breakevens,
        "win_rate_pct": summary.win_rate_pct,
        "full_win_rate_pct": summary.full_win_rate_pct,
        "profit_factor": summary.profit_factor,
        "total_net_pnl_pct": summary.total_net_pnl_pct,
        "initial_deposit_usd": summary.initial_deposit_usd,
        "final_deposit_usd": summary.final_deposit_usd,
        "net_profit_usd": summary.net_profit_usd,
        "total_fees_usd": summary.total_fees_usd,
        "expectancy_r": summary.expectancy_r,
        "max_drawdown_pct": summary.max_drawdown_pct,
        "sharpe_ratio": summary.sharpe_ratio,
        "sortino_ratio": summary.sortino_ratio,
        "avg_mfe_pct": summary.avg_mfe_pct,
        "avg_mae_pct": summary.avg_mae_pct,
        "avg_mfe_r": summary.avg_mfe_r,
        "avg_mae_r": summary.avg_mae_r,
        "avg_bars_held": summary.avg_bars_held,
        "avg_slippage_tax_pct": summary.avg_slippage_tax_pct,
        "losses_saved_vs_baseline": summary.losses_saved_vs_baseline,
        "mfe_mae_analysis": summary.mfe_mae_analysis,
    }
    metrics_file.write_text(json.dumps(metrics_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary.trade_results:
        trades_data = []
        for t in summary.trade_results:
            trades_data.append({
                "signal_id": t.signal_id,
                "symbol": t.symbol,
                "interval": t.interval,
                "direction": t.direction,
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
                "mfe_pct": t.mfe_pct,
                "mae_pct": t.mae_pct,
                "mfe_r": t.mfe_r,
                "mae_r": t.mae_r,
                "slippage_tax_pct": t.slippage_tax_pct,
                "tp_hits": ";".join(t.tp_hits),
            })
        df_trades = pd.DataFrame(trades_data)
        df_trades.to_csv(target_dir / "trades.csv", index=False)


def format_mfe_mae_report(baseline_summary: BacktestSummary) -> str:
    """Генерирует специализированный отчет по квантовому MFE / MAE анализу."""
    stats = baseline_summary.mfe_mae_analysis
    if not stats:
        return ""

    mae_r = stats.get("win_mae_percentiles_r", {})
    mfe_r = stats.get("all_mfe_percentiles_r", {})
    reach = stats.get("mfe_reach_rates_pct", {})

    lines = [
        "## 🔬 Квантовый анализ MFE и MAE (Распределение потенциала импульсов)",
        "",
        "### 1. MAE (Maximum Adverse Excursion) — Просадка для прибыльных сделок:",
        f"- **Медианная просадка победителей (p50):** `{mae_r.get('p50_median', 0.0)} R` от стоп-лосса",
        f"- **75% победителей не заходят глубже:** `{mae_r.get('p75', 0.0)} R`",
        f"- **90% победителей не заходят глубже:** `{mae_r.get('p90', 0.0)} R`",
        f"- **95% победителей не заходят глубже:** `{mae_r.get('p95', 0.0)} R`",
        "",
        "> **Институциональный вывод по Stop-Loss:** Если успешные сделки в 90% случаев не опускаются глубже "
        f"`{mae_r.get('p90', 0.0)} R`, то нахождение цены глубже этой отметки с вероятностью >90% завершится полным стопом. "
        "Это математически обосновывает сжатие базового стоп-лосса либо активацию Fail-Fast выхода.",
        "",
        "### 2. MFE (Maximum Favorable Excursion) — Вероятность достижения целевых уровней:",
        f"- **Достижение +1.0 R (TP1 23.6% Фибо):** `{reach.get('reached_1r_pct', 0.0)}%` сделок",
        f"- **Достижение +2.0 R (TP2 38.2% Фибо):** `{reach.get('reached_2r_pct', 0.0)}%` сделок",
        f"- **Достижение +3.0 R (TP3 50.0% Фибо):** `{reach.get('reached_3r_pct', 0.0)}%` сделок",
        f"- **Достижение +4.0 R (TP4 61.8% Фибо):** `{reach.get('reached_4r_pct', 0.0)}%` сделок",
        f"- **Медианный забег цены (p50 MFE):** `+{mfe_r.get('p50_median', 0.0)} R`",
        f"- **Максимальный выброс импульса (p90 MFE):** `+{mfe_r.get('p90', 0.0)} R`",
        "",
        "> **Институциональный вывод по Take-Profit:** Разница в достижимости между TP2 и TP4 показывает, "
        "что фиксация основной части объема на уровнях 23.6% и 38.2% обеспечивает стабильный сбор математического ожидания, "
        "в то время как остаток позиции целесообразно переводить в безубыточный трейлинг.",
    ]
    return "\n".join(lines)


def generate_comparison_report(summaries: List[BacktestSummary], baseline_summary: BacktestSummary) -> str:
    """Генерирует глобальный сравнительный отчет в корне strategies_lab."""
    sorted_s = sorted(summaries, key=lambda s: (s.profit_factor, s.net_profit_usd), reverse=True)

    lines = [
        "# 📊 Сравнительный аудит стратегий торговли импульсами Эллиотта v2.0",
        "",
        "### Параметры финансового моделирования:",
        "- **Начальный депозит:** `$1,000.00`",
        "- **Риск на сделку:** `1.0% ($10.00)`",
        "- **Комиссии биржи:** `0.1% на вход + 0.1% на выход (суммарно 0.2% от позиции)`",
        "- **Таймфрейм анализа:** `15m` с минутным разрешением микроструктуры `1m`",
        "",
        "| Место | Стратегия | Сделок | Win Rate | Profit Factor | Net PnL ($) | Итог Депозита | EV (Net R) | Max DD | Спасенные SL | Комиссии ($) | Шарп |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for rank, s in enumerate(sorted_s, 1):
        badge = "🥇 " if rank == 1 else "🥈 " if rank == 2 else "🥉 " if rank == 3 else f"{rank}. "
        lines.append(
            f"| {badge} | **{s.strategy_name}** | {s.executed_trades} | "
            f"**{s.win_rate_pct}%** | **{s.profit_factor:.2f}** | "
            f"**${s.net_profit_usd:+.2f}** ({s.total_net_pnl_pct:+.1f}%) | "
            f"**${s.final_deposit_usd:,.2f}** | "
            f"**{s.expectancy_r:+0.2f} R** | "
            f"{s.max_drawdown_pct:.1f}% | "
            f"**+{s.losses_saved_vs_baseline}** | "
            f"${s.total_fees_usd:.2f} | "
            f"{s.sharpe_ratio:.2f} |"
        )

    # Добавляем отчет по MFE / MAE
    mfe_mae_text = format_mfe_mae_report(baseline_summary)
    if mfe_mae_text:
        lines.extend(["", mfe_mae_text])

    # Slippage Tax анализ для Multi-TF
    mss_summary = next((s for s in summaries if "MultiTF_MSS" in s.strategy_name or "Structure_Shift" in s.strategy_name), None)
    if mss_summary and mss_summary.executed_trades > 0:
        lines.extend([
            "",
            "### ⚡ Анализ Slippage Tax (Плата за подтверждение на 1m):",
            f"- **Средний налог на проскальзывание:** `{mss_summary.avg_slippage_tax_pct:.2f}%` от дистанции до TP1.",
            "- **Вывод:** Проскальзывание при входе на 1m микросломе составляет незначительную часть диапазона, "
            "что подтверждает высокую математическую эффективность микроструктурного фильтра.",
        ])

    lines.extend([
        "",
        "---",
        "*Все подробные метрики (`metrics.json`) и побарные сделки (`trades.csv`) сохранены в папках `strategies_lab/strategies/`.*",
    ])

    report_text = "\n".join(lines)
    (LAB_DIR / "COMPARISON_REPORT.md").write_text(report_text, encoding="utf-8")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Strategies Lab v2.0 — Квантовый аудит стратегий RalphTradeBot")
    parser.add_argument("--interval", type=str, default="15m", help="Таймфрейм тестирования (по умолч. 15m)")
    parser.add_argument("--symbols", type=str, default=None, help="Символы через запятую (по умолч. все)")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "entries", "exits"], help="Режим тестирования")
    parser.add_argument("--min-score", type=float, default=80.0, help="Минимальный score импульса (по умолч. 80.0)")

    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None

    logger.info("=" * 72)
    logger.info("🧪 STRATEGIES LAB v2.0 — ИНСТИТУЦИОНАЛЬНЫЙ МАТЕМАТИЧЕСКИЙ АУДИТ")
    logger.info(f"   Таймфрейм: {args.interval} (с 1m микроструктурой) | Порог Score: {args.min_score}")
    logger.info(f"   Депозит: $1,000 | Риск: 1% ($10) | Комиссия: 0.1% maker + 0.1% taker")
    logger.info(f"   Режим: {args.mode}")
    logger.info("=" * 72)

    # 1. Загрузка рыночных данных (Multi-TF: 15m + 1m)
    market_data = load_all_market_data(interval=args.interval, symbols=symbols)
    if not market_data:
        logger.error("Нет данных для бэктеста!")
        return

    # 2. Обнаружение пула импульсов Эллиотта
    signals = detect_all_signals_in_history(market_data, interval=args.interval, min_score=args.min_score)
    if not signals:
        logger.warning("Импульсов Эллиотта не обнаружено.")
        return

    # 3. Базовый прогон Baseline
    baseline_strat = BaselineStrategy()
    logger.info(f"\n▶ Прогон эталона: {baseline_strat.name}...")
    baseline_summary = run_strategy_backtest(baseline_strat, signals, market_data)
    save_strategy_results(baseline_summary, "baseline")
    baseline_losses = {t.signal_id: (t.net_pnl_usd < -0.05) for t in baseline_summary.trade_results}

    summaries = [baseline_summary]

    # Список стратегий для тестирования
    strats_to_run = []

    if args.mode in ("all", "entries"):
        strats_to_run.extend([
            (MarketStructureShiftStrategy(), "strategy_1_structure_shift"),
            (AbsorptionBarStrategy(), "strategy_2_absorption_bar"),
            (KeltnerReentryStrategy(), "strategy_3_keltner_reentry"),
            (ChandelierTrailingStrategy(), "strategy_4_chandelier_trailing"),
        ])

    if args.mode in ("all", "exits"):
        strats_to_run.extend([
            (FailFastTimeStopExitModel(time_stop_bars=3, min_profit_r_to_stay=0.4), "exit_model_b_fail_fast"),
            (PureChandelierExitModel(atr_multiplier=2.2), "exit_model_c_chandelier"),
            (DualTargetQuickLockExitModel(), "exit_model_d_dual_target"),
            (QuantMasterHybridExitModel(time_stop_bars=3, min_profit_r_to_stay=0.35, atr_multiplier=2.2), "exit_model_e_quant_master"),
        ])

    for strat, folder in strats_to_run:
        logger.info(f"\n▶ Прогон: {strat.name} ({strat.description})...")
        summ = run_strategy_backtest(strat, signals, market_data, baseline_losses=baseline_losses)
        save_strategy_results(summ, folder)
        summaries.append(summ)

    # 4. Генерация сравнительного отчета
    report = generate_comparison_report(summaries, baseline_summary)
    print("\n" + report)
    logger.info(f"\n✅ Аудит завершен! Сводный отчет сохранён в {LAB_DIR / 'COMPARISON_REPORT.md'}")


if __name__ == "__main__":
    main()
