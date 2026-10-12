"""
run_optimization.py — Скрипт запуска мульти-эпохального исследования и оптимизации выходов для Стратегии 5.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from strategies_lab.strategies.strategy_5_sweep_engulf.optimization import (
    HISTORICAL_REGIMES,
    TOP_SYMBOLS,
    collect_filled_trades_across_regimes,
    run_full_optimization_matrix,
    analyze_mfe_distribution,
)
from strategies_lab.strategies.strategy_5_sweep_engulf.config import RESULTS_DIR

logger = logging.getLogger("RunOptimization")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    logger.info("🚀 ЗАПУСК МУЛЬТИ-ЭПОХАЛЬНОЙ ОПТИМИЗАЦИИ ВЫХОДОВ...")
    logger.info(f"Эпохи для стресс-теста: {[r[0] for r in HISTORICAL_REGIMES]}")
    logger.info(f"Инструменты: {TOP_SYMBOLS}")

    # 1. Сбор всех исполненных сделок
    filled_trades = collect_filled_trades_across_regimes(
        regimes=HISTORICAL_REGIMES,
        symbols=TOP_SYMBOLS,
        interval="15m",
        fill_window_bars=2,
        future_window_bars=120,
    )

    if not filled_trades:
        logger.error("❌ Не удалось собрать исполненные сделки.")
        return

    logger.info(f"\n✅ Успешно собрано {len(filled_trades)} сделок по всем 5 эпохам!")

    # 2. Анализ распределения MFE
    logger.info("\n📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ MFE (Maximum Favorable Excursion)...")
    mfe_stats = analyze_mfe_distribution(filled_trades)
    reach = mfe_stats["reach_stats"]
    logger.info(f"  Медианный MFE: {mfe_stats['median_mfe_r']} R | Средний MFE: {mfe_stats['mean_mfe_r']} R")
    for k, v in reach.items():
        logger.info(f"  Достигли {k}: {v['count']} сделок ({v['pct']}%)")

    # 3. Прогон матрицы оптимизации
    logger.info("\n🧪 СИМУЛЯЦИЯ МАТРИЦЫ АРХИТЕКТУР ВЫХОДОВ...")
    opt_res = run_full_optimization_matrix(filled_trades)
    model_results = opt_res["model_results"]
    detailed_rows = opt_res["detailed_rows"]

    # 4. Сохранение детальных сделок в CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(detailed_rows)
    csv_file = RESULTS_DIR / "optimization_trades.csv"
    trades_df.to_csv(csv_file, index=False)
    logger.info(f"💾 Детальные сделки сохранены: {csv_file}")

    # 5. Генерация красивого отчёта
    report_file = RESULTS_DIR / "optimization_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# 🔬 Исследование и Оптимизация Выходов: Стратегия 5 (Sweep & Engulf + 50% Limit)\n\n")
        f.write(f"### 📅 Выборка: 5 исторических эпох (2020–2026), 13 топ-монет, таймфрейм 15m\n")
        f.write(f"### 🎯 Всего исполненных входов (50% лимит, окно 1-2 свечи, SL за тенью): **{len(filled_trades)} сделок**\n\n")

        f.write("## 1. 📈 Анализ потенциала движения: MFE Distribution (Maximum Favorable Excursion)\n")
        f.write("Показывает, какую максимальную прибыль в R успевает дать сделка до выбивания начального стоп-лосса:\n\n")
        f.write("| Порог прибыли | Количество сделок | % от всех входов |\n")
        f.write("| :---: | :---: | :---: |\n")
        for k, v in reach.items():
            f.write(f"| **{k}** | {v['count']} | **{v['pct']}%** |\n")
        f.write(f"\n*Медианный MFE:* **{mfe_stats['median_mfe_r']} R** | *Средний MFE:* **{mfe_stats['mean_mfe_r']} R** | *75-й перцентиль:* **{mfe_stats['p75_mfe_r']} R**\n\n")

        f.write("## 2. 🏆 Сравнительная таблица моделей выхода (A/B Тест на идентичном пуле сделок)\n\n")
        f.write("| Модель выхода | Сделок | Win Rate % | Profit Factor | Net PnL ($) | Net EV (R) | Max DD % | Комиссии ($) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for m_name, res in model_results.items():
            pf_str = f"**🔥 {res['profit_factor']}**" if res['profit_factor'] >= 2.0 else f"{res['profit_factor']}"
            pnl_str = f"**${res['net_pnl_usd']:+,.2f}**"
            f.write(
                f"| {m_name} | {res['total_trades']} | {res['win_rate']}% | {pf_str} | "
                f"{pnl_str} | **{res['net_ev_r']:+0.2f} R** | {res['max_dd_pct']}% | ${res['total_fees_usd']:,.2f} |\n"
            )

        f.write("\n## 3. 🔍 Разбивка по историческим эпохам (Stress-Test)\n\n")
        # Анализ устойчивости лучшей модели по эпохам
        f.write("| Историческая эпоха | Тип рынка | Сделок | Win Rate % (1:3 TP) | Net PnL ($) (1:3 TP) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        reg_groups = trades_df[trades_df['model'] == "Fixed_3.0R (Базовый)"].groupby("regime")
        for reg_id, grp in reg_groups:
            reg_name = dict(HISTORICAL_REGIMES).get(reg_id, reg_id)
            n_t = len(grp)
            w_t = len(grp[grp['net_pnl_usd'] > 0])
            wr = (w_t / n_t * 100.0) if n_t > 0 else 0.0
            pnl = grp['net_pnl_usd'].sum()
            f.write(f"| {reg_id} | {reg_name} | {n_t} | {wr:.1f}% | **${pnl:+,.2f}** |\n")

    logger.info(f"📄 Итоговый отчет сохранен в: {report_file}")


if __name__ == "__main__":
    main()
