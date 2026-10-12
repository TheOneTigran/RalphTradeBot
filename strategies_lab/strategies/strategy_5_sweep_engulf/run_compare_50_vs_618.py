"""
run_compare_50_vs_618.py — Исполняемый скрипт для детального сравнительного аудита:
Вход 50% vs 61.8%, стоп в БУ на 2R, спасенные стопы, статистика по 13 монетам, помесячная динамика и сложный процент.
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

from strategies_lab.strategies.strategy_5_sweep_engulf.advanced_research import collect_raw_patterns
from strategies_lab.strategies.strategy_5_sweep_engulf.compare_50_vs_618 import (
    simulate_strategy_setup,
    analyze_trades_dataset,
    compute_instrument_breakdown,
    compute_monthly_compound,
)
from strategies_lab.strategies.strategy_5_sweep_engulf.config import RESULTS_DIR

logger = logging.getLogger("RunCompare")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    logger.info("🚀 ЗАПУСК СРАВНИТЕЛЬНОГО АУДИТА 50% VS 61.8%...")

    patterns = collect_raw_patterns()
    if not patterns:
        logger.error("❌ Паттерны не найдены.")
        return

    logger.info(f"✅ Найдено свечных паттернов S: {len(patterns)}")

    # ═════════════════════════════════════════════════════════════════════════
    # 1. СИМУЛЯЦИЯ 4 КОНФИГУРАЦИЙ
    # ═════════════════════════════════════════════════════════════════════════
    logger.info("▶ 1. Симуляция: Вход 50% | Стоп в БУ на 2R | Тейк строго 3R...")
    trades_50_tp = simulate_strategy_setup(patterns, retrace_ratio=0.50, target_r=3.0, use_reverse_signal_exit=False)

    logger.info("▶ 2. Симуляция: Вход 50% | Стоп в БУ на 2R | Тейк 3R + Выход по обратному сигналу...")
    trades_50_rev = simulate_strategy_setup(patterns, retrace_ratio=0.50, target_r=3.0, use_reverse_signal_exit=True)

    logger.info("▶ 3. Симуляция: Вход 61.8% | Стоп в БУ на 2R | Тейк строго 3R...")
    trades_618_tp = simulate_strategy_setup(patterns, retrace_ratio=0.618, target_r=3.0, use_reverse_signal_exit=False)

    logger.info("▶ 4. Симуляция: Вход 61.8% | Стоп в БУ на 2R | Тейк 3R + Выход по обратному сигналу...")
    trades_618_rev = simulate_strategy_setup(patterns, retrace_ratio=0.618, target_r=3.0, use_reverse_signal_exit=True)

    # ═════════════════════════════════════════════════════════════════════════
    # 2. РАСЧЁТ СВОДНЫХ МЕТРИК
    # ═════════════════════════════════════════════════════════════════════════
    metrics_50_tp = analyze_trades_dataset(trades_50_tp)
    metrics_50_rev = analyze_trades_dataset(trades_50_rev)
    metrics_618_tp = analyze_trades_dataset(trades_618_tp)
    metrics_618_rev = analyze_trades_dataset(trades_618_rev)

    # Инструменты
    coins_50_df = compute_instrument_breakdown(trades_50_tp)
    coins_618_df = compute_instrument_breakdown(trades_618_tp)

    # Помесячная динамика и сложный процент
    monthly_50_df = compute_monthly_compound(trades_50_tp, initial_deposit=1000.0, risk_rate=0.01)
    monthly_618_df = compute_monthly_compound(trades_618_tp, initial_deposit=1000.0, risk_rate=0.01)

    # Сохраняем в CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([t.__dict__ for t in trades_50_tp]).to_csv(RESULTS_DIR / "trades_50_tp.csv", index=False)
    pd.DataFrame([t.__dict__ for t in trades_618_tp]).to_csv(RESULTS_DIR / "trades_618_tp.csv", index=False)
    monthly_50_df.to_csv(RESULTS_DIR / "monthly_compound_50.csv", index=False)
    monthly_618_df.to_csv(RESULTS_DIR / "monthly_compound_618.csv", index=False)

    # ═════════════════════════════════════════════════════════════════════════
    # 3. ФОРМИРОВАНИЕ ПОДРОБНОГО ОТЧЁТА
    # ═════════════════════════════════════════════════════════════════════════
    report_file = RESULTS_DIR / "comparative_audit_50_vs_618.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# ⚖️ Сравнительный Аудит: Вход 50% vs Вход 61.8% (Стоп в БУ на 2R, 1:3 Тейк и Развороты)\n\n")
        f.write(f"### 📅 Историческая база: 5 эпох (2020–2026 гг.), 13 ключевых криптовалют, таймфрейм 15m\n")
        f.write(f"### 🎯 Обнаружено чистых свечных паттернов S: **{len(patterns)}**\n\n")

        # ── Сводная таблица 4 конфигураций ──
        f.write("## 1. 🏆 Сводное сравнение ключевых вариантов\n\n")
        f.write("| Параметр стратегии | Вход 50% (Тейк 3R + БУ@2R) | Вход 50% (3R / Разворот + БУ@2R) | Вход 61.8% (Тейк 3R + БУ@2R) | Вход 61.8% (3R / Разворот + БУ@2R) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Исполнено сделок** | **{metrics_50_tp['total_trades']}** | **{metrics_50_rev['total_trades']}** | **{metrics_618_tp['total_trades']}** | **{metrics_618_rev['total_trades']}** |\n")
        f.write(f"| **Fill Rate %** | {metrics_50_tp['total_trades']/len(patterns)*100:.1f}% | {metrics_50_rev['total_trades']/len(patterns)*100:.1f}% | {metrics_618_tp['total_trades']/len(patterns)*100:.1f}% | {metrics_618_rev['total_trades']/len(patterns)*100:.1f}% |\n")
        f.write(f"| **Прибыльные (Wins)** | {metrics_50_tp['wins']} ({metrics_50_tp['win_rate_pct']}%) | {metrics_50_rev['wins']} ({metrics_50_rev['win_rate_pct']}%) | {metrics_618_tp['wins']} ({metrics_618_tp['win_rate_pct']}%) | {metrics_618_rev['wins']} ({metrics_618_rev['win_rate_pct']}%) |\n")
        f.write(f"| **Безубытки (БУ)** | {metrics_50_tp['bes']} ({metrics_50_tp['be_rate_pct']}%) | {metrics_50_rev['bes']} ({metrics_50_rev['be_rate_pct']}%) | {metrics_618_tp['bes']} ({metrics_618_tp['be_rate_pct']}%) | {metrics_618_rev['bes']} ({metrics_618_rev['be_rate_pct']}%) |\n")
        f.write(f"| **Убыточные (Losses)** | {metrics_50_tp['losses']} ({metrics_50_tp['loss_rate_pct']}%) | {metrics_50_rev['losses']} ({metrics_50_rev['loss_rate_pct']}%) | {metrics_618_tp['losses']} ({metrics_618_tp['loss_rate_pct']}%) | {metrics_618_rev['losses']} ({metrics_618_rev['loss_rate_pct']}%) |\n")
        f.write(f"| **🛡️ СПАСЕНО СТОПОВ переносом в БУ** | **{metrics_50_tp['saved_from_sl']} сделок** ({metrics_50_tp['saved_from_sl']/metrics_50_tp['total_trades']*100:.1f}%) | **{metrics_50_rev['saved_from_sl']} сделок** | **{metrics_618_tp['saved_from_sl']} сделок** ({metrics_618_tp['saved_from_sl']/metrics_618_tp['total_trades']*100:.1f}%) | **{metrics_618_rev['saved_from_sl']} сделок** |\n")
        f.write(f"| **Profit Factor** | **🔥 {metrics_50_tp['profit_factor']}** | **🔥 {metrics_50_rev['profit_factor']}** | **🔥 {metrics_618_tp['profit_factor']}** | **🔥 {metrics_618_rev['profit_factor']}** |\n")
        f.write(f"| **Чистый Net PnL ($) (Фикс деп $1k)** | **${metrics_50_tp['net_pnl_usd']:+,.2f}** | **${metrics_50_rev['net_pnl_usd']:+,.2f}** | **${metrics_618_tp['net_pnl_usd']:+,.2f}** | **${metrics_618_rev['net_pnl_usd']:+,.2f}** |\n")
        f.write(f"| **Net EV на сделку** | **+{metrics_50_tp['net_ev_r']:0.2f} R** | **+{metrics_50_rev['net_ev_r']:0.2f} R** | **+{metrics_618_tp['net_ev_r']:0.2f} R** | **+{metrics_618_rev['net_ev_r']:0.2f} R** |\n")
        f.write(f"| **Max Drawdown %** | **{metrics_50_tp['max_dd_pct']}%** | **{metrics_50_rev['max_dd_pct']}%** | **{metrics_618_tp['max_dd_pct']}%** | **{metrics_618_rev['max_dd_pct']}%** |\n")

        fin_comp_50 = monthly_50_df.iloc[-1]['compound_end_usd']
        fin_comp_618 = monthly_618_df.iloc[-1]['compound_end_usd']
        f.write(f"| **Итоговый депозит СО СЛОЖНЫМ %** | **${fin_comp_50:,.2f} (x{fin_comp_50/1000:.2f})** | — | **${fin_comp_618:,.2f} (x{fin_comp_618/1000:.2f})** | — |\n\n")

        # ── Раздел 2: Спасенные стопы ──
        f.write("## 2. 🛡️ Сколько стопов мы реально предотвратили переносом в БУ на 2R?\n\n")
        f.write(f"- **Для входа 50%:** перенос в БУ на 2R спас от полного стоп-лосса **{metrics_50_tp['saved_from_sl']} сделок** из 300 (**{metrics_50_tp['saved_from_sl']/metrics_50_tp['total_trades']*100:.1f}%** всех входов)!\n")
        f.write(f"  * Это сделки, которые уверенно дали +2R, но не дошли до 3R, развернулись и пробили первоначальный стоп. Без правила БУ@2R мы бы потеряли в них **${metrics_50_tp['saved_from_sl'] * 10:,.2f}** депозита.\n")
        f.write(f"- **Для входа 61.8%:** перенос в БУ спас **{metrics_618_tp['saved_from_sl']} сделок** из 210 (**{metrics_618_tp['saved_from_sl']/metrics_618_tp['total_trades']*100:.1f}%** всех входов) — экономия **${metrics_618_tp['saved_from_sl'] * 10:,.2f}** риска.\n\n")

        # ── Раздел 3: Статистика по инструментам ──
        f.write("## 3. 🪙 Детальная статистика по всем 13 инструментам\n")
        f.write("Рейтинг активности и результативности монет (показаны варианты с входом 50% и 61.8%):\n\n")
        f.write("| Монета | Сделок (50%) | Win Rate (50%) | PnL $ (50%) | PF (50%) | Сделок (61.8%) | Win Rate (61.8%) | PnL $ (61.8%) | PF (61.8%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        c50_map = coins_50_df.set_index("symbol")
        c618_map = coins_618_df.set_index("symbol")

        for sym in coins_50_df['symbol']:
            r50 = c50_map.loc[sym]
            r618 = c618_map.loc[sym] if sym in c618_map.index else None
            t618 = r618['trades'] if r618 is not None else 0
            wr618 = r618['win_rate_pct'] if r618 is not None else 0.0
            pnl618 = r618['net_pnl_usd'] if r618 is not None else 0.0
            pf618 = r618['profit_factor'] if r618 is not None else 0.0

            pf50_str = f"**{r50['profit_factor']}**" if r50['profit_factor'] >= 2.0 else f"{r50['profit_factor']}"
            pf618_str = f"**{pf618}**" if pf618 >= 2.0 else f"{pf618}"

            f.write(
                f"| **{sym}** | {r50['trades']} | {r50['win_rate_pct']}% | ${r50['net_pnl_usd']:+,.2f} | {pf50_str} | "
                f"{t618} | {wr618}% | ${pnl618:+,.2f} | {pf618_str} |\n"
            )

        f.write("\n### 💡 Топ-монеты стратегии:\n")
        top3_pnl = coins_50_df.sort_values(by="net_pnl_usd", ascending=False).head(3)['symbol'].tolist()
        top3_act = coins_50_df.sort_values(by="trades", ascending=False).head(3)['symbol'].tolist()
        f.write(f"- **Самые прибыльные монеты:** `{', '.join(top3_pnl)}`\n")
        f.write(f"- **Самые активные монеты:** `{', '.join(top3_act)}`\n\n")

        # ── Раздел 4: Помесячная динамика и Сложный Процент ──
        f.write("## 4. 📆 Помесячная динамика и Сложный Процент: Вход 50% vs Вход 61.8%\n\n")
        f.write("| Месяц | Сделок (50%) | WR% (50%) | Доход $ (50%) | **Депозит Сложный (50%)** | Сделок (61.8%) | WR% (61.8%) | Доход $ (61.8%) | **Депозит Сложный (61.8%)** |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        m50_map = monthly_50_df.set_index("month")
        m618_map = monthly_618_df.set_index("month")
        all_months = sorted(list(set(monthly_50_df['month'].tolist() + monthly_618_df['month'].tolist())))

        for m in all_months:
            r50 = m50_map.loc[m] if m in m50_map.index else None
            r618 = m618_map.loc[m] if m in m618_map.index else None

            t50 = r50['trades'] if r50 is not None else 0
            wr50 = r50['win_rate_pct'] if r50 is not None else 0.0
            p50 = r50['fixed_pnl_usd'] if r50 is not None else 0.0
            cd50 = r50['compound_end_usd'] if r50 is not None else 0.0

            t618 = r618['trades'] if r618 is not None else 0
            wr618 = r618['win_rate_pct'] if r618 is not None else 0.0
            p618 = r618['fixed_pnl_usd'] if r618 is not None else 0.0
            cd618 = r618['compound_end_usd'] if r618 is not None else 0.0

            f.write(
                f"| {m} | {t50} | {wr50}% | ${p50:+,.2f} | **${cd50:,.2f}** | "
                f"{t618} | {wr618}% | ${p618:+,.2f} | **${cd618:,.2f}** |\n"
            )

        # ── Анализ лучших и худших месяцев ──
        best_m50 = monthly_50_df.sort_values(by="compound_roi_pct", ascending=False).iloc[0]
        worst_m50 = monthly_50_df.sort_values(by="compound_roi_pct", ascending=True).iloc[0]
        best_m618 = monthly_618_df.sort_values(by="compound_roi_pct", ascending=False).iloc[0]
        worst_m618 = monthly_618_df.sort_values(by="compound_roi_pct", ascending=True).iloc[0]

        f.write("\n### 🌟 Лучшие и худшие месяцы:\n")
        f.write(f"- **Вариант 50%:**\n")
        f.write(f"  * 🏆 **Лучший месяц:** `{best_m50['month']}` $\\rightarrow$ **+{best_m50['compound_roi_pct']}%** прироста (+${best_m50['compound_pnl_usd']:,.2f}, {best_m50['trades']} сделок, WR {best_m50['win_rate_pct']}%)\n")
        f.write(f"  * ⚠️ **Худший месяц:** `{worst_m50['month']}` $\\rightarrow$ **{worst_m50['compound_roi_pct']}%** просадки (${worst_m50['compound_pnl_usd']:,.2f}, {worst_m50['trades']} сделок, WR {worst_m50['win_rate_pct']}%)\n")
        f.write(f"- **Вариант 61.8%:**\n")
        f.write(f"  * 🏆 **Лучший месяц:** `{best_m618['month']}` $\\rightarrow$ **+{best_m618['compound_roi_pct']}%** прироста (+${best_m618['compound_pnl_usd']:,.2f}, {best_m618['trades']} сделок, WR {best_m618['win_rate_pct']}%)\n")
        f.write(f"  * ⚠️ **Худший месяц:** `{worst_m618['month']}` $\\rightarrow$ **{worst_m618['compound_roi_pct']}%** просадки (${worst_m618['compound_pnl_usd']:,.2f}, {worst_m618['trades']} сделок, WR {worst_m618['win_rate_pct']}%)\n")

    logger.info(f"📄 Отчет сохранен: {report_file}")


if __name__ == "__main__":
    main()
