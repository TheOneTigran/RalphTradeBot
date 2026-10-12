"""
generate_prod_charts_and_data.py — Экспорт графиков и подробных таблиц
репликации продакшена для отчета пользователю.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from strategies_lab.strategies.strategy_5_sweep_engulf.prod_replication_audit import run_prod_audit
from strategies_lab.strategies.strategy_5_sweep_engulf.chart_plotter import plot_sweep_engulf_trade
from strategies_lab.strategies.base_strategy import TradeSignal, TradeResult

ARTIFACTS_DIR = Path(r"C:\Users\user\.gemini\antigravity-ide\brain\2249d323-d907-4b4e-ae7d-74c8bae47d0a")
RESULTS_DIR = BASE_DIR / "strategies_lab" / "strategies" / "strategy_5_sweep_engulf" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    executed_trades, df_trades, df_monthly, df_coins, kpis, mdata = run_prod_audit()

    # Сохраняем CSV
    df_trades.to_csv(RESULTS_DIR / "prod_replication_trades.csv", index=False)
    df_monthly.to_csv(RESULTS_DIR / "prod_replication_monthly.csv", index=False)
    df_coins.to_csv(RESULTS_DIR / "prod_replication_coins.csv", index=False)

    print(f"Всего сделок: {len(executed_trades)}")

    # Выбираем характерные сделки для визуализации
    tp_trades = [t for t in executed_trades if t.exit_reason == "tp"]
    be_trades = [t for t in executed_trades if t.exit_reason == "be" and t.saved_from_sl]
    sl_trades = [t for t in executed_trades if t.exit_reason == "sl"]

    # 1. Эталонный TP (SHORT)
    sample_tp_short = next((t for t in tp_trades if t.direction == "SHORT" and t.symbol in ["SOLUSDT", "ETHUSDT", "BTCUSDT", "APTUSDT"]), tp_trades[0])
    # 2. Эталонный TP (LONG)
    sample_tp_long = next((t for t in tp_trades if t.direction == "LONG" and t.symbol in ["SOLUSDT", "ETHUSDT", "BTCUSDT", "AAVEUSDT"]), tp_trades[1])
    # 3. Безубыток со спасением от стопа
    sample_be = be_trades[0] if be_trades else [t for t in executed_trades if t.exit_reason == "be"][0]
    # 4. Стоп-лосс
    sample_sl = sl_trades[0]

    samples = [
        ("prod_sample_tp_short.png", sample_tp_short, "Эталонный Take-Profit 3R (SHORT)"),
        ("prod_sample_tp_long.png", sample_tp_long, "Эталонный Take-Profit 3R (LONG)"),
        ("prod_sample_be_saved.png", sample_be, "Безубыток БУ @ 2R (Спасение от стопа)"),
        ("prod_sample_sl.png", sample_sl, "Стоп-Лосс (SL) по кончику тени S"),
    ]

    for filename, trade, title in samples:
        sym = trade.symbol
        df = mdata[sym]
        cand = trade.candidate
        pat = trade.pattern

        w0_b = cand.algo_res.wave_indices.get("W0", cand.p1 - 20)
        sig = TradeSignal(
            symbol=sym,
            interval="15m",
            direction=cand.direction,
            w5_bar=cand.p2,
            w5_time=df.index[cand.p2],
            w5_price=cand.p2_price,
            w3_bar=cand.p1,
            w3_price=cand.p1_price,
            w0_bar=w0_b,
            w0_price=cand.algo_res.wave_points.get("W0", float(df['low'].iloc[w0_b] if cand.direction == "SHORT" else df['high'].iloc[w0_b])),
            algo_score=cand.algo_res.score,
            wave_points=cand.algo_res.wave_points,
            raw_df=df.iloc[:cand.p2 + 1],
            df_1m=None,
            wave_indices=cand.algo_res.wave_indices,
            details=cand.algo_res.details,
        )

        res = TradeResult(
            signal_id=f"T{trade.trade_id}",
            symbol=sym,
            interval="15m",
            direction=trade.direction,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            initial_sl=trade.initial_sl,
            final_pnl_pct=0.0,
            pnl_r=trade.net_pnl_r,
            exit_reason=trade.exit_reason,
            bars_held=trade.bars_held,
            mfe_pct=0.0,
            mae_pct=0.0,
            net_pnl_usd=trade.net_pnl_usd,
            fee_usd=trade.fee_usd,
        )

        save_p = ARTIFACTS_DIR / filename
        plot_sweep_engulf_trade(
            signal=sig,
            result=res,
            pattern=pat,
            full_df=df,
            save_path=save_p,
            lookback_before_w0=25,
            lookahead_after_exit=15,
        )
        print(f"Сохранен график: {save_p}")

if __name__ == "__main__":
    main()
