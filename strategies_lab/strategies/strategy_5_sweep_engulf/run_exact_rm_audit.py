"""
run_exact_rm_audit.py — Точное и прозрачное сравнение работы стратегии из run_portfolio_rm_simulation.py:
1. Сравнение 3 параллельных сделок vs 4 параллельных сделок.
2. Сравнение без лимита просадки vs с лимитом 5% в месяц.
3. Влияние риска на сделку (1.0%, 0.50%, 0.35%) при жестком лимите -5% в месяц.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(r"c:\Users\user\Desktop\RalphTradeBot")
sys.path.insert(0, str(BASE_DIR))

from run_portfolio_rm_simulation import (
    load_all_market_data,
    detect_all_raw_signals,
    ActivePosition,
    PendingOrder,
    ClosedTrade,
    SYMBOLS,
    FEE,
    RESULTS_DIR,
)

def run_simulation(mdata, signals, max_concurrent=3, cooldown_bars=6, risk_per_trade_pct=1.0, monthly_limit_pct=5.0):
    timeline = next(iter(mdata.values())).index
    sigs_by_time = {}
    for s in signals:
        sigs_by_time.setdefault(s["signal_time"], []).append(s)
        
    active_positions = []
    pending_orders = []
    closed_trades = []
    last_trade_bar = {}
    
    cur_month = ""
    current_depo = 1000.0
    month_start_depo = 1000.0
    month_realized_pnl_pct = 0.0
    month_locked = False
    
    skipped_max_concurrent = 0
    skipped_cooldown = 0
    skipped_monthly_dd = 0
    trade_id_seq = 0
    
    for bar_idx, cur_time in enumerate(timeline):
        m_str = cur_time.strftime("%Y-%m")
        if m_str != cur_month:
            cur_month = m_str
            month_start_depo = current_depo
            month_realized_pnl_pct = 0.0
            month_locked = False
            
        # 1. Active positions
        still_active = []
        for pos in active_positions:
            df = mdata[pos.symbol]
            if cur_time not in df.index:
                still_active.append(pos); continue
            row = df.loc[cur_time]
            bh, bl = float(row['high']), float(row['low'])
            pos.bars_held += 1
            
            if pos.direction == "SHORT":
                hit_sl = bh >= pos.cur_sl
                hit_tp = bl <= pos.tp_p
                hit_2r = bl <= pos.trig_2r
            else:
                hit_sl = bl <= pos.cur_sl
                hit_tp = bh >= pos.tp_p
                hit_2r = bh >= pos.trig_2r
                
            if hit_sl:
                reason = "be" if pos.be_active else "sl"
                exit_p = pos.cur_sl
                net_r = 0.0 if pos.be_active else -(pos.risk_frac + 2.0 * FEE) / pos.risk_frac
                trade_pnl_pct = net_r * risk_per_trade_pct
                closed_trades.append(ClosedTrade(
                    trade_id=pos.trade_id, symbol=pos.symbol, direction=pos.direction,
                    entry_time=pos.entry_time, exit_time=cur_time,
                    entry_p=pos.entry_p, exit_p=exit_p, sl_p=pos.sl_p, tp_p=pos.tp_p,
                    exit_reason=reason, net_r=net_r, net_pnl_usd=trade_pnl_pct * 10.0,
                    bars_held=pos.bars_held, month=m_str,
                ))
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                if monthly_limit_pct is not None and month_realized_pnl_pct <= -monthly_limit_pct:
                    month_locked = True
                    pending_orders.clear()
                continue
                
            if hit_tp:
                trade_pnl_pct = 3.0 * risk_per_trade_pct
                closed_trades.append(ClosedTrade(
                    trade_id=pos.trade_id, symbol=pos.symbol, direction=pos.direction,
                    entry_time=pos.entry_time, exit_time=cur_time,
                    entry_p=pos.entry_p, exit_p=pos.tp_p, sl_p=pos.sl_p, tp_p=pos.tp_p,
                    exit_reason="tp", net_r=3.0, net_pnl_usd=trade_pnl_pct * 10.0,
                    bars_held=pos.bars_held, month=m_str,
                ))
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                continue
                
            if hit_2r and not pos.be_active:
                pos.be_active = True
                pos.cur_sl = pos.be_p
                
            if pos.bars_held >= 120:
                closed_trades.append(ClosedTrade(
                    trade_id=pos.trade_id, symbol=pos.symbol, direction=pos.direction,
                    entry_time=pos.entry_time, exit_time=cur_time,
                    entry_p=pos.entry_p, exit_p=float(row['close']), sl_p=pos.sl_p, tp_p=pos.tp_p,
                    exit_reason="timeout", net_r=0.0, net_pnl_usd=0.0,
                    bars_held=pos.bars_held, month=m_str,
                ))
                continue
                
            still_active.append(pos)
        active_positions = still_active
        
        # 2. Pending orders
        still_pending = []
        for po in pending_orders:
            if month_locked:
                skipped_monthly_dd += 1; continue
            df = mdata[po.symbol]
            if cur_time not in df.index:
                if bar_idx < po.expire_bar: still_pending.append(po)
                continue
            row = df.loc[cur_time]
            bh, bl = float(row['high']), float(row['low'])
            if (po.direction == "SHORT" and bh > po.sl_p) or (po.direction == "LONG" and bl < po.sl_p):
                continue
            filled = (bh >= po.entry_p) if po.direction == "SHORT" else (bl <= po.entry_p)
            if filled:
                if len(active_positions) < max_concurrent:
                    trade_id_seq += 1
                    active_positions.append(ActivePosition(
                        trade_id=trade_id_seq, symbol=po.symbol, direction=po.direction,
                        entry_time=cur_time, entry_p=po.entry_p, sl_p=po.sl_p,
                        cur_sl=po.sl_p, tp_p=po.tp_p, be_p=po.be_p, trig_2r=po.trig_2r,
                        risk_frac=po.risk_frac,
                    ))
                    last_trade_bar[po.symbol] = bar_idx
                else:
                    skipped_max_concurrent += 1
                continue
            if bar_idx < po.expire_bar:
                still_pending.append(po)
        pending_orders = still_pending
        
        # 3. New signals
        if cur_time in sigs_by_time:
            for sig in sigs_by_time[cur_time]:
                sym = sig["symbol"]
                if month_locked:
                    skipped_monthly_dd += 1; continue
                last_b = last_trade_bar.get(sym, -999)
                if (bar_idx - last_b) < cooldown_bars:
                    skipped_cooldown += 1; continue
                if any(p.symbol == sym for p in active_positions):
                    skipped_max_concurrent += 1; continue
                if any(p.symbol == sym for p in pending_orders):
                    continue
                pending_orders.append(PendingOrder(
                    symbol=sym, direction=sig["direction"], signal_time=cur_time,
                    expire_bar=bar_idx + 2, entry_p=sig["entry_p"], sl_p=sig["sl_p"],
                    tp_p=sig["tp_p"], be_p=sig["be_p"], trig_2r=sig["trig_2r"],
                    risk_frac=sig["risk_frac"],
                ))
                
    return closed_trades, {
        "skipped_max_concurrent": skipped_max_concurrent,
        "skipped_cooldown": skipped_cooldown,
        "skipped_monthly_dd": skipped_monthly_dd,
        "final_depo": current_depo,
    }


def analyze(closed_trades, stats_meta, label, risk_pct):
    if not closed_trades:
        print(f"[{label}] 0 сделок")
        return
    df = pd.DataFrame([{
        "month": t.month, "net_r": t.net_r, "reason": t.exit_reason,
        "pnl_pct": t.net_r * risk_pct
    } for t in closed_trades])
    
    n_t = len(df)
    w = len(df[df['reason'] == "tp"])
    b = len(df[df['reason'] == "be"])
    l = len(df[df['reason'] == "sl"])
    tot_r = df['net_r'].sum()
    gw = df[df['net_r'] > 0]['net_r'].sum()
    gl = abs(df[df['net_r'] < 0]['net_r'].sum())
    pf = gw / gl if gl > 0 else 99.0
    
    # Расчет помесячной доходности
    months = sorted(df['month'].unique())
    m_pnl = []
    depo = 1000.0
    for m in months:
        sub = df[df['month'] == m]
        start_d = depo
        for p in sub['pnl_pct']:
            depo *= (1.0 + p / 100.0)
        pct = (depo - start_d) / start_d * 100.0
        m_pnl.append((m, len(sub), sub['net_r'].sum(), pct, depo))
        
    df_m = pd.DataFrame(m_pnl, columns=["month", "trades", "net_r", "month_pct", "end_depo"])
    pos_m = len(df_m[df_m['month_pct'] > 0])
    stop_m = len(df_m[df_m['month_pct'] <= -4.95])
    
    # Максимальная просадка
    curve = [1000.0]
    for p in df['pnl_pct']:
        curve.append(curve[-1] * (1.0 + p / 100.0))
    s = pd.Series(curve)
    max_dd = ((s.cummax() - s) / s.cummax()).max() * 100.0
    
    print(f"\n>>> {label} <<<")
    print(f"Сделок: {n_t} ({n_t/22:.1f}/мес) | Тейков: {w} ({w/n_t*100:.1f}%) | БУ: {b} ({b/n_t*100:.1f}%) | Стопов: {l} ({l/n_t*100:.1f}%)")
    print(f"Profit Factor: {pf:.2f} | Net R: {tot_r:+.2f} R | EV: {tot_r/n_t:+.3f} R")
    print(f"Депозит: ${depo:,.2f} ({depo/1000.0:+.2f}x) | Макс. просадка: {max_dd:.1f}%")
    print(f"Месяцев в плюсе: {pos_m} из {len(months)} | Месяцев с блокировкой 5%: {stop_m}")
    return {
        "label": label, "trades": n_t, "wr": round(w/n_t*100, 1), "pf": round(pf, 2),
        "net_r": round(tot_r, 1), "depo": round(depo, 2), "growth_x": round(depo/1000.0, 2),
        "max_dd_pct": round(max_dd, 1), "pos_m": pos_m, "stop_m": stop_m
    }

if __name__ == "__main__":
    mdata = load_all_market_data()
    signals = detect_all_raw_signals(mdata)
    
    scenarios = [
        # max_c, cd, risk_pct, monthly_limit, label
        (3, 6, 1.0, None, "1. Max 3 позиции | БЕЗ лимита месяца | Риск 1.0%"),
        (4, 6, 1.0, None, "2. Max 4 позиции | БЕЗ лимита месяца | Риск 1.0%"),
        (3, 6, 1.0, 5.0,  "3. Max 3 позиции | ЛИМИТ -5% в месяц | Риск 1.0%"),
        (4, 6, 1.0, 5.0,  "4. Max 4 позиции | ЛИМИТ -5% в месяц | Риск 1.0%"),
        (3, 6, 0.50, 5.0, "5. Max 3 позиции | ЛИМИТ -5% в месяц | Риск 0.50%"),
        (4, 6, 0.50, 5.0, "6. Max 4 позиции | ЛИМИТ -5% в месяц | Риск 0.50%"),
        (3, 6, 0.35, 5.0, "7. Max 3 позиции | ЛИМИТ -5% в месяц | Риск 0.35%"),
        (4, 6, 0.35, 5.0, "8. Max 4 позиции | ЛИМИТ -5% в месяц | Риск 0.35%"),
    ]
    
    summary = []
    for max_c, cd, r_pct, m_lim, lbl in scenarios:
        trades, meta = run_simulation(mdata, signals, max_concurrent=max_c, cooldown_bars=cd, risk_per_trade_pct=r_pct, monthly_limit_pct=m_lim)
        res = analyze(trades, meta, lbl, r_pct)
        summary.append(res)
        
    df_s = pd.DataFrame(summary)
    df_s.to_csv(RESULTS_DIR / "exact_rm_comparison_table.csv", index=False)
    print("\n" + "="*80)
    print("                      ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА")
    print("="*80)
    print(df_s[["label", "trades", "pf", "net_r", "depo", "growth_x", "max_dd_pct", "pos_m", "stop_m"]].to_string())
