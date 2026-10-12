"""
honest_retest.py — Перетест Стратегии 5 БЕЗ заглядывания в будущее.

Баг старой версии: пики W5 брались через argrelextrema(order=3) по всему ряду,
т.е. требовали, чтобы 3 СЛЕДУЮЩИЕ свечи не перебили high W5. Вход на W5+1..W5+2
со стопом за high W5 в такой выборке не мог быть выбит, а все провальные сетапы
(цена пошла дальше) заранее отбрасывались.

Здесь: кандидат W5 = бар t, у которого high[t] >= max(high[t-3:t]) (только левая сторона).
W3 (p1) — подтверждённый пивот (p1 <= t-3, известен на момент t). Все фильтры RSI/Эллиотта
считаются по данным [:t+1]. Свеча S должна закрыться, вход лимиткой на 50% S на 1-2 свече после.
Выход: БУ на +2R, тейк +3R net, стоп за тенью S. Комиссия 0.05% + 0.05%.

Портфель: $1000, риск 1% от баланса на начало месяца, лимит убытка -5% в месяц,
не более 5 одновременных позиций, не более 1 позиции на монету.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from signal_scanner import calculate_rsi_wilder
from elliott_detector import detect_elliott_impulse
from strategies_lab.engine import load_all_market_data
from strategies_lab.strategies.strategy_5_sweep_engulf.patterns import calculate_net_1_to_3_tp
from strategies_lab.strategies.strategy_5_sweep_engulf.config import RESULTS_DIR, MIN_ELLIOTT_SCORE

DATA_DIR = Path(r"C:\Users\user\Desktop\данные для бектестов\2025_2026_Full_Top_Coins")
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT',
           'AVAXUSDT', 'LINKUSDT', 'NEARUSDT', 'SUIUSDT', 'AAVEUSDT', 'APTUSDT', 'DOTUSDT', 'RENDERUSDT']
FEE = 0.0005
OUT = RESULTS_DIR / "honest"
CHARTS = OUT / "charts"
OUT.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)


def left_pivots(arr: np.ndarray, is_high: bool, k: int = 3) -> np.ndarray:
    """Бар t — кандидат, если он экстремум относительно k ПРЕДЫДУЩИХ баров (без будущего)."""
    n = len(arr)
    out = np.zeros(n, dtype=bool)
    for t in range(k, n):
        w = arr[t - k:t]
        out[t] = arr[t] >= w.max() if is_high else arr[t] <= w.min()
    return out


def confirmed_pivots(arr: np.ndarray, is_high: bool, k: int = 3) -> np.ndarray:
    """Классический пивот (k слева и k справа). Можно использовать только если t+k уже в прошлом."""
    n = len(arr)
    out = np.zeros(n, dtype=bool)
    for t in range(k, n - k):
        w = np.concatenate([arr[t - k:t], arr[t + 1:t + k + 1]])
        out[t] = arr[t] >= w.max() if is_high else arr[t] <= w.min()
    return out


def scan_symbol(sym: str, df: pd.DataFrame):
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    o = df['open'].values.astype(float)
    rsi = calculate_rsi_wilder(c, period=7)
    n = len(df)
    trades = []
    busy_until = -1

    for direction in ("SHORT", "LONG"):
        is_high = direction == "SHORT"
        src = h if is_high else l
        cand = left_pivots(src, is_high)
        conf = np.where(confirmed_pivots(src, is_high))[0]
        busy_until = -1
        for t in np.where(cand)[0]:
            if t <= busy_until or t >= n - 160:
                continue
            # W3 = последний подтверждённый пивот, известный к моменту t (p1 + 3 <= t)
            prior = conf[(conf + 3 <= t)]
            if len(prior) == 0:
                continue
            p1 = int(prior[-1])
            dist = t - p1
            if dist < 6 or dist > 45:
                continue
            if is_high:
                if h[t] <= h[p1] or np.max(h[p1 + 1:t]) > h[t]:
                    continue
                r1, r2 = rsi[p1], rsi[t]
                if r2 >= r1 or (r1 - r2) < 2.5 or r1 < 64 or r2 < 64:
                    continue
            else:
                if l[t] >= l[p1] or np.min(l[p1 + 1:t]) < l[t]:
                    continue
                r1, r2 = rsi[p1], rsi[t]
                if r2 <= r1 or (r2 - r1) < 2.5 or r1 > 36 or r2 > 36:
                    continue

            res = detect_elliott_impulse(high=h[:t + 1], low=l[:t + 1], close=c[:t + 1], open_p=o[:t + 1],
                                         w3_bar=p1, w5_bar=t, direction=direction, rsi=rsi[:t + 1])
            if not (res.is_valid and res.score >= MIN_ELLIOTT_SCORE):
                continue

            # Свеча S = сама W5: свип предыдущей свечи + закрытие за её телом (видно на закрытии бара t)
            pb_top, pb_bot = max(o[t - 1], c[t - 1]), min(o[t - 1], c[t - 1])
            if is_high:
                ok = h[t] > h[t - 1] and c[t] < o[t] and c[t] < pb_bot
            else:
                ok = l[t] < l[t - 1] and c[t] > o[t] and c[t] > pb_top
            if not ok:
                continue

            rng = h[t] - l[t]
            if rng <= 0:
                continue
            entry = l[t] + 0.5 * rng
            sl = h[t] if is_high else l[t]
            risk = abs(entry - sl)
            be_trig, _, _ = calculate_net_1_to_3_tp(entry, sl, direction, FEE, 2.0)
            tp, _, _ = calculate_net_1_to_3_tp(entry, sl, direction, FEE, 3.0)
            be_px = entry * (1 - 2 * FEE) if is_high else entry * (1 + 2 * FEE)

            # Лимит на 1-2 свече после S
            fill = -1
            for b in (t + 1, t + 2):
                if is_high:
                    if o[b] >= sl:
                        break
                    if h[b] >= entry:
                        fill = b
                        break
                else:
                    if o[b] <= sl:
                        break
                    if l[b] <= entry:
                        fill = b
                        break
            if fill == -1:
                continue

            cur_sl, hit2, reason, exit_px, exit_b = sl, False, "timeout", c[min(fill + 150, n - 1)], min(fill + 150, n - 1)
            for b in range(fill, min(fill + 150, n)):
                if not hit2:
                    if (is_high and h[b] >= cur_sl) or (not is_high and l[b] <= cur_sl):
                        reason, exit_px, exit_b = "sl", cur_sl, b
                        break
                    if (is_high and l[b] <= be_trig) or (not is_high and h[b] >= be_trig):
                        hit2, cur_sl = True, be_px
                        # консервативно: если на этой же свече и TP — не засчитываем
                        continue
                else:
                    if (is_high and h[b] >= cur_sl) or (not is_high and l[b] <= cur_sl):
                        reason, exit_px, exit_b = "be", cur_sl, b
                        break
                    if (is_high and l[b] <= tp) or (not is_high and h[b] >= tp):
                        reason, exit_px, exit_b = "tp", tp, b
                        break

            risk_frac = risk / entry
            pct = (entry - exit_px) / entry if is_high else (exit_px - entry) / entry
            # Net R на 1R риска: позиция = 1/risk_frac (в единицах риска)
            net_r = (pct - FEE - FEE * (1 + pct)) / risk_frac
            trades.append(dict(symbol=sym, direction=direction, w3=p1, s_bar=t, fill_bar=fill, exit_bar=exit_b,
                               entry_time=df.index[fill], exit_time=df.index[exit_b], entry=entry, sl=sl, tp=tp,
                               exit_px=exit_px, reason=reason, net_r=net_r, risk_pct=risk_frac * 100,
                               wave_indices=res.wave_indices, wave_points=res.wave_points))
            busy_until = exit_b
    return trades


def portfolio_sim(tr: pd.DataFrame, start=1000.0, risk=0.01, month_cap_r=-5.0, max_pos=5):
    tr = tr.sort_values("entry_time").reset_index(drop=True)
    bal = start
    open_pos = []  # (exit_time, symbol, r, risk_usd)
    taken = []
    cur_month, month_r, month_start_bal, month_locked = None, 0.0, start, False
    for _, t in tr.iterrows():
        # закрываем позиции, вышедшие до входа
        still = []
        for p in open_pos:
            if p[0] <= t.entry_time:
                bal += p[2] * p[3]
            else:
                still.append(p)
        open_pos = still
        m = t.entry_time.to_period("M")
        if m != cur_month:
            cur_month, month_r, month_start_bal, month_locked = m, 0.0, bal, False
        if month_locked or len(open_pos) >= max_pos or any(p[1] == t.symbol for p in open_pos):
            continue
        # худший сценарий: все открытые позиции могут стопнуться
        if month_r - len(open_pos) - 1 < month_cap_r - 1e-9:
            continue
        r_usd = month_start_bal * risk
        open_pos.append((t.exit_time, t.symbol, t.net_r, r_usd))
        month_r += t.net_r
        if month_r <= month_cap_r:
            month_locked = True
        taken.append({**t.to_dict(), "risk_usd": r_usd, "pnl_usd": t.net_r * r_usd})
    for p in open_pos:
        bal += p[2] * p[3]
    return pd.DataFrame(taken), bal


def prop_mc(r_list: np.ndarray, n_sims=20000, risk=0.01, target=0.08, max_dd=0.10, trades_cap=200, seed=1):
    """Monte Carlo фазы челленджа: риск 1% от начального, цель +8%, лимит общей просадки -10% (статический)."""
    rng = np.random.default_rng(seed)
    passed = failed = 0
    lengths = []
    for _ in range(n_sims):
        eq = 0.0
        for k in range(trades_cap):
            eq += rng.choice(r_list) * risk
            if eq >= target:
                passed += 1
                lengths.append(k + 1)
                break
            if eq <= -max_dd:
                failed += 1
                break
    return passed / n_sims, failed / n_sims, (np.median(lengths) if lengths else None)


def plot_trade(t: dict, df: pd.DataFrame, path: Path):
    w0 = t["wave_indices"].get("W0", t["w3"] - 30)
    a, b = max(0, w0 - 10), min(len(df), t["exit_bar"] + 15)
    d = df.iloc[a:b]
    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor="#131722")
    ax.set_facecolor("#131722")
    for i, (_, r) in enumerate(d.iterrows()):
        col = "#089981" if r.close >= r.open else "#F23645"
        ax.plot([i, i], [r.low, r.high], color=col, lw=1)
        ax.add_patch(patches.Rectangle((i - 0.3, min(r.open, r.close)), 0.6,
                                       max(abs(r.close - r.open), (r.high - r.low) * 0.02), color=col))
    xs, ys = [], []
    for k in ["W0", "W1", "W2", "W3", "W4", "W5"]:
        if k in t["wave_indices"] and k in t["wave_points"]:
            x = t["wave_indices"][k] - a
            if 0 <= x < len(d):
                xs.append(x); ys.append(t["wave_points"][k])
                ax.annotate(k[1], (x, t["wave_points"][k]), color="white", fontsize=10, ha="center",
                            xytext=(0, 10 if (k in "W1W3W5") == (t["direction"] == "LONG") else -14),
                            textcoords="offset points")
    ax.plot(xs, ys, color="#00E5FF", lw=1.6)
    x0, x1 = t["fill_bar"] - a, t["exit_bar"] - a + 3
    ax.hlines(t["entry"], x0 - 2, x1, colors="#FFD700", linestyles="-.", label="Entry 50%")
    ax.hlines(t["sl"], x0 - 2, x1, colors="#FF5252", linestyles="--", label="SL")
    ax.hlines(t["tp"], x0 - 2, x1, colors="#00E676", label="TP 3R")
    ax.axvline(t["s_bar"] - a, color="#888", lw=0.6, ls=":")
    ax.tick_params(colors="#D1D4DC")
    ax.legend(facecolor="#1E222D", labelcolor="#D1D4DC", fontsize=8, loc="upper left")
    ax.set_title(f"{t['symbol']} 15m {t['direction']} | {t['entry_time']:%Y-%m-%d %H:%M} | "
                 f"{t['reason'].upper()} {t['net_r']:+.2f}R", color="white")
    fig.tight_layout()
    fig.savefig(path, facecolor="#131722", dpi=100)
    plt.close(fig)


def main():
    md = load_all_market_data(data_dir=DATA_DIR, symbols=SYMBOLS, interval="15m")
    all_tr = []
    for sym, d in md.items():
        all_tr += scan_symbol(sym, d["main"])
    tr = pd.DataFrame(all_tr).sort_values("entry_time").reset_index(drop=True)

    def stats(x: pd.DataFrame, rcol="net_r"):
        r = x[rcol].values
        w, be, lo = (r > 0.05).sum(), (np.abs(r) <= 0.05).sum(), (r < -0.05).sum()
        gw, gl = r[r > 0].sum(), -r[r < 0].sum()
        eq = np.cumsum(r); dd = (np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq).max()
        streak = mx = 0
        for v in r:
            streak = streak + 1 if v < -0.05 else 0; mx = max(mx, streak)
        return dict(n=len(r), wins=int(w), be=int(be), losses=int(lo), wr=round(w / len(r) * 100, 1),
                    pf=round(gw / gl, 2) if gl else None, ev=round(r.mean(), 3), sum_r=round(r.sum(), 1),
                    max_dd_r=round(dd, 1), max_loss_streak=mx)

    raw = stats(tr)
    port, final_bal = portfolio_sim(tr)
    pst = stats(port)
    port["month"] = pd.to_datetime(port["entry_time"]).dt.tz_localize(None).dt.to_period("M")
    monthly = port.groupby("month").agg(trades=("net_r", "size"), sum_r=("net_r", "sum"),
                                        pnl_usd=("pnl_usd", "sum")).reset_index()
    by_sym = tr.groupby("symbol")["net_r"].agg(["size", "sum", "mean"]).sort_values("sum", ascending=False)
    by_dir = tr.groupby("direction")["net_r"].agg(["size", "sum", "mean"])

    p_pass, p_fail, med_len = prop_mc(tr["net_r"].values)
    p2_pass, p2_fail, med2 = prop_mc(tr["net_r"].values, target=0.05)

    tr.drop(columns=["wave_indices", "wave_points"]).to_csv(OUT / "honest_trades.csv", index=False)
    monthly.to_csv(OUT / "honest_monthly.csv", index=False)

    # Графики: 4 TP, 3 SL, 2 BE
    picks = pd.concat([tr[tr.reason == "tp"].sample(min(4, (tr.reason == "tp").sum()), random_state=3),
                       tr[tr.reason == "sl"].sample(min(3, (tr.reason == "sl").sum()), random_state=3),
                       tr[tr.reason == "be"].sample(min(2, (tr.reason == "be").sum()), random_state=3)])
    for i, (_, t) in enumerate(picks.iterrows()):
        plot_trade(t.to_dict(), md[t.symbol]["main"], CHARTS / f"{i+1:02d}_{t.symbol}_{t.direction}_{t.reason}.png")

    print("RAW:", raw)
    print("PORTFOLIO:", pst, "final_bal", round(final_bal, 2))
    print("BY_DIR:\n", by_dir)
    print("BY_SYM:\n", by_sym)
    print("MONTHLY:\n", monthly.to_string())
    print(f"PROP phase1 (+8%/-10%): pass {p_pass:.1%} fail {p_fail:.1%} median trades {med_len}")
    print(f"PROP phase2 (+5%/-10%): pass {p2_pass:.1%} fail {p2_fail:.1%} median trades {med2}")
    print("avg risk %:", round(tr.risk_pct.mean(), 3), "median:", round(tr.risk_pct.median(), 3))


if __name__ == "__main__":
    main()
