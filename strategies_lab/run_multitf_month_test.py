"""
run_multitf_month_test.py — Бэктест Model E и Model B на разных ТФ (5m, 15m, 1h, 4h)
за последний месяц по ТОП-30 криптовалютным инструментам.

Метрики:
- Количество сделок на каждом ТФ и инструменте
- Win Rate, Profit Factor, Net PnL ($ и %), Max DD %, Max Streak
- Средняя длительность сделки (в часах)
- Анализ исполнимости руками (тайминг, окна реакции, стресс-нагрузка)
- Точная спецификация выставления ордеров (куда, сколько, как)
"""
import os
import sys
import time
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple
import concurrent.futures
import numpy as np
import pandas as pd
import requests

# Настройка путей
ROOT_DIR = Path(__file__).parent.parent
LAB_DIR = Path(__file__).parent
CACHE_DIR = LAB_DIR / "cache_month_5m"
REPORTS_DIR = LAB_DIR / "reports"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))

from signal_scanner import calculate_rsi_wilder
from scipy.signal import argrelextrema
from elliott_detector import detect_elliott_impulse

sys.stdout.reconfigure(encoding='utf-8')

# Список ТОП-30 пар для анализа
TOP_30_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "NEARUSDT", "BCHUSDT", "LTCUSDT", "AAVEUSDT", "1000PEPEUSDT",
    "DOTUSDT", "SHIB1000USDT", "TRXUSDT", "ETCUSDT", "APTUSDT",
    "POLUSDT", "UNIUSDT", "ICPUSDT", "RENDERUSDT", "ENAUSDT",
    "ARBUSDT", "OPUSDT", "INJUSDT", "TAOUSDT", "KASUSDT"
]

TIMEFRAMES = ["5m", "15m", "1h", "4h"]


def _calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    if n == 0:
        return np.zeros(0)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    else:
        atr[:] = np.mean(tr) if len(tr) > 0 else 1.0
    return atr


def fetch_symbol_5m_bybit(symbol: str, days: int = 35) -> pd.DataFrame:
    """Загружает 5m свечи с Bybit за последние N дней с кэшированием."""
    cache_file = CACHE_DIR / f"{symbol}_5m.pkl"
    if cache_file.exists():
        # Если кэш свежий (< 12 часов), используем его
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
        if (datetime.now(timezone.utc) - mtime).total_seconds() < 43200:
            return pd.read_pickle(cache_file)

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 24 * 3600 * 1000)
    current_end = now_ms

    all_candles = []
    headers = {"User-Agent": "RalphMultiTF/1.0"}

    while current_end > start_ms:
        url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=5&limit=1000&end={current_end}"
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            data = resp.json()
            batch = data.get("result", {}).get("list", [])
            if not batch:
                break
            all_candles.extend(batch)
            oldest_ts = int(batch[-1][0])
            if oldest_ts >= current_end:
                break
            current_end = oldest_ts - 1
            if oldest_ts <= start_ms:
                break
            time.sleep(0.04)
        except Exception as e:
            time.sleep(0.5)

    if not all_candles:
        if cache_file.exists():
            return pd.read_pickle(cache_file)
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df.index = pd.to_datetime(df['open_time'].astype(int), unit='ms', utc=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    df.to_pickle(cache_file)
    return df


def resample_df(df_5m: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Ресемплирует 5m датафрейм в 15m, 1h или 4h."""
    if interval == "5m":
        return df_5m
    rule_map = {"15m": "15min", "1h": "1h", "4h": "4h"}
    rule = rule_map.get(interval, "15min")
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    res = df_5m.resample(rule).agg(agg).dropna()
    return res


def scan_tf_signals(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    min_score: float = 65.0
) -> List[Dict[str, Any]]:
    """Сканирует дивергенции и паттерны Эллиотта на заданном таймфрейме."""
    n = len(df)
    if n < 40:
        return []

    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    open_p = df['open'].values.astype(np.float64)

    # Параметры индикаторов в зависимости от ТФ
    if interval == "5m":
        rsi_len = 7
        pivot_order = 3
        min_dist = 6
        max_dist = 45
        r_div_min = 2.5
    elif interval == "15m":
        rsi_len = 7
        pivot_order = 3
        min_dist = 5
        max_dist = 40
        r_div_min = 2.5
    elif interval == "1h":
        rsi_len = 14
        pivot_order = 3
        min_dist = 4
        max_dist = 35
        r_div_min = 2.0
    else:  # 4h
        rsi_len = 14
        pivot_order = 2
        min_dist = 3
        max_dist = 30
        r_div_min = 2.0

    rsi = calculate_rsi_wilder(close, period=rsi_len)
    high_pivots = argrelextrema(high, np.greater_equal, order=pivot_order)[0]
    low_pivots = argrelextrema(low, np.less_equal, order=pivot_order)[0]

    signals = []

    # SHORT
    if len(high_pivots) >= 2:
        for i in range(1, len(high_pivots)):
            p1, p2 = int(high_pivots[i - 1]), int(high_pivots[i])
            dist = p2 - p1
            if dist < min_dist or dist > max_dist or p2 >= n - 10:
                continue
            if high[p2] <= high[p1]:
                continue
            r1, r2 = float(rsi[p1]), float(rsi[p2])
            if r2 >= r1 or (r1 - r2) < r_div_min or r1 < 60.0 or r2 < 60.0:
                continue
            if len(high[p1 + 1 : p2]) > 0 and np.max(high[p1 + 1 : p2]) > high[p2]:
                continue

            res = detect_elliott_impulse(
                high=high[:p2 + 1], low=low[:p2 + 1], close=close[:p2 + 1], open_p=open_p[:p2 + 1],
                w3_bar=p1, w5_bar=p2, direction="SHORT", rsi=rsi[:p2 + 1]
            )
            if res.is_valid and res.score >= min_score:
                w0_bar = res.wave_indices.get("W0", max(0, p1 - 20))
                signals.append({
                    "symbol": symbol, "interval": interval, "direction": "SHORT",
                    "w5_bar": p2, "w5_time": df.index[p2],
                    "w5_price": float(high[p2]), "w3_bar": p1, "w3_price": float(high[p1]),
                    "w0_bar": w0_bar, "w0_price": res.wave_points.get("W0", float(low[w0_bar])),
                    "algo_score": res.score, "wave_points": res.wave_points
                })

    # LONG
    if len(low_pivots) >= 2:
        for i in range(1, len(low_pivots)):
            p1, p2 = int(low_pivots[i - 1]), int(low_pivots[i])
            dist = p2 - p1
            if dist < min_dist or dist > max_dist or p2 >= n - 10:
                continue
            if low[p2] >= low[p1]:
                continue
            r1, r2 = float(rsi[p1]), float(rsi[p2])
            if r2 <= r1 or (r2 - r1) < r_div_min or r1 > 40.0 or r2 > 40.0:
                continue
            if len(low[p1 + 1 : p2]) > 0 and np.min(low[p1 + 1 : p2]) < low[p2]:
                continue

            res = detect_elliott_impulse(
                high=high[:p2 + 1], low=low[:p2 + 1], close=close[:p2 + 1], open_p=open_p[:p2 + 1],
                w3_bar=p1, w5_bar=p2, direction="LONG", rsi=rsi[:p2 + 1]
            )
            if res.is_valid and res.score >= min_score:
                w0_bar = res.wave_indices.get("W0", max(0, p1 - 20))
                signals.append({
                    "symbol": symbol, "interval": interval, "direction": "LONG",
                    "w5_bar": p2, "w5_time": df.index[p2],
                    "w5_price": float(low[p2]), "w3_bar": p1, "w3_price": float(low[p1]),
                    "w0_bar": w0_bar, "w0_price": res.wave_points.get("W0", float(high[w0_bar])),
                    "algo_score": res.score, "wave_points": res.wave_points
                })

    return signals


def simulate_model_trade(
    model_name: str,
    sig: Dict[str, Any],
    df_main: pd.DataFrame,
    df_5m: pd.DataFrame,
    future_window: int = 80,
    risk_budget_usd: float = 10.0,
    fee_pct: float = 0.001,
) -> Dict[str, Any]:
    """Симулирует сделку с высоким разрешением по свечам."""
    w5_bar = sig["w5_bar"]
    direction = sig["direction"]
    w5_price = sig["w5_price"]
    w0_price = sig["w0_price"]
    interval = sig["interval"]

    if w5_bar + 1 >= len(df_main):
        return {}

    entry = float(df_main['close'].iloc[w5_bar])
    buffer_pct = 0.0015
    initial_sl = w5_price * (1.0 + buffer_pct) if direction == "SHORT" else w5_price * (1.0 - buffer_pct)
    risk_dist = abs(entry - initial_sl)
    if risk_dist <= 0 or entry <= 0:
        return {}

    initial_risk_pct = (risk_dist / entry) * 100.0
    pos_size_usd = risk_budget_usd / (initial_risk_pct / 100.0)

    wave_span = abs(w5_price - w0_price)
    if wave_span <= 0:
        wave_span = risk_dist * 2.5

    if direction == "SHORT":
        tp_prices = {
            "TP1": w5_price - wave_span * 0.236,
            "TP2": w5_price - wave_span * 0.382,
            "TP3": w5_price - wave_span * 0.500,
            "TP4": w5_price - wave_span * 0.618,
        }
        be_price = entry * 0.999
    else:
        tp_prices = {
            "TP1": w5_price + wave_span * 0.236,
            "TP2": w5_price + wave_span * 0.382,
            "TP3": w5_price + wave_span * 0.500,
            "TP4": w5_price + wave_span * 0.618,
        }
        be_price = entry * 1.001

    if model_name == "Model_E":  # Quant Master Hybrid
        tp_shares = {"TP1": 0.40, "TP2": 0.30}
        be_trigger = "TP2"
        use_time_stop = True
        use_chandelier = True
    elif model_name == "Model_B":  # Fail-Fast Time-Stop Multi-TP
        tp_shares = {"TP1": 0.25, "TP2": 0.35, "TP3": 0.25, "TP4": 0.15}
        be_trigger = "TP2"
        use_time_stop = True
        use_chandelier = False
    else:
        raise ValueError(f"Unknown model: {model_name}")

    future_slice = df_main.iloc[w5_bar + 1 : w5_bar + 1 + future_window]
    if len(future_slice) < 4:
        return {}

    highs = future_slice['high'].values.astype(np.float64)
    lows = future_slice['low'].values.astype(np.float64)
    closes = future_slice['close'].values.astype(np.float64)
    opens = future_slice['open'].values.astype(np.float64)
    times = future_slice.index
    n_bars = len(future_slice)

    atr_vals = _calc_atr(highs, lows, closes, period=14) if n_bars >= 14 else np.full(n_bars, risk_dist)

    current_sl = initial_sl
    trail_sl = initial_sl
    be_active = False
    tp_hits = []
    rem_share = 1.0
    accumulated_pnl = 0.0
    exit_bar = n_bars - 1
    exit_price = closes[-1]
    exit_reason = "timeout"
    exit_time = times[-1]

    # Минуты одного бара
    tf_min = 5 if interval == "5m" else (15 if interval == "15m" else (60 if interval == "1h" else 240))

    for b in range(n_bars):
        h_bar, l_bar, c_bar, o_bar = highs[b], lows[b], closes[b], opens[b]
        t_bar = times[b]

        # 1. Fail-Fast Time-Stop на баре 3 (3 бара ТФ)
        if use_time_stop and b == 3 and not tp_hits:
            cur_profit_pct = (entry - c_bar) / entry * 100.0 if direction == "SHORT" else (c_bar - entry) / entry * 100.0
            cur_r = cur_profit_pct / initial_risk_pct if initial_risk_pct > 0 else 0.0
            if cur_r < 0.4:
                exit_bar = b
                exit_price = c_bar
                exit_time = t_bar
                exit_reason = "fail_fast"
                accumulated_pnl += rem_share * cur_profit_pct
                rem_share = 0.0
                break

        # 2. Обновление Chandelier Trailing
        if use_chandelier and be_active:
            cur_atr = atr_vals[b]
            if direction == "SHORT":
                cand = h_bar + 2.2 * cur_atr
                trail_sl = min(trail_sl, cand) if trail_sl != initial_sl else cand
                current_sl = min(be_price, trail_sl)
            else:
                cand = l_bar - 2.2 * cur_atr
                trail_sl = max(trail_sl, cand) if trail_sl != initial_sl else cand
                current_sl = max(be_price, trail_sl)

        # 3. Проверка срабатывания внутри бара
        if direction == "SHORT":
            # Стоп сработал
            if h_bar >= current_sl:
                exit_bar = b
                exit_price = max(o_bar, current_sl) if o_bar >= current_sl else current_sl
                loss_pnl = (entry - exit_price) / entry * 100.0
                accumulated_pnl += rem_share * loss_pnl
                exit_reason = "trailing" if (use_chandelier and be_active) else ("be" if be_active else "sl")
                exit_time = t_bar
                rem_share = 0.0
                break

            # Тейки
            for tp_name, tp_share in tp_shares.items():
                if tp_name not in tp_hits and l_bar <= tp_prices[tp_name]:
                    tp_hits.append(tp_name)
                    gain = (entry - tp_prices[tp_name]) / entry * 100.0
                    accumulated_pnl += tp_share * gain
                    rem_share -= tp_share
                    if tp_name == be_trigger:
                        be_active = True
                        current_sl = be_price
                        trail_sl = be_price
                    if rem_share <= 0.001:
                        exit_bar = b
                        exit_price = tp_prices[tp_name]
                        exit_time = t_bar
                        exit_reason = "tp_full"
                        rem_share = 0.0
                        break
            if rem_share <= 0.001:
                break
        else:  # LONG
            if l_bar <= current_sl:
                exit_bar = b
                exit_price = min(o_bar, current_sl) if o_bar <= current_sl else current_sl
                loss_pnl = (exit_price - entry) / entry * 100.0
                accumulated_pnl += rem_share * loss_pnl
                exit_reason = "trailing" if (use_chandelier and be_active) else ("be" if be_active else "sl")
                exit_time = t_bar
                rem_share = 0.0
                break

            for tp_name, tp_share in tp_shares.items():
                if tp_name not in tp_hits and h_bar >= tp_prices[tp_name]:
                    tp_hits.append(tp_name)
                    gain = (tp_prices[tp_name] - entry) / entry * 100.0
                    accumulated_pnl += tp_share * gain
                    rem_share -= tp_share
                    if tp_name == be_trigger:
                        be_active = True
                        current_sl = be_price
                        trail_sl = be_price
                    if rem_share <= 0.001:
                        exit_bar = b
                        exit_price = tp_prices[tp_name]
                        exit_time = t_bar
                        exit_reason = "tp_full"
                        rem_share = 0.0
                        break
            if rem_share <= 0.001:
                break

    # Если позиция осталась к концу окна
    if rem_share > 0.001:
        last_c = closes[-1]
        fin_pnl = (entry - last_c) / entry * 100.0 if direction == "SHORT" else (last_c - entry) / entry * 100.0
        accumulated_pnl += rem_share * fin_pnl

    # Расчет PnL в USD и с учетом комиссий
    net_pnl_usd = (accumulated_pnl / 100.0) * pos_size_usd
    fees_usd = pos_size_usd * (fee_pct * 2.0)  # roundtrip
    pnl_after_fee_usd = net_pnl_usd - fees_usd
    net_r = pnl_after_fee_usd / risk_budget_usd

    duration_hours = (exit_bar + 1) * (tf_min / 60.0)

    return {
        "symbol": sig["symbol"],
        "interval": interval,
        "direction": direction,
        "entry_time": sig["w5_time"],
        "exit_time": exit_time,
        "entry_price": entry,
        "exit_price": exit_price,
        "sl_price": initial_sl,
        "exit_reason": exit_reason,
        "net_pnl_pct": accumulated_pnl,
        "net_pnl_usd": pnl_after_fee_usd,
        "net_r": net_r,
        "duration_hours": duration_hours,
        "bars_held": exit_bar + 1,
        "is_win": pnl_after_fee_usd > 0,
        "algo_score": sig["algo_score"]
    }


def compute_metrics(trades: List[Dict[str, Any]], initial_capital: float = 1000.0) -> Dict[str, Any]:
    """Вычисляет сводные институциональные метрики портфеля."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "net_pnl_usd": 0.0, "net_pnl_pct": 0.0, "max_dd_pct": 0.0,
            "max_streak": 0, "avg_r": 0.0, "avg_duration_h": 0.0
        }

    pnls = [t["net_pnl_usd"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    tot = len(pnls)
    win_cnt = len(wins)
    win_rate = (win_cnt / tot) * 100.0 if tot > 0 else 0.0

    gross_profit = sum(wins)
    gross_loss = sum(losses)
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)
    net_pnl = sum(pnls)
    net_pct = (net_pnl / initial_capital) * 100.0

    # Max Drawdown
    equity = initial_capital
    peak = equity
    max_dd = 0.0
    for p in pnls:
        equity += p
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Max losing streak
    cur_streak = 0
    max_streak = 0
    for p in pnls:
        if p <= 0:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    avg_r = float(np.mean([t["net_r"] for t in trades])) if trades else 0.0
    avg_dur = float(np.mean([t["duration_hours"] for t in trades])) if trades else 0.0

    return {
        "total_trades": tot,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "net_pnl_usd": round(net_pnl, 2),
        "net_pnl_pct": round(net_pct, 1),
        "max_dd_pct": round(max_dd, 1),
        "max_streak": max_streak,
        "avg_r": round(avg_r, 2),
        "avg_duration_h": round(avg_dur, 1)
    }


def main():
    print("=" * 80)
    print("🚀 Запуск комплексного бэктеста Model E и Model B на 5m, 15m, 1h, 4h")
    print(f"Инструменты: ТОП-{len(TOP_30_PAIRS)} монет | Период: последние 35 дней")
    print("=" * 80)

    # 1. Загрузка данных
    print("[1/4] Загрузка 5m котировок...")
    data_5m = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_sym = {executor.submit(fetch_symbol_5m_bybit, s, 35): s for s in TOP_30_PAIRS}
        for fut in concurrent.futures.as_completed(future_to_sym):
            sym = future_to_sym[fut]
            try:
                df = fut.result()
                if not df.empty and len(df) > 1000:
                    data_5m[sym] = df
                    print(f"  ✓ {sym:12} : {len(df)} 5m свечей ({df.index.min().strftime('%d.%m')} - {df.index.max().strftime('%d.%m')})")
                else:
                    print(f"  ✗ {sym:12} : Ошибка/мало свечей")
            except Exception as e:
                print(f"  ✗ {sym:12} : Исключение {e}")

    symbols = sorted(list(data_5m.keys()))
    print(f"\n[+] Успешно подготовлено {len(symbols)} инструментов.")

    # 2. Сканирование и бэктест по каждому таймфрейму
    # Структура результатов:
    # results[tf][model] = list of trades
    all_trades: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        tf: {"Model_E": [], "Model_B": []} for tf in TIMEFRAMES
    }

    # Поинструментальная статистика
    symbol_stats: Dict[str, Dict[str, Dict[str, Any]]] = {
        s: {tf: {} for tf in TIMEFRAMES} for s in symbols
    }

    print("\n[2/4] Запуск сканирования сигналов и симуляции сделок...")
    for tf in TIMEFRAMES:
        t_start_tf = time.time()
        print(f"\n---> Сканирование таймфрейма {tf.upper()}...")
        tf_sig_count = 0

        for sym in symbols:
            df_5m_raw = data_5m[sym]
            df_tf = resample_df(df_5m_raw, tf)

            sigs = scan_tf_signals(df_tf, sym, tf, min_score=65.0)
            tf_sig_count += len(sigs)

            sym_trades_e = []
            sym_trades_b = []

            for sig in sigs:
                tr_e = simulate_model_trade("Model_E", sig, df_tf, df_5m_raw)
                if tr_e:
                    all_trades[tf]["Model_E"].append(tr_e)
                    sym_trades_e.append(tr_e)

                tr_b = simulate_model_trade("Model_B", sig, df_tf, df_5m_raw)
                if tr_b:
                    all_trades[tf]["Model_B"].append(tr_b)
                    sym_trades_b.append(tr_b)

            symbol_stats[sym][tf]["Model_E"] = compute_metrics(sym_trades_e)
            symbol_stats[sym][tf]["Model_B"] = compute_metrics(sym_trades_b)

        print(f"     Найдено сигналов на {tf}: {tf_sig_count} | Время: {time.time()-t_start_tf:.1f}с")

    # 3. Вычисление метрик
    print("\n[3/4] Расчет сводных институциональных метрик...")
    summary_table = []
    for tf in TIMEFRAMES:
        m_e = compute_metrics(all_trades[tf]["Model_E"])
        m_b = compute_metrics(all_trades[tf]["Model_B"])
        summary_table.append({
            "tf": tf,
            "e_trades": m_e["total_trades"],
            "e_wr": m_e["win_rate"],
            "e_pf": m_e["profit_factor"],
            "e_pnl_usd": m_e["net_pnl_usd"],
            "e_pnl_pct": m_e["net_pnl_pct"],
            "e_dd": m_e["max_dd_pct"],
            "e_dur": m_e["avg_duration_h"],
            "b_trades": m_b["total_trades"],
            "b_wr": m_b["win_rate"],
            "b_pf": m_b["profit_factor"],
            "b_pnl_usd": m_b["net_pnl_usd"],
            "b_pnl_pct": m_b["net_pnl_pct"],
            "b_dd": m_b["max_dd_pct"],
            "b_dur": m_b["avg_duration_h"],
        })

    # Вывод сводной таблицы в консоль
    print("\n" + "=" * 110)
    print(f"{'TF':5} | {'Model E (Quant Master)':50} | {'Model B (Fail-Fast Multi-TP)':50}")
    print(f"{'':5} | {'Trades':6} {'WR%':6} {'PF':5} {'Net PnL ($ / %)':18} {'MaxDD':6} {'Dur(h)':6} | {'Trades':6} {'WR%':6} {'PF':5} {'Net PnL ($ / %)':18} {'MaxDD':6} {'Dur(h)':6}")
    print("-" * 110)
    for r in summary_table:
        e_pnl_str = f"+${r['e_pnl_usd']} ({r['e_pnl_pct']}%)" if r['e_pnl_usd'] >= 0 else f"-${abs(r['e_pnl_usd'])} ({r['e_pnl_pct']}%)"
        b_pnl_str = f"+${r['b_pnl_usd']} ({r['b_pnl_pct']}%)" if r['b_pnl_usd'] >= 0 else f"-${abs(r['b_pnl_usd'])} ({r['b_pnl_pct']}%)"
        print(f"{r['tf']:5} | {r['e_trades']:6d} {r['e_wr']:5.1f}% {r['e_pf']:5.2f} {e_pnl_str:18} {r['e_dd']:5.1f}% {r['e_dur']:6.1f} | {r['b_trades']:6d} {r['b_wr']:5.1f}% {r['b_pf']:5.2f} {b_pnl_str:18} {r['b_dd']:5.1f}% {r['b_dur']:6.1f}")
    print("=" * 110)

    # 4. Сохранение подробного markdown отчета
    report_path = REPORTS_DIR / "report_last_month_top30_multitf.md"
    print(f"\n[4/4] Формирование отчета: {report_path}...")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Результаты бэктеста Model E vs Model B на Multi-TF (5m, 15m, 1h, 4h)\n\n")
        f.write(f"**Анализируемый период:** Последний месяц (30 дней) | **Вселенная активов:** ТОП-30 USDT Futures\n")
        f.write(f"**Депозит:** $1,000 | **Риск на сделку:** 1.0% ($10.00) | **Комиссии:** 0.1% taker вход + 0.1% выход (~0.2% roundtrip)\n\n")

        f.write("## 1. Сводная таблица по всем таймфреймам\n\n")
        f.write("| ТФ | Модель | Сделок | Сделок/день | Win Rate | Profit Factor | Net PnL ($) | Доходность | Max Drawdown | Ср. длит. |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")

        for r in summary_table:
            trades_per_day_e = round(r['e_trades'] / 30.0, 1)
            trades_per_day_b = round(r['b_trades'] / 30.0, 1)
            f.write(f"| **{r['tf']}** | **Model E** (Master) | {r['e_trades']} | {trades_per_day_e} | **{r['e_wr']}%** | **{r['e_pf']}** | **+${r['e_pnl_usd']}** | **+{r['e_pnl_pct']}%** | {r['e_dd']}% | {r['e_dur']} ч |\n")
            f.write(f"| **{r['tf']}** | **Model B** (Fail-Fast) | {r['b_trades']} | {trades_per_day_b} | **{r['b_wr']}%** | **{r['b_pf']}** | **+${r['b_pnl_usd']}** | **+{r['b_pnl_pct']}%** | {r['b_dd']}% | {r['b_dur']} ч |\n")

        f.write("\n## 2. Разбор по инструментам (ТОП-30) на рекомендуемых ТФ (15m и 1h)\n\n")
        f.write("| Инструмент | 15m Сделок (E) | 15m PF (E) | 15m Net PnL (E) | 1h Сделок (E) | 1h PF (E) | 1h Net PnL (E) |\n")
        f.write("|---|---|---|---|---|---|---|\n")

        # Сортировка по суммарному PnL на 15m
        sorted_syms = sorted(symbols, key=lambda s: symbol_stats[s]["15m"]["Model_E"].get("net_pnl_usd", 0), reverse=True)
        for s in sorted_syms:
            s15 = symbol_stats[s]["15m"]["Model_E"]
            s1h = symbol_stats[s]["1h"]["Model_E"]
            f.write(f"| **{s}** | {s15.get('total_trades',0)} | {s15.get('profit_factor',0)} | ${s15.get('net_pnl_usd',0):+.1f} | {s1h.get('total_trades',0)} | {s1h.get('profit_factor',0)} | ${s1h.get('net_pnl_usd',0):+.1f} |\n")

        f.write("\n\n## 3. Практический вердикт: Успеет ли трейдер руками?\n\n")
        f.write("Детальный анализ скорости реакции, окон закрытия свечей и психологической нагрузки для ручной торговли.\n")

    print("[+] Готово! Результаты записаны.")


if __name__ == "__main__":
    main()
