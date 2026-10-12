"""
test_w5_resweep_engulf.py — Тестирование модели пересвипа 5-й волны:
1. Ищем импульс W0..W5 с RSI-дивергенцией по продакшен-логике (Score >= 65).
   Вершина W5 подтверждена на баре p2 (через order=3, то есть на баре p2+3).
2. Начиная с p2+3 мониторим свечи t (в окне от p2+3 до p2+max_bars):
   - Свеча t пересвипывает экстремум W5 (High[t] > High[p2] для SHORT / Low[t] < Low[p2] для LONG).
   - Свеча t закрывается поглощением (bearish engulfing для SHORT / bullish engulfing для LONG).
3. На ЗАКРЫТИИ свечи t даем сигнал:
   - Лимит на 50% свечи t (окно 1-2 свечи: t+1, t+2).
   - Стоп за кончик тени свечи t (без буфера).
   - Тейк 3.0R Net, БУ на +2.0R Net.
   - Консервативная симуляция (SL бьет первым при конфликте внутри свечи).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"c:\Users\user\Desktop\RalphTradeBot")
sys.path.insert(0, str(BASE_DIR))

from signal_scanner import calculate_rsi_wilder
from elliott_detector import detect_elliott_impulse

DATA_DIR = Path(r"C:\Users\user\Desktop\данные для бектестов\2025_2026_Full_Top_Coins")
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'NEARUSDT',
    'SUIUSDT', 'AAVEUSDT', 'APTUSDT', 'DOTUSDT', 'RENDERUSDT',
]

FEE = 0.0005  # 0.05% taker/maker


def load_data():
    mdata = {}
    for sym in SYMBOLS:
        p = DATA_DIR / f"{sym}_15m.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if 'open_time' in df.columns:
            df.index = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        elif 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], utc=True)
        else:
            df.index = pd.to_datetime(df.iloc[:, 0], utc=True)
        mdata[sym] = df[['open', 'high', 'low', 'close', 'volume']]
    return mdata


def run_experiment(max_sweep_bars_after_w5=10, retrace_ratio=0.50):
    mdata = load_data()
    print(f"Загружено монет: {len(mdata)}")
    
    total_impulses = 0
    total_sweeps_found = 0
    total_fills = 0
    
    trades = []
    
    for sym in SYMBOLS:
        df = mdata[sym]
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        close = df['close'].values.astype(np.float64)
        open_p = df['open'].values.astype(np.float64)
        n = len(df)
        
        rsi = calculate_rsi_wilder(close, period=7)
        
        high_pivots = argrelextrema(high, np.greater_equal, order=3)[0]
        low_pivots = argrelextrema(low, np.less_equal, order=3)[0]
        
        # 1. SHORT: Ищем импульсы вверх с подтвержденным W5 (p2)
        if len(high_pivots) >= 2:
            for i in range(1, len(high_pivots)):
                p1 = int(high_pivots[i - 1])
                p2 = int(high_pivots[i])
                
                # В реальном времени p2 подтверждается только на закрытии бара p2 + 3!
                conf_bar = p2 + 3
                if conf_bar >= n - 20:
                    continue
                
                dist = p2 - p1
                if dist < 4 or dist > 45:
                    continue
                if high[p2] <= high[p1]:
                    continue
                
                r1, r2 = float(rsi[p1]), float(rsi[p2])
                if r2 >= r1 or (r1 - r2) < 2.5:
                    continue
                if r1 < 60.0 or r2 < 45.0:
                    continue
                if len(high[p1 + 1 : p2]) > 0 and np.max(high[p1 + 1 : p2]) > high[p2]:
                    continue
                if len(rsi[p1 + 1 : p2]) > 0 and np.min(rsi[p1 + 1 : p2]) < 30.0:
                    continue
                
                # Проверка импульса до p2
                algo_res = detect_elliott_impulse(
                    high=high[:p2 + 1], low=low[:p2 + 1], close=close[:p2 + 1], open_p=open_p[:p2 + 1],
                    w3_bar=p1, w5_bar=p2, direction="SHORT", rsi=rsi[:p2 + 1],
                )
                if not algo_res.is_valid or algo_res.score < 65:
                    continue
                
                total_impulses += 1
                w5_high = high[p2]
                
                # Мониторим свечи начиная с момента подтверждения (p2 + 3)
                # или начиная со свечей после p2
                # Ищем свечу S, которая пересвипнула W5: High[t] > w5_high
                sweep_bar = None
                for t in range(conf_bar, min(n - 20, p2 + max_sweep_bars_after_w5 + 1)):
                    # Проверяем, не ушла ли цена слишком глубоко вниз (отмена структуры)
                    # Если пробито начало волны 4, импульс сломан
                    w4_p = algo_res.wave_points.get("W4", low[p2])
                    if low[t] < w4_p:
                        break  # слом структуры
                    
                    # Условие пересвипа: свеча t пробивает High W5
                    if high[t] > w5_high:
                        # Условие поглощения: свеча медвежья и закрывается ниже тела предыдущей свечи
                        prev_body_bottom = min(open_p[t - 1], close[t - 1])
                        is_bearish = close[t] < open_p[t]
                        is_engulf = close[t] < prev_body_bottom
                        
                        if is_bearish and is_engulf:
                            sweep_bar = t
                            break
                        else:
                            # Если свеча пробила High W5, но закрылась бычьей (продолжение тренда) —
                            # она обновила максимум W5, и если дальше нет поглощения, отменяем
                            w5_high = high[t]
                
                if sweep_bar is None:
                    continue
                
                total_sweeps_found += 1
                
                # Свеча S найдена на баре sweep_bar!
                # Сигнал генерируется на ЗАКРЫТИИ sweep_bar (start_bar = sweep_bar + 1)
                s_high = high[sweep_bar]
                s_low = low[sweep_bar]
                s_range = s_high - s_low
                if s_range <= 0:
                    continue
                
                limit_entry = s_low + retrace_ratio * s_range
                sl_price = s_high
                risk_dist = sl_price - limit_entry
                if risk_dist <= 0:
                    continue
                
                risk_frac = risk_dist / limit_entry
                reward_frac = (3.0 * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp_price = limit_entry - limit_entry * reward_frac
                
                # Trigger +2R Net
                trigger_2r = limit_entry - 2.0 * risk_dist
                be_price = limit_entry * (1.0 - 2.0 * FEE)
                
                # Проверяем исполнение лимита на барах sweep_bar+1 и sweep_bar+2
                fill_bar = None
                for fb in (sweep_bar + 1, sweep_bar + 2):
                    if fb >= n:
                        break
                    # Если до входа пробит стоп — отмена
                    if high[fb] > sl_price:
                        break
                    if high[fb] >= limit_entry:
                        fill_bar = fb
                        break
                
                if fill_bar is None:
                    continue
                
                total_fills += 1
                
                # Симуляция позиции с консервативным intrabar правилом
                be_active = False
                cur_sl = sl_price
                exit_res = "timeout"
                exit_bar = fill_bar
                
                for b in range(fill_bar, min(n, fill_bar + 160)):
                    bh, bl = high[b], low[b]
                    
                    # Консервативно: сначала проверяем стоп / БУ!
                    if bh >= cur_sl:
                        exit_res = "be" if be_active else "sl"
                        exit_bar = b
                        break
                    
                    if bl <= tp_price:
                        exit_res = "tp"
                        exit_bar = b
                        break
                    
                    if not be_active and bl <= trigger_2r:
                        be_active = True
                        cur_sl = be_price
                
                # Расчет R
                if exit_res == "tp":
                    r_val = 3.0
                elif exit_res == "be":
                    r_val = 0.0
                elif exit_res == "sl":
                    r_val = -(risk_frac + 2.0 * FEE) / risk_frac
                else:
                    r_val = 0.0
                
                trades.append({
                    "symbol": sym,
                    "direction": "SHORT",
                    "p2_bar": p2,
                    "sweep_bar": sweep_bar,
                    "fill_bar": fill_bar,
                    "entry_time": df.index[fill_bar],
                    "entry_price": limit_entry,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "exit_res": exit_res,
                    "r_val": r_val,
                })
        
        # 2. LONG: Ищем импульсы вниз с подтвержденным W5 (p2)
        if len(low_pivots) >= 2:
            for i in range(1, len(low_pivots)):
                p1 = int(low_pivots[i - 1])
                p2 = int(low_pivots[i])
                
                conf_bar = p2 + 3
                if conf_bar >= n - 20:
                    continue
                
                dist = p2 - p1
                if dist < 4 or dist > 45:
                    continue
                if low[p2] >= low[p1]:
                    continue
                
                r1, r2 = float(rsi[p1]), float(rsi[p2])
                if r2 <= r1 or (r2 - r1) < 2.5:
                    continue
                if r1 > 40.0 or r2 > 55.0:
                    continue
                if len(low[p1 + 1 : p2]) > 0 and np.min(low[p1 + 1 : p2]) < low[p2]:
                    continue
                if len(rsi[p1 + 1 : p2]) > 0 and np.max(rsi[p1 + 1 : p2]) > 70.0:
                    continue
                
                algo_res = detect_elliott_impulse(
                    high=high[:p2 + 1], low=low[:p2 + 1], close=close[:p2 + 1], open_p=open_p[:p2 + 1],
                    w3_bar=p1, w5_bar=p2, direction="LONG", rsi=rsi[:p2 + 1],
                )
                if not algo_res.is_valid or algo_res.score < 65:
                    continue
                
                total_impulses += 1
                w5_low = low[p2]
                
                sweep_bar = None
                for t in range(conf_bar, min(n - 20, p2 + max_sweep_bars_after_w5 + 1)):
                    w4_p = algo_res.wave_points.get("W4", high[p2])
                    if high[t] > w4_p:
                        break
                    
                    if low[t] < w5_low:
                        prev_body_top = max(open_p[t - 1], close[t - 1])
                        is_bullish = close[t] > open_p[t]
                        is_engulf = close[t] > prev_body_top
                        
                        if is_bullish and is_engulf:
                            sweep_bar = t
                            break
                        else:
                            w5_low = low[t]
                
                if sweep_bar is None:
                    continue
                
                total_sweeps_found += 1
                
                s_high = high[sweep_bar]
                s_low = low[sweep_bar]
                s_range = s_high - s_low
                if s_range <= 0:
                    continue
                
                limit_entry = s_low + retrace_ratio * s_range
                sl_price = s_low
                risk_dist = limit_entry - sl_price
                if risk_dist <= 0:
                    continue
                
                risk_frac = risk_dist / limit_entry
                reward_frac = (3.0 * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp_price = limit_entry + limit_entry * reward_frac
                
                trigger_2r = limit_entry + 2.0 * risk_dist
                be_price = limit_entry * (1.0 + 2.0 * FEE)
                
                fill_bar = None
                for fb in (sweep_bar + 1, sweep_bar + 2):
                    if fb >= n:
                        break
                    if low[fb] < sl_price:
                        break
                    if low[fb] <= limit_entry:
                        fill_bar = fb
                        break
                
                if fill_bar is None:
                    continue
                
                total_fills += 1
                
                be_active = False
                cur_sl = sl_price
                exit_res = "timeout"
                exit_bar = fill_bar
                
                for b in range(fill_bar, min(n, fill_bar + 160)):
                    bh, bl = high[b], low[b]
                    
                    if bl <= cur_sl:
                        exit_res = "be" if be_active else "sl"
                        exit_bar = b
                        break
                    
                    if bh >= tp_price:
                        exit_res = "tp"
                        exit_bar = b
                        break
                    
                    if not be_active and bh >= trigger_2r:
                        be_active = True
                        cur_sl = be_price
                
                if exit_res == "tp":
                    r_val = 3.0
                elif exit_res == "be":
                    r_val = 0.0
                elif exit_res == "sl":
                    r_val = -(risk_frac + 2.0 * FEE) / risk_frac
                else:
                    r_val = 0.0
                
                trades.append({
                    "symbol": sym,
                    "direction": "LONG",
                    "p2_bar": p2,
                    "sweep_bar": sweep_bar,
                    "fill_bar": fill_bar,
                    "entry_time": df.index[fill_bar],
                    "entry_price": limit_entry,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "exit_res": exit_res,
                    "r_val": r_val,
                })
    
    print(f"\n--- РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА (Окно мониторинга: {max_sweep_bars_after_w5} баров, Вход: {retrace_ratio*100:.0f}%) ---")
    print(f"Всего импульсов W0..W5: {total_impulses}")
    print(f"Найдено пересвипов W5 с поглощением: {total_sweeps_found}")
    print(f"Исполнено лимитов: {total_fills}")
    
    if not trades:
        print("Сделок не найдено.")
        return
    
    df_res = pd.DataFrame(trades)
    wins = len(df_res[df_res['exit_res'] == 'tp'])
    bes = len(df_res[df_res['exit_res'] == 'be'])
    sls = len(df_res[df_res['exit_res'] == 'sl'])
    to = len(df_res[df_res['exit_res'] == 'timeout'])
    
    tot_r = df_res['r_val'].sum()
    ev_r = tot_r / len(df_res)
    
    gw = df_res[df_res['r_val'] > 0]['r_val'].sum()
    gl = abs(df_res[df_res['r_val'] < 0]['r_val'].sum())
    pf = (gw / gl) if gl > 0 else 99.0
    
    print(f"Сделок: {len(df_res)}")
    print(f"Тейки (+3R): {wins} ({wins/len(df_res)*100:.1f}%)")
    print(f"Безубытки (0R): {bes} ({bes/len(df_res)*100:.1f}%)")
    print(f"Стопы (-1R..-1.2R): {sls} ({sls/len(df_res)*100:.1f}%)")
    print(f"Таймауты: {to}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Сумма R: {tot_r:.2f} R")
    print(f"EV на сделку: {ev_r:.3f} R")

if __name__ == "__main__":
    for window in [6, 10, 15]:
        run_experiment(max_sweep_bars_after_w5=window, retrace_ratio=0.50)
