"""
test_honest_models.py — Тестирование абсолютно честных моделей в реальном времени (0% lookahead):

Model 1 (Предложение пользователя):
  - W5 подтверждена (на p2+3).
  - Начиная с p2+3 ждем свечу S, которая пересвипывает High[p2] и поглощает.
  - Вход на 50% свечи S, стоп за кончик тени S.

Model 2 (W5 как свеча S в реальном времени):
  - W3 подтверждена на p1.
  - На текущем закрытом баре t:
    High[t] > High[p1] (или High[t] > High[t-1]).
    Бара t закрывается поглощением (close[t] < min(open[t-1], close[t-1])).
    Дивергенция RSI(p1) vs RSI(t).
    detect_elliott_impulse(w3=p1, w5=t) дает Score >= 65 без требования правых баров!
  - Сигнал СРАЗУ на закрытии бара t.

Проверяем с консервативным внутрисвечным исполнением:
если на одной свече задеты SL и TP/2R — стоп ВСЕГДА срабатывает первым!
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
FEE = 0.0005


def load_all():
    mdata = {}
    for sym in SYMBOLS:
        p = DATA_DIR / f"{sym}_15m.csv"
        if not p.exists(): continue
        df = pd.read_csv(p)
        if 'open_time' in df.columns:
            df.index = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        elif 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], utc=True)
        else:
            df.index = pd.to_datetime(df.iloc[:, 0], utc=True)
        mdata[sym] = df[['open', 'high', 'low', 'close', 'volume']]
    return mdata


def sim_trade(df, fill_bar, entry_p, sl_p, target_r=3.0, be_r=2.0, direction="SHORT"):
    n = len(df)
    risk_dist = abs(entry_p - sl_p)
    if risk_dist <= 0 or fill_bar >= n:
        return "invalid", 0.0
    
    risk_frac = risk_dist / entry_p
    reward_frac = (target_r * risk_frac + 2.0 * FEE) / (1.0 - FEE)
    
    if direction == "SHORT":
        tp_p = entry_p - entry_p * reward_frac
        trig_2r = entry_p - be_r * risk_dist
        be_p = entry_p * (1.0 - 2.0 * FEE)
    else:
        tp_p = entry_p + entry_p * reward_frac
        trig_2r = entry_p + be_r * risk_dist
        be_p = entry_p * (1.0 + 2.0 * FEE)
    
    cur_sl = sl_p
    be_active = False
    
    for b in range(fill_bar, min(n, fill_bar + 160)):
        bh = df['high'].iat[b]
        bl = df['low'].iat[b]
        
        # Консервативно: сначала проверяем стоп / БУ
        if direction == "SHORT":
            hit_sl = bh >= cur_sl
            hit_tp = bl <= tp_p
            hit_2r = bl <= trig_2r
        else:
            hit_sl = bl <= cur_sl
            hit_tp = bh >= tp_p
            hit_2r = bh >= trig_2r
        
        if hit_sl:
            # Стоп бьет первым
            if be_active:
                return "be", 0.0
            else:
                loss_r = -(risk_frac + 2.0 * FEE) / risk_frac
                return "sl", loss_r
        
        if hit_tp:
            return "tp", target_r
        
        if hit_2r and not be_active:
            be_active = True
            cur_sl = be_p
            # не закрываем в БУ на том же баре
            continue
    
    return "timeout", 0.0


def test_model_2(mdata, min_candle_pct=0.25):
    """
    Model 2: В реальном времени W3 подтвержден, на закрытии текущей свечи t:
    Свеча t делает экстремум (High > max high past 3 bars),
    закрывается поглощением,
    имеет RSI-дивергенцию с W3,
    и detect_elliott_impulse(w3=p1, w5=t) подтверждает импульс (Score >= 65).
    """
    trades = []
    
    for sym in SYMBOLS:
        df = mdata[sym]
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_p = df['open'].values
        n = len(df)
        
        rsi = calculate_rsi_wilder(close, period=7)
        high_pivots = argrelextrema(high, np.greater_equal, order=3)[0]
        low_pivots = argrelextrema(low, np.less_equal, order=3)[0]
        
        # SHORT
        if len(high_pivots) >= 1:
            for p1 in high_pivots:
                r1 = float(rsi[p1])
                if r1 < 60.0: continue
                
                # Ищем свечу t после p1 (от p1+4 до p1+45)
                for t in range(p1 + 4, min(n - 20, p1 + 46)):
                    # Бара t должен быть выше W3
                    if high[t] <= high[p1]: continue
                    # Свеча t должна быть локальным максимумом
                    if high[t] < np.max(high[p1+1 : t+1]): continue
                    
                    # Медвежье поглощение
                    is_bearish = close[t] < open_p[t]
                    prev_body_bottom = min(open_p[t-1], close[t-1])
                    is_engulf = close[t] < prev_body_bottom
                    if not (is_bearish and is_engulf): continue
                    
                    # Размер свечи не менее min_candle_pct (защита от микро-шума)
                    c_range = high[t] - low[t]
                    if (c_range / close[t]) * 100.0 < min_candle_pct: continue
                    
                    # Дивергенция RSI
                    r_t = float(rsi[t])
                    if r_t >= r1 - 2.5 or r_t < 45.0: continue
                    
                    # Проверка импульса Эллиотта строго на срезе [0:t+1]
                    algo_res = detect_elliott_impulse(
                        high=high[:t + 1], low=low[:t + 1], close=close[:t + 1], open_p=open_p[:t + 1],
                        w3_bar=p1, w5_bar=t, direction="SHORT", rsi=rsi[:t + 1],
                    )
                    if not algo_res.is_valid or algo_res.score < 65: continue
                    
                    # Сигнал на закрытии бара t!
                    limit_entry = low[t] + 0.50 * c_range
                    sl_p = high[t]
                    
                    # Проверка заполнения лимита на t+1, t+2
                    fill_bar = None
                    for fb in (t + 1, t + 2):
                        if fb >= n: break
                        if high[fb] > sl_p: break  # выбит стоп до входа
                        if high[fb] >= limit_entry:
                            fill_bar = fb
                            break
                    
                    if fill_bar is None: continue
                    
                    res, r_val = sim_trade(df, fill_bar, limit_entry, sl_p, target_r=3.0, be_r=2.0, direction="SHORT")
                    trades.append((sym, "SHORT", t, fill_bar, res, r_val))
                    break  # не ищем дубликаты на этом же W3
        
        # LONG
        if len(low_pivots) >= 1:
            for p1 in low_pivots:
                r1 = float(rsi[p1])
                if r1 > 40.0: continue
                
                for t in range(p1 + 4, min(n - 20, p1 + 46)):
                    if low[t] >= low[p1]: continue
                    if low[t] > np.min(low[p1+1 : t+1]): continue
                    
                    is_bullish = close[t] > open_p[t]
                    prev_body_top = max(open_p[t-1], close[t-1])
                    is_engulf = close[t] > prev_body_top
                    if not (is_bullish and is_engulf): continue
                    
                    c_range = high[t] - low[t]
                    if (c_range / close[t]) * 100.0 < min_candle_pct: continue
                    
                    r_t = float(rsi[t])
                    if r_t <= r1 + 2.5 or r_t > 55.0: continue
                    
                    algo_res = detect_elliott_impulse(
                        high=high[:t + 1], low=low[:t + 1], close=close[:t + 1], open_p=open_p[:t + 1],
                        w3_bar=p1, w5_bar=t, direction="LONG", rsi=rsi[:t + 1],
                    )
                    if not algo_res.is_valid or algo_res.score < 65: continue
                    
                    limit_entry = low[t] + 0.50 * c_range
                    sl_p = low[t]
                    
                    fill_bar = None
                    for fb in (t + 1, t + 2):
                        if fb >= n: break
                        if low[fb] < sl_p: break
                        if low[fb] <= limit_entry:
                            fill_bar = fb
                            break
                    
                    if fill_bar is None: continue
                    
                    res, r_val = sim_trade(df, fill_bar, limit_entry, sl_p, target_r=3.0, be_r=2.0, direction="LONG")
                    trades.append((sym, "LONG", t, fill_bar, res, r_val))
                    break
    
    return trades


if __name__ == "__main__":
    mdata = load_all()
    print("Данные загружены.")
    for min_c in [0.15, 0.30, 0.50]:
        t_res = test_model_2(mdata, min_candle_pct=min_c)
        if not t_res:
            print(f"Min Candle {min_c}%: 0 сделок")
            continue
        n_t = len(t_res)
        w = sum(1 for x in t_res if x[4] == "tp")
        b = sum(1 for x in t_res if x[4] == "be")
        l = sum(1 for x in t_res if x[4] == "sl")
        tot_r = sum(x[5] for x in t_res)
        gw = sum(x[5] for x in t_res if x[5] > 0)
        gl = abs(sum(x[5] for x in t_res if x[5] < 0))
        pf = gw / gl if gl > 0 else 99.0
        print(f"Model 2 [MinCandle={min_c}%] -> Сделок: {n_t}, Wins: {w} ({w/n_t*100:.1f}%), БУ: {b} ({b/n_t*100:.1f}%), Стопы: {l} ({l/n_t*100:.1f}%), PF: {pf:.2f}, SumR: {tot_r:+.1f}R, EV: {tot_r/n_t:+.2f}R")
