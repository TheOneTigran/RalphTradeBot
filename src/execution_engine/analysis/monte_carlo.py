"""
monte_carlo.py — Анализ устойчивости (Симуляция Монте-Карло).

1. Block Bootstrap: Рандомизация сделок блоками по 5 штук (защита от IID Fallacy).
2. Расчет вероятности просадки (Risk of Ruin >= 20%).
3. Стресс-тест исполнения: искусственное ухудшение PnL за счёт скрытых комиссий/спреда.
"""

import math
import random
import pandas as pd
import numpy as np


def compute_profit_factor(pnls: list) -> float:
    gross_profit = sum([p for p in pnls if p > 0])
    gross_loss = abs(sum([p for p in pnls if p < 0]))
    if gross_loss == 0:
        return 999.0
    return gross_profit / gross_loss


def calculate_max_drawdown(pnls: list, starting_balance: float = 10000.0) -> float:
    """Возвращает максимальную просадку в процентах по серии PnL."""
    balance = starting_balance
    peak = starting_balance
    max_dd = 0.0
    
    for p in pnls:
        balance += p
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
            
    return max_dd


def block_bootstrap(pnls: list, block_size: int = 5) -> list:
    """Перемешивает массив сохраняя кластеры зависимых сделок (убыточных серий)."""
    n = len(pnls)
    if n <= block_size:
        # Если данных мало, возвращаем как есть или просто шаффлим
        return random.sample(pnls, n)
        
    num_blocks = math.ceil(n / block_size)
    resampled = []
    
    for _ in range(num_blocks):
        # Случайная стартовая точка для блока
        start_idx = random.randint(0, n - block_size)
        resampled.extend(pnls[start_idx:start_idx + block_size])
        
    # Обрезаем если блоков получилось чуть больше из-за ceil
    return resampled[:n]


def run_monte_carlo(csv_path: str, starting_balance: float = 10000.0, iterations: int = 10000, risk_ruin_threshold: float = 20.0):
    print(f"--- Симуляция Монте-Карло ({iterations} прогонов) ---")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Ошибка: Не найден файл {csv_path}")
        return
        
    if df.empty or 'PnL_USD' not in df.columns:
        print("Недостаточно данных для Монте-Карло симуляции.")
        return
        
    # Только закрытые сделки
    df = df[df["Status"].isin(["FILLED_TP", "FILLED_SL"])]
    original_pnls = df["PnL_USD"].tolist()
    
    if not original_pnls:
        print("Нет закрытых сделок (TP/SL) для симуляции.")
        return
        
    # --- 1. Block Bootstrap ---
    ruin_count = 0
    max_dd_list = []
    
    for _ in range(iterations):
        simulated_pnls = block_bootstrap(original_pnls, block_size=5)
        sim_dd = calculate_max_drawdown(simulated_pnls, starting_balance)
        max_dd_list.append(sim_dd)
        
        if sim_dd >= risk_ruin_threshold:
            ruin_count += 1
            
    avg_sim_dd = sum(max_dd_list) / iterations
    ruin_prob = (ruin_count / iterations) * 100.0
    
    print("\n[ Block Bootstrap Итоги ]")
    print(f"Средняя симулированная Макс. Просадка: {avg_sim_dd:.2f}%")
    print(f"Вероятность просадки > {risk_ruin_threshold}% (Risk of Ruin): {ruin_prob:.2f}%")
    if ruin_prob > 5.0:
        print("❌ СТРАТЕГИЯ СЛИШКОМ РИСКОВАННАЯ (>5% шанс сильной просадки)")
    else:
        print("✅ РИСК RUIN В ПРЕДЕЛАХ НОРМЫ")
        
        
    # --- 2. Stress Test (Slippage Penalty) ---
    print("\n[ Имитация Жесткого Спреда / Проскальзывания ]")
    pf_original = compute_profit_factor(original_pnls)
    print(f"Оригинальный Profit Factor: {pf_original:.2f}")
    
    # Имитация: ухудшение прибыли на 0.1% от цены, увеличение убытков на 0.1% от цены.
    # Так как PnL_USD в CSV уже записан в абсолютном значении, сделаем грубое пенальти в % от самого PnL
    # Но более точный стресс-тест - это вычесть размер проскальзывания из входа и выхода.
    # Entry_Price * 0.001 * Position_Size.
    
    penalized_pnls = []
    try:
        for idx, row in df.iterrows():
            entry = float(row['Entry_Price'])
            size = float(row['Position_Size'])
            original_pnl = float(row['PnL_USD'])
            
            # Пенальти: 0.1% при входе + 0.1% при выходе = 0.2% от нотионального объема
            notional = entry * size
            penalty_usd = notional * 0.002
            
            penalized_pnls.append(original_pnl - penalty_usd)
            
        pf_stressed = compute_profit_factor(penalized_pnls)
        print(f"Стресс Profit Factor (0.1% penalty in/out): {pf_stressed:.2f}")
        
        if pf_stressed > 1.2:
            print("✅ СТРАТЕГИЯ ВЫЖИВАЕТ НА РЕАЛЬНОМ РЫНКЕ С КОМИССИЯМИ")
        else:
            print("❌ СТРАТЕГИЯ ПОГИБНЕТ ОТ СПРЕДОВ/КОМИССИЙ (Stressed PF < 1.2)")
    except Exception as e:
        print(f"Не удалось рассчитать стресс-спред по колонкам CSV: {e}")

if __name__ == "__main__":
    run_monte_carlo("backtest_results.csv")
