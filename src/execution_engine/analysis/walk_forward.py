"""
walk_forward.py — Walk-Forward Analysis (WFA) симулятор.

Предотвращает переобучение (Curve Fitting). Оптимизирует параметры на In-Sample
(IS) окне длиной в 4+ месяца и запускает лучшие параметры на слепом OOS окне.
Целевая функция: Recovery Factor (PnL / Max Drawdown).
"""

import os
import itertools
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

from src.wave_strategy.pipeline import run_wave_strategy
from src.execution_engine.backtester import run_backtest

# Сетка параметров для оптимизации
PARAM_GRID = {
    "entry_atr_mult": [0.2, 0.5, 0.8],
    "sl_atr_mult": [0.5, 0.8, 1.2],
}


def create_folds(candles: List[Dict], is_months: int = 4, oos_months: int = 1) -> List[dict]:
    """Разбивает массив свечей на IS и OOS окна."""
    # Конвертируем ts в datetime
    df = pd.DataFrame(candles)
    if 'ts' in df.columns:
        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        raise ValueError("No timestamp column found")
        
    df = df.sort_values('datetime').reset_index(drop=True)
    
    start_date = df['datetime'].iloc[0]
    end_date = df['datetime'].iloc[-1]
    
    folds = []
    current_start = start_date
    
    while True:
        is_end = current_start + pd.DateOffset(months=is_months)
        oos_end = is_end + pd.DateOffset(months=oos_months)
        
        if oos_end > end_date:
            break
            
        fold = {
            'is_mask': (df['datetime'] >= current_start) & (df['datetime'] < is_end),
            'oos_mask': (df['datetime'] >= is_end) & (df['datetime'] < oos_end),
            'is_start_dt': current_start,
            'is_end_dt': is_end,
            'oos_start_dt': is_end,
            'oos_end_dt': oos_end
        }
        folds.append(fold)
        
        # Сдвиг окна на OOS (т.е на 1 месяц)
        current_start += pd.DateOffset(months=oos_months)
        
    return folds, df


def run_wfa(
    symbol: str, 
    candles: List[Dict], 
    macro_tfs: List[str] = ["1h"], 
    micro_tfs: List[str] = ["15m"]
):
    print(f"--- Начало Walk Forward Analysis ({symbol}) ---")
    folds, df_master = create_folds(candles, is_months=4, oos_months=1)
    
    if not folds:
        print("Недостаточно данных для формирования даже одного фолда WFA (нужно > 5 месяцев).")
        return
        
    print(f"Найдено фолдов: {len(folds)} (IS=4m, OOS=1m)")
    
    # Генерация всех комбинаций параметров
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Сеток параметров для прогона в IS: {len(combinations)}")
    
    all_oos_stats = []
    
    for idx, fold in enumerate(folds):
        print(f"\n[Fold {idx+1}/{len(folds)}] IS: {fold['is_start_dt'].date()} -> {fold['is_end_dt'].date()} | OOS: {fold['oos_start_dt'].date()} -> {fold['oos_end_dt'].date()}")
        
        is_candles = df_master[fold['is_mask']].to_dict('records')
        oos_candles = df_master[fold['oos_mask']].to_dict('records')
        
        best_rf = -9999
        best_params = None
        best_stats = None
        
        # 1. OPTIMIZATION ON IN-SAMPLE
        for params in combinations:
            print(f"  Оптимизация IS с параметрами: {params}...")
            # В реальности scanner лучше закэшировать, но для TDD оставляем так
            candles_by_tf = {tf: is_candles for tf in macro_tfs + micro_tfs}
            
            plans = run_wave_strategy(
                symbol=symbol,
                candles_by_tf=candles_by_tf,
                macro_tfs=macro_tfs,
                micro_tfs=micro_tfs,
                min_rr=1.5,
                entry_atr_mult=params['entry_atr_mult'],
                sl_atr_mult=params['sl_atr_mult'],
                verbose=False
            )
            
            if not plans:
                continue
                
            stats = run_backtest(plans, is_candles, csv_output_path="null_is.csv")
            
            pnl = stats["Final Balance"] - 10000.0  # Уязвимость: берем стартовый как константу
            max_dd = stats["Max Drawdown %"]
            
            # The Objective Function: Recovery Factor
            if max_dd <= 0:
                rf = pnl # Если нет просадки (идеально), просто берем PnL
            else:
                rf = pnl / max_dd
                
            if rf > best_rf:
                best_rf = rf
                best_params = params
                best_stats = stats
                
        if not best_params:
            print(f"  В IS ({fold['is_start_dt'].date()}) не найдено ни одной профитной настройки.")
            continue
            
        print(f"  > Лучшие настройки IS: {best_params} (Recovery Factor = {best_rf:.2f})")
        
        # 2. OUT OF SAMPLE BLIND TEST
        print(f"  > Слепой тест на OOS ({fold['oos_start_dt'].date()})...")
        candles_by_tf_oos = {tf: oos_candles for tf in macro_tfs + micro_tfs}
        
        oos_plans = run_wave_strategy(
            symbol=symbol,
            candles_by_tf=candles_by_tf_oos,
            macro_tfs=macro_tfs,
            micro_tfs=micro_tfs,
            min_rr=1.5,
            entry_atr_mult=best_params['entry_atr_mult'],
            sl_atr_mult=best_params['sl_atr_mult'],
            verbose=False
        )
        
        oos_stats = run_backtest(oos_plans, oos_candles, csv_output_path=f"oos_fold_{idx}.csv")
        
        pnl_oos = oos_stats["Final Balance"] - 10000.0
        print(f"  > Результат OOS: PnL = {pnl_oos:.2f} USD, WinRate = {oos_stats['Win Rate %']}%")
        
        all_oos_stats.append({
            "fold_idx": idx,
            "best_params": best_params,
            "oos_pnl": pnl_oos,
            "oos_dd": oos_stats["Max Drawdown %"],
            "oos_trades": oos_stats["Total Trades"]
        })
        
    print("\n=== Итоги Walk-Forward Analysis ===")
    total_oos_pnl = sum([s["oos_pnl"] for s in all_oos_stats])
    total_oos_trades = sum([s["oos_trades"] for s in all_oos_stats])
    print(f"Суммарный слепой профит (OOS): {total_oos_pnl:.2f} USD")
    print(f"Всего слепых сделок: {total_oos_trades}")
    if total_oos_pnl > 0:
        print("✅ СТРАТЕГИЯ УСТОЙЧИВА К РЫНКУ (Прибыль на OOS подтверждена)")
    else:
        print("❌ СТРАТЕГИЯ ПЕРЕОБУЧИЛАСЬ (Убыток на OOS)")
