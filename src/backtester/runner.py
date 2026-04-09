"""
runner.py — Главный модуль запуска бэктестинга.
"""
import sys
import os
# Добавляем корень проекта в пути поиска модулей
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
import argparse
from datetime import datetime, timezone
import os
import pandas as pd
from typing import List, Dict

from src.backtester.data_manager import fetch_all_needed_history
from src.backtester.time_machine import TimeMachine
from src.backtester.evaluator import evaluate_plan, BacktestResult
from src.core.pipeline import run_full_ai_pipeline, run_math_pipeline
from src.math_engine.math_preprocessor import preprocess_all
from src.core.models import LLMContext
from src.core.config import MAX_LEVERAGE,  CANDLE_LIMIT, INITIAL_DEPOSIT, RISK_PER_TRADE_PERCENT, MIN_RR_RATIO, EXCHANGE_FEE
from src.validator.hard_validator import validate_plan as hard_validate_plan
from src.utils.visualizer import plot_wave_structure

# Настройка логирования для бэктеста (более компактная)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("backtester")

TF_MS = {
    "1d": 86400000,
    "4h": 14400000,
    "1h": 3600000,
    "15m": 900000,
    "5m": 300000
}

def _extract_price(text) -> float:
    """Вспомогательная функция для извлечения цены из строки или числа (поддерживает научную нотацию)."""
    if text is None: return 0.0
    if isinstance(text, (int, float)): return float(text)
    import re
    # Регулярка для обычных чисел и научной нотации (типа 1.02e+05)
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    m = re.search(pattern, str(text).replace(',', ''))
    return float(m.group(0)) if m else 0.0

def run_backtest(
    symbol: str, 
    timeframes: List[str], 
    start_date: str, 
    end_date: str, 
    step_minutes: int = 240,
    use_critic: bool = True,
    use_math_only: bool = False
):
    """
    Основной цикл бэктестинга.
    """
    logger.info(f"=== ЗАПУСК БЭКТЕСТА: {symbol} ===")
    logger.info(f"Период: {start_date} - {end_date} | Шаг: {step_minutes} мин")

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    # 1. Загрузка данных (нужно 300 свечей ПРЕДЫСТОРИИ для самого крупного ТФ)
    # Ищем самый крупный ТФ
    max_tf = "1d" if "1d" in timeframes else timeframes[0]
    lookback_ms = 300 * TF_MS.get(max_tf, 86400000)
    history_since_ms = start_ts - lookback_ms

    # Загружаем историю, включая "будущее" (для проверки TP/SL)
    # end_ts + еще запас для отработки сделки (например, неделя)
    future_buffer_ms = 7 * 24 * 60 * 60 * 1000 
    history_until_ms = end_ts + future_buffer_ms

    history_dfs = fetch_all_needed_history(symbol, timeframes, since_ms=history_since_ms, until_ms=history_until_ms)
    
    # Инициализируем Машины Времени для каждого ТФ
    machines = {tf: TimeMachine(df, candle_limit=CANDLE_LIMIT or 300) for tf, df in history_dfs.items()}
    
    # 2. Определяем временные точки для анализа
    # Используем 15m (базовый ТФ) для определения шагов времени
    base_tf = "15m" if "15m" in timeframes else timeframes[-1]
    all_ts = [ts for ts in machines[base_tf].get_all_timestamps(step=1) if start_ts <= ts <= end_ts]
    
    # Прореживаем согласно шагу (step_minutes)
    step_candles = max(1, step_minutes // (TF_MS.get(base_tf) // 60000))
    test_ts = all_ts[::step_candles]
    
    logger.info(f"Всего точек для анализа: {len(test_ts)}")

    results: List[Dict] = []
    current_balance = INITIAL_DEPOSIT
    logger.info(f"💰 Стартовый баланс: ${current_balance:.2f} (Риск: {RISK_PER_TRADE_PERCENT}% на сделку)")
    
    # 3. Главный цикл
    total_points = len(test_ts)
    busy_until_ts = 0  # Время, до которого бот "занят" открытой сделкой
    traded_patterns = set() # Уникальные ключи (direction + levels), чтобы не входить дважды в одно и то же

    for i, ts in enumerate(test_ts):
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
        
        # --- ПРОВЕРКА СОСТОЯНИЯ (ЗАНЯТОСТЬ) ---
        if ts < busy_until_ts:
            continue

        # Получаем снимки по всем ТФ
        all_snapshots = {}
        valid_history = True
        for tf in timeframes:
            snap = machines[tf].get_snapshot(ts)
            if snap is None:
                valid_history = False
                break
            all_snapshots[tf] = snap
            
        if not valid_history:
            continue
            
        # Математический препроцессинг 
        try:
            import time
            t0 = time.time()
            context = preprocess_all(
                symbol=symbol,
                all_candles=all_snapshots,
                timeframes=timeframes,
                orderbook_walls=None
            )
            t_math = time.time() - t0
            
            # Логируем прогресс (раз в 100 шагов или если есть паттерн)
            curr_p = context.timeframes[-1].vectors[-1].end_price if context.timeframes[-1].vectors else 0
            has_any_pattern = any("МАТЕМАТИЧЕСКИ ОПРЕДЕЛЁННЫЕ СТРУКТУРЫ" in tf_data.mathematical_wave_state for tf_data in context.timeframes)
            
            if has_any_pattern or i % 100 == 0:
                logger.info(f"--- [ {i+1}/{total_points} ] [ {dt.strftime('%Y-%m-%d %H:%M')} ] Анализ {symbol} @ {curr_p:.2f} ---")

            # Пайплайн выбора
            if use_math_only:
                plan = run_math_pipeline(context, use_ai_confirm=False)
            else:
                plan = run_full_ai_pipeline(context, use_critic=use_critic)

            # 2. Hard-валидация уровней и R:R
            if not hard_validate_plan(plan):
                continue

            direction = str(plan.trade_params.get("direction", "")).upper()
            if not direction: continue

            if "WAIT" in direction:
                if plan.wave_count_label == "RR Filtered":
                    # Логируем отсеянный по RR паттерн
                    results.append({
                        "ts": ts,
                        "date": dt.strftime('%Y-%m-%d %H:%M'),
                        "direction": "WAIT (RR)",
                        "status": "RR_FILTERED",
                        "rr_ratio": 0,
                        "pnl_pct": 0,
                        "pnl_usd": 0,
                        "fee_usd": 0,
                        "balance": round(current_balance, 2),
                        "pos_size_usd": 0,
                        "leverage": 0,
                        "reasoning": plan.detailed_logic[:200]
                    })
                continue

            entry_p = _extract_price(plan.trigger_prices.get("confirmation_level")) or _extract_price(plan.trigger_prices.get("entry_zone"))
            sl_p = _extract_price(plan.trade_params.get("stop_loss"))
            tps = plan.trade_params.get("take_profit_levels", [])
            tp1_p = _extract_price(tps[0]) if tps else 0.0

            # --- ЗАЩИТА ОТ ПОВТОРНЫХ ВХОДОВ ---
            pattern_key = f"{direction}_{entry_p:.1f}_{sl_p:.1f}_{tp1_p:.1f}"
            if pattern_key in traded_patterns:
                continue

            logger.info(f"  🎯 СИГНАЛ: {direction} | Entry: {entry_p:.2f} | TP: {tp1_p:.2f} | SL: {sl_p:.2f}")

            # Проверка исполнения (смотрим на будущие 1000 свечей младшего ТФ)
            future = machines[base_tf].get_future_candles(ts, limit=1000)

            # --- ВИЗУАЛИЗАЦИЯ ---
            try:
                found_s = None
                for tf_data in context.timeframes:
                    state = tf_data.mathematical_wave_state
                    if state and isinstance(state, dict) and state.get('structures'):
                        found_s = state['structures'][0]
                        break

                if found_s:
                    tf_candles = machines[base_tf].get_snapshot(ts)
                    print(f"DEBUG: Plotting {found_s.pattern_type} for {symbol}")
                    plot_wave_structure(symbol, f"Structure_{found_s.pattern_type}_{base_tf}", tf_candles, found_s)
            except Exception as plot_e:
                logger.warning(f"Ошибка при попытке визуализации: {plot_e}")

            outcome = evaluate_plan(plan, future)
            
            # Блокируем время в зависимости от исхода
            if outcome.status in ["PROFIT", "LOSS", "EXPIRED"]:
                if outcome.exit_ts > ts:
                    busy_until_ts = outcome.exit_ts
                traded_patterns.add(pattern_key) 
            elif outcome.status == "CANCELLED":
                # Если отмена по инвалидации (цена ушла не туда до входа) — не блокируем надолго
                pass

            # Расчет прибыли в USD на основе риска и КОМИССИИ
            pnl_usd = 0.0
            pos_size_usd = 0.0
            fee_total_usd = 0.0
            rr_actual = 0.0
            leverage = 0.0
            
            if entry_p > 0 and sl_p > 0 and entry_p != sl_p:
                risk_dist = abs(entry_p - sl_p)
                reward_dist = abs(tp1_p - entry_p) if tp1_p > 0 else 0
                rr_actual = reward_dist / risk_dist if risk_dist > 0 else 0
                
                # Позиция на основе риска (1% от баланса)
                stop_pct = (risk_dist / entry_p) * 100
                risk_amount = (current_balance * (RISK_PER_TRADE_PERCENT / 100)) if current_balance > 0 else 0
                pos_size_usd = (risk_amount / (stop_pct / 100)) if stop_pct > 0 else 0
                
                # КЭП ПО ПЛЕЧУ: Не более 3x от баланса
                max_pos = current_balance * 3.0
                if pos_size_usd > max_pos:
                    pos_size_usd = max_pos
                    
                leverage = (pos_size_usd / current_balance) if current_balance > 0 else 0
                
                # Комиссия (вход + выход)
                if outcome.status in ["PROFIT", "LOSS", "EXPIRED"]:
                    fee_total_usd = pos_size_usd * (EXCHANGE_FEE / 100) * 2
                    pnl_usd = (pos_size_usd * (outcome.pnl_pct / 100)) - fee_total_usd
                else: 
                    pnl_usd = 0.0 
                
                current_balance += pnl_usd
            
            logic_summary = plan.detailed_logic[:150].replace('\n', ' ') + "..." if plan.detailed_logic else ""
            logger.info(f"  💵 PnL: ${pnl_usd:+.2f} (Fee: ${fee_total_usd:.2f}) | Bal: ${current_balance:.2f} | R:R: {rr_actual:.2f}")

            results.append({
                "ts": ts,
                "date": dt.strftime('%Y-%m-%d %H:%M'),
                "direction": direction,
                "status": outcome.status,
                "rr_ratio": round(rr_actual, 2),
                "pnl_pct": round(outcome.pnl_pct, 4),
                "pnl_usd": round(pnl_usd, 2),
                "fee_usd": round(fee_total_usd, 2),
                "balance": round(current_balance, 2),
                "pos_size_usd": round(pos_size_usd, 0),
                "leverage": round(leverage, 1),
                "reasoning": logic_summary
            })
            
            # Промежуточное сохранение
            pd.DataFrame(results).to_csv("reports/backtest_latest.csv", index=False)
            
        except Exception as e:
            # logger.error(f"Ошибка в точке {dt}: {e}")
            continue

    # 4. Итоги
    if not results:
        logger.warning("Сделок не найдено.")
        return

    df_results = pd.DataFrame(results)
    
    # Фильтруем только исполненные сделки (не CANCELLED)
    executed_trades = df_results[df_results['status'].isin(["PROFIT", "LOSS", "EXPIRED"])]
    cancelled_count = len(df_results[df_results['status'] == "CANCELLED"])
    
    total_executed = len(executed_trades)
    total_pnl_usd = current_balance - INITIAL_DEPOSIT
    
    # Win Rate считаем только по PROFIT/LOSS
    completed_trades = executed_trades[executed_trades['status'].isin(["PROFIT", "LOSS"])]
    win_rate = (len(completed_trades[completed_trades['status'] == "PROFIT"]) / len(completed_trades)) * 100 if len(completed_trades) > 0 else 0
    
    logger.info("="*40)
    logger.info("ФИНАЛЬНЫЙ ОТЧЕТ БЭКТЕСТА")
    logger.info(f"Начальный баланс: ${INITIAL_DEPOSIT:.2f}")
    logger.info(f"Конечный баланс:  ${current_balance:.2f}")
    logger.info(f"Общий PnL ($):    ${total_pnl_usd:+.2f} ({ (total_pnl_usd/INITIAL_DEPOSIT*100):+.2f}%)")
    logger.info(f"Исполнено сделок: {total_executed}")
    logger.info(f"  - Профит:       {len(completed_trades[completed_trades['status'] == 'PROFIT'])}")
    logger.info(f"  - Убыток:       {len(completed_trades[completed_trades['status'] == 'LOSS'])}")
    logger.info(f"  - Истекло (EXP):  {len(executed_trades[executed_trades['status'] == 'EXPIRED'])}")
    logger.info(f"Отменено (CANC):  {cancelled_count} (цена не дошла до входа)")
    logger.info(f"Win Rate:         {win_rate:.1f}%")
    logger.info("="*40)
    
    # Сохраняем результаты
    report_path = f"reports/backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(report_path, index=False)
    logger.info(f"Подробный отчет сохранен в {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Бэктестинг RalphTradeBot V2")
    parser.add_argument("--symbol", default="BTCUSDT", help="Символ")
    parser.add_argument("--start", default="2024-12-01", help="Дата начала (ГГГГ-ММ-ДД)")
    parser.add_argument("--end", default="2025-01-01", help="Дата конца (ГГГГ-ММ-ДД)")
    parser.add_argument("--step", type=int, default=240, help="Шаг анализа в минутах (например 240 = 4 часа)")
    parser.add_argument("--no-critic", action="store_true", help="Отключить Критика для скорости")
    parser.add_argument("--math", action="store_true", help="Только математический анализ (без ИИ)")
    
    args = parser.parse_args()
    
    tfs = ["1d", "4h", "1h", "15m", "5m"]
    run_backtest(
        symbol=args.symbol,
        timeframes=tfs,
        start_date=args.start,
        end_date=args.end,
        step_minutes=args.step,
        use_critic=not args.no_critic,
        use_math_only=args.math
    )
