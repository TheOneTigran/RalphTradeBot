"""
backtester.py — Событийный цикл.

Принимает ЗАРАНЕЕ сгенерированный список DtwTradePlan (чтобы не было O(N^2) сканирования).
Итерируется ТОЛЬКО по минутным свечам, точно имитируя реалии рынка.
"""

import csv
import logging
from typing import List, Dict

from .models import SimulatedOrder, OrderStatus
from .matcher import process_candle
from .portfolio import PortfolioManager
from .config import STARTING_BALANCE

logger = logging.getLogger(__name__)


def run_backtest(
    trade_plans: List["DtwTradePlan"],
    candles_1m: List[Dict],
    csv_output_path: str = "backtest_results.csv"
):
    """
    Симулирует торговлю на исторических свечах.
    """
    # Сортируем сгенерированные сканером планы хронологически
    trade_plans = sorted(trade_plans, key=lambda x: x.creation_time)
    
    portfolio = PortfolioManager()
    active_orders: List[SimulatedOrder] = []
    closed_orders: List[SimulatedOrder] = []
    
    plan_index = 0
    total_plans = len(trade_plans)
    
    logger.info(f"Начинаем бэктест на {len(candles_1m)} свечах (ТФ 1m/5m). Планов на исполнение: {total_plans}")
    
    last_trade_times = {} # tracking duplicates {symbol_direction: last_time}
    
    for c_idx, candle in enumerate(candles_1m):
        c_time = candle.get("timestamp", candle.get("ts", 0))
        
        # 1. Поступление новых планов от Сигнального движка (Сканера)
        while plan_index < total_plans and trade_plans[plan_index].creation_time <= c_time:
            new_plan = trade_plans[plan_index]
            
            # Дедупликация: 60 минут (3600000 мс)
            key = f"{new_plan.symbol}_{new_plan.direction}"
            last_time = last_trade_times.get(key, 0)
            if c_time - last_time < 3600000:
                plan_index += 1
                continue # Уже брали сделку на этой структуре недавно
            
            # Проверяем доступность капитала/лимитов
            currently_in_trade = sum(1 for o in active_orders if o.status in [OrderStatus.ACTIVE, OrderStatus.PENDING])
            if portfolio.can_open_position(currently_in_trade):
                # Создаем ордер
                order = SimulatedOrder(plan=new_plan)
                
                # Считаем размер
                size = portfolio.calculate_position_size(
                    entry=new_plan.entry_price, 
                    sl=new_plan.sl_aggressive
                )
                order.position_size = size
                
                active_orders.append(order)
                last_trade_times[key] = c_time
            else:
                # Нет маржи или превышен лимит позиций
                rej_order = SimulatedOrder(plan=new_plan, status=OrderStatus.REJECTED_MARGIN)
                closed_orders.append(rej_order)
                
            plan_index += 1
            
        # 2. Обработка всех текущих ордеров по текущей (одной!) свече
        to_remove = []
        for order in active_orders:
            # Матчинг внутри свечи
            process_candle(order, candle)
            
            # Если ордер завершил свой цикл — применяем финансы
            if order.status in [
                OrderStatus.FILLED_TP, 
                OrderStatus.FILLED_SL, 
                OrderStatus.CANCELLED_TIME, 
                OrderStatus.CANCELLED_INVALID
            ]:
                portfolio.apply_result(order)
                to_remove.append(order)
                
        # 3. Чистка
        for o in to_remove:
            active_orders.remove(o)
            closed_orders.append(o)
            
    # Принудительно закрываем те, что остались в рынке на момент конца истории
    for order in active_orders:
        if order.status == OrderStatus.ACTIVE:
            # Закрываем по последней известной цене
            last_close = candles_1m[-1]["close"]
            # Считается как закрыто вручную (FILLED_SL просто как маркер)
            from .matcher import _close
            _close(order, last_close, OrderStatus.CANCELLED_TIME, candles_1m[-1].get("timestamp", 0))
            portfolio.apply_result(order)
            closed_orders.append(order)
            
    # Выводим CSV
    _save_to_csv(closed_orders, csv_output_path)
    
    # Сводка
    stats = portfolio.get_stats()
    logger.info("Бэктест завершен! Статистика:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
        
    return stats


def _save_to_csv(orders: List[SimulatedOrder], path: str):
    """
    Экспорт отчета с требуемыми колонками: 
    Entry_Time, Exit_Time, Status, PnL_USD, Drawdown_Pct (вычисляемый), Trade_Duration_Bars
    """
    import datetime
    
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Symbol", "Direction", "Creation_Time", "Entry_Time", "Exit_Time", 
            "Status", "Entry_Price", "Exit_Price", "PnL_USD", "Drawdown_Pct", "Position_Size", "Trade_Duration_Bars"
        ])
        
        for o in orders:
            p = o.plan
            c_time_str = datetime.datetime.fromtimestamp(p.creation_time/1000.0).strftime('%Y-%m-%d %H:%M') if p.creation_time else ""
            ent_time_str = datetime.datetime.fromtimestamp(o.entry_time/1000.0).strftime('%Y-%m-%d %H:%M') if o.entry_time else ""
            ex_time_str = datetime.datetime.fromtimestamp(o.exit_time/1000.0).strftime('%Y-%m-%d %H:%M') if o.exit_time else ""
            
            writer.writerow([
                p.symbol,
                p.direction,
                c_time_str,
                ent_time_str,
                ex_time_str,
                o.status.name,
                round(o.fill_price, 4) if o.fill_price else "",
                round(o.close_price, 4) if o.close_price else "",
                o.pnl_usd,
                o.drawdown_pct,
                round(o.position_size, 6),
                o.bars_held
            ])
