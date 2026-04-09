"""
test_matcher.py — Юнит-тесты на проверку сложной логики сведения (Matcher).
Проверяют Пессимистичный сценарий (Spike Rule) и Ценовые Гэпы (Gap Handling).
"""
import pytest
from src.wave_strategy.models import DtwTradePlan
from src.execution_engine.models import SimulatedOrder, OrderStatus
from src.execution_engine.matcher import process_candle


def create_mock_plan(direction="LONG", entry=100.0, sl_cons=90.0, sl_agg=95.0, tp=110.0):
    return DtwTradePlan(
        symbol="BTC/USDT",
        direction=direction,
        entry_price=entry,
        sl_aggressive=sl_agg,
        sl_conservative=sl_cons,
        take_profit=tp,
        creation_time=1000000,
        risk_reward_ratio=2.0,
        macro_context="mock",
        micro_trigger="mock"
    )

def test_long_spike_hits_both_sl_and_tp():
    """Правило Шпильки: внутри одной свечи цена задевает и SL, и TP -> Убыток."""
    plan = create_mock_plan("LONG", entry=100, sl_agg=90, tp=120)
    order = SimulatedOrder(plan=plan, status=OrderStatus.ACTIVE)
    
    # Свеча: open=105, high=130 (пробил TP=120), low=80 (пробил SL=90), close=100
    candle = {"ts": 2000000, "open": 105.0, "high": 130.0, "low": 80.0, "close": 100.0}
    
    process_candle(order, candle)
    
    # Ожидаем пессимистичный СТОП ЛОСС
    assert order.status == OrderStatus.FILLED_SL
    # Проверка проскальзывания на SL: SL * (1 - 0.05%) -> 90 * 0.9995 = 89.955
    assert order.close_price < 90.0

def test_long_gap_open_below_sl():
    """Гэповое открытие ниже Стоп Лосса: цена исполнения = open свечи (а не уровень SL)."""
    plan = create_mock_plan("LONG", entry=100, sl_agg=90, tp=120)
    order = SimulatedOrder(plan=plan, status=OrderStatus.ACTIVE)
    
    # Свеча: открылась сразу на 85 (ниже SL=90), High даже не дошел до SL
    candle = {"ts": 2000000, "open": 85.0, "high": 88.0, "low": 80.0, "close": 82.0}
    
    process_candle(order, candle)
    
    assert order.status == OrderStatus.FILLED_SL
    # Исполнено по 85 с проскальзыванием вниз
    assert order.close_price < 85.0
    
def test_long_buy_stop_gap_over_entry():
    """Гэповое открытие выше уровня Buy Stop (Entry): исполнение по Open, а не по Entry."""
    plan = create_mock_plan("LONG", entry=100, sl_cons=90, sl_agg=95, tp=120)
    order = SimulatedOrder(plan=plan, status=OrderStatus.PENDING)
    
    # Открылась на 105 (перепрыгнув Entry 100)
    candle = {"ts": 2000000, "open": 105.0, "high": 110.0, "low": 102.0, "close": 108.0}
    
    process_candle(order, candle)
    
    assert order.status == OrderStatus.ACTIVE
    # Вход осуществлен по худшей цене (open свечи + проскальзывание вверх)
    assert order.fill_price > 105.0

def test_invalidation_before_entry():
    """Слом паттерна до входа в сделку."""
    plan = create_mock_plan("LONG", entry=100, sl_cons=80, sl_agg=90, tp=120)
    order = SimulatedOrder(plan=plan, status=OrderStatus.PENDING)
    
    # Свеча пробила Conservative SL = 80, но не дошла до Entry = 100
    candle = {"ts": 2000000, "open": 95.0, "high": 98.0, "low": 75.0, "close": 85.0}
    
    process_candle(order, candle)
    
    assert order.status == OrderStatus.CANCELLED_INVALID
