
import sys
import os
sys.path.append('c:/Users/user/Desktop/RalphTradeBot')

import logging
logging.basicConfig(level=logging.INFO)

from src.core.models import LLMContext, TimeframeData, Vector
from src.core.math_actor import get_math_trade_plan

# Создаем фейковый контекст с зигзагом
# 0 -> 100, A -> 110, B -> 105, C -> 115
zigzag_vectors = [
    Vector(start_price=100, end_price=110, start_time=1000, end_time=2000, is_bullish=True, volume_anomaly=False, price_change_percent=10, rsi_at_end=50),
    Vector(start_price=110, end_price=105, start_time=2000, end_time=3000, is_bullish=False, volume_anomaly=False, price_change_percent=-5, rsi_at_end=50),
    Vector(start_price=105, end_price=115, start_time=3000, end_time=4000, is_bullish=True, volume_anomaly=False, price_change_percent=10, rsi_at_end=60),
]

tf_data = TimeframeData(
    timeframe="1h",
    vectors=zigzag_vectors,
    current_price=115,
    current_rsi=50,
    mathematical_wave_state=""
)

context = LLMContext(
    symbol="BTCUSDT",
    timeframes=[tf_data]
)

plan = get_math_trade_plan(context)
print(f"Plan Direction: {plan.trade_params.get('direction')}")
print(f"Logic: {plan.detailed_logic}")
