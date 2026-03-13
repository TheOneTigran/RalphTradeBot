import logging
from src.backtester.data_manager import get_or_fetch_history

logging.basicConfig(level=logging.INFO)

# Тест загрузки BTC за последний месяц (1h)
df = get_or_fetch_history("BTCUSDT", "1h", months=1)
print(f"Загружено {len(df)} свечей. Первая: {df['ts'].iloc[0]}, Последняя: {df['ts'].iloc[-1]}")
