"""
Интеграционный тест всего пайплайна RalphTradeBot на синтетических данных.
Проверяет: indicators → math_preprocessor → ai_prompt_builder → trading_plan_generator
Без вызовов Bybit API и LLM.
"""
import json
import math
import random

random.seed(42)

# ─── 1. Генерация синтетических свечей ───────────────────────────────────────

def generate_candles(n=100, start_price=40000.0, trend=1):
    """Генерирует синтетический OHLCV-массив с реалистичным случайным движением."""
    candles = []
    price = start_price
    base_ts = 1_700_000_000_000
    for i in range(n):
        change = random.gauss(0.001 * trend, 0.012)
        open_ = price
        close = price * (1 + change)
        high = max(open_, close) * (1 + abs(random.gauss(0, 0.003)))
        low  = min(open_, close) * (1 - abs(random.gauss(0, 0.003)))
        vol  = random.uniform(100, 1000)
        # Искусственно добавим пару аномальных объёмов
        if i in (30, 65):
            vol *= 4
        candles.append({
            "ts": base_ts + i * 3_600_000,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        })
        price = close
    return candles

# ─── 2. Тест индикаторов ─────────────────────────────────────────────────────

from src.math_engine.indicators import rsi, sma, volume_anomaly_mask, detect_rsi_divergence

candles_1h = generate_candles(150, 40000, trend=1)

# Форсируем аномальный объём в конкретных позициях (независимо от RNG-состояния)
for anom_idx in (30, 65, 100):
    avg_vol = sum(c["volume"] for c in candles_1h) / len(candles_1h)
    candles_1h[anom_idx]["volume"] = avg_vol * 6.0   # гарантированно > 2.5x SMA

closes = [c["close"] for c in candles_1h]
vols   = [c["volume"] for c in candles_1h]

rsi_vals = rsi(closes, 14)
sma_vals = sma(closes, 20)
vol_mask = volume_anomaly_mask(vols, 20, 2.5)
div_mask = detect_rsi_divergence(closes, rsi_vals, window=5)

assert len(rsi_vals) == len(closes), "RSI длина не совпадает"
n_anomalies = sum(vol_mask)
assert n_anomalies > 0, "Аномальный объём не обнаружен"
print(f"✅ Индикаторы: RSI[-1]={rsi_vals[-1]:.2f}, аномалий объёма={n_anomalies}, RSI-дивергенций={sum(div_mask)}")

# ─── 3. Тест math_preprocessor ────────────────────────────────────────────────

from src.math_engine.math_preprocessor import preprocess_timeframe, preprocess_all, calc_fib_levels

tf_data = preprocess_timeframe(candles_1h, "1h")
assert len(tf_data.vectors) > 0, "Векторы не построены"
assert len(tf_data.fib_levels) > 0, "Уровни Фибоначчи не рассчитаны"
print(f"✅ Препроцессор [1h]: векторов={len(tf_data.vectors)}, уровней Fib={len(tf_data.fib_levels)}, RSI={tf_data.current_rsi}")

candles_4h  = generate_candles(120, 40000, trend=1)
candles_1d  = generate_candles(80,  40000, trend=1)
all_candles = {"1d": candles_1d, "4h": candles_4h, "1h": candles_1h}
timeframes  = ["1d", "4h", "1h"]

context = preprocess_all("BTCUSDT", all_candles, timeframes,
                         orderbook_walls={"bid_wall": 38500.0, "ask_wall": 42000.0})
assert len(context.timeframes) == 3, "Не все таймфреймы обработаны"
print(f"✅ Контекст LLM: символ={context.symbol}, ТФ={[t.timeframe for t in context.timeframes]}")

# ─── 4. Тест ai_prompt_builder ────────────────────────────────────────────────

from src.ai.ai_prompt_builder import build_messages, SYSTEM_PROMPT

messages = build_messages(context)
assert messages[0]["role"] == "system"
assert "ИМПУЛЬС" in messages[0]["content"]
assert "BTCUSDT" in messages[1]["content"]
print(f"✅ Промпт собран: {len(SYSTEM_PROMPT)} символов системного промпта")

# ─── 5. Тест trading_plan_generator ───────────────────────────────────────────

from src.trader.trading_plan_generator import parse_llm_response, format_plan_for_user

mock_llm_response = json.dumps({
    "wave_count_label": "[1W: (3)] [1D: iii of 3] [4H: (v) of iii]",
    "detailed_logic": "На 1D формируется волна (3) восходящего импульса. Fib-кластер 40200 (strength=3) является ключевой поддержкой. RSI дивергенция подтверждает завершение коррекции.",
    "main_scenario": "После завершения коррекции (iv) ожидается волна (v) вверх с целями 41800, 43500 и 46000.",
    "alternative_scenario": "Пробой 39500 отменяет сценарий. Вероятно начало волны (iv) large degree.",
    "trigger_prices": {
        "confirmation_level": "40800 (BOS выше локального хая)",
        "entry_zone": "40200 - 40500",
        "invalid_level": "39500"
    },
    "trade_params": {
        "direction": "LONG",
        "stop_loss": "39350",
        "take_profit_levels": ["41800 (TP1, 1:2)", "43500 (TP2, 1:3)", "46000 (TP3, 1.618 ext)"],
        "risk_reward_ratio": "1:3.3"
    }
})

plan = parse_llm_response(mock_llm_response)
assert plan.trade_params.get("direction") == "LONG"
assert len(plan.trade_params.get("take_profit_levels", [])) == 3
formatted = format_plan_for_user(plan, "BTCUSDT")
print(f"✅ Парсинг плана: {plan.trade_params.get('direction')}, R:R={plan.trade_params.get('risk_reward_ratio')}")

# ─── 6. Итог ─────────────────────────────────────────────────────────────────

print("\n" + "═" * 60)
print(formatted)
print("═" * 60)
print("\n🎉 Все тесты прошли успешно! Пайплайн работает корректно.")
