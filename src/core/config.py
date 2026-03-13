"""
config.py — Единая точка конфигурации приложения.
Читает переменные из .env файла.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Bybit ---
BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")

# --- Telegram ---
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

# --- LLM ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# --- Анализ ---
SYMBOLS: list[str] = [s.strip() for s in os.getenv("SYMBOLS", "BTCUSDT").split(",")]
TIMEFRAMES: list[str] = [t.strip() for t in os.getenv("TIMEFRAMES", "1w,1d,4h,1h,15m,5m").split(",")]
CANDLE_LIMIT: int = int(os.getenv("CANDLE_LIMIT", "500"))

# --- Объем ---
VOLUME_ANOMALY_MULTIPLIER: float = float(os.getenv("VOLUME_ANOMALY_MULTIPLIER", "2.5"))
VOLUME_ANOMALY_PERIOD: int = int(os.getenv("VOLUME_ANOMALY_PERIOD", "20"))

# --- ATR-Fractal: настройки поиска свингов по таймфреймам ---
# fractal_n  — половина окна фрактала Вильямса (2 → окно 5 баров, 3 → 7 баров)
# atr_mult   — минимальный размер значимого вектора = atr_mult × ATR
#              Повышение множителя отсеивает мелкие коррекции и шум
# atr_period — период Average True Range
#
# Логика выбора параметров:
#   1W/1D: свечи "тяжёлые", большое временное плечо → мультипликатор 0.8–1.0
#   4H/1H: средние ТФ, нужно видеть волны но убрать шум → 1.5–2.0
#   15m/5m: алго-трейдинг, много ложных движений → 2.5–3.0 + окно fractal_n=3
ATR_FRACTAL_SETTINGS: dict[str, dict] = {
    "1w":  {"fractal_n": 2, "atr_mult": 0.8,  "atr_period": 14},
    "1d":  {"fractal_n": 2, "atr_mult": 1.0,  "atr_period": 14},
    "4h":  {"fractal_n": 2, "atr_mult": 1.5,  "atr_period": 14},
    "1h":  {"fractal_n": 2, "atr_mult": 2.0,  "atr_period": 14},
    "15m": {"fractal_n": 3, "atr_mult": 2.5,  "atr_period": 14},
    "5m":  {"fractal_n": 3, "atr_mult": 3.0,  "atr_period": 14},
}

# Допуск совпадения уровней для определения Фибо-кластера (0.5% по умолчанию)
# Два уровня с разных свингов считаются кластером если |price_A - price_B| / price_A <= tolerance
FIB_CLUSTER_TOLERANCE: float = float(os.getenv("FIB_CLUSTER_TOLERANCE", "0.005"))

# --- Риск-менеджмент ---
INITIAL_DEPOSIT: float = float(os.getenv("INITIAL_DEPOSIT", "10000"))
RISK_PER_TRADE_PERCENT: float = float(os.getenv("RISK_PER_TRADE_PERCENT", "1.0"))
MIN_RR_RATIO: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "2.5"))
EXCHANGE_FEE: float = float(os.getenv("EXCHANGE_FEE_PERCENT", "0.1"))
