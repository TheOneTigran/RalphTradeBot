"""
config.py — Единая точка конфигурации приложения V3.
Читает переменные из .env файла.

V3 Changes:
  - Удалены LLM-настройки (OPENAI_*)
  - Добавлены: DuckDB, Redis, ML, HITL, WS
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# Bybit (Primary Exchange)
# ═══════════════════════════════════════════════════════════════════════════
BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")

# ═══════════════════════════════════════════════════════════════════════════
# Telegram
# ═══════════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ═══════════════════════════════════════════════════════════════════════════
# LLM (только для Reporting Agent — форматирование текстовых отчётов)
# ═══════════════════════════════════════════════════════════════════════════
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")

# ═══════════════════════════════════════════════════════════════════════════
# Storage: DuckDB (embedded columnar OLAP)
# ═══════════════════════════════════════════════════════════════════════════
DUCKDB_PATH: str = os.getenv("DUCKDB_PATH", os.path.join("data", "ralph.duckdb"))

# ═══════════════════════════════════════════════════════════════════════════
# Storage: Redis (in-memory cache, optional)
# ═══════════════════════════════════════════════════════════════════════════
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ═══════════════════════════════════════════════════════════════════════════
# Анализ
# ═══════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════
# Риск-менеджмент
# ═══════════════════════════════════════════════════════════════════════════
INITIAL_DEPOSIT: float = float(os.getenv("INITIAL_DEPOSIT", "10000"))
RISK_PER_TRADE_PERCENT: float = float(os.getenv("RISK_PER_TRADE_PERCENT", "1.0"))
MIN_RR_RATIO: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.5"))
EXCHANGE_FEE: float = float(os.getenv("EXCHANGE_FEE_PERCENT", "0.1"))
MAX_LEVERAGE: float = float(os.getenv("MAX_LEVERAGE", "10.0"))

# ═══════════════════════════════════════════════════════════════════════════
# ML Scoring (XGBoost / LightGBM)
# ═══════════════════════════════════════════════════════════════════════════
ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", os.path.join("data", "ml_model.joblib"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

# Веса фильтров для Confluence Layer (должны суммироваться в 1.0)
FILTER_WEIGHT_WAVE_LOGIC: float = 0.40
FILTER_WEIGHT_FIBO_ZONE: float = 0.20
FILTER_WEIGHT_LIQUIDITY_SWEEP: float = 0.20
FILTER_WEIGHT_CLUSTER_VOLUME: float = 0.20

# ═══════════════════════════════════════════════════════════════════════════
# HITL (Human-in-the-Loop)
# ═══════════════════════════════════════════════════════════════════════════
HITL_MIN_LABELS_FOR_TRAINING: int = int(os.getenv("HITL_MIN_LABELS", "100"))
HITL_RETRAIN_INTERVAL_HOURS: int = int(os.getenv("HITL_RETRAIN_HOURS", "168"))  # 7 дней

# ═══════════════════════════════════════════════════════════════════════════
# WebSocket Streaming
# ═══════════════════════════════════════════════════════════════════════════
WS_RECONNECT_DELAY: float = float(os.getenv("WS_RECONNECT_DELAY", "5.0"))
WS_MAX_RECONNECT_ATTEMPTS: int = int(os.getenv("WS_MAX_RECONNECT_ATTEMPTS", "10"))
