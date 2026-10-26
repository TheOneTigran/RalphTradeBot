"""
config.py — Конфигурация торговой системы RalphTradeBot.

Централизованные настройки: пути, API ключи, Telegram бот, пороги фильтрации,
цветовая палитра TradingView и пресеты для топ-15 криптовалютных пар.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════════
# Пути
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CONFIRMED_DIR = RESULTS_DIR / "confirmed_signals"
REJECTED_DIR = RESULTS_DIR / "rejected_signals"
VISION_CACHE_FILE = BASE_DIR / "vision_cache.json"
ANALYTICS_DB_PATH = DATA_DIR / "ralph_analytics.db"

for d in [RESULTS_DIR, CONFIRMED_DIR, REJECTED_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

# ═══════════════════════════════════════════════════════════════════════
# Telegram Bot Настройки
# ═══════════════════════════════════════════════════════════════════════
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8590403293:AAEz9jdkfe0dkHpcTZVzTIRph18pwEWLqDo")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-5254263991")
TELEGRAM_MIN_SCORE = 70         # Порог качества алгоритмического детектора для отправки (>= 70)

# ═══════════════════════════════════════════════════════════════════════
# Vision AI Ключи и Провайдеры (Отключен по запросу пользователя)
# ═══════════════════════════════════════════════════════════════════════
ENABLE_VISION_FILTER = False    # Переключатель Vision AI: отключен (модуль сохранен, но не блокирует сделки)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

OPENROUTER_MODELS = [
    "inclusionai/ling-3.0-flash-vl",
    "google/gemini-2.5-flash-lite",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-26b-a4b-it:free",
]

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
]

GROQ_MODELS = []

# ═══════════════════════════════════════════════════════════════════════
# Пороги фильтрации RalphTradeBot
# ═══════════════════════════════════════════════════════════════════════
MIN_ALGO_ELLIOTT_SCORE = 70    # Мин. балл детектора органического импульса (>= 70)
MIN_VISION_CONFIRM_SCORE = 70  # Мин. балл Vision AI
VISION_TIMEOUT = 15            # Таймаут запроса в секундах
VISION_TEMPERATURE = 0.1       # Температура генерации

# ═══════════════════════════════════════════════════════════════════════
# Параметры торгового плана Sweep & Engulfing (Модель 2)
# ═══════════════════════════════════════════════════════════════════════
SL_ZONE_FROM_EXTREME = 0.17    # Стоп-лосс внутри свечи S на уровне 83% коррекции (0.17 от экстремума W5 внутрь свечи)
SL_BUFFER_PCT = 0.17           # Коэффициент отступа стопа от экстремума W5 внутрь свечи (0.17 * range = 83% свечи)
TP1_MULT = 1.5                 # TP1 = 1:1.5 от базового полудиапазона свечи (50%) -> перенос стопа в безубыток
TP2_MULT = 3.5                 # TP2 = 1:3.5 от базового полудиапазона свечи (50%) -> финальная цель
MIN_RR_RATIO = 1.5             # Минимальное соотношение Риск/Прибыль (R:R)
ACTIVE_MODEL = "Model_2_Sweep_Engulf" # Проверенная модель Sweep & Engulfing
TP_SHARES = [0.40, 0.60]       # 40% на TP1 (1.5x | +2.27R Net), 60% на TP2 (3.5x | +5.30R Net) -> +4.09R Full TP
BREAKEVEN_AFTER_TP1 = True     # Перенос стопа остатка 70% в безубыток сразу при достижении TP1
BREAKEVEN_AFTER_TP2 = False
BREAKEVEN_OFFSET_PCT = 0.05    # Смещение безубытка для компенсации комиссии биржи
DEFAULT_DEPOSIT_USD = 500.0    # Депозит по умолчанию $500
DEFAULT_RISK_PCT = 0.5         # Риск по умолчанию 0.5% от депозита
RISK_BUDGET_USD = 2.5          # Базовый риск на сделку в $ (0.5% от $500 = $2.50)
DEFAULT_LEVERAGE = 10          # Плечо для расчета рекомендуемой маржи в сигнале
COMMISSION_RATE = 0.0005       # Комиссия биржи 0.05% Taker (0.10% roundtrip)
OUTCOME_TRACKER_INTERVAL_SEC = 30  # Интервал проверки отработки активных сигналов
OUTCOME_MAX_BARS_TTL = 120     # Макс. количество баров до экспирации активного сигнала
PRE_CLOSE_SIGNAL_SEC = 60      # Отправка сигнала за 60 секунд (1 минуту) до закрытия свечи
PRE_CLOSE_WINDOW_MIN_SEC = 2   # Мин. остаток до закрытия свечи для pre-close сканирования (сек)
PRE_CLOSE_WINDOW_MAX_SEC = 68  # Макс. остаток до закрытия свечи для pre-close сканирования (сек)
POST_CLOSE_FALLBACK_SEC = 55   # Окно сканирования строго после закрытия свечи поглощения (сек)
ENTRY_MAX_WAIT_BARS = 8        # Макс. количество свечей ТФ для ожидания лимитного входа 50%
EXCLUDE_SUNDAY_MSK = True      # Полный запрет на открытие сделок в воскресенье (МСК)
PRIME_INTRADAY_HOURS_MSK = (8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23)  # Активные часы МСК (08:00-10:00 и 15:00-24:00)
FORBIDDEN_4H_HOURS_MSK = (7, 11)  # Исключённые слоты 4h по МСК

# ═══════════════════════════════════════════════════════════════════════
# Параметры рендера графиков
# ═══════════════════════════════════════════════════════════════════════
LOOKBACK_CANDLES = 200         # Количество свечей в окне графика
CHART_WIDTH_PX = 1400          # Ширина основного полотна
CHART_HEIGHT_PX = 700          # Высота основного полотна
CHART_DPI = 100

# Цветовая палитра TradingView Dark
BG_DARK = "#131722"
GRID_CLR = "#363A45"
TEXT_CLR = "#D1D4DC"
CANDLE_UP = "#089981"
CANDLE_DN = "#F23645"
RSI_CLR = "#7B61FF"
RSI_OB_CLR = "#FF6B6B"
RSI_OS_CLR = "#51CF66"
SIGNAL_LINE_CLR = "#FFD700"
ORIGIN_DIV_CLR = "#00E5FF"     # Яркий аквамариновый цвет для W0 Origin Divergence
PIVOT_HIGH_CLR = "#EF5350"
PIVOT_LOW_CLR = "#26A69A"

# ═══════════════════════════════════════════════════════════════════════
# Списки криптовалютных пар и таймфреймов
# ═══════════════════════════════════════════════════════════════════════
# ТОП-12 лидеров по результатам бэктестов за последний месяц (PF 2.4 - 12.0)
TOP_12_LEADERS = [
    "NEARUSDT", "SUIUSDT", "KASUSDT", "DOGEUSDT", "UNIUSDT",
    "APTUSDT", "OPUSDT", "1000PEPEUSDT", "LTCUSDT", "BCHUSDT",
    "TAOUSDT", "ICPUSDT",
]

TOP_15_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "NEARUSDT", "BCHUSDT", "LTCUSDT", "AAVEUSDT", "1000PEPEUSDT",
]

TOP_30_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "NEARUSDT", "BCHUSDT", "LTCUSDT", "AAVEUSDT", "1000PEPEUSDT",
    "DOTUSDT", "1000SHIBUSDT", "TRXUSDT", "ETCUSDT", "APTUSDT",
    "POLUSDT", "UNIUSDT", "ICPUSDT", "RENDERUSDT", "FETUSDT",
    "ARBUSDT", "OPUSDT", "INJUSDT", "TAOUSDT", "KASUSDT",
]

# Активный рабочий пул для сканера (ТОП-12 лидеров)
ACTIVE_SYMBOLS = TOP_30_SYMBOLS

# Рабочие таймфреймы для ручной торговли (15m и 1h)
LIVE_TIMEFRAMES = ["15m", "30m", "1h", "4h", "1d"]

DEFAULT_PRESETS = {
    "BTCUSDT_5m": {
        "symbol": "BTCUSDT", "interval": "5m",
        "left_bars": 20, "right_bars": 2, "rsi_len": 7,
        "rsi_ob": 69.92, "rsi_os": 26.88, "tp_pct": 2.3, "sl_pct": 3.132,
    },
    "BTCUSDT_15m": {
        "symbol": "BTCUSDT", "interval": "15m",
        "left_bars": 12, "right_bars": 3, "rsi_len": 7,
        "rsi_ob": 68.0, "rsi_os": 26.0, "tp_pct": 2.5, "sl_pct": 3.5,
    },
    "ETHUSDT_5m": {
        "symbol": "ETHUSDT", "interval": "5m",
        "left_bars": 10, "right_bars": 3, "rsi_len": 6,
        "rsi_ob": 68.0, "rsi_os": 25.0, "tp_pct": 1.8, "sl_pct": 3.2,
    },
    "ETHUSDT_15m": {
        "symbol": "ETHUSDT", "interval": "15m",
        "left_bars": 3, "right_bars": 9, "rsi_len": 4,
        "rsi_ob": 65.02, "rsi_os": 21.35, "tp_pct": 1.3, "sl_pct": 6.53,
    },
    "SOLUSDT_15m": {
        "symbol": "SOLUSDT", "interval": "15m",
        "left_bars": 5, "right_bars": 5, "rsi_len": 8,
        "rsi_ob": 71.04, "rsi_os": 33.28, "tp_pct": 1.5, "sl_pct": 3.8,
    },
    "BNBUSDT_15m": {
        "symbol": "BNBUSDT", "interval": "15m",
        "left_bars": 8, "right_bars": 4, "rsi_len": 7,
        "rsi_ob": 68.5, "rsi_os": 26.5, "tp_pct": 1.5, "sl_pct": 3.0,
    },
    "XRPUSDT_15m": {
        "symbol": "XRPUSDT", "interval": "15m",
        "left_bars": 8, "right_bars": 4, "rsi_len": 7,
        "rsi_ob": 70.0, "rsi_os": 28.0, "tp_pct": 1.8, "sl_pct": 3.5,
    },
    "DOGEUSDT_15m": {
        "symbol": "DOGEUSDT", "interval": "15m",
        "left_bars": 8, "right_bars": 4, "rsi_len": 7,
        "rsi_ob": 72.0, "rsi_os": 27.0, "tp_pct": 2.0, "sl_pct": 4.0,
    },
    "ADAUSDT_15m": {
        "symbol": "ADAUSDT", "interval": "15m",
        "left_bars": 13, "right_bars": 10, "rsi_len": 8,
        "rsi_ob": 67.61, "rsi_os": 19.74, "tp_pct": 1.2, "sl_pct": 3.8,
    },
    "AVAXUSDT_15m": {
        "symbol": "AVAXUSDT", "interval": "15m",
        "left_bars": 8, "right_bars": 4, "rsi_len": 7,
        "rsi_ob": 69.0, "rsi_os": 26.0, "tp_pct": 1.8, "sl_pct": 3.5,
    },
    "LINKUSDT_15m": {
        "symbol": "LINKUSDT", "interval": "15m",
        "left_bars": 8, "right_bars": 4, "rsi_len": 7,
        "rsi_ob": 69.0, "rsi_os": 26.0, "tp_pct": 1.8, "sl_pct": 3.5,
    },
    "SUIUSDT_15m": {
        "symbol": "SUIUSDT", "interval": "15m",
        "left_bars": 10, "right_bars": 4, "rsi_len": 6,
        "rsi_ob": 69.0, "rsi_os": 24.0, "tp_pct": 2.0, "sl_pct": 3.5,
    },
    "NEARUSDT_15m": {
        "symbol": "NEARUSDT", "interval": "15m",
        "left_bars": 9, "right_bars": 6, "rsi_len": 8,
        "rsi_ob": 72.0, "rsi_os": 25.0, "tp_pct": 1.8, "sl_pct": 4.0,
    },
    "BCHUSDT_15m": {
        "symbol": "BCHUSDT", "interval": "15m",
        "left_bars": 10, "right_bars": 6, "rsi_len": 8,
        "rsi_ob": 67.12, "rsi_os": 24.46, "tp_pct": 1.7, "sl_pct": 3.011,
    },
    "LTCUSDT_15m": {
        "symbol": "LTCUSDT", "interval": "15m",
        "left_bars": 10, "right_bars": 5, "rsi_len": 6,
        "rsi_ob": 68.0, "rsi_os": 24.0, "tp_pct": 1.5, "sl_pct": 3.5,
    },
    "AAVEUSDT_15m": {
        "symbol": "AAVEUSDT", "interval": "15m",
        "left_bars": 8, "right_bars": 4, "rsi_len": 7,
        "rsi_ob": 69.0, "rsi_os": 26.0, "tp_pct": 2.0, "sl_pct": 4.0,
    },
    "1000PEPEUSDT_15m": {
        "symbol": "1000PEPEUSDT", "interval": "15m",
        "left_bars": 8, "right_bars": 4, "rsi_len": 7,
        "rsi_ob": 72.0, "rsi_os": 25.0, "tp_pct": 2.2, "sl_pct": 4.5,
    },
}
