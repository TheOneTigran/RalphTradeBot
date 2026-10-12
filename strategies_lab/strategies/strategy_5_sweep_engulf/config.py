"""
config.py — Конфигурация параметров стратегии 5 (Sweep Engulf + 50% Retracement).
"""
from pathlib import Path

STRATEGY_DIR = Path(__file__).parent
RESULTS_DIR = STRATEGY_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

for d in [RESULTS_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Параметры детекции свечи свипа и поглощения ──
MAX_SWEEP_LOOKAHEAD = 3     # Окно поиска свечи свипа после W5 (1..3 свечи)
RETRACE_RATIO = 0.50        # Откат на 50% полного диапазона (High - Low) свечи S
RR_TARGET = 3.0             # Целевое соотношение Риск/Прибыль (1:3)
FEE_RATE = 0.0005           # Фиксированная комиссия 0.05% на вход и 0.05% на выход (0.10% roundtrip)

# ── Окно ожидания заполнения лимита (в свечах после S) ──
DEFAULT_FILL_WINDOW = 3     # Базовое окно (1-3 свечи)
TEST_FILL_WINDOWS = [1, 2, 3, 5]  # Сетка для сравнительного анализа

# ── Риск-менеджмент ──
ACCOUNT_DEPOSIT_USD = 1000.0  # Стартовый депозит для симуляции
RISK_PER_TRADE_USD = 10.0     # Риск на сделку 1.0% ($10)
TIMEOUT_BARS = 60             # Максимальное удержание позиции (в свечах)
MIN_ELLIOTT_SCORE = 65.0      # Минимальный скор импульса Эллиотта для анализа
