"""
main.py — Точка входа RalphTradeBot V2.

Запуск:
    python main.py
    python main.py --symbols BTCUSDT ETHUSDT --timeframes 1d 4h 1h
    python main.py --no-critic    # пропустить Self-Correction (быстрее)
    python main.py --no-oi        # пропустить загрузку Open Interest

Пайплайн:
  1. data_fetcher      → OHLCV + стакан + OI + Funding Rate
  2. math_preprocessor → ATR-Fractal свинги, Fib-кластеры, AO, OI-дивергенция
  3. ai_prompt_builder → JSON + промпт для LLM Actor
  4. Actor LLM         → генерирует черновой TradePlan
  5. Critic LLM        → проверяет план на нарушения 89WAVES
  6. (optional) Actor  → исправляет план по замечаниям Критика
  7. trading_plan_gen  → парсит, форматирует, сохраняет отчёт
"""
from __future__ import annotations

import sys
import os
# Добавляем корень проекта в пути поиска модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if "src" not in os.listdir(os.getcwd()) and "src" in os.listdir(os.path.dirname(os.path.abspath(__file__))):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    sys.path.append(os.getcwd())

import argparse
import json
import logging
import sys
import os
from datetime import datetime

from openai import OpenAI

from src.core.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    SYMBOLS,
    TIMEFRAMES,
)
from src.core.exceptions import WaveEngineError, PlanParsingError
from src.core.models import CriticFeedback, TradePlan
from src.fetcher.data_fetcher import (
    fetch_all_timeframes,
    fetch_orderbook_walls,
    fetch_open_interest,
)
from src.math_engine.math_preprocessor import preprocess_all
from src.ai.ai_prompt_builder import build_messages
from src.ai.critic_prompt import build_critic_messages, build_correction_messages
from src.trader.trading_plan_generator import parse_llm_response, format_plan_for_user
from src.visualizer.chart_generator import generate_chart
from src.validator.hard_validator import validate_plan as hard_validate_plan

# ─── Логирование ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ─── LLM клиент ──────────────────────────────────────────────────────────────

# ─── LLM и Pipeline импортируются из src.core.pipeline ─────────────
from src.core.pipeline import run_full_ai_pipeline, build_context_summary


# ─── Основная функция анализа ─────────────────────────────────────────────────

def analyze_symbol(
    symbol: str,
    timeframes: list[str],
    use_critic: bool = True,
    use_oi: bool = True,
) -> None:
    """
    Полный цикл анализа для одной торговой пары.
    Включает: Actor LLM → Critic → (Correction) → отчёт.

    Args:
        symbol:     Торговая пара (например BTCUSDT).
        timeframes: Список таймфреймов от старшего к младшему.
        use_critic: Запускать ли Self-Correction (Critic) проход.
        use_oi:     Загружать ли данные Open Interest.
    """
    print(f"\n{'═' * 60}")
    print(f"  🔍 Анализирую {symbol}...")
    print(f"{'═' * 60}")

    # ── 1. Загрузка данных ────────────────────────────────────────────────
    logger.info("[1/5] Загрузка рыночных данных с Bybit...")
    all_candles = fetch_all_timeframes(symbol)
    orderbook_walls = fetch_orderbook_walls(symbol)

    # Open Interest (необязательный)
    oi_snapshot = None
    if use_oi:
        logger.info("[+OI] Загрузка Open Interest...")
        oi_snapshot = fetch_open_interest(symbol, period="1d", limit=30)

    # ── 2. Математическая обработка ──────────────────────────────────────
    logger.info("[2/5] ATR-Fractal обработка (свинги, Fib-кластеры, AO)...")
    context = preprocess_all(
        symbol=symbol,
        all_candles=all_candles,
        timeframes=timeframes,
        orderbook_walls=orderbook_walls,
    )

    # Прикрепляем OI к младшему ТФ (самый свежий контекст)
    if oi_snapshot and context.timeframes:
        context.timeframes[-1].oi_data = oi_snapshot

    # OI-дивергенция: если OI и цена движутся в разные стороны
    if oi_snapshot and oi_snapshot.oi_change_pct_24h is not None:
        last_tf = context.timeframes[-1]
        if last_tf.vectors:
            last_vec = last_tf.vectors[-1]
            price_up = last_vec.is_bullish
            oi_up = oi_snapshot.oi_change_pct_24h > 0
            oi_snapshot.oi_price_divergence = price_up != oi_up

    # Сериализуем контекст один раз (нужен для Correction-шага)
    market_data_json = context.model_dump_json(indent=2)

    # ── 3. Actor/Critic/Correction AI Pipeline ───────────────────────────
    from src.core.pipeline import run_full_ai_pipeline
    
    logger.info("[3/5] Запуск AI Пайплайна (Actor + Self-Healing)...")
    plan = run_full_ai_pipeline(context, use_critic=use_critic)

    # ── Вывод и сохранение ────────────────────────────────────────────────
    formatted_plan = format_plan_for_user(plan, symbol)
    print(formatted_plan)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Используем "" вместо llm_text, так как пайплайн возвращает уже распарсенный объект
    _save_report(symbol, formatted_plan, "", timestamp, plan)

    # ── 6. Визуализация (Phase 4) ─────────────────────────────────────────
    logger.info("[6/6] Генерация графиков...")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Строим графики для ВСЕХ анализируемых ТФ
    tfs_to_plot = timeframes
    
    for tf in tfs_to_plot:
        tf_data = next((t for t in context.timeframes if t.timeframe == tf), None)
        if tf_data:
            chart_path = os.path.join(reports_dir, f"{symbol}_{tf}_chart_{timestamp}.png")
            generate_chart(
                symbol=symbol,
                timeframe=tf,
                candles=all_candles.get(tf, []),
                vectors=tf_data.vectors,
                fib_clusters=tf_data.fib_clusters,
                save_path=chart_path,
                plan=plan
            )


# ─── Сохранение отчётов ──────────────────────────────────────────────────────

def _save_report(
    symbol: str,
    text_plan: str,
    raw_llm: str,
    timestamp: str,
    plan: TradePlan | None = None,
) -> None:
    """Сохраняет отчёт в .txt и .json (расширенный с метаданными плана)."""
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    txt_path = os.path.join(reports_dir, f"{symbol}_plan_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_plan)

    # JSON: сырой ответ Actor + метаданные плана
    json_path = os.path.join(reports_dir, f"{symbol}_raw_{timestamp}.json")
    json_payload: dict = {"raw_llm_response": raw_llm}
    if plan:
        json_payload["parsed_plan"] = plan.model_dump()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    logger.info("Отчеты сохранены:\n - %s\n - %s", txt_path, json_path)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RalphTradeBot V2 — Elliott Wave анализ на базе LLM + Self-Correction"
    )
    parser.add_argument("--symbols",  nargs="+", default=SYMBOLS,     help="Торговые пары")
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES, help="Таймфреймы")
    parser.add_argument("--no-critic", action="store_true", help="Отключить Self-Correction (Critic)")
    parser.add_argument("--no-oi",     action="store_true", help="Отключить загрузку Open Interest")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    symbols: list[str]    = args.symbols
    timeframes: list[str] = args.timeframes
    use_critic: bool      = not args.no_critic
    use_oi: bool          = not args.no_oi

    logger.info(
        "Старт RalphTradeBot V2 | Монеты: %s | ТФ: %s | Critic: %s | OI: %s",
        ", ".join(symbols), ", ".join(timeframes),
        "ВКЛ" if use_critic else "ВЫКЛ",
        "ВКЛ" if use_oi else "ВЫКЛ",
    )

    for symbol in symbols:
        try:
            analyze_symbol(symbol, timeframes, use_critic=use_critic, use_oi=use_oi)
        except WaveEngineError as e:
            logger.error("Ошибка анализа %s: %s", symbol, e)
        except KeyboardInterrupt:
            logger.info("Прервано пользователем.")
            break

    print("\n✅ Анализ завершён.")
