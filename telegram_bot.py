"""
telegram_bot.py — Telegram интерфейс для RalphTradeBot V2.

Команды бота:
/start - Приветствие и справка
/analyze <SYMBOL> [TIMEFRAMES] - Запуск полного анализа (с Critic и графиками)
Пример: /analyze BTCUSDT 1d,4h,1h
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import FSInputFile
from aiogram.utils.markdown import hbold, hcode

from src.core.config import TELEGRAM_BOT_TOKEN
from src.core.exceptions import WaveEngineError, PlanParsingError
from src.fetcher.data_fetcher import fetch_all_timeframes, fetch_orderbook_walls, fetch_open_interest
from src.math_engine.math_preprocessor import preprocess_all
from src.ai.ai_prompt_builder import build_messages
from src.trader.trading_plan_generator import parse_llm_response, format_plan_for_user
from src.visualizer.chart_generator import generate_chart

# Импортируем LLM-логику из main
from main import (
    _call_llm,
    _run_critic,
    _run_correction,
    _build_context_summary,
    _save_report,
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN не найден в .env. Бот не может быть запущен.")
    sys.exit(1)

bot = Bot(token=TELEGRAM_BOT_TOKEN, parse_mode="HTML")
dp = Dispatcher()


def run_analysis_sync(symbol: str, timeframes: list[str]) -> tuple[str | None, list[str]]:
    """
    Синхронная функция, выполняющая весь пайплайн V2.
    Возвращает (отформатированный_отчет, пути_к_графикам).
    """
    logger.info("Bot API: Начало анализа %s", symbol)

    # 1. Fetch data
    all_candles = fetch_all_timeframes(symbol)
    orderbook_walls = fetch_orderbook_walls(symbol)
    oi_snapshot = fetch_open_interest(symbol, period="1d", limit=30)

    # 2. Math processing
    context = preprocess_all(
        symbol=symbol,
        all_candles=all_candles,
        timeframes=timeframes,
        orderbook_walls=orderbook_walls,
    )

    if oi_snapshot and context.timeframes:
        context.timeframes[-1].oi_data = oi_snapshot
        if oi_snapshot.oi_change_pct_24h is not None and context.timeframes[-1].vectors:
            last_vec = context.timeframes[-1].vectors[-1]
            oi_snapshot.oi_price_divergence = last_vec.is_bullish != (oi_snapshot.oi_change_pct_24h > 0)

    market_data_json = context.model_dump_json(indent=2)

    # 3. Actor LLM
    messages = build_messages(context)
    llm_text = _call_llm(messages)

    # 4. Parsing
    try:
        plan = parse_llm_response(llm_text)
    except PlanParsingError as e:
        logger.error("Ошибка парсинга Actor-ответа: %s", e)
        return None, []

    # 5. Critic (Self-Correction)
    context_summary = _build_context_summary(context)
    feedback = _run_critic(plan, context_summary)

    if feedback is not None:
        plan.critic_validated = feedback.is_valid
        plan.critic_warnings = feedback.warnings if feedback.warnings else None

        if not feedback.is_valid and feedback.critical_errors:
            corrected = _run_correction(plan, feedback, market_data_json)
            if corrected:
                corrected.critic_validated = True
                corrected.critic_warnings = feedback.warnings or None
                plan = corrected

    # 6. Formatting & Charts
    formatted_plan = format_plan_for_user(plan, symbol)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _save_report(symbol, formatted_plan, llm_text, timestamp, plan)

    chart_paths = []
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    tfs_to_plot = [tf for tf in ["1d", "4h", "1h", "15m"] if tf in timeframes][:2]

    for tf in tfs_to_plot:
        tf_data = next((t for t in context.timeframes if t.timeframe == tf), None)
        if tf_data:
            c_path = os.path.join(reports_dir, f"{symbol}_{tf}_chart_{timestamp}.png")
            generate_chart(
                symbol=symbol,
                timeframe=tf,
                candles=all_candles.get(tf, []),
                vectors=tf_data.vectors,
                fib_clusters=tf_data.fib_clusters,
                save_path=c_path,
                plan=plan
            )
            if os.path.exists(c_path):
                chart_paths.append(c_path)

    return formatted_plan, chart_paths


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    text = (
        f"{hbold('🤖 Добро пожаловать в RalphTradeBot V2!')}\n\n"
        "Я бот для продвинутого фрактального Волнового Анализа Эллиотта.\n"
        "Использую ATR-свинги, кластеры Фибоначчи и Actor-Critic LLM пайплайн.\n\n"
        f"Команда: {hcode('/analyze BTCUSDT 1d,4h,1h')}\n"
    )
    await message.answer(text)


@dp.message(Command("analyze"))
async def cmd_analyze(message: types.Message):
    parts = message.text.split()
    if len(parts) < 2:
        await message.answer("Укажите пару. Пример: /analyze BTCUSDT 1d,4h,1h")
        return

    symbol = parts[1].upper()
    timeframes = ["1d", "4h", "1h", "15m", "5m"]
    if len(parts) > 2:
        timeframes = [t.strip().lower() for t in parts[2].split(",")]

    processing_msg = await message.answer(
        f"⏳ Начинаю глубокий анализ {hbold(symbol)}.\nЭто займет 1-2 минуты (LLM + Critic + графики)..."
    )

    try:
        # Запускаем синхронную логику в отдельном потоке (asyncio.to_thread), 
        # чтобы бот не зависал и мог обрабатывать команды других юзеров.
        formatted_plan, chart_paths = await asyncio.to_thread(
            run_analysis_sync, symbol, timeframes
        )

        if not formatted_plan:
            await processing_msg.edit_text("❌ Ошибка парсинга или получения ответа от LLM.")
            return

        # Телеграм поддерживает моноширинный текст через <code>
        response_text = f"<code>{formatted_plan}</code>"

        # Если есть графики, отправляем их альбомом (MediaGroup)
        if chart_paths:
            media = []
            for i, path in enumerate(chart_paths):
                # Подпись только к первой фотке
                caption = response_text if i == 0 else ""
                # Если текст слишком длинный (>1024 символов), Telegram обрежет caption.
                # Поэтому мы обрежем текст, а потом отправим полную версию отдельным сообщением.
                is_long = len(response_text) > 1000
                
                media.append(
                    types.InputMediaPhoto(
                        type='photo',
                        media=FSInputFile(path),
                        caption=caption if not is_long else "",
                        parse_mode="HTML"
                    )
                )
            
            await bot.send_media_group(chat_id=message.chat.id, media=media)

            # Если текст был длинный, отправляем отдельным моноширинным сообщением
            if len(response_text) > 1000 or not media:
                # Разбиваем на чанки по 4000 символов
                for x in range(0, len(response_text), 4000):
                    await message.answer(f"<code>{formatted_plan[x:x+4000]}</code>")
        else:
            await message.answer(response_text)

        await processing_msg.delete()

    except Exception as e:
        logger.exception("Ошибка при обработке запроса /analyze")
        await processing_msg.edit_text(f"❌ Критическая ошибка: {type(e).__name__} - {e}")


async def main():
    logger.info("Бот запущен.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
