"""
trading_plan_generator.py — Парсинг ответа LLM и формирование итогового торгового плана.

Принимает сырую строку от LLM (JSON), валидирует через Pydantic-схему TradePlan
и готовит читабельный вывод для пользователя.
"""
from __future__ import annotations

import json
import re
import logging

from src.core.models import TradePlan
from src.core.exceptions import PlanParsingError

logger = logging.getLogger(__name__)


def parse_llm_response(llm_text: str) -> TradePlan:
    """
    Парсит сырой ответ LLM в объект TradePlan.

    Алгоритм:
      1. Извлекает чистый JSON из ответа (LLM может добавить markdown-разметку).
      2. Очищает типичные «грехи» бесплатных моделей (числа с комментарием без кавычек).
      3. Валидирует через Pydantic TradePlan.

    Args:
        llm_text: Сырой текст ответа от LLM.

    Returns:
        Объект TradePlan.

    Raises:
        PlanParsingError: Если JSON невалиден или не соответствует схеме.
    """
    json_str = _extract_json(llm_text)

    try:
        data = json.loads(json_str)
        plan = TradePlan(**data)
    except json.JSONDecodeError as e:
        raise PlanParsingError(
            f"LLM вернул невалидный JSON: {e}\n\nИсходный текст:\n{llm_text}"
        ) from e
    except Exception as e:
        raise PlanParsingError(
            f"Ошибка валидации торгового плана: {e}\n\nJSON:\n{json_str}"
        ) from e

    direction = plan.trade_params.get("direction", "?") if plan.trade_params else "?"
    rr = plan.trade_params.get("risk_reward_ratio", "?") if plan.trade_params else "?"
    main_prob = plan.main_scenario_probability
    logger.info(
        "Торговый план: %s | %s | R:R=%s | P=%s%%",
        direction, plan.wave_count_label, rr,
        main_prob if main_prob is not None else "?",
    )
    return plan


def _extract_json(text: str) -> str:
    """
    Живучий JSON-экстрактор для 'шумных' бесплатных моделей.

    Стратегия — безопасная, построчная:
    1. Извлекаем текст между первым '{' и последним '}'.
    2. Построчно: если значение = число + скобки без кавычек → оборачиваем.
    3. Убираем висячие запятые.
    """
    # 1. Извлекаем между крайними { и }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        raise PlanParsingError("JSON-блок не найден в ответе LLM.")

    raw_json = text[start_idx:end_idx + 1]

    # 2. Построчное исправление:
    # Паттерн: "ключ": 12345.6 (текст без кавычек) → "ключ": "12345.6 (текст)"
    lines = raw_json.split('\n')
    fixed_lines = []
    for line in lines:
        m = re.match(r'^(\s*"[\w_]+")\s*:\s*([\d][\d.]*\s*\(.+)$', line.rstrip(' ,'))
        if m:
            suffix = ',' if line.rstrip().endswith(',') else ''
            line = f'{m.group(1)}: "{m.group(2).strip()}"{suffix}'
        fixed_lines.append(line)

    raw_json = '\n'.join(fixed_lines)

    # 3. Убираем висячие запятые перед ] и }
    raw_json = re.sub(r',\s*([\]}])', r'\1', raw_json)

    return raw_json


def format_plan_for_user(plan: TradePlan, symbol: str) -> str:
    """
    Форматирует TradePlan в профессиональный аналитический отчет.
    Включает: вероятности сценариев (Multi-Count), статус проверки Критика.
    """
    trig = plan.trigger_prices or {}
    params = plan.trade_params or {}

    tp_list = params.get("take_profit_levels", [])
    tp_str = "\n".join([f"    • {tp}" for tp in tp_list]) if tp_list else "    • N/A"

    direction = str(params.get("direction", "N/A")).upper()
    is_wait = False
    
    if "LONG" in direction or "ЛОНГ" in direction:
        dir_emoji = "🟢 ЛОНГ"
    elif "SHORT" in direction or "ШОРТ" in direction:
        dir_emoji = "🔴 ШОРТ"
    elif "WAIT" in direction or "ЖДЕМ" in direction or "ОЖИДАНИЕ" in direction:
        dir_emoji = "⏳ РЕЖИМ ОЖИДАНИЯ (Нет надежного сетапа)"
        is_wait = True
    else:
        dir_emoji = f"⚪ {direction}"

    # Вероятности сценариев (Multi-Count)
    main_prob = plan.main_scenario_probability
    alt_prob  = plan.alternative_scenario_probability
    main_prob_str = f" [{main_prob}%]" if main_prob is not None else ""
    alt_prob_str  = f" [{alt_prob}%]"  if alt_prob  is not None else ""

    # Статус Критика (Self-Correction)
    if plan.critic_validated is True:
        critic_badge = "✅ VERIFIED BY CRITIC (нарушений правил не найдено)"
    elif plan.critic_validated is False:
        critic_badge = "🔧 CORRECTED BY CRITIC (план скорректирован после аудита)"
    else:
        critic_badge = "⬜ Без проверки Критика (используй --critic для аудита)"

    warnings_block = ""
    if plan.critic_warnings:
        warnings_block = "\n║\n║  💬 ЗАМЕЧАНИЯ КРИТИКА:\n" + "\n".join(
            [f"║    ⚠ {w}" for w in plan.critic_warnings]
        )

    # Координаты волн
    waves_block = ""
    if plan.waves_breakdown:
        waves_str = []
        for w in plan.waves_breakdown:
            import datetime
            dt_s = datetime.datetime.fromtimestamp(getattr(w, 'start_time', 0)/1000, tz=datetime.timezone.utc).strftime('%H:%M %d.%m') if getattr(w, 'start_time', 0) else 'N/A'
            dt_e = datetime.datetime.fromtimestamp(getattr(w, 'end_time', 0)/1000, tz=datetime.timezone.utc).strftime('%H:%M %d.%m') if getattr(w, 'end_time', 0) else 'N/A'
            waves_str.append(f"║    • {w.timeframe}: {w.wave_name} ({w.start_price:.2f} [{dt_s}] → {w.end_price:.2f} [{dt_e}])")
        waves_block = "\n╠══════════════════════════════════════════════════════════════════════════════\n║  КЛЮЧЕВЫЕ ВОЛНЫ:\n" + "\n".join(waves_str)

    # Multi-TF Confluence таблица
    confluence_block = ""
    if plan.waves_breakdown:
        from collections import OrderedDict
        tf_summary = OrderedDict()
        tf_order = ["1w", "1d", "4h", "1h", "15m", "5m"]
        
        for w in plan.waves_breakdown:
            tf_key = w.timeframe.lower()
            if tf_key not in tf_summary:
                tf_summary[tf_key] = w.wave_name
            else:
                tf_summary[tf_key] = w.wave_name  # берём последнюю волну как текущую фазу

        if tf_summary:
            confluence_lines = ["║  📊 СИНЕРГИЯ ТАЙМФРЕЙМОВ:"]
            for tf in tf_order:
                if tf in tf_summary:
                    wave = tf_summary[tf]
                    confluence_lines.append(f"║    {tf.upper():>4s}: Волна {wave}")
            confluence_block = "\n╠══════════════════════════════════════════════════════════════════════════════\n" + "\n".join(confluence_lines)

    # Разделяем HARD ошибки и LLM предупреждения
    warnings_block = ""
    if plan.critic_warnings:
        hard_warns = [w for w in plan.critic_warnings if w.startswith("[HARD]")]
        llm_warns = [w for w in plan.critic_warnings if not w.startswith("[HARD]")]
        
        parts = []
        if hard_warns:
            parts.append("║\n║  🔴 ПРОГРАММНАЯ ПРОВЕРКА (Hard Validator):")
            parts.extend([f"║    ✗ {w.replace('[HARD] ', '')}" for w in hard_warns])
        if llm_warns:
            parts.append("║\n║  💬 ЗАМЕЧАНИЯ LLM-КРИТИКА:")
            parts.extend([f"║    ⚠ {w}" for w in llm_warns])
        
        if parts:
            warnings_block = "\n" + "\n".join(parts)

    # Разный вывод для сделки и режима ожидания
    if is_wait:
        trade_block = f"""
СТАТУС:   {dir_emoji}
║  Ждем зону:    {trig.get("entry_zone", "N/A")} 
║  Подтверждение:{trig.get("confirmation_level", "N/A")}
""".strip('\n')
    else:
        trade_block = f"""
СДЕЛКА: {dir_emoji}
║  Стоп-лосс:    {params.get("stop_loss", "N/A")}
║  Цели:
{tp_str}
║  Риск/Прибыль: {params.get("risk_reward_ratio", "N/A")}
""".strip('\n')

    output = f"""
╔══════════════════════════════════════════════════════════════════════════════
║  RALPHTRADEBOT V2 │ ПРОФЕССИОНАЛЬНЫЙ АНАЛИЗ │ {symbol}
║  {critic_badge}
╠══════════════════════════════════════════════════════════════════════════════
║  РАЗМЕТКА: {plan.wave_count_label}{waves_block}{confluence_block}
╠══════════════════════════════════════════════════════════════════════════════
║  🧠 ЛОГИКА И ОБОСНОВАНИЕ:
║  {plan.detailed_logic}
╠══════════════════════════════════════════════════════════════════════════════
║  📈 ОСНОВНОЙ СЦЕНАРИЙ{main_prob_str}:
║  {plan.main_scenario}
╠══════════════════════════════════════════════════════════════════════════════
║  ⚠️  АЛЬТЕРНАТИВА{alt_prob_str}:{warnings_block}
║  {plan.alternative_scenario}
╠══════════════════════════════════════════════════════════════════════════════
║  🎯 ТРИГГЕРЫ И ЗОНЫ ДЕЙСТВИЯ:
║  • СИГНАЛ ВХОДА:    {trig.get("confirmation_level", "N/A")}
║  • ЗОНА НАБОРА:     {trig.get("entry_zone", "N/A")}
║  • ОТМЕНА ПЛАНА:    {trig.get("invalid_level", "N/A")}
╠══════════════════════════════════════════════════════════════════════════════
║  📊 {trade_block}
╚══════════════════════════════════════════════════════════════════════════════""".strip()

    return output

