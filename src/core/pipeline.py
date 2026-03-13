"""
pipeline.py — Ядро AI-пайплайна (Actor, Critic, Correction).
"""
import logging
import json
import os
import sys
from openai import OpenAI
from src.core.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from src.core.models import TradePlan, CriticFeedback, LLMContext
from src.ai.ai_prompt_builder import build_messages
from src.ai.critic_prompt import build_critic_messages, build_correction_messages
from src.trader.trading_plan_generator import parse_llm_response
from src.validator.hard_validator import validate_plan as hard_validate_plan
from src.core.math_actor import get_math_trade_plan

logger = logging.getLogger(__name__)

# Глобальные клиенты для кеширования соединений
_clients: dict[str, OpenAI] = {}

def get_llm_client(base_url: str, api_key: str) -> OpenAI:
    if base_url not in _clients:
        is_google = "googleapis.com" in (base_url or "")
        _clients[base_url] = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/ralphtradebot",
                "X-Title": "RalphTradeBot",
            } if not is_google else {},
            timeout=30.0,
        )
    return _clients[base_url]

def call_llm(messages: list[dict], temperature: float = 0.1) -> str:
    """
    Вызывает LLM через OpenRouter (стабильный канал для Gemini).
    """
    try:
        client = get_llm_client(OPENAI_BASE_URL, OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"❌ Ошибка вызова LLM ({OPENAI_MODEL}): {e}")
        return ""

def run_critic(draft_plan: TradePlan, context_summary: str) -> CriticFeedback | None:
    draft_json = draft_plan.model_dump_json(indent=2)
    messages = build_critic_messages(draft_json, context_summary)
    logger.info("🔍 Critic: проверка плана...")
    try:
        critic_text = call_llm(messages, temperature=0.0)
        start = critic_text.find('{')
        end = critic_text.rfind('}')
        if start == -1 or end == -1: return None
        critic_data = json.loads(critic_text[start:end + 1])
        return CriticFeedback(**critic_data)
    except Exception as e:
        logger.warning("Не удалось разобрать ответ Критика: %s", e)
        return None

def run_correction(draft_plan: TradePlan, feedback: CriticFeedback, market_data_json: str) -> TradePlan | None:
    draft_json = draft_plan.model_dump_json(indent=2)
    critic_json = feedback.model_dump_json(indent=2)
    messages = build_correction_messages(draft_json, critic_json, market_data_json)
    logger.info("🔧 Correction: Actor исправляет план...")
    try:
        corrected_text = call_llm(messages, temperature=0.1)
        return parse_llm_response(corrected_text)
    except Exception: return None

def build_context_summary(context: LLMContext) -> str:
    lines = [f"Символ: {context.symbol}"]
    for tf_data in context.timeframes:
        vecs = tf_data.vectors
        if vecs:
            last = vecs[-1]
            lines.append(
                f"[{tf_data.timeframe}] Последний вектор: {last.start_price:.4g}→{last.end_price:.4g} "
                f"({last.price_change_percent:+.1f}%), RSI={tf_data.current_rsi:.1f}"
            )
    return "\n".join(lines)

def run_full_ai_pipeline(context: LLMContext, use_critic: bool = True) -> TradePlan:
    """
    Запускает оптимизированный цикл: Actor -> Hard Validator -> (Optional Critic) -> Correction.
    """
    # 1. Actor: первичный анализ
    messages = build_messages(context)
    llm_text = call_llm(messages)
    plan = parse_llm_response(llm_text)
    
    # --- FAST-TRACK: Мгновенный выход на WAIT ---
    direction = str(plan.trade_params.get("direction", "")).upper()
    if "WAIT" in direction or not direction:
        return plan

    # 2. Hard Validation & Self-Healing (Актер исправляет свои технические ошибки)
    market_data_json = context.model_dump_json(indent=2)
    
    MAX_HARD_RETRIES = 1
    for i in range(MAX_HARD_RETRIES):
        hard_errors = hard_validate_plan(plan)
        if not hard_errors:
            break
            
        logger.warning(f"  [Hard-Fix {i+1}] Обнаружены ошибки: {hard_errors[:2]}")
        
        # Создаем фиктивный фидбек для исправления
        fake_feedback = CriticFeedback(
            is_valid=False,
            critical_errors=hard_errors,
            recommendations=["Исправь указанные технические ошибки в уровнях или волновой структуре."]
        )
        
        corrected = run_correction(plan, fake_feedback, market_data_json)
        if corrected:
            plan = corrected
            # Если после коррекции стал WAIT — выходим
            if "WAIT" in str(plan.trade_params.get("direction", "")).upper():
                return plan
        else:
            break

    if not use_critic:
        return plan

    # 3. LLM Critic: интеллектуальная проверка (только если план прошел Hard Validation)
    context_summary = build_context_summary(context)
    feedback = run_critic(plan, context_summary)
    
    if feedback and (feedback.critical_errors or not feedback.is_valid):
        logger.warning(f"  [Critic-Fix] Ошибки LLM: {feedback.critical_errors[:2]}")
        corrected = run_correction(plan, feedback, market_data_json)
        if corrected:
            plan = corrected

    return plan

def run_math_pipeline(context: LLMContext, use_ai_confirm: bool = False) -> TradePlan:
    """
    Запускает математический поиск сигналов (Fast & Reliable).
    """
    # 1. Math Actor: ищем паттерны кодом
    plan = get_math_trade_plan(context)
    
    # Если на входе WAIT — сразу выходим
    if "WAIT" in str(plan.trade_params.get("direction", "")).upper():
        return plan

    # 2. Опционально: ИИ-Критик подтверждает математический сетап
    if use_ai_confirm:
        context_summary = build_context_summary(context)
        feedback = run_critic(plan, context_summary)
        
        if feedback and (feedback.critical_errors or not feedback.is_valid):
            logger.warning(f"  [AI-Rejection] ИИ отклонил мат. сигнал: {feedback.critical_errors[:1]}")
            # Если ИИ против — превращаем в WAIT
            plan.trade_params["direction"] = "WAIT"
            plan.detailed_logic += f" | ОТКЛОНЕНО ИИ: {feedback.critical_errors}"

    return plan
