"""
vision_filter.py — Vision AI клиент и валидатор импульсов для RalphTradeBot.

Каскад провайдеров: OpenRouter → Gemini Direct → Groq.
SHA256-кэширование для ускорения и экономии токенов.
"""
import os
import json
import base64
import hashlib
import logging
import socket
from pathlib import Path
from typing import Optional, Dict, Any

import requests

from config import (
    OPENROUTER_API_KEY, GEMINI_API_KEY, GROQ_API_KEY,
    OPENROUTER_API_URL, GROQ_API_URL,
    OPENROUTER_MODELS, GEMINI_MODELS, GROQ_MODELS,
    VISION_CACHE_FILE, MIN_VISION_CONFIRM_SCORE,
    VISION_TIMEOUT, VISION_TEMPERATURE,
)

socket.setdefaulttimeout(VISION_TIMEOUT)
logger = logging.getLogger(__name__)


def _load_cache() -> dict:
    if VISION_CACHE_FILE.exists():
        try:
            with open(VISION_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    try:
        with open(VISION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Ошибка сохранения кэша: {e}")


def _compute_hash(png_bytes: bytes, prompt_text: str) -> str:
    return hashlib.sha256(png_bytes + prompt_text.encode("utf-8")).hexdigest()


def _parse_json_response(content: str) -> Dict[str, Any]:
    if not content:
        raise ValueError("Пустой ответ модели")
        
    if "```json" in content:
        clean = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        clean = content.split("```")[1].split("```")[0].strip()
    else:
        clean = content.strip()
    
    try:
        return json.loads(clean)
    except Exception:
        s = content.find("{")
        e = content.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(content[s : e + 1])
        raise ValueError(f"JSON не найден в ответе: {content[:120]}...")


def _try_openrouter(data_url: str, prompt: str) -> Optional[Dict[str, Any]]:
    if not OPENROUTER_API_KEY:
        return None
    
    session = requests.Session()
    session.trust_env = False
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ralphtradebot.local",
        "X-Title": "RalphTradeBot",
    }
    
    for model in OPENROUTER_MODELS:
        try:
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                "temperature": VISION_TEMPERATURE,
                "max_tokens": 1500,
            }
            
            resp = session.post(
                OPENROUTER_API_URL, headers=headers, 
                json=payload, timeout=(5.0, 15.0)
            )
            
            if resp.status_code == 200:
                result = resp.json()
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    msg = choice.get("message", {})
                    candidates = []
                    if msg.get("content"):
                        candidates.append(msg["content"])
                    if msg.get("reasoning"):
                        candidates.append(msg["reasoning"])
                    
                    for text in candidates:
                        try:
                            parsed = _parse_json_response(text)
                            parsed["_provider"] = f"OpenRouter/{model}"
                            logger.info(f"✅ OpenRouter ({model}): Score {parsed.get('score', '?')}")
                            return parsed
                        except Exception:
                            continue
            else:
                logger.warning(f"OpenRouter ({model}) HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            logger.warning(f"OpenRouter ({model}) ошибка: {e}")
            continue
            
    return None


def _try_gemini(b64_img: str, prompt: str) -> Optional[Dict[str, Any]]:
    if not GEMINI_API_KEY:
        return None
    
    for model in GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64_img}}
                    ]
                }],
                "generationConfig": {
                    "temperature": VISION_TEMPERATURE,
                    "maxOutputTokens": 600,
                }
            }
            resp = requests.post(url, json=payload, timeout=VISION_TIMEOUT)
            if resp.status_code == 200:
                content = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                parsed = _parse_json_response(content)
                parsed["_provider"] = f"Gemini/{model}"
                logger.info(f"✅ Gemini ({model}): Score {parsed.get('score', '?')}")
                return parsed
        except Exception as e:
            logger.warning(f"Gemini ({model}) ошибка: {e}")
            continue
            
    return None


def _try_groq(data_url: str, prompt: str) -> Optional[Dict[str, Any]]:
    return None


def evaluate_elliott_impulse(
    png_bytes: bytes,
    prompt: str,
    min_score: int = MIN_VISION_CONFIRM_SCORE,
) -> Dict[str, Any]:
    """Главная точка входа: оценивает график через Vision AI с кэшированием."""
    img_hash = _compute_hash(png_bytes, prompt)
    cache = _load_cache()

    if img_hash in cache and cache[img_hash].get("provider") != "fallback":
        cached = cache[img_hash]
        c_score = cached.get("score", 0)
        passed = c_score >= min_score
        return {
            "passed": passed,
            "score": c_score,
            "reason": cached.get("reason", "Из кэша"),
            "provider": cached.get("provider", "cache"),
            "wave_direction": cached.get("wave_direction", "unclear"),
            "impulse_detected": cached.get("impulse_detected", passed),
            "rule_violations": cached.get("rule_violations", "none"),
        }

    b64_img = base64.b64encode(png_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64_img}"

    parsed = _try_openrouter(data_url, prompt)
    if parsed is None:
        parsed = _try_gemini(b64_img, prompt)
    if parsed is None:
        parsed = _try_groq(data_url, prompt)

    if parsed is not None:
        score = int(parsed.get("score", 0))
        reason = parsed.get("reason", "Паттерн подтвержден")
        provider = parsed.get("_provider", "vision_ai")
        wave_dir = parsed.get("wave_direction", "unclear")
        passed = score >= min_score

        cache[img_hash] = {
            "score": score,
            "reason": reason,
            "provider": provider,
            "wave_direction": wave_dir,
            "impulse_detected": passed,
            "rule_violations": parsed.get("rule_violations", "none"),
        }
        _save_cache(cache)

        return {
            "passed": passed,
            "score": score,
            "reason": reason,
            "provider": provider,
            "wave_direction": wave_dir,
            "impulse_detected": passed,
            "rule_violations": parsed.get("rule_violations", "none"),
        }

    # Fallback при недоступности внешних API
    return {
        "passed": False,
        "score": 50,
        "reason": "Все Vision API временно недоступны",
        "provider": "fallback",
        "wave_direction": "unclear",
        "impulse_detected": False,
        "rule_violations": "api_offline",
    }
