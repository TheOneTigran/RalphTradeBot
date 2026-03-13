
import os

path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the freshness loop with a more logged version
old_loop = r'''        # Фильтруем только "свежие" паттерны (используем исходные extrema для точности времени)
        recent_ts = {round(float(e.timestamp)) for e in extrema[-10:]} # Увеличили окно свежести
        for s in tf_structures:
            logger.info(f"  [MATH] Найден паттерн {s.pattern_type} ({tf}) {s.direction}, конфиденс: {s.confidence:.1f}")
            if round(float(s.points[-1].timestamp)) in recent_ts:'''

new_loop = r'''        # Фильтруем только "свежие" паттерны
        recent_ts = {round(float(e.timestamp)) for e in extrema[-10:]}
        logger.info(f"    [DEBUG] TF:{tf} | Extrema total:{len(extrema)} | Structures found:{len(tf_structures)}")
        for s in tf_structures:
            last_ts = round(float(s.points[-1].timestamp))
            is_recent = last_ts in recent_ts
            logger.info(f"  [MATH] Найден паттерн {s.pattern_type} ({tf}) {s.direction}, конфиденс: {s.confidence:.1f} | Fresh:{is_recent}")
            if is_recent:'''

text = text.replace(old_loop, new_loop)

# Also ensure RR filtering logs are visible
text = text.replace('logger.warning(f"    ↪ Пропуск: План не прошел валидацию', 'logger.info(f"    [RR_FAIL] План не прошел валидацию')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Applied better debug logging to math_actor.py")
