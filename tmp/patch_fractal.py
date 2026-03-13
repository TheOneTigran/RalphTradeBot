
import os
import re

path = r'c:\Users\user\Desktop\RalphTradeBot\src\math_engine\wave_analyzer.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add _check_wxy_core
wxy_core = r'''
def _check_wxy_core(pts: List[Extremum]) -> bool:
    """Быстрая проверка 4 точек на W-X-Y."""
    p0, pW, pX, pY = pts
    bullish = pW.price > p0.price
    lenW = abs(pW.price - p0.price)
    lenX = abs(pX.price - pW.price)
    lenY = abs(pY.price - pX.price)
    if lenW == 0: return False
    if lenX >= lenW: return False
    if bullish and pY.price <= pW.price: return False
    if not bullish and pY.price >= pW.price: return False
    return True
'''

# Insert before _check_impulse
content = content.replace('def _check_impulse(', wxy_core + '\ndef _check_impulse(')

# 2. Update _fractal_validate_wave logic
frac_old = r'''    if expected_internal == "impulse":
        # Ищем 5-волновой импульс внутри
        if len(sub_extrema) >= 6:
            for start_idx in range(len(sub_extrema) - 5):
                sub_pts = sub_extrema[start_idx: start_idx + 6]
                result = _check_impulse_core(sub_pts)
                if result:
                    return 10.0, f"✓ Фрактал: внутри найден {result} на младшем ТФ"
            return -10.0, "✗ Фрактал: внутри НЕ найден 5-волновой импульс"
        return -5.0, f"Недостаточно экстремумов для фрактальной проверки ({len(sub_extrema)})"'''

frac_new = r'''    if expected_internal == "impulse":
        # Ищем 5-волновой импульс внутри
        has_correct = False
        has_wrong = False
        if len(sub_extrema) >= 6:
            for start_idx in range(len(sub_extrema) - 5):
                sub_pts = sub_extrema[start_idx: start_idx + 6]
                if _check_impulse_core(sub_pts):
                    has_correct = True; break
        
        # Проверяем "не-импульсную" природу (WXY/Zigzag) - КРИТИКА Пункт 5
        if not has_correct and len(sub_extrema) >= 4:
            for start_idx in range(len(sub_extrema) - 3):
                sub_pts = sub_extrema[start_idx: start_idx + 4]
                if _check_zigzag_core(sub_pts) or _check_wxy_core(sub_pts):
                    has_wrong = True; break

        if has_correct:
            return 15.0, "✓ Фрактал: подтвержден внутренний микро-импульс"
        if has_wrong:
            return -20.0, "✗ Фрактал: вместо импульса обнаружена тройка (WXY/Zigzag)"
        return -5.0, f"Фрактал неясен (всего {len(sub_extrema)} точек)"'''

content = content.replace(frac_old, frac_new)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully applied fractal improvements to wave_analyzer.py")
