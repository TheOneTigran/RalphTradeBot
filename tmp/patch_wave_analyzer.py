
import os
import re

path = r'c:\Users\user\Desktop\RalphTradeBot\src\math_engine\wave_analyzer.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update _check_diagonal for Expanding Diagonals
diagonal_old = r'''    len1 = abs\(p1.price - p0.price\)
    len2 = abs\(p2.price - p1.price\)
    len3 = abs\(p3.price - p2.price\)
    len4 = abs\(p4.price - p3.price\)
    len5 = abs\(p5.price - p4.price\)

    # Волна 4 ДОЛЖНА зайти в зону волны 1 \(ключ для diagonal\)
    if bullish and p4.price > p1.price:
        return None
    if not bullish and p4.price < p1.price:
        return None

    # Убывание длин: 1 > 3 > 5 и 2 > 4
    if not \(len1 > len3 > len5 \* 0.8\):
        return None
    if not \(len2 > len4 \* 0.8\):
        return None

    confidence = 55.0
    details_parts = \[f"W1={len1:.2f}>W3={len3:.2f}>W5={len5:.2f}. W4 зашла в зону W1."\]'''

diagonal_new = r'''    len1 = abs(p1.price - p0.price)
    len2 = abs(p2.price - p1.price)
    len3 = abs(p3.price - p2.price)
    len4 = abs(p4.price - p3.price)
    len5 = abs(p5.price - p4.price)

    # Волна 4 ДОЛЖНА зайти в зону волны 1 (ключ для diagonal)
    if bullish and p4.price > p1.price:
        return None
    if not bullish and p4.price < p1.price:
        return None

    # --- ПУНКТ 2: Сходящаяся vs Расширяющаяся Диагональ ---
    is_contracting = len1 > len3 > len5 * 0.8
    is_expanding = len1 < len3 < len5 * 1.2
    
    if not (is_contracting or is_expanding):
        return None
    
    if is_expanding:
        # В расширяющейся Волна 4 обычно глубже Волны 2
        if bullish and p4.price > p2.price: return None
        if not bullish and p4.price < p2.price: return None
        subtype = "Расширяющаяся"
    else:
        subtype = "Сходящаяся"

    confidence = 65.0
    details_parts = [f"{subtype} Диагональ: W1={len1:.0f}, W3={len3:.0f}, W5={len5:.0f}. W4 зашла в зону W1."]'''

content = re.sub(diagonal_old, diagonal_new, content)

# 2. Update _check_zigzag for Origin target
zigzag_old = r'''    # Тейк-профиты
    if bullish:
        fibo_targets = \[
            pC.price - \(total_len \* 0.382\),
            pC.price - \(total_len \* 0.618\)
        \]
    else:
        fibo_targets = \[
            pC.price \+ \(total_len \* 0.382\),
            pC.price \+ \(total_len \* 0.618\)
        \]'''

zigzag_new = r'''    # Тейк-профиты:
    # 1. ПУНКТ 4: Точка начала (Origin) - консервативная цель разворота
    fibo_targets = [round(p0.price, 2)]
    
    if bullish:
        fibo_targets.extend([
            round(pC.price - (total_len * 0.618), 2),
            round(pC.price - (total_len * 1.0), 2)
        ])
    else:
        fibo_targets.extend([
            round(pC.price + (total_len * 0.618), 2),
            round(pC.price + (total_len * 1.0), 2)
        ])'''

content = re.sub(zigzag_old, zigzag_new, content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Successfully patched wave_analyzer.py")
