
import os

path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Properly initialize tf_structures and pass sub_tf_vectors
old_block = r'''        # --- 6-точечные паттерны ---
        if len(recent_6) >= 6:
            for i in range(len(recent_6) - 5):
                pts = recent_6[i:i+6]
                for func in [_check_impulse, _check_diagonal, _check_triangle]:
                    res = func(pts, tf) if func != _check_impulse else func(pts, tf, vectors)
                    if res: tf_structures.append(res)'''

new_block = r'''        tf_structures = []
        tf_atr = tf_data.current_atr or 0
        
        # Находим sub_tf_vectors для фрактальной проверки
        sub_tf_map = {"1d": "4h", "4h": "1h", "1h": "15m", "15m": "5m", "5m": None}
        sub_tf_name = sub_tf_map.get(tf)
        sub_tf_vectors = next((t.vectors for t in context.timeframes if t.timeframe == sub_tf_name), None) if sub_tf_name else None

        # --- 6-точечные паттерны ---
        if len(recent_6) >= 6:
            for i in range(len(recent_6) - 5):
                pts = recent_6[i:i+6]
                for func in [_check_impulse, _check_diagonal, _check_triangle]:
                    # Передаем вектори и суб-векторы для фракталов и объема
                    if func == _check_impulse:
                        res = func(pts, tf, vectors, sub_tf_vectors)
                    elif func == _check_diagonal:
                        res = func(pts, tf, vectors)
                    else:
                        res = func(pts, tf)
                    if res: tf_structures.append(res)'''

text = text.replace(old_block, new_block)

# 2. Fix 4-point patterns to also use tf_structures (already using it but good to check)
# 3. Fix _check_forming_123 to match signature
text = text.replace('res = _check_forming_123(extrema[-4:], tf)', 'res = _check_forming_123(extrema[-4:], tf, sub_tf_vectors)')

# 4. Remove debug logs that might clutter
# (Keep some for now)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed initialization and data passing in math_actor.py")
 domestic_cat_test = False # Placeholder
 domestic_cat_test = False # Placeholder
 domestic_cat_test = False # Placeholder
 domestic_cat_test = False # Placeholder
