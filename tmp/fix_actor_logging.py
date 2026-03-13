
import os

path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_until = -1

for i, line in enumerate(lines):
    if i < skip_until: continue

    # 1. Disable redundant ZigZag filter, use extrema directly
    if '# --- ПУНКТ 3: Настоящая ZigZag-фильтрация (Macro-filter) ---' in line:
        new_lines.append('        # Используем экстремумы напрямую (уже профильтрованы в preprocessor)\n')
        new_lines.append('        final_extrema = extrema\n')
        # Skip original filtering block (31 to 72 approx)
        for j in range(i, len(lines)):
            if 'recent_6 = final_extrema[-15:]' in lines[j]:
                skip_until = j
                break
        continue

    # 2. Improve logging in _convert_structure_to_plan
    if 'if not sl or sl == entry: return None' in line:
        new_lines.append('    if not sl or sl == entry:\n')
        new_lines.append('        logger.warning(f"  [PLAN] Отказ: SL == Entry ({sl})")\n')
        new_lines.append('        return None\n')
        continue
    if 'if direction == "LONG" and sl >= entry: return None' in line:
        new_lines.append('    if direction == "LONG" and sl >= entry:\n')
        new_lines.append('        logger.warning(f"  [PLAN] Отказ: LONG SL({sl}) >= Entry({entry})")\n')
        new_lines.append('        return None\n')
        continue
    if 'if direction == "SHORT" and sl <= entry: return None' in line:
        new_lines.append('    if direction == "SHORT" and sl <= entry:\n')
        new_lines.append('        logger.warning(f"  [PLAN] Отказ: SHORT SL({sl}) <= Entry({entry})")\n')
        new_lines.append('        return None\n')
        continue

    new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Simplified math_actor and added more plan-rejection logging.")
