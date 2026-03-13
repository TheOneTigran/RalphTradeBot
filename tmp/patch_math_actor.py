
import os

path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_until = -1

for i, line in enumerate(lines):
    if i < skip_until:
        continue
    
    # fix scoring loop
    if 'for tf, s in all_valid_structures:' in line:
        new_lines.append('    for tf, s, atr in all_valid_structures:\n')
    
    # fix score_structure inner function
    elif 'def score_structure(item):' in line:
        new_lines.append(line)
        new_lines.append('        tf, s, atr = item\n')
        skip_until = i + 2
    
    # fix section 3: selection
    elif 'best_tf, best_s = all_valid_structures[0]' in line:
        new_lines.append('    current_idx = 0\n')
        new_lines.append('    best_tf, best_s, best_atr = all_valid_structures[current_idx]\n')
        new_lines.append('    main_plan = _convert_structure_to_plan(best_s, context, best_tf, best_atr)\n')
        # Skip original best_tf and main_plan lines (135, 136)
        skip_until = i + 2
        
    # fix R:R loop
    elif 'if not main_plan:' in line and 'Если лучший не прошел R:R' in lines[i+1]:
        # Replacing lines 143-151 approx
        new_lines.append('    if not main_plan:\n')
        new_lines.append('        for i_alt in range(1, len(all_valid_structures)):\n')
        new_lines.append('            best_tf, best_s, best_atr = all_valid_structures[i_alt]\n')
        new_lines.append('            main_plan = _convert_structure_to_plan(best_s, context, best_tf, best_atr)\n')
        new_lines.append('            if main_plan:\n')
        new_lines.append('                current_idx = i_alt\n')
        new_lines.append('                break\n')
        new_lines.append('    if not main_plan:\n')
        new_lines.append('        return TradePlan(wave_count_label="RR Filtered", main_scenario="WAIT", trade_params={"direction": "WAIT"})\n')
        
        # Also need alt_desc update
        new_lines.append('\n    alt_desc = "Нет альтернатив"\n')
        new_lines.append('    rem_idx = 0 if current_idx != 0 else 1\n')
        new_lines.append('    if len(all_valid_structures) > 1:\n')
        new_lines.append('        alt_tf, alt_s, _alt_atr = all_valid_structures[rem_idx]\n')
        new_lines.append('        alt_desc = f"Альтернатива: {alt_s.pattern_type} {alt_s.direction} на {alt_tf} ({alt_s.confidence:.0f}%)"\n')
        
        # Skip until section 153
        skip_until = i + 9 

    # fix function signature
    elif 'def _convert_structure_to_plan(s: WaveStructure, context: LLMContext, tf: str) -> Optional[TradePlan]:' in line:
        new_lines.append('def _convert_structure_to_plan(s: WaveStructure, context: LLMContext, tf: str, atr: float = 0) -> Optional[TradePlan]:\n')
    
    # fix entry logic
    elif 'if is_reversal:' in line and i > 170:
        new_lines.append(line)
        new_lines.append('        sl = last_p.price\n')
        new_lines.append('        offset = atr * 0.1 if atr > 0 else (prev_p.price * 0.0005)\n')
        new_lines.append('        if direction == "LONG":\n')
        new_lines.append('            entry = prev_p.price + offset\n')
        new_lines.append('        else:\n')
        new_lines.append('            entry = prev_p.price - offset\n')
        skip_until = i + 8 # Skip until sl_p
    else:
        new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Successfully patched math_actor.py")
