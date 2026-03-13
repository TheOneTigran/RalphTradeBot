
import os

path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'recent_ts = {round(float(e.timestamp)) for e in extrema[-10:]}' in line:
        new_lines.append(line)
        new_lines.append('        logger.info(f"    [DEBUG] TF:{tf} | tf_structures:{len(tf_structures)} | recent_ts count:{len(recent_ts)}")\n')
        continue
    
    if 'if round(float(s.points[-1].timestamp)) in recent_ts:' in line:
        new_lines.append('            ts_val = round(float(s.points[-1].timestamp))\n')
        new_lines.append('            is_recent = ts_val in recent_ts\n')
        new_lines.append('            logger.info(f"    [DEBUG] Pattern:{s.pattern_type} | TS:{ts_val} | In Recent:{is_recent}")\n')
        new_lines.append('            if is_recent:\n')
        continue
    
    # Avoid duplicate 'if is_recent'
    if 'if round(float(s.points[-1].timestamp)) in recent_ts:' in line:
        continue

    new_lines.append(line)

# Also fix the selection logic if it's returning WAIT
with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Added more debug logging to math_actor.py")
