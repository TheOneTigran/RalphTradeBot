
import os

path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Make RR rejection VERY visible
text = text.replace('return TradePlan(wave_count_label="RR Filtered", main_scenario="WAIT", trade_params={"direction": "WAIT"})',
                    'print(f"!!! [RR_FILTER] Plan rejected for {best_tf} {best_s.pattern_type}"); return TradePlan(wave_count_label="RR Filtered", main_scenario="WAIT", trade_params={"direction": "WAIT"})')

# Log the rejection in _convert_structure_to_plan too
text = text.replace('logger.info(f"    [RR_FAIL] План не прошел валидацию', 'print(f"    [RR_FAIL] {direction} {tf} {s.pattern_type} RR={reward/risk:.2f}"); logger.info(f"    [RR_FAIL] План не прошел валидацию')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Added visible print debugging to math_actor.py")
