
import os
import re

# 1. Patch config.py
c_path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\config.py'
with open(c_path, 'r', encoding='utf-8') as f:
    c_content = f.read()
c_content = c_content.replace('MIN_RR_RATIO: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "2.5"))', 
                              'MIN_RR_RATIO: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.8"))')
with open(c_path, 'w', encoding='utf-8') as f:
    f.write(c_content)

# 2. Patch math_actor.py (offset 0.1 -> 0.05)
m_path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(m_path, 'r', encoding='utf-8') as f:
    m_content = f.read()
m_content = m_content.replace('offset = atr * 0.1 if atr > 0 else (prev_p.price * 0.0005)',
                              'offset = atr * 0.05 if atr > 0 else (prev_p.price * 0.0005)')
with open(m_path, 'w', encoding='utf-8') as f:
    f.write(m_content)

# 3. Patch wave_analyzer.py (Add more targets)
w_path = r'c:\Users\user\Desktop\RalphTradeBot\src\math_engine\wave_analyzer.py'
with open(w_path, 'r', encoding='utf-8') as f:
    w_content = f.read()

# Aggressive targets for Impulse
imp_old = r'''    for f in \[0.382, 0.5, 0.618\]:
        target = p5.price - \(total_len \* f\) if bullish else p5.price \+ \(total_len \* f\)
        fibo_targets.append\(round\(target, 2\)\)'''

imp_new = r'''    for f in [0.382, 0.618, 1.0, 1.618]:
        target = p5.price - (total_len * f) if bullish else p5.price + (total_len * f)
        fibo_targets.append(round(target, 2))'''

w_content = re.sub(imp_old, imp_new, w_content)

# Aggressive targets for Zigzag
zig_old = r'''        fibo_targets.extend\(\[
            round\(pC.price - \(total_len \* 0.618\), 2\),
            round\(pC.price - \(total_len \* 1.0\), 2\)
        \]\)'''

zig_new = r'''        fibo_targets.extend([
            round(pC.price - (total_len * 0.618), 2),
            round(pC.price - (total_len * 1.618), 2),
            round(pC.price - (total_len * 2.618), 2)
        ])'''

w_content = re.sub(zig_old, zig_new, w_content)

with open(w_path, 'w', encoding='utf-8') as f:
    f.write(w_content)

print("Applied global improvements patch.")
