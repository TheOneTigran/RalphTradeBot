
import os
import re

# 1. Patch config.py
c_path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\config.py'
with open(c_path, 'r', encoding='utf-8') as f:
    c_content = f.read()
c_content = re.sub(r'MIN_RR_RATIO: float = float\(os.getenv\("MIN_RISK_REWARD_RATIO", ".*"\)\)', 
                   'MIN_RR_RATIO: float = float(os.getenv("MIN_RISK_REWARD_RATIO", "1.5"))', c_content)
with open(c_path, 'w', encoding='utf-8') as f:
    f.write(c_content)

# 2. Patch math_actor.py (offset -> 0.07)
m_path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(m_path, 'r', encoding='utf-8') as f:
    m_content = f.read()
m_content = re.sub(r'offset = atr \* .* if atr > 0 else \(prev_p\.price \* 0\.0005\)',
                   'offset = atr * 0.07 if atr > 0 else (prev_p.price * 0.0005)', m_content)
with open(m_path, 'w', encoding='utf-8') as f:
    f.write(m_content)

print("Applied final parameter adjustments (RR=1.5, offset=0.07).")
