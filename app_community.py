"""
app_community.py — RalphTradeBot HITL: публичный дашборд для Elliott Wave разметки.

Деплой: Streamlit Community Cloud (share.streamlit.io)
БД: SQLite (персистентно на диске Streamlit Cloud)
OHLCV: загружается с Bybit через ccxt с защитой от rate-limit
"""
import json
import sqlite3
import os
import sys
import time
import uuid

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, ".")

# ─────────────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("SQLITE_PATH", "data/community.sqlite3")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

st.set_page_config(
    page_title="RalphTradeBot — Wave HITL",
    layout="wide",
    page_icon="🌊",
    initial_sidebar_state="expanded",
)

# ── Стили ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Фон */
.stApp { background: #0d1117; }
section[data-testid="stSidebar"] { background: #161b22 !important; }

/* Хедер */
.ralph-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.ralph-header h1 {
    font-size: 1.6rem !important;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 !important;
}
.ralph-header p {
    font-size: 0.85rem;
    color: #8b949e;
    margin: 4px 0 0 0;
}

/* Карточки метрик */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-card .label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #e6edf3; }
.metric-card .sub   { font-size: 0.75rem; color: #58a6ff; }

/* Кнопки вердикта */
div[data-testid="stButton"] button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    height: 52px !important;
    border: 1px solid transparent !important;
    transition: transform 0.1s, box-shadow 0.1s;
}
div[data-testid="stButton"] button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.4); }

/* Accept */
[data-testid="baseButton-secondary"]:nth-child(1) button { background: #238636 !important; color: #fff !important; border-color: #2ea043 !important; }
/* Reject */
[data-testid="baseButton-secondary"]:nth-child(2) button { background: #da3633 !important; color: #fff !important; border-color: #f85149 !important; }
/* Skip */
[data-testid="baseButton-secondary"]:nth-child(3) button { background: #21262d !important; color: #8b949e !important; border-color: #30363d !important; }

/* Sidebar */
.sidebar-metric {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.sidebar-metric .s-label { color: #8b949e; font-size: 0.82rem; }
.sidebar-metric .s-value { color: #58a6ff; font-weight: 600; font-size: 1rem; }

/* Info/warn бокс */
.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 16px 20px;
    color: #c9d1d9;
    font-size: 0.92rem;
}
.stExpander { border: 1px solid #30363d !important; border-radius: 10px !important; }
.stExpander summary { font-weight: 600; color: #8b949e !important; }

/* Wave label badge */
.wave-badge {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #8b949e;
    margin-right: 6px;
}
.bull { border-color: #238636; color: #3fb950; background: #0f2f1c; }
.bear { border-color: #da3633; color: #f85149; background: #2f0f0f; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS ohlcv (
        symbol TEXT, timeframe TEXT, ts INTEGER,
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY (symbol, timeframe, ts))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS labeling_queue (
        id TEXT PRIMARY KEY, symbol TEXT, timeframe TEXT,
        pattern_type TEXT, score REAL, is_bullish INTEGER,
        wave_points_json TEXT, features_json TEXT,
        status TEXT DEFAULT 'pending')""")
    conn.execute("""CREATE TABLE IF NOT EXISTS labeled_setups (
        id TEXT PRIMARY KEY, created_at TEXT, symbol TEXT, timeframe TEXT,
        pattern_type TEXT, features_json TEXT, label INTEGER,
        wave_points_json TEXT, notes TEXT, reviewer TEXT)""")
    conn.commit()
    return conn


db = get_db()


# ── OHLCV Fetch (с защитой от rate-limit) ─────────────────────────────
@st.cache_data(ttl=3600)
def fetch_ohlcv_safe(symbol: str, timeframe: str, limit: int = 2000):
    """
    Загружает OHLCV с Bybit с паузами.
    Streamlit Cloud заблокирован на Binance (US IP), поэтому используем Bybit.
    Лимит батча Bybit — строго 200 свечей, иначе Rate Limit.
    """
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "rateLimit": 500,
        "options": {"defaultType": "linear"},
    })
    tf_sec = exchange.parse_timeframe(timeframe)
    batch_size = 200  # Bybit strict chunk size
    all_raw = []

    # Ccxt sometimes needs /USDT format depending on version, 
    # but Bybit linear accepts standard. We'll use ccxt standard.
    ccxt_symbol = symbol.replace("USDT", "/USDT:USDT") if "USDT" in symbol else symbol

    try:
        latest = exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=1)
        if not latest:
            return []
        since = latest[0][0] - (limit * tf_sec * 1000)

        while len(all_raw) < limit:
            remaining = limit - len(all_raw)
            fetch_n = min(batch_size, remaining)
            batch = exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=since, limit=fetch_n)
            if not batch:
                break
            all_raw.extend(batch)
            since = batch[-1][0] + tf_sec * 1000
            if len(batch) < fetch_n:
                break
            time.sleep(0.5)

    except Exception as e:
        logger.error(f"Fetch Error: {e}")
        if not all_raw:
            return []
        # Возвращаем что успели загрузить

    unique = {c[0]: c for c in all_raw}
    sorted_raw = sorted(unique.values(), key=lambda x: x[0])[-limit:]
    candles = [{"ts": c[0], "open": c[1], "high": c[2], "low": c[3],
                "close": c[4], "volume": c[5]} for c in sorted_raw]

    db.executemany(
        "INSERT OR REPLACE INTO ohlcv (symbol, timeframe, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
        [(symbol, timeframe, c["ts"], c["open"], c["high"], c["low"], c["close"], c["volume"]) for c in candles],
    )
    db.commit()
    return candles


def get_ohlcv_range(symbol, timeframe, since_ts, until_ts):
    rows = db.execute(
        "SELECT ts,open,high,low,close,volume FROM ohlcv WHERE symbol=? AND timeframe=? AND ts>=? AND ts<=? ORDER BY ts",
        (symbol, timeframe, since_ts, until_ts),
    ).fetchall()
    return [{"ts": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in rows]


# ── Wave Engine ────────────────────────────────────────────────────────
def build_queue(symbol, timeframe, candles):
    from src.wave_engine.extremum_finder import ExtremumFinder
    from src.wave_engine.hypothesis_dag import HypothesisDAG

    high  = np.array([c["high"]  for c in candles])
    low   = np.array([c["low"]   for c in candles])
    close = np.array([c["close"] for c in candles])
    ts    = np.array([c["ts"]    for c in candles])

    extrema = ExtremumFinder(mode="single").find(high, low, close, ts, fractal_n=5, atr_mult=3.0)
    dag = HypothesisDAG()
    for ext in extrema:
        dag.ingest_extremum(ext)

    added = 0
    for h in dag.completed_hypotheses:
        if h.confidence < 0.4 or len(h.points) < 4:
            continue
        p0, p1 = h.points[0].price, h.points[1].price
        if abs(p1 - p0) / p0 < 0.03:
            continue
        all_p = [p.price for p in h.points]
        if (max(all_p) - min(all_p)) / min(all_p) < 0.05:
            continue
        pts = [{"index": p.index, "price": p.price, "timestamp": p.timestamp, "is_high": p.is_high}
               for p in h.points]
        if not db.execute("SELECT id FROM labeling_queue WHERE id=?", (h.id,)).fetchone():
            db.execute(
                "INSERT OR IGNORE INTO labeling_queue (id,symbol,timeframe,pattern_type,score,is_bullish,wave_points_json,features_json) VALUES (?,?,?,?,?,?,?,?)",
                (h.id, symbol, timeframe, h.pattern_type.value, h.confidence,
                 int(h.is_bullish), json.dumps(pts), json.dumps(h.features)),
            )
            added += 1
    db.commit()
    return added


# ═══════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="ralph-header">
  <div>🌊</div>
  <div>
    <h1>RalphTradeBot — Elliott Wave HITL</h1>
    <p>Совместная разметка волновых гипотез · Строим золотой датасет для ML</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Параметры")
    symbol    = st.selectbox("Символ", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    timeframe = st.selectbox("Таймфрейм", ["1h", "4h", "1d"])
    reviewer  = st.text_input("Ваше имя (волновик)", value="anonymous",
                               placeholder="Введи ник — он сохранится в датасет")

    st.markdown("---")

    if st.button("📥 Загрузить историю и сканировать паттерны", use_container_width=True):
        if not reviewer.strip() or reviewer.strip().lower() == "anonymous":
            st.error("⚠️ Пожалуйста, введите ваше имя (ник) перед загрузкой!")
        else:
            with st.spinner(f"Загрузка {symbol} {timeframe} с Bybit (может занять 5-10 сек)..."):
                candles = fetch_ohlcv_safe(symbol, timeframe, limit=2000)
            if not candles:
                st.error("Не удалось загрузить историю. Попробуй позже (Bybit rate limit).")
            else:
                with st.spinner(f"Wave Engine анализирует {len(candles)} свечей..."):
                    n = build_queue(symbol, timeframe, candles)
                st.success(f"✅ Добавлено {n} новых гипотез в очередь!")
                st.rerun()

    st.markdown("---")

    pending  = db.execute("SELECT COUNT(*) FROM labeling_queue WHERE status='pending'").fetchone()[0]
    labeled  = db.execute("SELECT COUNT(*) FROM labeled_setups").fetchone()[0]
    accept_n = db.execute("SELECT COUNT(*) FROM labeled_setups WHERE label=1").fetchone()[0]
    reject_n = labeled - accept_n

    st.markdown(f"""
    <div class="sidebar-metric"><span class="s-label">В очереди</span><span class="s-value">{pending}</span></div>
    <div class="sidebar-metric"><span class="s-label">Размечено</span><span class="s-value">{labeled}</span></div>
    <div class="sidebar-metric"><span class="s-label">Accept ✅</span><span class="s-value" style="color:#3fb950">{accept_n}</span></div>
    <div class="sidebar-metric"><span class="s-label">Reject ❌</span><span class="s-value" style="color:#f85149">{reject_n}</span></div>
    """, unsafe_allow_html=True)

    if labeled > 0:
        pct = accept_n / labeled * 100
        st.progress(int(pct), text=f"Accept rate: {pct:.0f}%")


# ── Main: Current Item ─────────────────────────────────────────────────
row = db.execute(
    "SELECT id,symbol,timeframe,pattern_type,score,is_bullish,wave_points_json,features_json "
    "FROM labeling_queue WHERE status='pending' ORDER BY score DESC LIMIT 1"
).fetchone()

if not row:
    st.markdown("""
    <div class="info-box">
        ✅ <strong>Очередь пуста!</strong><br>
        Нажми <strong>«Загрузить историю»</strong> в боковой панели слева чтобы сгенерировать новые гипотезы.
    </div>""", unsafe_allow_html=True)
else:
    item_id, sym, tf, pat, score, is_bull, wp_json, feat_json = row
    pts  = json.loads(wp_json)
    feats = json.loads(feat_json)

    # Info bar
    bull_cls  = "bull" if is_bull else "bear"
    direction = "LONG" if is_bull else "SHORT"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
      <span class="wave-badge">{sym}</span>
      <span class="wave-badge">{tf}</span>
      <span class="wave-badge {bull_cls}">{direction}</span>
      <span class="wave-badge">{pat}</span>
      <span class="wave-badge" style="color:#f0c040;border-color:#9e7c00;background:#1f1a00">Score {score:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    if pts:
        start_ts  = pts[0]["timestamp"]
        end_ts    = pts[-1]["timestamp"]
        wave_span = max(end_ts - start_ts, 3_600_000)
        tf_ms = {"1h": 3_600_000, "4h": 14_400_000, "15m": 900_000, "1d": 86_400_000}.get(tf, 3_600_000)
        pad_l = max(wave_span * 3.0, 150 * tf_ms)
        pad_r = max(wave_span * 0.5,  50 * tf_ms)

        candles = get_ohlcv_range(sym, tf, int(start_ts - pad_l), int(end_ts + pad_r))
        if len(candles) > 600:
            candles = candles[-600:]

        if not candles:
            st.markdown("""<div class="info-box">
                📭 Нет свечных данных для визуализации.<br>
                Нажми <strong>«Загрузить историю»</strong> в боковой панели.
            </div>""", unsafe_allow_html=True)
        else:
            df = pd.DataFrame(candles)
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            d = df["close"].diff()
            g = d.where(d > 0, 0).rolling(14).mean()
            l = (-d.where(d < 0, 0)).rolling(14).mean()
            df["rsi"] = 100 - (100 / (1 + g / l))

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.75, 0.25],
            )
            # Свечи
            fig.add_trace(go.Candlestick(
                x=df["ts"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                increasing_line_color="#3fb950", decreasing_line_color="#f85149",
                name="Price",
            ), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(
                x=df["ts"], y=df["rsi"],
                line=dict(color="#ab63fa", width=1.8), name="RSI",
            ), row=2, col=1)
            fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.05)",
                          line_width=0, row=2, col=1)
            fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(63,185,80,0.05)",
                          line_width=0, row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="rgba(248,81,73,0.4)", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="rgba(63,185,80,0.4)", row=2, col=1)

            # Волна
            wave_x = [pd.to_datetime(p["timestamp"], unit="ms") for p in pts]
            wave_y = [p["price"]                                  for p in pts]
            lc   = "#3fb950" if is_bull else "#f85149"
            lbls = ["0","1","2","3","4","5"][:len(pts)]
            fig.add_trace(go.Scatter(
                x=wave_x, y=wave_y, mode="lines+markers+text",
                line=dict(color=lc, width=2.5, dash="solid"),
                marker=dict(size=10, color="#0d1117", line=dict(width=2.5, color=lc)),
                text=lbls, textposition="top center",
                textfont=dict(size=13, color=lc, family="Inter"),
                name="Wave",
            ), row=1, col=1)

            # Границы паттерна
            for vx in [wave_x[0], wave_x[-1]]:
                fig.add_vline(x=vx, line_dash="dot",
                              line_color="rgba(240,192,64,0.35)", row="all", col=1)

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=660,
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                margin=dict(l=8, r=8, t=12, b=8),
                showlegend=False,
                font=dict(family="Inter"),
                xaxis2=dict(showgrid=True, gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                yaxis2=dict(gridcolor="#21262d", range=[0, 100]),
            )
            fig.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    # Features
    with st.expander("🔬 Фичи — что увидел алгоритм", expanded=False):
        st.json(feats)

    st.markdown("---")

    # Verdict
    notes = st.text_input("💬 Комментарий (необязательно)", key="notes_input",
                           placeholder="Например: W4 слишком мелкая, нет чередования...")

    def submit(label: int):
        if not reviewer.strip() or reviewer.strip().lower() == "anonymous":
            st.toast("⚠️ Пожалуйста, введите ваше имя в боковой панели слева!")
            return
            
        db.execute(
            "INSERT INTO labeled_setups (id,created_at,symbol,timeframe,pattern_type,features_json,label,wave_points_json,notes,reviewer) "
            "VALUES (?,datetime('now'),?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), sym, tf, pat, feat_json, label, wp_json, notes, reviewer),
        )
        db.execute("UPDATE labeling_queue SET status='labeled' WHERE id=?", (item_id,))
        db.commit()
        st.rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅  ACCEPT — Разметка верна", use_container_width=True):
            submit(1)
    with c2:
        if st.button("❌  REJECT — Мусор", use_container_width=True):
            submit(0)
    with c3:
        if st.button("⏭  SKIP — Не уверен", use_container_width=True):
            db.execute("UPDATE labeling_queue SET status='labeled' WHERE id=?", (item_id,))
            db.commit()
            st.rerun()
