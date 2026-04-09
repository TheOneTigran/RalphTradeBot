"""
app_community.py — Standalone HITL-дашборд для деплоя на Streamlit Community Cloud.

Отличия от локальной версии:
- Использует SQLite (не DuckDB) для совместимости с облачным дисковым слоем Streamlit
- Данные OHLCV подгружаются напрямую с Bybit через ccxt при каждом запуске
- Labels хранятся в SQLite (персистентно через st.session_state + backends)
"""
import json
import sqlite3
import os
import time
import logging

import ccxt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Path to a writable SQLite DB (Streamlit Cloud allows writes to /tmp or repo root)
DB_PATH = os.environ.get("SQLITE_PATH", "data/community.sqlite3")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

logger = logging.getLogger(__name__)

# ── Страница ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RalphTradeBot — Wave HITL", layout="wide", page_icon="🌊")

st.markdown("""
<style>
h1 { font-size: 1.5rem !important; }
.block-container { padding-top: 1rem; }
.stButton > button { width: 100%; font-weight: bold; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── SQLite хранилище ─────────────────────────────────────────────────────────
@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT, timeframe TEXT, ts INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, timeframe, ts)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labeling_queue (
            id TEXT PRIMARY KEY, symbol TEXT, timeframe TEXT,
            pattern_type TEXT, score REAL, is_bullish INTEGER,
            wave_points_json TEXT, features_json TEXT,
            status TEXT DEFAULT 'pending'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS labeled_setups (
            id TEXT PRIMARY KEY, created_at TEXT, symbol TEXT, timeframe TEXT,
            pattern_type TEXT, features_json TEXT, label INTEGER,
            wave_points_json TEXT, notes TEXT, reviewer TEXT
        )
    """)
    conn.commit()
    return conn


db = get_db()


# ── Загрузка истории с Bybit ─────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Загружаю историю с Bybit...")
def fetch_and_cache_ohlcv(symbol: str, timeframe: str, limit: int = 5000):
    """Загружает OHLCV и кэширует на 1 час."""
    exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})
    tf_seconds = exchange.parse_timeframe(timeframe)
    
    all_raw = []
    latest = exchange.fetch_ohlcv(symbol, timeframe, limit=1)
    if not latest:
        return []
    last_ts = latest[0][0]
    since = last_ts - (limit * tf_seconds * 1000)
    
    while len(all_raw) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=min(1000, limit - len(all_raw)))
        if not batch:
            break
        all_raw.extend(batch)
        since = batch[-1][0] + tf_seconds * 1000
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    
    # Deduplicate and sort
    unique = {c[0]: c for c in all_raw}
    sorted_raw = sorted(unique.values(), key=lambda x: x[0])[-limit:]
    
    candles = [{"ts": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]} for c in sorted_raw]
    
    # Store in SQLite
    db.executemany(
        "INSERT OR REPLACE INTO ohlcv (symbol, timeframe, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
        [(symbol, timeframe, c["ts"], c["open"], c["high"], c["low"], c["close"], c["volume"]) for c in candles]
    )
    db.commit()
    return candles


def get_ohlcv_range(symbol, timeframe, since_ts, until_ts):
    rows = db.execute(
        "SELECT ts,open,high,low,close,volume FROM ohlcv WHERE symbol=? AND timeframe=? AND ts>=? AND ts<=? ORDER BY ts",
        (symbol, timeframe, since_ts, until_ts)
    ).fetchall()
    return [{"ts": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in rows]


# ── Генерация очереди из истории ──────────────────────────────────────────────
def build_queue(symbol, timeframe, candles):
    """Run local Wave Engine on fetched candles and populate queue."""
    import sys
    sys.path.insert(0, ".")
    from src.wave_engine.extremum_finder import ExtremumFinder
    from src.wave_engine.hypothesis_dag import HypothesisDAG

    high = np.array([c["high"] for c in candles])
    low = np.array([c["low"] for c in candles])
    close = np.array([c["close"] for c in candles])
    timestamps = np.array([c["ts"] for c in candles])

    finder = ExtremumFinder(mode="single")
    extrema = finder.find(high, low, close, timestamps, fractal_n=5, atr_mult=3.0)

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
        existing = db.execute("SELECT id FROM labeling_queue WHERE id=?", (h.id,)).fetchone()
        if not existing:
            db.execute(
                "INSERT OR IGNORE INTO labeling_queue (id,symbol,timeframe,pattern_type,score,is_bullish,wave_points_json,features_json) VALUES (?,?,?,?,?,?,?,?)",
                (h.id, symbol, timeframe, h.pattern_type.value, h.confidence, int(h.is_bullish),
                 json.dumps(pts), json.dumps(h.features))
            )
            added += 1
    db.commit()
    return added


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🌊 RalphTradeBot — Elliott Wave HITL Labeling")
st.caption("Совместная разметка волновых гипотез для обучения ML-слоя")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Настройки")
    symbol = st.selectbox("Символ", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    timeframe = st.selectbox("Таймфрейм", ["1h", "4h", "1d"])
    reviewer = st.text_input("Ваше имя (волновик)", value="anonymous")

    if st.button("🔄 Загрузить историю и сгенерировать паттерны"):
        with st.spinner("Загрузка данных с Bybit..."):
            candles = fetch_and_cache_ohlcv(symbol, timeframe, limit=5000)
        with st.spinner("Запуск Wave Engine..."):
            n = build_queue(symbol, timeframe, candles)
        st.success(f"Добавлено {n} новых гипотез в очередь.")
        st.rerun()

    pending = db.execute("SELECT COUNT(*) FROM labeling_queue WHERE status='pending'").fetchone()[0]
    labeled = db.execute("SELECT COUNT(*) FROM labeled_setups").fetchone()[0]
    accept_n = db.execute("SELECT COUNT(*) FROM labeled_setups WHERE label=1").fetchone()[0]
    st.divider()
    st.metric("В очереди", pending)
    st.metric("Размечено всего", labeled)
    st.metric("Процент Accept", f"{(accept_n/labeled*100):.0f}%" if labeled else "—")


# Main area
row = db.execute(
    "SELECT id,symbol,timeframe,pattern_type,score,is_bullish,wave_points_json,features_json FROM labeling_queue WHERE status='pending' ORDER BY score DESC LIMIT 1"
).fetchone()

if not row:
    st.info("✅ Очередь пуста! Нажми «Загрузить историю» в боковой панели чтобы сгенерировать новые гипотезы.")
else:
    item_id, sym, tf, pat, score, is_bull, wp_json, feat_json = row
    pts = json.loads(wp_json)
    feats = json.loads(feat_json)

    # Header metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Символ / ТФ", f"{sym} ({tf})")
    c2.metric("Паттерн", pat)
    c3.metric("Score", f"{score:.2f}", "▲" if is_bull else "▼")

    # Load candles and render chart
    if pts:
        start_ts = pts[0]["timestamp"]
        end_ts = pts[-1]["timestamp"]
        wave_span = max(end_ts - start_ts, 3_600_000)
        tf_ms = {"1h": 3_600_000, "4h": 14_400_000, "15m": 900_000, "1d": 86_400_000}.get(tf, 3_600_000)
        pad_l = max(wave_span * 3.0, 150 * tf_ms)
        pad_r = max(wave_span * 0.5, 50 * tf_ms)

        candles = get_ohlcv_range(sym, tf, int(start_ts - pad_l), int(end_ts + pad_r))
        if len(candles) > 600:
            candles = candles[-600:]

        if not candles:
            st.warning("Нет данных для этого паттерна. Нажми «Загрузить историю» в боковой панели.")
        else:
            df = pd.DataFrame(candles)
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            d = df["close"].diff()
            g = d.where(d > 0, 0).rolling(14).mean()
            l = (-d.where(d < 0, 0)).rolling(14).mean()
            df["rsi"] = 100 - (100 / (1 + g / l))

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03, subplot_titles=("Price", "RSI (14)"),
                                row_width=[0.25, 0.75])
            fig.add_trace(go.Candlestick(x=df["ts"], open=df["open"], high=df["high"],
                                          low=df["low"], close=df["close"], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["ts"], y=df["rsi"],
                                     line=dict(color="#ab63fa", width=2), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(150,150,150,0.4)", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(150,150,150,0.4)", row=2, col=1)

            wave_x = [pd.to_datetime(p["timestamp"], unit="ms") for p in pts]
            wave_y = [p["price"] for p in pts]
            lc = "rgba(0,220,100,0.9)" if is_bull else "rgba(255,80,80,0.9)"
            labels = ["0","1","2","3","4","5"][:len(pts)]
            fig.add_trace(go.Scatter(x=wave_x, y=wave_y, mode="lines+markers+text",
                                     line=dict(color=lc, width=2.5),
                                     marker=dict(size=9, color="white", line=dict(width=2, color=lc)),
                                     text=labels, textposition="top center",
                                     textfont=dict(size=13, color=lc), name="Wave"), row=1, col=1)
            fig.add_vline(x=wave_x[0], line_dash="dot", line_color="rgba(255,255,0,0.3)")
            fig.add_vline(x=wave_x[-1], line_dash="dot", line_color="rgba(255,255,0,0.3)")
            fig.update_layout(xaxis_rangeslider_visible=False, height=680,
                              template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
            fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

    # Features expander
    with st.expander("🔬 Фичи (что увидел бот)", expanded=False):
        st.json(feats)

    # Verdict
    st.divider()
    st.subheader("Ваш вердикт")
    notes = st.text_input("Комментарий (необязательно)", key="notes_input")

    def submit(label: int):
        import uuid as _uuid
        db.execute(
            "INSERT INTO labeled_setups (id,created_at,symbol,timeframe,pattern_type,features_json,label,wave_points_json,notes,reviewer) VALUES (?,datetime('now'),?,?,?,?,?,?,?,?)",
            (str(_uuid.uuid4()), sym, tf, pat, feat_json, label, wp_json, notes, reviewer)
        )
        db.execute("UPDATE labeling_queue SET status='labeled' WHERE id=?", (item_id,))
        db.commit()
        st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ ACCEPT", use_container_width=True):
            submit(1)
    with col2:
        if st.button("❌ REJECT", use_container_width=True):
            submit(0)
    with col3:
        if st.button("⏭ SKIP", use_container_width=True):
            db.execute("UPDATE labeling_queue SET status='labeled' WHERE id=?", (item_id,))
            db.commit()
            st.rerun()
