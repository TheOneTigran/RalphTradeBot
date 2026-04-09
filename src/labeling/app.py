"""
app.py — Streamlit HITL Dashboard.

Позволяет эксперту (вас) валидировать волновые разметки бота, 
формируя идеально чистый датасет для ML-слоя (XGBoost).
Запуск: streamlit run src/labeling/app.py
"""
import json
import logging
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys
import os

# Фикс импортов для запуска из корня проекта через streamlit
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.labeling.feedback_store import FeedbackStore
from src.labeling.historical_labeler import get_queue_items, mark_labeled
from src.storage.duckdb_store import get_store

# Настройка страницы
st.set_page_config(page_title="RalphTradeBot HITL", layout="wide")

@st.cache_resource
def get_feedback_store():
    return FeedbackStore()

fb_store = get_feedback_store()
db_store = get_store()

# --- Стилизация ---
st.markdown("""
<style>
.stButton>button { width: 100%; font-weight: bold; }
.btn-accept { background-color: #28a745 !important; color: white !important; }
.btn-reject { background-color: #dc3545 !important; color: white !important; }
.btn-correct { background-color: #ffc107 !important; color: black !important; }
</style>
""", unsafe_allow_html=True)

# Загрузка одной гипотезы из очереди
items = get_queue_items(limit=1)

st.title("RalphTradeBot V3 — Human-In-The-Loop Labeling")

if not items:
    st.success("Очередь пуста! Вы всё разметили. Сгенерируйте новую выборку через `historical_labeler.py`.")
    
    # Показать статистику
    stats = fb_store.get_stats()
    st.metric("Total Labeled", stats["total_labels"])
    st.metric("Accept Rate", f"{stats['accept_rate']:.1f}%")
else:
    item = items[0]
    
    # 1. Инфо-панель
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Symbol / TF", f"{item['symbol']} ({item['timeframe']})")
    with col_info2:
        st.metric("Pattern Type", item['pattern_type'])
    with col_info3:
        # Убираем эмодзи или используем гарантированно поддерживаемые Streamlit
        st.metric("Bot Score", f"{item['score']:.2f}", 
                  "+" if item['is_bullish'] else "-")

    # 2. График
    pts = item["wave_points"]
    if pts:
        start_ts = pts[0]["timestamp"]
        end_ts   = pts[-1]["timestamp"]
        wave_span = max(end_ts - start_ts, 3_600_000)  # минимум 1h

        # Берём 150 свечей левого контекста + весь паттерн + 50 свечей правого контекста
        tf_ms = {"1h": 3_600_000, "4h": 14_400_000, "15m": 900_000, "1d": 86_400_000}.get(
            item["timeframe"], 3_600_000
        )
        pad_left  = max(wave_span * 3.0, 150 * tf_ms)
        pad_right = max(wave_span * 0.5,  50 * tf_ms)

        fetch_since = int(start_ts - pad_left)
        fetch_until = int(end_ts   + pad_right)

        candles = db_store.get_ohlcv(
            item["symbol"], item["timeframe"],
            since_ts=fetch_since,
            until_ts=fetch_until,
            limit=700,
        )

        # Оставляем не больше 600 свечей (последние 600 — ближайшие к паттерну)
        if len(candles) > 600:
            candles = candles[-600:]

        if not candles:
            st.warning(
                f"⚠️ Нет OHLCV-данных для этого паттерна.\n"
                f"W0 timestamp: {start_ts}, W-last timestamp: {end_ts}.\n"
                "Запусти `python main.py --mode fetch` чтобы подгрузить историю."
            )
        else:
            df = pd.DataFrame(candles)
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")

            # RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df["rsi"] = 100 - (100 / (1 + gain / loss))

            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=("Price", "RSI (14)"),
                row_width=[0.25, 0.75],
            )

            # Свечи
            fig.add_trace(go.Candlestick(
                x=df["ts"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name="Price",
            ), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(
                x=df["ts"], y=df["rsi"],
                line=dict(color="#ab63fa", width=2), name="RSI",
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(150,150,150,0.5)", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(150,150,150,0.5)", row=2, col=1)

            # Волновая разметка
            wave_x = [pd.to_datetime(p["timestamp"], unit="ms") for p in pts]
            wave_y = [p["price"] for p in pts]
            line_color = "rgba(0,220,100,0.9)" if item["is_bullish"] else "rgba(255,80,80,0.9)"
            wave_labels = ["0","1","2","3","4","5"][:len(pts)]

            fig.add_trace(go.Scatter(
                x=wave_x, y=wave_y,
                mode="lines+markers+text",
                line=dict(color=line_color, width=2.5),
                marker=dict(size=9, color="white", line=dict(width=2, color=line_color)),
                text=wave_labels,
                textposition="top center",
                textfont=dict(size=13, color=line_color),
                name="Wave",
            ), row=1, col=1)

            # Вертикальные линии начала и конца паттерна
            fig.add_vline(x=wave_x[0],  line_dash="dot", line_color="rgba(255,255,0,0.4)", row=1, col=1)
            fig.add_vline(x=wave_x[-1], line_dash="dot", line_color="rgba(255,255,0,0.4)", row=1, col=1)

            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=700,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=False,
            )
            fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)
            
    # 3. Фичи (что увидел бот)
    st.subheader("Features")
    st.json(item["features"])
    
    st.markdown("---")
    
    # 4. Action Buttons (Форма для обновления стейта)
    st.subheader("Your Verdict")
    notes = st.text_input("Notes (Optional)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    def process_action(label: int, is_correct: bool = False):
        fb_store.submit_label(
            symbol=item['symbol'],
            timeframe=item['timeframe'],
            hypothesis_type=item['pattern_type'],
            features=item['features'],
            label=label,
            source="human_corrected" if is_correct else "algorithm",
            wave_points=item['wave_points'],
            notes=notes
        )
        mark_labeled(item["id"])
        st.rerun()

    with col1:
        if st.button("✅ ACCEPT (Valid)", use_container_width=True):
            process_action(1, False)
            
    with col2:
        if st.button("❌ REJECT (Trash)", use_container_width=True):
            process_action(0, False)
            
    with col3:
        if st.button("⏭ SKIP (Not Sure)", use_container_width=True):
            mark_labeled(item["id"])  # Просто убираем из очереди
            st.rerun()
            
    with col4:
        st.button("✏️ CORRECT (Manual)", use_container_width=True, 
                  help="This will open a manual coordinate editor in the final MVP", disabled=True)
        
    st.markdown("---")
    stats = fb_store.get_stats()
    st.caption(f"Labeled: {stats['total_labels']} | Accept Rate: {stats['accept_rate']:.1f}%")
