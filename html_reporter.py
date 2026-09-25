"""
html_reporter.py — Генератор интерактивного HTML-отчета для RalphTradeBot.

Создает стильный автономный дашборд 'RalphTradeBot backtest.html'
в эстетике TradingView / Bloomberg Terminal:
- Ключевые KPI (Win Rate, PnL, частота сигналов, средняя длительность импульса).
- Интерактивная таблица с живым поиском и фильтрами (монета, ТФ, тир качества).
- Встроенный модальный просмотр графиков сигналов высокого разрешения.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RalphTradeBot Backtest Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-main: #0c0f17;
      --bg-card: #141926;
      --bg-card-hover: #1b2234;
      --border: #242c42;
      --border-light: #323d5a;
      --text-main: #e2e8f0;
      --text-dim: #94a3b8;
      --text-muted: #64748b;
      --cyan: #00e5ff;
      --gold: #ffd700;
      --green: #089981;
      --red: #f23645;
      --purple: #7b61ff;
      --font-main: 'Outfit', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-main);
      color: var(--text-main);
      font-family: var(--font-main);
      line-height: 1.5;
      padding: 28px;
    }
    .container { max-width: 1560px; margin: 0 auto; }
    
    /* Header */
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .brand-logo {
      width: 46px; height: 46px;
      background: linear-gradient(135deg, var(--cyan), var(--purple));
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 24px; font-weight: 800; color: #000;
      box-shadow: 0 0 24px rgba(0, 229, 255, 0.35);
    }
    .brand-title h1 { font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }
    .brand-title p { font-size: 13px; color: var(--text-dim); }
    .header-meta { text-align: right; font-size: 12px; color: var(--text-muted); font-family: var(--font-mono); }
    .badge-tg {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 5px 12px; border-radius: 20px;
      background: rgba(0, 229, 255, 0.12); color: var(--cyan);
      border: 1px solid rgba(0, 229, 255, 0.3); font-size: 12px; font-weight: 600;
    }

    /* KPI Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }
    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); border-color: var(--border-light); }
    .kpi-card::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
      background: var(--kpi-color, var(--cyan));
    }
    .kpi-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
    .kpi-value { font-size: 28px; font-weight: 800; margin: 6px 0 2px; font-family: var(--font-mono); color: var(--kpi-color, var(--text-main)); }
    .kpi-sub { font-size: 11px; color: var(--text-muted); }

    /* Filter Toolbar */
    .toolbar {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 20px;
      margin-bottom: 24px;
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      justify-content: space-between;
    }
    .filter-group { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .filter-label { font-size: 12px; color: var(--text-muted); font-weight: 600; margin-right: 4px; }
    .btn-pill {
      background: var(--bg-main);
      border: 1px solid var(--border);
      color: var(--text-dim);
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-pill:hover { border-color: var(--cyan); color: var(--text-main); }
    .btn-pill.active { background: var(--cyan); color: #000; border-color: var(--cyan); }
    .search-input {
      background: var(--bg-main);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 7px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-family: var(--font-mono);
      min-width: 220px;
    }
    .search-input:focus { outline: none; border-color: var(--cyan); }

    /* Table */
    .table-wrap {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow-x: auto;
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th {
      background: #111520;
      color: var(--text-dim);
      font-weight: 600;
      text-align: left;
      padding: 14px 16px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }
    td {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      vertical-align: middle;
    }
    tr:hover td { background: var(--bg-card-hover); }

    .tag-tier {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700;
    }
    .tier-textbook { background: rgba(8, 153, 129, 0.18); color: var(--green); border: 1px solid rgba(8, 153, 129, 0.4); }
    .tier-good { background: rgba(0, 229, 255, 0.15); color: var(--cyan); border: 1px solid rgba(0, 229, 255, 0.35); }
    .tier-reject { background: rgba(242, 54, 69, 0.15); color: var(--red); border: 1px solid rgba(242, 54, 69, 0.35); }

    .symbol-chip {
      font-family: var(--font-mono); font-weight: 700;
      padding: 3px 7px; border-radius: 4px;
      background: rgba(255,255,255,0.06); color: var(--text-main);
    }
    .dir-badge { font-weight: 700; display: inline-flex; align-items: center; gap: 4px; }
    .dir-short { color: var(--red); }
    .dir-long { color: var(--green); }

    .score-badge {
      font-family: var(--font-mono); font-weight: 800; font-size: 14px;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .pnl-badge { font-family: var(--font-mono); font-weight: 700; }
    .pnl-win { color: var(--green); }
    .pnl-loss { color: var(--red); }

    .reason-cell { max-width: 380px; font-size: 12px; color: var(--text-dim); line-height: 1.4; }
    .btn-view {
      background: rgba(0, 229, 255, 0.12);
      border: 1px solid rgba(0, 229, 255, 0.35);
      color: var(--cyan);
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-view:hover { background: var(--cyan); color: #000; }

    /* Modal */
    .modal {
      display: none; position: fixed; inset: 0; z-index: 1000;
      background: rgba(0,0,0,0.88); backdrop-filter: blur(8px);
      align-items: center; justify-content: center; padding: 24px;
    }
    .modal.active { display: flex; }
    .modal-content {
      max-width: 95vw; max-height: 92vh; position: relative;
      background: var(--bg-card); border-radius: 14px; border: 1px solid var(--border-light);
      padding: 12px; box-shadow: 0 16px 48px rgba(0,0,0,0.7);
    }
    .modal-content img { max-width: 100%; max-height: 84vh; border-radius: 8px; display: block; }
    .modal-close {
      position: absolute; top: -14px; right: -14px;
      width: 32px; height: 32px; border-radius: 50%;
      background: var(--red); color: #fff; border: none; font-size: 18px;
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
  </style>
</head>
<body>
  <div class="container">
    
    <!-- Header -->
    <header class="header">
      <div class="brand">
        <div class="brand-logo">R</div>
        <div class="brand-title">
          <h1>RalphTradeBot Dashboard</h1>
          <p>Автономный аудит 5-волновых импульсов Эллиотта & RSI-дивергенций v2.1</p>
        </div>
      </div>
      <div class="header-meta">
        <div class="badge-tg">🚀 Telegram Bot: @RalphTraderBot (Score ≥ 85%)</div>
        <div style="margin-top: 6px;">Сгенерировано: __GENERATED_DATE__</div>
      </div>
    </header>

    <!-- KPI Grid -->
    <section class="kpi-grid">
      <div class="kpi-card" style="--kpi-color: var(--cyan);">
        <div class="kpi-label">Всего сигналов</div>
        <div class="kpi-value">__TOTAL_SIGNALS__</div>
        <div class="kpi-sub">Сканирование Топ-15 инструментов</div>
      </div>
      <div class="kpi-card" style="--kpi-color: var(--green);">
        <div class="kpi-label">Эталонные (Score ≥ 85)</div>
        <div class="kpi-value">__TEXTBOOK_COUNT__</div>
        <div class="kpi-sub">Отправлены в Telegram-канал</div>
      </div>
      <div class="kpi-card" style="--kpi-color: var(--green);">
        <div class="kpi-label">Win Rate (Score ≥ 85)</div>
        <div class="kpi-value">__TEXTBOOK_WR__%</div>
        <div class="kpi-sub">Высшая точность входа</div>
      </div>
      <div class="kpi-card" style="--kpi-color: var(--gold);">
        <div class="kpi-label">Общий Win Rate</div>
        <div class="kpi-value">__OVERALL_WR__%</div>
        <div class="kpi-sub">Все подтвержденные (Score ≥ 70)</div>
      </div>
      <div class="kpi-card" style="--kpi-color: var(--purple);">
        <div class="kpi-label">Ср. время импульса</div>
        <div class="kpi-value">__AVG_DURATION__</div>
        <div class="kpi-sub">От точки W0 до триггера W5</div>
      </div>
      <div class="kpi-card" style="--kpi-color: var(--cyan);">
        <div class="kpi-label">Частота сигналов</div>
        <div class="kpi-value">__FREQ_PER_DAY__ / день</div>
        <div class="kpi-sub">На портфель из 15 монет</div>
      </div>
    </section>

    <!-- Toolbar -->
    <div class="toolbar">
      <div class="filter-group">
        <span class="filter-label">Категория:</span>
        <button class="btn-pill active" onclick="setFilter('tier', 'all')">Все (__TOTAL_COUNT__)</button>
        <button class="btn-pill" onclick="setFilter('tier', 'textbook')">Эталонные ≥ 85 (__TEXTBOOK_COUNT__)</button>
        <button class="btn-pill" onclick="setFilter('tier', 'good')">Хорошие 70-84 (__GOOD_COUNT__)</button>
        <button class="btn-pill" onclick="setFilter('tier', 'reject')">Отклонённые (__REJECT_COUNT__)</button>
      </div>

      <div class="filter-group">
        <span class="filter-label">Таймфрейм:</span>
        <button class="btn-pill active" onclick="setFilter('tf', 'all')">Все ТФ</button>
        <button class="btn-pill" onclick="setFilter('tf', '5m')">5м</button>
        <button class="btn-pill" onclick="setFilter('tf', '15m')">15м</button>
        <button class="btn-pill" onclick="setFilter('tf', '30m')">30м</button>
      </div>

      <input type="text" id="searchInput" class="search-input" placeholder="🔍 Поиск по монете, ТФ, тексту..." oninput="applyFilters()">
    </div>

    <!-- Table -->
    <div class="table-wrap">
      <table id="signalsTable">
        <thead>
          <tr>
            <th>Качество</th>
            <th>Инструмент</th>
            <th>ТФ</th>
            <th>Сигнал</th>
            <th>Цена</th>
            <th>Формирование импульса (W0 → W5)</th>
            <th>Оценка ИИ</th>
            <th>W0 Исток / Дивергенция</th>
            <th>Исход</th>
            <th>Обоснование ИИ (на русском)</th>
            <th>График</th>
          </tr>
        </thead>
        <tbody id="tableBody">
          __TABLE_ROWS__
        </tbody>
      </table>
    </div>

  </div>

  <!-- Modal Preview -->
  <div id="chartModal" class="modal" onclick="closeModal()">
    <div class="modal-content" onclick="event.stopPropagation()">
      <button class="modal-close" onclick="closeModal()">&times;</button>
      <img id="modalImg" src="" alt="Chart">
    </div>
  </div>

  <script>
    let currentTier = 'all';
    let currentTf = 'all';

    function setFilter(type, val) {
      if (type === 'tier') currentTier = val;
      if (type === 'tf') currentTf = val;

      document.querySelectorAll(`.toolbar .btn-pill`).forEach(btn => {
        const txt = btn.innerText.toLowerCase();
        if (type === 'tier' && btn.getAttribute('onclick').includes('tier')) {
          btn.classList.toggle('active', btn.getAttribute('onclick').includes(`'${val}'`));
        }
        if (type === 'tf' && btn.getAttribute('onclick').includes('tf')) {
          btn.classList.toggle('active', btn.getAttribute('onclick').includes(`'${val}'`));
        }
      });
      applyFilters();
    }

    function applyFilters() {
      const q = document.getElementById('searchInput').value.toLowerCase();
      const rows = document.querySelectorAll('#tableBody tr');

      rows.forEach(r => {
        const tier = r.getAttribute('data-tier');
        const tf = r.getAttribute('data-tf');
        const text = r.innerText.toLowerCase();

        const matchTier = (currentTier === 'all') || (tier === currentTier);
        const matchTf = (currentTf === 'all') || (tf === currentTf);
        const matchSearch = !q || text.includes(q);

        r.style.display = (matchTier && matchTf && matchSearch) ? '' : 'none';
      });
    }

    function openModal(imgSrc) {
      document.getElementById('modalImg').src = imgSrc;
      document.getElementById('chartModal').classList.add('active');
    }
    function closeModal() {
      document.getElementById('chartModal').classList.remove('active');
    }
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
  </script>
</body>
</html>
"""


def generate_html_report(
    signals_data: List[Dict[str, Any]],
    output_path: Path,
    days_span: float = 30.0,
) -> Path:
    """Формирует автономный HTML дашборд."""
    total_signals = len(signals_data)
    textbook_signals = [s for s in signals_data if s.get("score", 0) >= 85]
    good_signals = [s for s in signals_data if 70 <= s.get("score", 0) < 85]
    reject_signals = [s for s in signals_data if s.get("score", 0) < 70]
    confirmed_signals = textbook_signals + good_signals

    textbook_wins = sum(1 for s in textbook_signals if s.get("trade_result") == "win")
    textbook_wr = (textbook_wins / len(textbook_signals) * 100.0) if textbook_signals else 0.0

    confirmed_wins = sum(1 for s in confirmed_signals if s.get("trade_result") == "win")
    overall_wr = (confirmed_wins / len(confirmed_signals) * 100.0) if confirmed_signals else 0.0

    # Средняя длительность формирования импульса (в барах и часах)
    durations_h = [s.get("duration_hours", 0.0) for s in confirmed_signals if s.get("duration_hours")]
    avg_dur_h = (sum(durations_h) / len(durations_h)) if durations_h else 0.0
    avg_dur_str = f"{avg_dur_h:.1f} ч" if avg_dur_h > 0 else "N/A"

    freq_per_day = f"{(total_signals / days_span):.1f}" if days_span > 0 else "N/A"

    # Формирование строк таблицы
    rows_html = []
    for s in signals_data:
        score = s.get("score", 0)
        if score >= 85:
            tier_class = "tier-textbook"
            tier_name = "Textbook ≥ 85"
            data_tier = "textbook"
        elif score >= 70:
            tier_class = "tier-good"
            tier_name = "Good 70-84"
            data_tier = "good"
        else:
            tier_class = "tier-reject"
            tier_name = "Rejected"
            data_tier = "reject"

        sym = s.get("symbol", "")
        tf = s.get("interval", "")
        dir_val = s.get("direction", "")
        dir_icon = "🔴 SHORT" if dir_val == "SHORT" else "🟢 LONG"
        dir_class = "dir-short" if dir_val == "SHORT" else "dir-long"

        price = s.get("signal_price", 0.0)
        price_str = f"${price:,.4f}" if price < 10.0 else f"${price:,.2f}"

        dur_bars = s.get("duration_bars", 0)
        dur_h = s.get("duration_hours", 0.0)
        dur_text = f"<b>{dur_h:.1f} ч</b> ({dur_bars} бар.)" if dur_h > 0 else f"{dur_bars} бар."

        w0_rsi = s.get("w0_rsi")
        w0_stat = s.get("w0_status", "")
        orig_div = s.get("origin_div", False)
        w0_info = f"RSI: {w0_rsi:.1f} ({w0_stat})" if w0_rsi else "N/A"
        if orig_div:
            w0_info += " <span style='color:var(--cyan);font-weight:700;'>[W0-DIV ★]</span>"

        res = s.get("trade_result", "")
        pnl = s.get("trade_pnl_pct", 0.0)
        if res == "win":
            pnl_html = f"<span class='pnl-badge pnl-win'>WIN ({pnl:+.2f}%)</span>"
        elif res == "loss":
            pnl_html = f"<span class='pnl-badge pnl-loss'>LOSS ({pnl:+.2f}%)</span>"
        else:
            pnl_html = f"<span class='pnl-badge'>OPEN ({pnl:+.2f}%)</span>"

        reason = s.get("reason", "")
        img_rel = s.get("chart_img_rel", "")
        btn_img = f"<button class='btn-view' onclick=\"openModal('{img_rel}')\">👁️ График</button>" if img_rel else "-"

        row = f"""<tr data-tier="{data_tier}" data-tf="{tf}">
          <td><span class="tag-tier {tier_class}">{tier_name}</span></td>
          <td><span class="symbol-chip">{sym}</span></td>
          <td><b>{tf}</b></td>
          <td><span class="dir-badge {dir_class}">{dir_icon}</span></td>
          <td>{price_str}</td>
          <td>{dur_text}</td>
          <td><span class="score-badge" style="color: {'var(--green)' if score>=85 else 'var(--cyan)' if score>=70 else 'var(--red)'}">{score}/100</span></td>
          <td>{w0_info}</td>
          <td>{pnl_html}</td>
          <td class="reason-cell">{reason}</td>
          <td>{btn_img}</td>
        </tr>"""
        rows_html.append(row)

    html = HTML_TEMPLATE
    html = html.replace("__GENERATED_DATE__", datetime.now().strftime("%d.%m.%Y %H:%M"))
    html = html.replace("__TOTAL_SIGNALS__", str(total_signals))
    html = html.replace("__TEXTBOOK_COUNT__", str(len(textbook_signals)))
    html = html.replace("__TEXTBOOK_WR__", f"{textbook_wr:.1f}")
    html = html.replace("__OVERALL_WR__", f"{overall_wr:.1f}")
    html = html.replace("__AVG_DURATION__", avg_dur_str)
    html = html.replace("__FREQ_PER_DAY__", freq_per_day)
    html = html.replace("__TOTAL_COUNT__", str(total_signals))
    html = html.replace("__GOOD_COUNT__", str(len(good_signals)))
    html = html.replace("__REJECT_COUNT__", str(len(reject_signals)))
    html = html.replace("__TABLE_ROWS__", "\n".join(rows_html))

    output_path.write_text(html, encoding="utf-8")
    logger.info(f"✅ Интерактивный HTML-отчет сохранен: {output_path}")
    return output_path
