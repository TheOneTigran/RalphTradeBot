#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Daily Backup & Diagnostics for TigrVPN.play2go.cloud (Old Server: 31.77.158.246)
Sends SQLite DBs and data to Telegram Chat (-5382213057)
"""

import os
import sys
import json
import glob
import time
import shutil
import sqlite3
import tarfile
import urllib.parse
import urllib.request
import subprocess
from datetime import datetime, timezone, timedelta

BOT_TOKEN = "8739997369:AAEDr6nJ6DJe5HiDnC_c5S_cccxnkbQ06zE"
CHAT_ID = "-5382213057"
BACKUP_DIR = "/opt/server-backup"
MSG_IDS_FILE = os.path.join(BACKUP_DIR, "last_msg_ids.txt")
MAX_TG_SIZE = 48 * 1024 * 1024

os.makedirs(BACKUP_DIR, exist_ok=True)

msk_tz = timezone(timedelta(hours=3))
now_msk = datetime.now(msk_tz).strftime('%Y-%m-%d %H:%M MSK')
hostname = subprocess.getoutput("hostname").strip()

def tg_delete_msg(msg_id):
    if not msg_id:
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage"
        data = urllib.parse.urlencode({"chat_id": CHAT_ID, "message_id": msg_id}).encode()
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Error deleting msg {msg_id}: {e}")

def tg_send_msg(text):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true"
        }).encode()
        req = urllib.request.Request(url, data=data)
        resp = urllib.request.urlopen(req, timeout=15)
        res_json = json.loads(resp.read().decode('utf-8'))
        if res_json.get("ok"):
            return res_json.get("result", {}).get("message_id")
        else:
            print(f"Failed to send message: {res_json}")
            return None
    except Exception as e:
        print(f"Error sending telegram message: {e}")
        return None

def tg_send_document(filepath, caption=""):
    try:
        if not os.path.exists(filepath):
            return None
        file_size = os.path.getsize(filepath)
        if file_size > MAX_TG_SIZE:
            print(f"File {filepath} too large ({file_size} bytes), skipping.")
            return None
            
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
        cmd = [
            "curl", "-s", "-X", "POST", url,
            "-F", f"chat_id={CHAT_ID}",
            "-F", f"document=@{filepath}",
            "-F", f"caption={caption}"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        res_json = json.loads(res.stdout)
        if res_json.get("ok"):
            return res_json.get("result", {}).get("message_id")
        else:
            print(f"Failed to send {filepath}: {res.stdout}")
            return None
    except Exception as e:
        print(f"Error uploading {filepath}: {e}")
        return None

def cleanup_old_messages():
    if os.path.exists(MSG_IDS_FILE):
        print("Cleaning up previous backup messages...")
        with open(MSG_IDS_FILE, "r") as f:
            for line in f:
                mid = line.strip()
                if mid:
                    tg_delete_msg(mid)
        os.remove(MSG_IDS_FILE)

def collect_container_diagnostics():
    try:
        cmd = ["docker", "inspect"] + subprocess.getoutput("docker ps -aq").split()
        if len(cmd) <= 1:
            return "• <i>Нет контейнеров Docker</i>"
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        raw = res.stdout
        json_start = raw.find('[')
        json_end = raw.rfind(']')
        if json_start == -1 or json_end == -1:
            return "• <i>Ошибка получения статуса контейнеров</i>"
        
        containers = json.loads(raw[json_start:json_end+1])
        containers.sort(key=lambda x: x.get("Name", "").lstrip("/").lower())
        
        lines = []
        for c in containers:
            name = c.get("Name", "").lstrip("/")
            state = c.get("State", {})
            running = state.get("Running", False)
            exit_code = state.get("ExitCode", 0)
            oom = state.get("OOMKilled", False)
            error = state.get("Error", "")
            finished_at = state.get("FinishedAt", "")[:19].replace("T", " ")
            health = state.get("Health", {}).get("Status")
            
            if running:
                h_text = f", {health}" if health else ""
                lines.append(f"✅ <b>{name}</b>: Работает (Up{h_text})")
            else:
                if oom:
                    reason = f"Нехватка памяти (OOMKilled, код {exit_code})"
                elif exit_code == 0:
                    reason = f"Остановлен штатно (ExitCode 0, {finished_at})"
                elif error:
                    reason = f"Ошибка: {error} (ExitCode {exit_code}, {finished_at})"
                else:
                    reason = f"Аварийно остановлен (ExitCode {exit_code}, {finished_at})"
                lines.append(f"❌ <b>{name}</b>: {reason}")
        return "\n".join(lines)
    except Exception as e:
        return f"• <i>Ошибка диагностики: {e}</i>"

def backup_sqlite_safe(src_path, dst_path):
    if not os.path.exists(src_path):
        return False
    try:
        src_conn = sqlite3.connect(src_path)
        dst_conn = sqlite3.connect(dst_path)
        with dst_conn:
            src_conn.backup(dst_conn)
        dst_conn.close()
        src_conn.close()
        return True
    except Exception as e:
        print(f"Live sqlite backup failed for {src_path}: {e}, doing copy...")
        shutil.copy2(src_path, dst_path)
        return True

def main():
    print(f"=== Starting Daily Backup & Diagnostics: {now_msk} ===")
    sent_msg_ids = []
    
    # Step 1: Clean old messages
    cleanup_old_messages()
    
    # Step 2: Prepare backups
    temp_files_to_clean = []
    uploaded_files_summary = []
    
    # 1. HedgeTradeBot DB
    hedge_db = "/root/hedgetradebot/data/bot_data.sqlite"
    hedge_backup = os.path.join(BACKUP_DIR, "hedgetrade_bot.sqlite")
    if backup_sqlite_safe(hedge_db, hedge_backup):
        print("Sending hedgetrade_bot.sqlite...")
        mid = tg_send_document(hedge_backup, f"📈 HedgeTradeBot DB | {now_msk}")
        if mid:
            sent_msg_ids.append(mid)
            uploaded_files_summary.append("• hedgetrade_bot.sqlite")

    # 2. 3x-ui VPN DB
    xui_db = "/etc/x-ui/x-ui.db"
    xui_backup = os.path.join(BACKUP_DIR, "x-ui-backup.db")
    if backup_sqlite_safe(xui_db, xui_backup):
        print("Sending x-ui-backup.db...")
        mid = tg_send_document(xui_backup, f"🔐 3x-ui VPN DB | {now_msk}")
        if mid:
            sent_msg_ids.append(mid)
            uploaded_files_summary.append("• x-ui-backup.db")

    # 3. МойМалыш PocketBase DB
    pb_dir = "/var/lib/docker/volumes/mylittleone_pb_data/_data"
    pb_backup = os.path.join(BACKUP_DIR, "moymalysh_pocketbase.tar.gz")
    if os.path.isdir(pb_dir):
        with tarfile.open(pb_backup, "w:gz") as tar:
            for item in ["data.db", "auxiliary.db"]:
                p = os.path.join(pb_dir, item)
                if os.path.exists(p):
                    tar.add(p, arcname=item)
        if os.path.exists(pb_backup) and os.path.getsize(pb_backup) > 0:
            print("Sending moymalysh_pocketbase.tar.gz...")
            mid = tg_send_document(pb_backup, f"👶 МойМалыш PocketBase DB | {now_msk}")
            if mid:
                sent_msg_ids.append(mid)
                uploaded_files_summary.append("• moymalysh_pocketbase.tar.gz")
            temp_files_to_clean.append(pb_backup)

    # 4. МойМалыш Uploads (if under 50MB)
    uploads_dir = "/var/lib/docker/volumes/mylittleone_content_uploads/_data"
    uploads_backup = os.path.join(BACKUP_DIR, "moymalysh_uploads.tar.gz")
    if os.path.isdir(uploads_dir) and os.listdir(uploads_dir):
        with tarfile.open(uploads_backup, "w:gz") as tar:
            tar.add(uploads_dir, arcname="uploads")
        if os.path.exists(uploads_backup):
            up_size = os.path.getsize(uploads_backup)
            if up_size < MAX_TG_SIZE:
                print("Sending moymalysh_uploads.tar.gz...")
                mid = tg_send_document(uploads_backup, f"👶 МойМалыш Uploads | {now_msk}")
                if mid:
                    sent_msg_ids.append(mid)
                    uploaded_files_summary.append("• moymalysh_uploads.tar.gz")
            else:
                print("МойМалыш uploads too large, skipped.")
            temp_files_to_clean.append(uploads_backup)

    # 5. RalphTradeBot Analytics DB
    ralph_db = "/root/RalphTradeBot/data/ralph_analytics.db"
    ralph_backup = os.path.join(BACKUP_DIR, "ralph_analytics.db")
    if backup_sqlite_safe(ralph_db, ralph_backup):
        print("Sending ralph_analytics.db...")
        mid = tg_send_document(ralph_backup, f"📊 RalphTradeBot Analytics DB | {now_msk}")
        if mid:
            sent_msg_ids.append(mid)
            uploaded_files_summary.append("• ralph_analytics.db (RalphTradeBot)")

    # Step 3: Collect system and container diagnostics
    disk_info = subprocess.getoutput("df -h / | tail -1 | awk '{print $3\"/\"$2\" (\"$5\" used)\"}'").strip()
    ram_info = subprocess.getoutput("free -h | awk '/Mem:/ {print $3\"/\"$2}'").strip()
    load_avg = subprocess.getoutput("uptime | sed -E 's/.*load average: (.*)/\\1/'").strip()
    container_diag = collect_container_diagnostics()

    # RalphTradeBot Analytics Summary
    ralph_stats_text = ""
    if os.path.exists(ralph_db):
        try:
            r_conn = sqlite3.connect(f"file:{ralph_db}?mode=ro", uri=True)
            r_cur = r_conn.cursor()
            total_sigs = r_cur.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            active_sigs = r_cur.execute("SELECT COUNT(*) FROM signal_outcomes WHERE status = 'active'").fetchone()[0]
            closed_tp = r_cur.execute("SELECT COUNT(*) FROM signal_outcomes WHERE status = 'closed_tp'").fetchone()[0]
            closed_be = r_cur.execute("SELECT COUNT(*) FROM signal_outcomes WHERE status = 'closed_be'").fetchone()[0]
            closed_sl = r_cur.execute("SELECT COUNT(*) FROM signal_outcomes WHERE status = 'closed_sl'").fetchone()[0]
            tp2_cnt = r_cur.execute("SELECT COALESCE(SUM(tp2_hit), 0) FROM signal_outcomes").fetchone()[0]
            last_h = r_cur.execute("SELECT iteration, pairs_scanned, scan_duration_sec FROM scanner_health ORDER BY id DESC LIMIT 1").fetchone()
            r_conn.close()

            decided = closed_tp + closed_be + closed_sl
            wr = round((closed_tp + closed_be) / decided * 100, 1) if decided > 0 else 0.0
            h_info = f"Итерация #{last_h[0]} ({last_h[1]} пар, {last_h[2]}с)" if last_h else "Ожидание сканирования"

            ralph_stats_text = (
                f"\n\n📈 <b>RalphTradeBot Аналитика:</b>\n"
                f"• Статус сканера: <code>{h_info}</code>\n"
                f"• Сигналов в БД: <code>{total_sigs}</code> (Активно: <code>{active_sigs}</code>)\n"
                f"• Win Rate: <code>{wr}%</code> (TP4: {closed_tp}, BE: {closed_be}, SL: {closed_sl})\n"
                f"• Достигли TP2 (Безубыток): <code>{tp2_cnt}</code>"
            )
        except Exception as e:
            ralph_stats_text = f"\n\n📈 <b>RalphTradeBot:</b> <i>Ошибка чтения статистики ({e})</i>"
    
    files_text = "\n".join(uploaded_files_summary) if uploaded_files_summary else "• <i>Файлы не найдены</i>"
    
    report_text = (
        f"🛡 <b>Отчет диагностики и резервного копирования</b>\n"
        f"🖥 <b>Сервер:</b> <code>{hostname} (31.77.158.246)</code>\n"
        f"🕒 <b>Дата:</b> <code>{now_msk}</code>\n\n"
        f"📦 <b>Статус проектов:</b>\n"
        f"{container_diag}\n\n"
        f"📊 <b>Ресурсы системы:</b>\n"
        f"• Диск: <code>{disk_info}</code>\n"
        f"• RAM: <code>{ram_info}</code>\n"
        f"• Load Avg: <code>{load_avg}</code>"
        f"{ralph_stats_text}\n\n"
        f"📁 <b>Отправленные бэкапы:</b>\n"
        f"{files_text}"
    )
    
    print("Sending final summary report...")
    summary_mid = tg_send_msg(report_text)
    if summary_mid:
        sent_msg_ids.append(summary_mid)
        
    # Step 4: Save sent message IDs
    with open(MSG_IDS_FILE, "w") as f:
        for mid in sent_msg_ids:
            f.write(f"{mid}\n")
            
    # Cleanup temp archives
    for tf in temp_files_to_clean:
        if os.path.exists(tf):
            try:
                os.remove(tf)
            except Exception:
                pass

    print(f"=== Backup & Diagnostics finished successfully! ({len(sent_msg_ids)} messages saved) ===")

if __name__ == "__main__":
    main()
