"""Notification helpers for trade alerts and weekly summaries."""
from __future__ import annotations

import hashlib
import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd
import requests


RequestPost = Callable[..., Any]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _fmt_money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "n/a"


def _fmt_btc(value: Any) -> str:
    try:
        return f"{float(value):.8f} BTC"
    except Exception:
        return "n/a"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "n/a"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _read_csv_if_available(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _trade_id(kind: str, event: Dict[str, Any]) -> str:
    stable_parts = [
        kind,
        str(event.get("time", "")),
        str(event.get("position_id", "")),
        str(event.get("price", event.get("exit_price", ""))),
        str(event.get("btc_bought", event.get("btc_sold", ""))),
        str(event.get("usd_spent", event.get("net_usd", ""))),
    ]
    raw = "|".join(stable_parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def format_trade_alert(kind: str, event: Dict[str, Any], summary: Optional[Dict[str, Any]] = None) -> str:
    """Format one trade event for Telegram."""
    title = "BUY" if kind == "buy" else "SELL"
    lines = [
        f"Bitcoin trade alert: {title}",
        f"Time: {event.get('time', 'n/a')}",
        f"Position: {event.get('type', 'n/a')} ({event.get('position_id', 'n/a')})",
    ]
    if kind == "buy":
        lines.extend(
            [
                f"Price: {_fmt_money(event.get('price'))}",
                f"USD spent: {_fmt_money(event.get('usd_spent'))}",
                f"BTC bought: {_fmt_btc(event.get('btc_bought'))}",
            ]
        )
        if event.get("stop_loss") not in {None, ""}:
            lines.append(f"Stop loss: {_fmt_money(event.get('stop_loss'))}")
        if event.get("take_profit") not in {None, ""}:
            lines.append(f"Take profit: {_fmt_money(event.get('take_profit'))}")
    else:
        lines.extend(
            [
                f"Entry: {_fmt_money(event.get('entry_price'))}",
                f"Exit: {_fmt_money(event.get('exit_price'))}",
                f"BTC sold: {_fmt_btc(event.get('btc_sold'))}",
                f"Net proceeds: {_fmt_money(event.get('net_usd'))}",
                f"PnL: {_fmt_money(event.get('pnl'))}",
            ]
        )
    lines.extend(
        [
            f"Fee: {_fmt_money(event.get('fee'))}",
            f"Remaining cash: {_fmt_money(event.get('remaining_cash'))}",
            f"Reason: {event.get('reason', 'n/a')}",
        ]
    )
    if summary:
        lines.append(f"Portfolio equity: {_fmt_money(summary.get('final_equity'))}")
        lines.append(f"Total return: {_fmt_pct(summary.get('total_return_pct'))}")
    return "\n".join(lines)


def send_telegram_message(
    text: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    request_post: RequestPost = requests.post,
    timeout: int = 15,
) -> bool:
    """Send a Telegram message. Returns False when credentials are not configured."""
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = request_post(url, json={"chat_id": chat, "text": text}, timeout=timeout)
    response.raise_for_status()
    return True


def notify_trade_events(
    buys: Sequence[Dict[str, Any]] | pd.DataFrame,
    sells: Sequence[Dict[str, Any]] | pd.DataFrame,
    summary: Optional[Dict[str, Any]] = None,
    state_path: str | Path = "reports/telegram_notified.json",
    enabled: Optional[bool] = None,
    request_post: RequestPost = requests.post,
) -> int:
    """Send Telegram alerts for trade rows that have not been notified yet."""
    if enabled is None:
        enabled = _as_bool(os.getenv("TELEGRAM_NOTIFICATIONS_ENABLED"), True)
    if not enabled:
        return 0

    state_file = Path(os.getenv("TELEGRAM_NOTIFICATIONS_STATE_PATH", str(state_path)))
    sent_ids: Set[str] = set(_load_json(state_file, []))
    new_sent_ids = set(sent_ids)
    sent_count = 0

    for kind, rows in (("buy", buys), ("sell", sells)):
        records = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
        for event in records:
            trade_id = _trade_id(kind, event)
            if trade_id in sent_ids:
                continue
            message = format_trade_alert(kind, event, summary)
            if send_telegram_message(message, request_post=request_post):
                new_sent_ids.add(trade_id)
                sent_count += 1

    if new_sent_ids != sent_ids:
        _save_json(state_file, sorted(new_sent_ids))
    return sent_count


def build_weekly_summary_email(report_dir: str | Path = "reports/latest_run") -> tuple[str, str]:
    """Build the Gmail weekly summary subject/body from the latest report files."""
    path = Path(report_dir)
    summary = _load_json(path / "summary.json", {})
    buys = _read_csv_if_available(path / "buys.csv")
    sells = _read_csv_if_available(path / "sells.csv")

    subject = "Bitcoin trading weekly summary"
    body = "\n".join(
        [
            "Bitcoin trading weekly summary",
            "",
            f"Final equity: {_fmt_money(summary.get('final_equity'))}",
            f"Total return: {_fmt_pct(summary.get('total_return_pct'))}",
            f"Max drawdown: {_fmt_pct(summary.get('max_drawdown_pct'))}",
            f"Realized PnL: {_fmt_money(summary.get('realized_pnl'))}",
            f"Remaining cash: {_fmt_money(summary.get('remaining_cash'))}",
            f"Open BTC: {_fmt_btc(summary.get('total_open_btc'))}",
            f"Buys: {int(summary.get('num_buys', 0) or len(buys))}",
            f"Sells: {int(summary.get('num_sells', 0) or len(sells))}",
            f"Paused: {summary.get('paused', False)}",
            f"Pause reason: {summary.get('pause_reason') or 'n/a'}",
            "",
            f"Report directory: {path}",
        ]
    )
    return subject, body


def send_gmail_summary(
    subject: str,
    body: str,
    sender: Optional[str] = None,
    app_password: Optional[str] = None,
    recipients: Optional[Iterable[str]] = None,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 465,
) -> bool:
    """Send a weekly summary through Gmail SMTP using an app password."""
    sender_email = sender or os.getenv("GMAIL_SENDER")
    password = app_password or os.getenv("GMAIL_APP_PASSWORD")
    if isinstance(recipients, str):
        recipient_text = recipients
    else:
        recipient_text = ",".join(recipients) if recipients is not None else os.getenv("GMAIL_RECIPIENTS", "")
    recipient_list = [item.strip() for item in recipient_text.split(",") if item.strip()]
    if not sender_email or not password or not recipient_list:
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = ", ".join(recipient_list)
    message.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
        server.login(sender_email, password)
        server.send_message(message)
    return True


def send_weekly_summary(report_dir: str | Path = "reports/latest_run") -> bool:
    subject, body = build_weekly_summary_email(report_dir)
    return send_gmail_summary(subject, body)
