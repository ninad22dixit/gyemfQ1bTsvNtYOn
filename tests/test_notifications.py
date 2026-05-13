import json

from trading_pipeline.notifications import (
    build_weekly_summary_email,
    format_trade_alert,
    notify_trade_events,
    send_gmail_summary,
    weekly_summary_status,
)


class DummyResponse:
    def raise_for_status(self):
        return None


def test_format_trade_alert_includes_buy_details():
    message = format_trade_alert(
        "buy",
        {
            "time": "2026-05-10",
            "type": "dca",
            "position_id": "dca_1",
            "price": 100,
            "usd_spent": 500,
            "btc_bought": 5,
            "fee": 1,
            "remaining_cash": 99500,
            "reason": "test buy",
        },
        {"final_equity": 101000, "total_return_pct": 1.0},
    )

    assert "Bitcoin trade alert: BUY" in message
    assert "USD spent: $500.00" in message
    assert "BTC bought: 5.00000000 BTC" in message
    assert "Portfolio equity: $101,000.00" in message


def test_notify_trade_events_deduplicates(tmp_path, monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return DummyResponse()

    state_path = tmp_path / "telegram_notified.json"
    buys = [
        {
            "time": "2026-05-10",
            "type": "swing",
            "position_id": "swing_1",
            "price": 100,
            "usd_spent": 1000,
            "btc_bought": 10,
        }
    ]

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    assert notify_trade_events(buys, [], state_path=state_path, request_post=fake_post) == 0
    assert calls == []

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    assert notify_trade_events(buys, [], state_path=state_path, request_post=fake_post) == 1
    assert notify_trade_events(buys, [], state_path=state_path, request_post=fake_post) == 0

    assert len(calls) == 1
    assert json.loads(state_path.read_text(encoding="utf-8"))


def test_build_weekly_summary_email_handles_empty_trade_csvs(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"final_equity": 100000, "total_return_pct": 0, "num_buys": 0, "num_sells": 0}),
        encoding="utf-8",
    )
    (tmp_path / "buys.csv").write_text("\n", encoding="utf-8")
    (tmp_path / "sells.csv").write_text("\n", encoding="utf-8")

    subject, body = build_weekly_summary_email(tmp_path)

    assert subject == "Bitcoin trading weekly summary"
    assert "Final equity: $100,000.00" in body
    assert "Buys: 0" in body


def test_send_gmail_summary_returns_false_without_credentials(monkeypatch):
    monkeypatch.delenv("GMAIL_SENDER", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    monkeypatch.delenv("GMAIL_RECIPIENTS", raising=False)

    assert send_gmail_summary("subject", "body") is False


def test_weekly_summary_status_skips_missing_report(tmp_path):
    status = weekly_summary_status(tmp_path / "missing")

    assert status["sent"] is False
    assert status["status"] == "skipped"
    assert "missing report file" in status["reason"]
