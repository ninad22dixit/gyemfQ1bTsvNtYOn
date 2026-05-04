import pandas as pd

from trading_pipeline.strategy_core import prepare_feature_table, accounting_check


def test_prepare_feature_table_has_expected_columns():
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=80, freq="D"),
        "open": range(100, 180),
        "high": range(105, 185),
        "low": range(95, 175),
        "close": range(100, 180),
        "volume": [10_000] * 80,
    })
    out = prepare_feature_table(df)
    assert {"ATR", "rsi_14", "macd", "ema_100", "atr_pct"}.issubset(out.columns)
    assert len(out) > 0


def test_accounting_check_detects_mismatch():
    equity = pd.DataFrame({
        "remaining_cash": [100.0],
        "total_btc": [1.0],
        "close": [50.0],
        "total_equity": [200.0],
    })
    bugs = accounting_check(equity)
    assert len(bugs) == 1
