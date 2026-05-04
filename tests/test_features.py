import numpy as np
import pandas as pd

from trading_pipeline.strategy_core import calculate_atr, calculate_macd, calculate_rsi, prepare_feature_table


def sample_df(rows=120):
    close = np.linspace(100, 130, rows)
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "open": close,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": np.linspace(1000, 2000, rows),
        }
    )


def test_indicators_have_expected_length():
    df = sample_df()
    assert len(calculate_atr(df)) == len(df)
    assert len(calculate_rsi(df["close"])) == len(df)
    macd, signal, hist = calculate_macd(df["close"])
    assert len(macd) == len(signal) == len(hist) == len(df)


def test_prepare_feature_table_adds_core_columns():
    out = prepare_feature_table(sample_df())
    for col in ["ATR", "rsi_14", "ema_20", "ema_50", "macd_hist", "atr_pct", "drawdown_from_peak"]:
        assert col in out.columns
