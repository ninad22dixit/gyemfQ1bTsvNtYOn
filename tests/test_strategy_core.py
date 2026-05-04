import numpy as np
import pandas as pd

from trading_pipeline.strategy_core import (
    CONFIG,
    accounting_check,
    apply_llm_guardrails,
    execute_buy,
    execute_sell,
    initialize_portfolio,
    prepare_feature_table,
    record_equity,
    run_hybrid_llm_quant_strategy,
)


def sample_df(rows=120):
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0.1, 1.0, rows))
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "open": close,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": np.full(rows, 1000.0),
        }
    )


def test_buy_sell_accounting_balances():
    df = prepare_feature_table(sample_df())
    cfg = dict(CONFIG, min_rows_before_trading=1, use_llm=False)
    portfolio = initialize_portfolio(df, cfg)
    row0 = df.iloc[20].copy(); row0.name = 20
    row1 = df.iloc[21].copy(); row1.name = 21
    assert execute_buy(row0, portfolio, "dca", 1000, "test", cfg)
    pos = portfolio["open_positions"][0]
    assert execute_sell(row1, portfolio, pos, pos["btc_remaining"] / 2, float(row1["close"]), "test", cfg)
    record_equity(row1, portfolio)
    bad = accounting_check(pd.DataFrame(portfolio["equity_curve"]))
    assert bad.empty


def test_guardrails_filter_unknown_features_and_low_confidence():
    df = prepare_feature_table(sample_df())
    cfg = dict(CONFIG, use_llm=False)
    portfolio = initialize_portfolio(df, cfg)
    row = df.iloc[60].copy(); row.name = 60
    decision = apply_llm_guardrails(
        row,
        {"regime": "swing_trading", "signal": "buy", "selected_features": ["rsi_14", "made_up"], "risk_multiplier": 2, "confidence": 0.1},
        portfolio,
        cfg,
    )
    assert "made_up" not in decision["selected_features"]
    assert decision["signal"] == "hold"
    assert 0 <= decision["risk_multiplier"] <= 1


def test_full_strategy_runs():
    df = prepare_feature_table(sample_df())
    cfg = dict(CONFIG, use_llm=False, min_rows_before_trading=30, llm_every_n_candles=5)
    _, equity_df, *_ = run_hybrid_llm_quant_strategy(df, cfg)
    assert not equity_df.empty
    assert accounting_check(equity_df).empty
