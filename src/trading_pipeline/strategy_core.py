"""Core trading strategy logic for a hybrid LLM + quantitative BTC agent.

The module is intentionally deterministic by default.  Set ``use_llm=True`` and
run Ollama locally to enable the LLM decision layer; otherwise the strategy uses
safe heuristic fallback logic.
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)
np.random.seed(42)

CONFIG: Dict[str, Any] = {
    "product_id": "BTC-USD",
    "granularity": 86400,
    "days_back": 300,
    "initial_budget": 100000.0,
    "max_portfolio_drawdown_pause": 0.25,
    "fee_rate": 0.001,
    "dca_budget_fraction": 0.75,
    "dca_buy_amount": 5000.0,
    "dca_drop_trigger_pct": 0.03,
    "dca_exit_rsi": 75.0,
    "dca_exit_fraction": 0.75,
    "swing_budget_fraction": 0.25,
    "swing_buy_amount": 10000.0,
    "swing_max_open_positions": 10,
    "swing_atr_stop_multiple": 1.5,
    "swing_take_profit_atr_multiple": 1.0,
    "swing_trailing_stop": True,
    "use_llm": False,
    "ollama_url": "http://localhost:11434/api/generate",
    "ollama_model": "llama3.2:1b",
    "ollama_timeout": 60,
    "llm_temperature": 0.1,
    "llm_every_n_candles": 25,
    "min_llm_confidence_to_trade": 0.55,
    "risk_multiplier_min": 0.10,
    "risk_multiplier_max": 1.00,
    "high_vol_atr_pct": 0.06,
    "medium_vol_atr_pct": 0.04,
    "soft_drawdown_throttle": 0.15,
    "regime_min_hold_candles": 5,
    "min_candles_between_trades": 3,
    "require_indicator_confirmation": True,
    "min_rows_before_trading": 50,
}

VALID_REGIMES = {"value_investing", "swing_trading", "hold"}
VALID_SIGNALS = {"buy", "sell", "hold"}

ALLOWED_LLM_FEATURES = {
    "close", "volume", "return_1", "return_3", "return_7", "volatility_14",
    "drawdown_from_peak", "sma_20", "sma_50", "ema_20", "ema_50", "ema_100",
    "rsi_14", "macd", "macd_signal", "macd_hist", "ATR", "atr_pct",
    "range_pct", "volume_z_20", "dist_ema20", "dist_ema50", "dist_ema100",
}


def get_btc_historical_candles(product_id: str = "BTC-USD", granularity: int = 86400, days_back: int = 300) -> pd.DataFrame:
    """Fetch historical OHLCV candles from Coinbase Exchange public API."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {
        "start": start_time.isoformat().replace("+00:00", "Z"),
        "end": end_time.isoformat().replace("+00:00", "Z"),
        "granularity": granularity,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    rows = response.json()
    candles = [
        {
            "time": datetime.fromtimestamp(row[0], tz=timezone.utc),
            "low": float(row[1]),
            "high": float(row[2]),
            "open": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        for row in rows
    ]
    return pd.DataFrame(candles).sort_values("time").reset_index(drop=True)


def load_candles_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df.sort_values("time").reset_index(drop=True)


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd - macd_signal


def prepare_feature_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy().sort_values("time").reset_index(drop=True)
    df["ATR"] = calculate_atr(df, 14)
    df["rsi_14"] = calculate_rsi(df["close"], 14)
    for window in (20, 50, 100):
        df[f"sma_{window}"] = df["close"].rolling(window).mean()
        df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
    df["macd"], df["macd_signal"], df["macd_hist"] = calculate_macd(df["close"])
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_7"] = df["close"].pct_change(7)
    df["volatility_14"] = df["return_1"].rolling(14).std()
    df["atr_pct"] = df["ATR"] / df["close"]
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["close_to_high_pct"] = (df["high"] - df["close"]) / df["close"]
    df["close_to_low_pct"] = (df["close"] - df["low"]) / df["close"]
    df["volume_z_20"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    for window in (20, 50, 100):
        df[f"dist_ema{window}"] = (df["close"] - df[f"ema_{window}"]) / df[f"ema_{window}"]
    df["rolling_peak"] = df["close"].cummax()
    df["drawdown_from_peak"] = (df["rolling_peak"] - df["close"]) / df["rolling_peak"]
    df["momentum_score"] = (
        np.sign(df["return_3"].fillna(0))
        + np.sign(df["macd_hist"].fillna(0))
        + np.sign(df["dist_ema20"].fillna(0))
    )
    return df


def feature_snapshot(row: pd.Series) -> Dict[str, Optional[float]]:
    cols = sorted(ALLOWED_LLM_FEATURES | {"momentum_score"})
    return {c: None if pd.isna(row.get(c, np.nan)) else float(row.get(c)) for c in cols}


def extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("LLM response is not text")
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text[:300]}")
    return json.loads(match.group(0))


def normalize_llm_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    regime = str(decision.get("regime", "hold")).strip().lower()
    signal = str(decision.get("signal", "hold")).strip().lower()
    if regime not in VALID_REGIMES:
        regime = "hold"
    if signal not in VALID_SIGNALS:
        signal = "hold"
    selected_features = decision.get("selected_features", [])
    if not isinstance(selected_features, list):
        selected_features = []
    try:
        risk_multiplier = float(decision.get("risk_multiplier", 0.5))
    except Exception:
        risk_multiplier = 0.5
    try:
        confidence = float(decision.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    return {
        "regime": regime,
        "signal": signal,
        "selected_features": [str(x) for x in selected_features[:8]],
        "risk_multiplier": float(np.clip(risk_multiplier, 0.0, 1.0)),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "reason": str(decision.get("reason", ""))[:500],
        "source": decision.get("source", "llm"),
    }


def heuristic_decision(row: pd.Series) -> Dict[str, Any]:
    close = float(row["close"])
    ema_20 = row.get("ema_20", np.nan)
    ema_50 = row.get("ema_50", np.nan)
    ema_100 = row.get("ema_100", np.nan)
    rsi = row.get("rsi_14", np.nan)
    macd_hist = row.get("macd_hist", np.nan)
    drawdown = row.get("drawdown_from_peak", np.nan)
    atr_pct = row.get("atr_pct", np.nan)

    if pd.notna(drawdown) and drawdown >= 0.15:
        regime = "value_investing"
    elif pd.notna(ema_100) and close < ema_100:
        regime = "value_investing"
    elif pd.notna(ema_20) and pd.notna(ema_50) and close > ema_50 and ema_20 > ema_50:
        regime = "swing_trading"
    else:
        regime = "hold"

    signal = "hold"
    if regime == "value_investing" and pd.notna(rsi) and rsi < 45:
        signal = "buy"
    elif regime == "swing_trading" and pd.notna(macd_hist) and macd_hist > 0 and pd.notna(rsi) and 45 <= rsi <= 72:
        signal = "buy"
    elif pd.notna(rsi) and rsi > 78:
        signal = "sell"

    risk_multiplier = 0.6
    if pd.notna(atr_pct):
        if atr_pct > 0.06:
            risk_multiplier = 0.25
        elif atr_pct > 0.04:
            risk_multiplier = 0.4
        elif atr_pct < 0.025:
            risk_multiplier = 0.8

    return normalize_llm_decision(
        {
            "regime": regime,
            "signal": signal,
            "selected_features": ["rsi_14", "macd_hist", "ema_20", "ema_50", "ema_100", "atr_pct", "drawdown_from_peak"],
            "risk_multiplier": risk_multiplier,
            "confidence": 0.55,
            "reason": "Deterministic fallback based on RSI, MACD, EMA trend, drawdown, and ATR%.",
            "source": "heuristic_fallback",
        }
    )


def call_ollama_decision(row: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    if not config.get("use_llm", True):
        return heuristic_decision(row)
    prompt = f"""
You are a conservative crypto trading risk assistant.
Return ONLY valid JSON with this schema:
{{"regime":"value_investing|swing_trading|hold","signal":"buy|sell|hold","selected_features":["feature"],"risk_multiplier":0.0,"confidence":0.0,"reason":"brief reason"}}
Rules: lower risk when ATR%, volatility, or drawdown is high.
Current candle features: {json.dumps(feature_snapshot(row), indent=2)}
""".strip()
    payload = {
        "model": config.get("ollama_model", "llama3.2:1b"),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": config.get("llm_temperature", 0.1)},
    }
    try:
        resp = requests.post(config["ollama_url"], json=payload, timeout=config.get("ollama_timeout", 60))
        resp.raise_for_status()
        normalized = normalize_llm_decision(extract_json_object(resp.json().get("response", "")))
        normalized["source"] = "ollama"
        return normalized
    except Exception as exc:
        fallback = heuristic_decision(row)
        fallback["reason"] = f"Ollama failed; using fallback. Error: {exc}"
        fallback["source"] = "heuristic_fallback_after_llm_error"
        return fallback


def make_safe_hold_decision(reason: str, source: str = "guardrail") -> Dict[str, Any]:
    return {
        "regime": "hold",
        "signal": "hold",
        "selected_features": [],
        "risk_multiplier": 0.0,
        "confidence": 0.0,
        "reason": reason[:500],
        "source": source,
        "guardrails_applied": [reason],
    }


def indicator_confirmation(row: pd.Series, decision: Dict[str, Any]) -> Tuple[bool, str]:
    signal = decision.get("signal", "hold")
    regime = decision.get("regime", "hold")
    if signal == "hold":
        return True, "hold requires no confirmation"

    close = float(row.get("close", np.nan))
    rsi = row.get("rsi_14", np.nan)
    macd_hist = row.get("macd_hist", np.nan)
    ema_20 = row.get("ema_20", np.nan)
    ema_50 = row.get("ema_50", np.nan)
    ema_100 = row.get("ema_100", np.nan)
    if any(pd.isna(x) for x in [close, rsi, macd_hist]):
        return False, "missing required indicator values"

    if signal == "buy" and regime == "swing_trading":
        trend_ok = pd.notna(ema_20) and pd.notna(ema_50) and close > ema_50 and ema_20 > ema_50
        momentum_ok = macd_hist > 0 and 45 <= rsi <= 75
        return bool(trend_ok and momentum_ok), "swing buy needs close>EMA50, EMA20>EMA50, MACD hist>0, and RSI 45-75"

    if signal == "buy" and regime == "value_investing":
        value_ok = (rsi <= 55) or (pd.notna(ema_100) and close < ema_100)
        return bool(value_ok), "DCA/value buy needs RSI<=55 or price below EMA100"

    if signal == "sell":
        sell_ok = (rsi >= 70) or (macd_hist < 0) or (pd.notna(ema_20) and close < ema_20)
        return bool(sell_ok), "sell needs RSI>=70, MACD hist<0, or close<EMA20"

    return False, "unsupported signal/regime combination"


def initialize_portfolio(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    initial_budget = float(config["initial_budget"])
    return {
        "initial_budget": initial_budget,
        "remaining_cash": initial_budget,
        "dca_budget_limit": initial_budget * float(config["dca_budget_fraction"]),
        "swing_budget_limit": initial_budget * float(config["swing_budget_fraction"]),
        "dca_capital_deployed": 0.0,
        "swing_capital_deployed": 0.0,
        "dca_reference_price": float(df.loc[0, "close"]),
        "open_positions": [],
        "buy_events": [],
        "sell_events": [],
        "equity_curve": [],
        "llm_decisions": [],
        "guardrail_events": [],
        "current_regime": "hold",
        "regime_duration": 0,
        "last_trade_idx": -10**9,
        "paused": False,
        "pause_reason": None,
    }


def active_positions(portfolio: Dict[str, Any], position_type: Optional[str] = None) -> List[Dict[str, Any]]:
    positions = [p for p in portfolio["open_positions"] if p.get("btc_remaining", 0.0) > 1e-12]
    return [p for p in positions if p.get("type") == position_type] if position_type else positions


def mark_to_market_value(portfolio: Dict[str, Any], current_price: float) -> float:
    btc_value = sum(p["btc_remaining"] * current_price for p in active_positions(portfolio))
    return float(portfolio["remaining_cash"]) + btc_value


def unrealized_pnl(portfolio: Dict[str, Any], current_price: float) -> float:
    return float(sum((current_price - p["entry_price"]) * p["btc_remaining"] for p in active_positions(portfolio)))


def realized_pnl(portfolio: Dict[str, Any]) -> float:
    return float(sum(e.get("pnl", 0.0) for e in portfolio["sell_events"]))


def apply_llm_guardrails(row: pd.Series, raw_decision: Dict[str, Any], portfolio: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize an LLM/fallback decision before it affects trading."""
    applied: List[str] = []
    try:
        decision = normalize_llm_decision(raw_decision)
    except Exception as exc:
        decision = make_safe_hold_decision(f"invalid LLM decision schema: {exc}")
        applied.append("invalid_schema_safe_hold")

    original_features = decision.get("selected_features", [])
    filtered_features = [f for f in original_features if f in ALLOWED_LLM_FEATURES]
    if len(filtered_features) != len(original_features):
        applied.append("feature_whitelist_filtered")
    decision["selected_features"] = filtered_features[:8]

    confidence = float(np.clip(decision.get("confidence", 0.0), 0.0, 1.0))
    risk_min = float(config.get("risk_multiplier_min", 0.10))
    risk_max = float(config.get("risk_multiplier_max", 1.00))
    decision["confidence"] = confidence
    decision["risk_multiplier"] = float(np.clip(decision.get("risk_multiplier", 0.0), risk_min, risk_max))

    min_conf = float(config.get("min_llm_confidence_to_trade", 0.55))
    if confidence < min_conf and decision.get("signal") in {"buy", "sell"}:
        decision["signal"] = "hold"
        decision["risk_multiplier"] = min(decision["risk_multiplier"], 0.25)
        applied.append(f"low_confidence_to_hold<{min_conf}")

    atr_pct = row.get("atr_pct", np.nan)
    if pd.notna(atr_pct):
        if atr_pct >= float(config.get("high_vol_atr_pct", 0.06)):
            decision["risk_multiplier"] *= 0.50
            applied.append("high_atr_volatility_50pct_risk")
        elif atr_pct >= float(config.get("medium_vol_atr_pct", 0.04)):
            decision["risk_multiplier"] *= 0.75
            applied.append("medium_atr_volatility_75pct_risk")

    current_equity = mark_to_market_value(portfolio, float(row["close"]))
    drawdown = max(0.0, (portfolio["initial_budget"] - current_equity) / portfolio["initial_budget"])
    if drawdown >= float(config.get("max_portfolio_drawdown_pause", 0.25)):
        decision = make_safe_hold_decision("hard drawdown pause threshold reached", source="guardrail_hard_pause")
        applied.append("hard_drawdown_safe_hold")
    elif drawdown >= float(config.get("soft_drawdown_throttle", 0.15)):
        decision["risk_multiplier"] *= 0.50
        applied.append("soft_drawdown_50pct_risk")

    current_regime = portfolio.get("current_regime", "hold")
    regime_duration = int(portfolio.get("regime_duration", 0))
    min_hold = int(config.get("regime_min_hold_candles", 5))
    proposed_regime = decision.get("regime", "hold")
    if current_regime != "hold" and proposed_regime != current_regime and regime_duration < min_hold:
        decision["regime"] = current_regime
        decision["signal"] = "hold"
        decision["risk_multiplier"] *= 0.50
        applied.append(f"regime_persistence_hold_{regime_duration}_of_{min_hold}")
    else:
        portfolio["regime_duration"] = regime_duration + 1 if proposed_regime == current_regime else 1
        portfolio["current_regime"] = proposed_regime

    if config.get("require_indicator_confirmation", True):
        ok, why = indicator_confirmation(row, decision)
        if not ok and decision.get("signal") in {"buy", "sell"}:
            decision["signal"] = "hold"
            decision["risk_multiplier"] *= 0.50
            applied.append(f"indicator_confirmation_failed: {why}")

    idx = int(row.name) if isinstance(row.name, (int, np.integer)) else 0
    candles_since_trade = idx - int(portfolio.get("last_trade_idx", -10**9))
    min_gap = int(config.get("min_candles_between_trades", 3))
    if candles_since_trade < min_gap and decision.get("signal") in {"buy", "sell"}:
        decision["signal"] = "hold"
        decision["risk_multiplier"] *= 0.50
        applied.append(f"trade_cooldown_{candles_since_trade}_of_{min_gap}")

    decision["risk_multiplier"] = float(np.clip(decision.get("risk_multiplier", 0.0), 0.0, risk_max))
    decision["guardrails_applied"] = applied
    if applied:
        portfolio.setdefault("guardrail_events", []).append(
            {
                "time": row.get("time"),
                "idx": idx,
                "close": float(row["close"]),
                "raw_regime": raw_decision.get("regime"),
                "raw_signal": raw_decision.get("signal"),
                "final_regime": decision.get("regime"),
                "final_signal": decision.get("signal"),
                "confidence": decision.get("confidence"),
                "risk_multiplier": decision.get("risk_multiplier"),
                "guardrails_applied": "; ".join(applied),
            }
        )
    return decision


def record_equity(row: pd.Series, portfolio: Dict[str, Any]) -> None:
    price = float(row["close"])
    equity = mark_to_market_value(portfolio, price)
    portfolio["equity_curve"].append(
        {
            "time": row["time"],
            "close": price,
            "remaining_cash": float(portfolio["remaining_cash"]),
            "total_btc": sum(p["btc_remaining"] for p in active_positions(portfolio)),
            "total_equity": equity,
            "unrealized_pnl": unrealized_pnl(portfolio, price),
            "realized_pnl": realized_pnl(portfolio),
            "drawdown_from_initial_budget": max(0.0, (portfolio["initial_budget"] - equity) / portfolio["initial_budget"]),
            "paused": portfolio["paused"],
        }
    )


def update_pause_state(row: pd.Series, portfolio: Dict[str, Any], config: Dict[str, Any]) -> None:
    equity = mark_to_market_value(portfolio, float(row["close"]))
    dd = (portfolio["initial_budget"] - equity) / portfolio["initial_budget"]
    if dd >= float(config["max_portfolio_drawdown_pause"]):
        portfolio["paused"] = True
        portfolio["pause_reason"] = f"Portfolio value dropped {dd:.2%} from initial budget. New entries paused."


def reduce_capital_deployed(portfolio: Dict[str, Any], position: Dict[str, Any], sold_fraction: float) -> None:
    amount = position["initial_usd"] * sold_fraction
    key = "dca_capital_deployed" if position["type"] == "dca" else "swing_capital_deployed"
    portfolio[key] = max(0.0, portfolio[key] - amount)


def execute_buy(row: pd.Series, portfolio: Dict[str, Any], position_type: str, usd_amount: float, reason: str, config: Dict[str, Any]) -> bool:
    if portfolio["paused"]:
        return False
    price = float(row["close"])
    fee_rate = float(config.get("fee_rate", 0.0))
    cap_key = "dca_capital_deployed" if position_type == "dca" else "swing_capital_deployed"
    limit_key = "dca_budget_limit" if position_type == "dca" else "swing_budget_limit"
    if position_type not in {"dca", "swing"}:
        raise ValueError("position_type must be 'dca' or 'swing'")
    usd_amount = min(float(usd_amount), portfolio["remaining_cash"], portfolio[limit_key] - portfolio[cap_key])
    if usd_amount <= 0:
        return False
    fee = usd_amount * fee_rate
    btc_amount = (usd_amount - fee) / price
    atr = row.get("ATR", np.nan)
    stop_loss = take_profit = None
    if position_type == "swing" and pd.notna(atr):
        stop_loss = price - float(config["swing_atr_stop_multiple"]) * float(atr)
        take_profit = price + float(config["swing_take_profit_atr_multiple"]) * float(atr)
    position = {
        "id": f"{position_type}_{pd.Timestamp(row['time']).strftime('%Y%m%d%H%M%S')}_{len(portfolio['open_positions']) + 1}",
        "type": position_type,
        "entry_time": row["time"],
        "entry_price": price,
        "initial_usd": usd_amount,
        "fee_entry": fee,
        "btc_initial": btc_amount,
        "btc_remaining": btc_amount,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "highest_price_seen": price,
        "entry_reason": reason,
    }
    portfolio["open_positions"].append(position)
    portfolio["remaining_cash"] -= usd_amount
    portfolio[cap_key] += usd_amount
    if position_type == "dca":
        portfolio["dca_reference_price"] = price
    portfolio["last_trade_idx"] = int(row.name) if isinstance(row.name, (int, np.integer)) else portfolio["last_trade_idx"]
    portfolio["buy_events"].append(
        {"time": row["time"], "type": position_type, "position_id": position["id"], "price": price, "usd_spent": usd_amount, "fee": fee, "btc_bought": btc_amount, "stop_loss": stop_loss, "take_profit": take_profit, "remaining_cash": portfolio["remaining_cash"], "reason": reason}
    )
    return True


def execute_sell(row: pd.Series, portfolio: Dict[str, Any], position: Dict[str, Any], btc_to_sell: float, exit_price: float, reason: str, config: Dict[str, Any]) -> bool:
    btc_to_sell = float(min(btc_to_sell, position["btc_remaining"]))
    if btc_to_sell <= 1e-12:
        return False
    gross = btc_to_sell * float(exit_price)
    fee = gross * float(config.get("fee_rate", 0.0))
    net = gross - fee
    pnl = net - (btc_to_sell * position["entry_price"])
    sold_fraction = btc_to_sell / max(position["btc_initial"], 1e-12)
    position["btc_remaining"] -= btc_to_sell
    portfolio["remaining_cash"] += net
    reduce_capital_deployed(portfolio, position, sold_fraction)
    portfolio["last_trade_idx"] = int(row.name) if isinstance(row.name, (int, np.integer)) else portfolio["last_trade_idx"]
    portfolio["sell_events"].append(
        {"time": row["time"], "type": position["type"], "position_id": position["id"], "entry_time": position["entry_time"], "entry_price": position["entry_price"], "exit_price": float(exit_price), "btc_sold": btc_to_sell, "gross_usd": gross, "fee": fee, "net_usd": net, "pnl": pnl, "remaining_cash": portfolio["remaining_cash"], "reason": reason}
    )
    return True


def process_swing_exits(row: pd.Series, portfolio: Dict[str, Any], llm_decision: Dict[str, Any], config: Dict[str, Any]) -> None:
    low, high, close = float(row["low"]), float(row["high"]), float(row["close"])
    atr = row.get("ATR", np.nan)
    for pos in list(active_positions(portfolio, "swing")):
        pos["highest_price_seen"] = max(float(pos.get("highest_price_seen", pos["entry_price"])), high)
        if config.get("swing_trailing_stop", True) and pd.notna(atr):
            trailing_stop = pos["highest_price_seen"] - float(config["swing_atr_stop_multiple"]) * float(atr)
            pos["stop_loss"] = trailing_stop if pos.get("stop_loss") is None else max(float(pos["stop_loss"]), trailing_stop)
        stop_hit = pos.get("stop_loss") is not None and low <= float(pos["stop_loss"])
        target_hit = pos.get("take_profit") is not None and high >= float(pos["take_profit"])
        bearish_exit = pd.notna(row.get("macd_hist", np.nan)) and row["macd_hist"] < 0 and close < row.get("ema_20", close)
        if stop_hit:
            execute_sell(row, portfolio, pos, pos["btc_remaining"], float(pos["stop_loss"]), "swing_atr_stop_loss", config)
        elif target_hit:
            execute_sell(row, portfolio, pos, pos["btc_remaining"], float(pos["take_profit"]), "swing_atr_take_profit", config)
        elif bearish_exit or llm_decision.get("signal") == "sell":
            execute_sell(row, portfolio, pos, pos["btc_remaining"], close, "swing_exit", config)


def process_dca_exits(row: pd.Series, portfolio: Dict[str, Any], llm_decision: Dict[str, Any], config: Dict[str, Any]) -> None:
    close = float(row["close"])
    rsi = row.get("rsi_14", np.nan)
    should_trim = (pd.notna(rsi) and rsi >= float(config["dca_exit_rsi"])) or llm_decision.get("signal") == "sell"
    if not should_trim:
        return
    for pos in list(active_positions(portfolio, "dca")):
        execute_sell(row, portfolio, pos, pos["btc_remaining"] * float(config["dca_exit_fraction"]), close, "dca_trim", config)


def process_entries(row: pd.Series, portfolio: Dict[str, Any], llm_decision: Dict[str, Any], config: Dict[str, Any]) -> None:
    if portfolio["paused"]:
        return
    close = float(row["close"])
    risk_multiplier = float(llm_decision.get("risk_multiplier", 0.5))
    regime = llm_decision.get("regime", "hold")
    signal = llm_decision.get("signal", "hold")
    if close > portfolio["dca_reference_price"]:
        portfolio["dca_reference_price"] = close
    dca_drop = (portfolio["dca_reference_price"] - close) / max(portfolio["dca_reference_price"], 1e-12)
    if regime == "value_investing" and signal in {"buy", "hold"} and dca_drop >= float(config["dca_drop_trigger_pct"]):
        execute_buy(row, portfolio, "dca", float(config["dca_buy_amount"]) * max(risk_multiplier, 0.25), f"DCA drop trigger: {dca_drop:.2%}", config)
    trend_ok = close > row.get("ema_50", close) and row.get("ema_20", close) > row.get("ema_50", close)
    momentum_ok = row.get("macd_hist", -1) > 0 and 45 <= row.get("rsi_14", 50) <= 75
    max_positions_ok = len(active_positions(portfolio, "swing")) < int(config["swing_max_open_positions"])
    if regime == "swing_trading" and signal == "buy" and trend_ok and momentum_ok and max_positions_ok:
        execute_buy(row, portfolio, "swing", float(config["swing_buy_amount"]) * risk_multiplier, "Swing entry", config)


def run_hybrid_llm_quant_strategy(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy().sort_values("time").reset_index(drop=True)
    portfolio = initialize_portfolio(df, config)
    last_decision = heuristic_decision(df.iloc[0])
    for idx, row in df.iterrows():
        if idx < int(config["min_rows_before_trading"]):
            record_equity(row, portfolio)
            continue
        raw_decision = call_ollama_decision(row, config) if idx % int(config["llm_every_n_candles"]) == 0 else last_decision
        last_decision = apply_llm_guardrails(row, raw_decision, portfolio, config)
        decision_record = dict(last_decision)
        decision_record.update({"time": row["time"], "idx": idx, "close": float(row["close"])})
        portfolio["llm_decisions"].append(decision_record)
        process_swing_exits(row, portfolio, last_decision, config)
        process_dca_exits(row, portfolio, last_decision, config)
        update_pause_state(row, portfolio, config)
        process_entries(row, portfolio, last_decision, config)
        update_pause_state(row, portfolio, config)
        record_equity(row, portfolio)
    return (
        portfolio,
        pd.DataFrame(portfolio["equity_curve"]),
        pd.DataFrame(portfolio["buy_events"]),
        pd.DataFrame(portfolio["sell_events"]),
        pd.DataFrame(portfolio["llm_decisions"]),
        pd.DataFrame(portfolio["guardrail_events"]),
    )


def summarize_performance(portfolio: Dict[str, Any], equity_df: pd.DataFrame, buys_df: pd.DataFrame, sells_df: pd.DataFrame) -> Dict[str, Any]:
    if equity_df.empty:
        return {}
    initial = float(portfolio["initial_budget"])
    final_equity = float(equity_df["total_equity"].iloc[-1])
    running_peak = equity_df["total_equity"].cummax()
    max_dd = ((running_peak - equity_df["total_equity"]) / running_peak).max()
    return {
        "initial_budget": initial,
        "final_equity": final_equity,
        "total_return_pct": (final_equity / initial - 1.0) * 100,
        "max_drawdown_pct": float(max_dd) * 100,
        "remaining_cash": float(portfolio["remaining_cash"]),
        "total_open_btc": sum(p["btc_remaining"] for p in active_positions(portfolio)),
        "realized_pnl": realized_pnl(portfolio),
        "num_buys": 0 if buys_df.empty else len(buys_df),
        "num_sells": 0 if sells_df.empty else len(sells_df),
        "paused": portfolio["paused"],
        "pause_reason": portfolio["pause_reason"],
    }


def accounting_check(equity_df: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    out = equity_df.copy()
    out["btc_market_value"] = out["total_btc"] * out["close"]
    out["recomputed_equity"] = out["remaining_cash"] + out["btc_market_value"]
    out["equity_diff"] = out["total_equity"] - out["recomputed_equity"]
    return out[out["equity_diff"].abs() > tolerance]
