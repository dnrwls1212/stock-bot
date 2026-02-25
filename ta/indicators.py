from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.utils.yf_silent import safe_download


def _to_float(x) -> float:
    """
    pandas/numpy scalar, 단일 원소 Series, 단일 원소 ndarray 모두 안전하게 float로 변환.
    FutureWarning/TypeError 방지용.
    """
    try:
        return float(x)
    except TypeError:
        return float(np.asarray(x).item())


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing (EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def fetch_daily_ta(ticker: str, lookback: str = "9mo") -> Dict[str, Any]:
    """
    일봉 TA 계산용 스냅샷을 반환.
    - close, prev_close, sma20, sma60, rsi14, atr14 포함
    - yfinance 노이즈(예: 'possibly delisted')를 stdout/stderr로 뿌리는 경우가 있어 silent wrapper 사용
    """
    t = (ticker or "").strip().upper()
    if not t:
        return {"ok": False, "error": "empty ticker"}

    r = safe_download(
        tickers=t,
        period=lookback,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if not r.get("ok"):
        return {"ok": False, "ticker": t, "error": r.get("error")}

    df = r.get("df")
    if df is None or len(df) < 70:
        return {"ok": False, "ticker": t, "error": f"not enough rows: {0 if df is None else len(df)}"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            return {"ok": False, "ticker": t, "error": f"missing column: {col}"}

    close = df["Close"].astype(float)
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    rsi14 = _rsi(close, 14)
    atr14 = _atr(df, 14)

    last_i = -1
    prev_i = -2

    last_dt = df.index[last_i]
    asof = str(getattr(last_dt, "date", lambda: last_dt)())

    return {
        "ok": True,
        "ticker": t,
        "asof": asof,
        "close": _to_float(close.iloc[last_i]),
        "prev_close": _to_float(close.iloc[prev_i]),
        "sma20": _to_float(sma20.iloc[last_i]),
        "sma60": _to_float(sma60.iloc[last_i]),
        "rsi14": _to_float(rsi14.iloc[last_i]),
        "atr14": _to_float(atr14.iloc[last_i]),
    }


def ta_score(daily: Dict[str, Any]) -> Dict[str, Any]:
    """
    daily(=fetch_daily_ta 결과)를 받아 TA 점수/라벨을 계산.
    - ta_score: -1.0 ~ +1.0 (대략)
    - ta_label: strong_bullish/bullish/neutral/bearish/strong_bearish
    """
    if not isinstance(daily, dict) or not daily.get("ok"):
        return {
            "ok": False,
            "ta_score": 0.0,
            "ta_label": "unknown",
            "error": daily.get("error") if isinstance(daily, dict) else "invalid",
        }

    close = float(daily["close"])
    sma20 = float(daily["sma20"])
    sma60 = float(daily["sma60"])
    rsi14 = float(daily["rsi14"])

    score = 0.0

    # Trend (SMA)
    if close > sma20 > sma60:
        score += 0.6
    elif close > sma20 and sma20 >= sma60:
        score += 0.35
    elif close < sma20 < sma60:
        score -= 0.6
    elif close < sma20 and sma20 <= sma60:
        score -= 0.35

    # Momentum (RSI)
    if rsi14 >= 70:
        score += 0.25
    elif rsi14 >= 60:
        score += 0.15
    elif rsi14 <= 30:
        score -= 0.25
    elif rsi14 <= 40:
        score -= 0.15

    score = float(max(-1.0, min(1.0, score)))

    if score >= 0.75:
        label = "strong_bullish"
    elif score >= 0.25:
        label = "bullish"
    elif score <= -0.75:
        label = "strong_bearish"
    elif score <= -0.25:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "ok": True,
        "ticker": daily.get("ticker"),
        "asof": daily.get("asof"),
        "ta_score": score,
        "ta_label": label,
        "rsi14": rsi14,
        "close": close,
        "sma20": sma20,
        "sma60": sma60,
        "atr14": float(daily.get("atr14", 0.0)),
    }