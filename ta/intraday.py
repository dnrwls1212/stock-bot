# src/ta/intraday.py
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from src.utils.yf_silent import safe_download

def _to_float(x) -> float:
    try:
        return float(x)
    except TypeError:
        return float(np.asarray(x).item())

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def fetch_intraday_ta(
    ticker: str,
    interval: str = "1m",
    period: Optional[str] = None,
) -> Dict[str, Any]:
    """
    intraday 캔들 기반 스냅샷. (VWAP 수급 지표 포함)
    """
    t = (ticker or "").strip().upper()
    if not t:
        return {"ok": False, "error": "empty ticker"}

    # MTF 분석을 위해 interval에 따라 안전한 period 자동 설정
    if period is None:
        if interval == "1m": period = "1d"
        elif interval == "15m": period = "5d"
        elif interval == "60m": period = "1mo"
        else: period = "5d"

    r = safe_download(
        tickers=t,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if not r.get("ok"):
        return {"ok": False, "ticker": t, "interval": interval, "error": r.get("error")}

    df = r.get("df")
    if df is None or df.empty or len(df) < 20:
        return {"ok": False, "ticker": t, "error": f"not enough rows"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            return {"ok": False, "ticker": t, "error": f"missing column: {col}"}

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    rsi14 = _rsi(close, 14)

    # 🚀 VWAP (거래량 가중 평균가) 계산
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()

    last_dt = df.index[-1]

    return {
        "ok": True,
        "ticker": t,
        "interval": interval,
        "period": period,
        "bar_time": str(last_dt),
        "close": _to_float(close.iloc[-1]),
        "ema9": _to_float(ema9.iloc[-1]),
        "ema21": _to_float(ema21.iloc[-1]),
        "rsi14": _to_float(rsi14.iloc[-1]),
        "vwap": _to_float(vwap.iloc[-1]), # 기관 수급 진입 타점용
    }