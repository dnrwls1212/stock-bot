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
    intraday 캔들 기반 스냅샷.
    - 기본: 1분봉(interval=1m)
    - yfinance 제약 때문에 1m은 보통 period=1d가 안전
    - yfinance가 종종 stdout/stderr로 노이즈(예: 'possibly delisted')를 출력하므로 silent wrapper 사용
    """
    t = (ticker or "").strip().upper()
    if not t:
        return {"ok": False, "error": "empty ticker"}

    if period is None:
        period = "1d" if interval == "1m" else "5d"

    r = safe_download(
        tickers=t,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if not r.get("ok"):
        return {"ok": False, "ticker": t, "interval": interval, "period": period, "error": r.get("error")}

    df = r.get("df")
    if df is None or df.empty or len(df) < 80:
        return {"ok": False, "ticker": t, "error": f"not enough rows: {0 if df is None else len(df)}"}

    # yfinance가 멀티인덱스로 오는 경우가 있어 단일 티커로 정리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            return {"ok": False, "ticker": t, "error": f"missing column: {col}"}

    close = df["Close"].astype(float)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    rsi14 = _rsi(close, 14)

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
    }