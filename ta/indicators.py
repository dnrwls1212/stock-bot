# src/ta/indicators.py
from __future__ import annotations

from typing import Dict, Any
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
    return (100 - (100 / (1 + rs))).fillna(50.0)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def fetch_daily_ta(ticker: str, lookback: str = "9mo") -> Dict[str, Any]:
    t = (ticker or "").strip().upper()
    if not t: return {"ok": False, "error": "empty ticker"}

    r = safe_download(tickers=t, period=lookback, interval="1d", auto_adjust=False, progress=False, threads=True)
    if not r.get("ok"): return {"ok": False, "ticker": t, "error": r.get("error")}

    df = r.get("df")
    if df is None or len(df) < 70: return {"ok": False, "ticker": t, "error": "not enough rows"}
    if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]

    close = df["Close"].astype(float)
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    rsi14 = _rsi(close, 14)
    atr14 = _atr(df, 14)

    last_dt = df.index[-1]
    return {
        "ok": True, "ticker": t, "asof": str(getattr(last_dt, "date", lambda: last_dt)()),
        "close": _to_float(close.iloc[-1]), "prev_close": _to_float(close.iloc[-2]),
        "sma20": _to_float(sma20.iloc[-1]), "sma60": _to_float(sma60.iloc[-1]),
        "rsi14": _to_float(rsi14.iloc[-1]), "atr14": _to_float(atr14.iloc[-1]),
    }

def ta_score(daily: Dict[str, Any]) -> Dict[str, Any]:
    # 기존 일봉 기준 TA 스코어 (생략 없이 원본 유지)
    if not isinstance(daily, dict) or not daily.get("ok"):
        return {"ok": False, "ta_score": 0.0, "ta_label": "unknown", "error": "invalid"}

    close = float(daily["close"])
    sma20 = float(daily["sma20"])
    sma60 = float(daily["sma60"])
    rsi14 = float(daily["rsi14"])
    score = 0.0

    if close > sma20 > sma60: score += 0.6
    elif close > sma20 and sma20 >= sma60: score += 0.35
    elif close < sma20 < sma60: score -= 0.6
    elif close < sma20 and sma20 <= sma60: score -= 0.35

    if rsi14 >= 70: score += 0.25
    elif rsi14 >= 60: score += 0.15
    elif rsi14 <= 30: score -= 0.25
    elif rsi14 <= 40: score -= 0.15

    score = float(max(-1.0, min(1.0, score)))
    label = "strong_bullish" if score >= 0.75 else "bullish" if score >= 0.25 else "strong_bearish" if score <= -0.75 else "bearish" if score <= -0.25 else "neutral"

    return {"ok": True, "ticker": daily.get("ticker"), "ta_score": score, "ta_label": label, "rsi14": rsi14, "close": close, "sma20": sma20, "sma60": sma60, "atr14": float(daily.get("atr14", 0.0))}

# 🚀 [추가] MTF (Multi-Timeframe) 종합 분석 로직
def mtf_score(daily: Dict[str, Any], hourly: Dict[str, Any], min15: Dict[str, Any]) -> Dict[str, Any]:
    """
    일봉(추세) + 1시간봉(타점) + 15분봉(VWAP 수급)을 결합하여 매수 타점을 평가합니다.
    """
    score = 0.0
    reasons = []

    if not (daily.get("ok") and hourly.get("ok") and min15.get("ok")):
        return {"mtf_score": 0.0, "mtf_label": "neutral", "reason": "MTF 데이터 수집 실패"}

    # 1. 일봉 (대추세 파악)
    d_close, d_sma20 = daily["close"], daily["sma20"]
    if d_close > d_sma20:
        score += 0.4
        reasons.append("일봉 단기상승추세")
    else:
        score -= 0.3
        reasons.append("일봉 단기하락추세")

    # 2. 1시간봉 (중기 눌림목 타점 파악)
    h_close, h_ema9, h_rsi = hourly["close"], hourly["ema9"], hourly["rsi14"]
    if 40 <= h_rsi <= 65 and h_close > h_ema9:
        score += 0.3
        reasons.append("1H 눌림목 반등")
    elif h_rsi > 70:
        score -= 0.3
        reasons.append("1H 단기 과매수")

    # 3. 15분봉 (VWAP 기준 실시간 수급 파악)
    m_close = min15["close"]
    m_vwap = min15.get("vwap", m_close) # 없으면 close로 fallback
    if m_close > m_vwap:
        score += 0.3
        reasons.append("15m VWAP 돌파(수급유입)")
    else:
        score -= 0.3
        reasons.append("15m VWAP 하회(수급저항)")

    score = float(max(-1.0, min(1.0, score)))
    label = "strong_buy" if score >= 0.7 else "buy" if score >= 0.3 else "strong_sell" if score <= -0.7 else "sell" if score <= -0.3 else "neutral"

    return {
        "mtf_score": score,
        "mtf_label": label,
        "reason": " | ".join(reasons)
    }