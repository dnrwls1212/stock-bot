from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf


@dataclass
class RegimeState:
    score: float
    label: str  # risk_on / neutral / risk_off


def _to_float_scalar(x) -> float:
    """
    pandas scalar / numpy scalar / 1-element Series 안전 변환
    """
    try:
        if isinstance(x, pd.Series):
            if len(x) == 0:
                return float("nan")
            return float(x.iloc[-1])
        return float(x)
    except Exception:
        try:
            import numpy as np
            return float(np.asarray(x).item())
        except Exception:
            return float("nan")


class RegimeFilter:
    """
    매우 단순한 market regime 판단기:
    - S&P500 (^GSPC) 기준
    - 종가 vs 20/60일선
    - 5일 수익률
    """

    def __init__(self, symbol: str = "^GSPC"):
        self.symbol = symbol
        self._last_state: Optional[RegimeState] = None

    def get(self) -> RegimeState:
        try:
            df = yf.download(self.symbol, period="6mo", interval="1d", progress=False)
        except Exception:
            return RegimeState(score=0.0, label="neutral")

        if df is None or len(df) < 70:
            return RegimeState(score=0.0, label="neutral")

        close = df["Close"]
        sma20 = close.rolling(20).mean()
        sma60 = close.rolling(60).mean()

        c = _to_float_scalar(close.iloc[-1])
        s20 = _to_float_scalar(sma20.iloc[-1])
        s60 = _to_float_scalar(sma60.iloc[-1])

        if len(close) >= 6:
            c5 = _to_float_scalar(close.iloc[-6])
        else:
            c5 = _to_float_scalar(close.iloc[0])

        # ----- 점수 계산 -----
        score = 0.0

        # 1) 추세 구조
        if c > s20:
            score += 0.3
        else:
            score -= 0.3

        if s20 > s60:
            score += 0.3
        else:
            score -= 0.3

        # 2) 단기 모멘텀
        if c5 > 0:
            ret5 = (c - c5) / c5
            if ret5 > 0:
                score += 0.2
            else:
                score -= 0.2

        # clamp
        score = max(-1.0, min(1.0, score))

        if score >= 0.3:
            label = "risk_on"
        elif score <= -0.3:
            label = "risk_off"
        else:
            label = "neutral"

        state = RegimeState(score=score, label=label)
        self._last_state = state
        return state