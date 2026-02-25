# src/strategy/scoring.py
from __future__ import annotations

import os
from typing import Any, Dict


EVENT_BASE = {
    "earnings": 2,
    "guidance": 2,
    "contract_partnership": 1,
    "regulation_export": 2,
    "mna": 2,
    "financing_offering": 1,
    "product_launch": 1,
    "litigation_investigation": 1,
    "macro_sector": 1,
    "other": 0,
}

SENTIMENT_SIGN = {
    "bullish": +1,
    "neutral": 0,
    "bearish": -1,
}


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)).strip())
    except Exception:
        return default


def event_score(analysis: Dict[str, Any]) -> float:
    """
    LLM이 준 event_type/sentiment/confidence 기반 이벤트 점수
    """
    et = analysis.get("event_type", "other")
    sent = analysis.get("sentiment", "neutral")

    try:
        conf = float(analysis.get("confidence", 0.5))
    except Exception:
        conf = 0.5

    base = EVENT_BASE.get(et, 0)
    sign = SENTIMENT_SIGN.get(sent, 0)

    # neutral은 방향성 약하니 약하게 반영
    if sign == 0:
        raw = base * 0.2
    else:
        raw = base * sign

    # confidence로 스케일링
    return float(raw) * max(0.0, min(1.0, conf))


# ✅ main.py 호환: score_event 이름 제공
def score_event(analysis: Dict[str, Any]) -> float:
    return event_score(analysis)


# ✅ main.py 호환: total_score 제공
def total_score(event_score: float, value_score: float, ta_score: float, intra_score: float = 0.0) -> float:
    """
    가중합 (env 가중치 사용)
    - W_NEWS, W_VAL, W_DAILY, W_INTRA
    """
    w_news = _env_float("W_NEWS", 0.55)
    w_val = _env_float("W_VAL", 0.55)
    w_daily = _env_float("W_DAILY", 0.60)
    w_intra = _env_float("W_INTRA", 0.70)

    denom = (w_news + w_val + w_daily + w_intra) or 1.0
    total = (
        (w_news * float(event_score))
        + (w_val * float(value_score))
        + (w_daily * float(ta_score))
        + (w_intra * float(intra_score))
    ) / denom

    return float(total)
