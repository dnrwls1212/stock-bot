# src/valuation/fair_value.py
from __future__ import annotations

from typing import Any, Dict, Optional


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def value_band_from_pe(pe: Optional[float]) -> Dict[str, Any]:
    """
    MVP용 단순 밸류에이션 밴드:
    - PE 낮을수록 +, 높을수록 -
    """
    if pe is None:
        return {
            "value_label": "unknown",
            "value_score": 0.0,
            "reason": "PE data not available",
        }

    pe = float(pe)
    if pe < 15:
        return {"value_label": "undervalued", "value_score": +1.0, "reason": f"PE={pe:.1f} is low"}
    if pe < 25:
        return {"value_label": "fair", "value_score": +0.2, "reason": f"PE={pe:.1f} is moderate"}
    if pe < 40:
        return {"value_label": "expensive", "value_score": -0.4, "reason": f"PE={pe:.1f} is high"}
    return {"value_label": "very_expensive", "value_score": -0.8, "reason": f"PE={pe:.1f} is very high"}


def compute_fair_value_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    snapshot에 trailing_pe/forward_pe 등이 있으면 사용.
    """
    pe = _to_float(snapshot.get("forward_pe")) or _to_float(snapshot.get("trailing_pe")) or _to_float(snapshot.get("pe"))
    band = value_band_from_pe(pe)
    return {"pe_used": pe, **band}


# ✅ main.py 호환: value_score 함수 제공
def value_score(snapshot: Dict[str, Any]) -> float:
    """
    main.py가 import하는 함수.
    snapshot(dict) -> value score(float)
    """
    try:
        out = compute_fair_value_snapshot(snapshot)
        return float(out.get("value_score", 0.0))
    except Exception:
        return 0.0
