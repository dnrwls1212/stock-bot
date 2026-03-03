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
    기본 밸류에이션 밴드:
    - PE 낮을수록 +, 높을수록 -
    """
    if pe is None:
        return {"value_label": "unknown", "value_score": 0.0, "reason": "PE data not available"}

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
    PER 기반 점수를 계산한 뒤, 실적 성장(Growth)과 애널리스트 목표가(Upside)를 검증하여
    떨어지는 칼날(Value Trap)을 방지합니다.
    """
    pe = _to_float(snapshot.get("forward_pe")) or _to_float(snapshot.get("trailing_pe")) or _to_float(snapshot.get("pe"))
    price = _to_float(snapshot.get("price"))
    target_price = _to_float(snapshot.get("target_price"))
    eg = _to_float(snapshot.get("earnings_growth"))
    rg = _to_float(snapshot.get("revenue_growth"))

    band = value_band_from_pe(pe)
    vscore = float(band.get("value_score", 0.0))
    reason = str(band.get("reason", ""))

    upside = 0.0
    if price is not None and target_price is not None and price > 0:
        upside = (target_price - price) / price

    # 🚨 [핵심] Value Trap (가짜 낙폭과대) 필터링 🚨
    # PER이 낮아서 고득점(vscore >= 0.6)을 받았더라도 펀더멘탈이 구리면 강제 탈락시킵니다.
    if vscore >= 0.6:
        # 1. 성장성 검증 (매출이나 이익 중 하나라도 역성장이 아니어야 함)
        is_growing = False
        if (eg is not None and eg > 0.0) or (rg is not None and rg > 0.0):
            is_growing = True
            
        if not is_growing:
            vscore = 0.0 # 점수 박탈
            reason += " | Rejected: 역성장 우려 (No Growth)"
        # 2. 목표가 괴리율 검증 (애널리스트 평균 목표가가 현재가보다 최소 10% 이상 높아야 함)
        elif target_price is not None and upside < 0.10:
            vscore = 0.0 # 점수 박탈
            reason += f" | Rejected: 상승 여력 부족 (Upside {upside*100:.1f}% < 10%)"
        else:
            reason += f" | Verified: 펀더멘탈 우수 (Upside {upside*100:.1f}%)"

    return {
        "pe_used": pe,
        "value_score": vscore,
        "value_label": band.get("value_label", "unknown"),
        "reason": reason,
        "upside": upside,
        "earnings_growth": eg,
        "revenue_growth": rg
    }

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