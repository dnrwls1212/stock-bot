# src/valuation/fair_value.py
from __future__ import annotations

from typing import Any, Dict, Optional

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

def compute_fair_value_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    단순 PER이 아닌 PEG(성장 대비 가치), 월가 투자의견(Consensus), 
    이익 성장률(Earnings Growth)을 입체적으로 평가하여 점수를 산출합니다.
    """
    pe = _to_float(snapshot.get("forward_pe")) or _to_float(snapshot.get("trailing_pe"))
    peg = _to_float(snapshot.get("peg"))
    price = _to_float(snapshot.get("price"))
    target_price = _to_float(snapshot.get("target_price"))
    eg = _to_float(snapshot.get("earnings_growth"))
    rg = _to_float(snapshot.get("revenue_growth"))
    rec_mean = _to_float(snapshot.get("recommendation_mean"))

    vscore = 0.0
    reason_parts = []

    # 🚀 1. PEG Ratio 평가 (고성장주 우대)
    if peg is not None and peg > 0:
        if peg <= 1.2:
            vscore += 0.6
            reason_parts.append(f"PEG {peg:.2f} (초고속 성장 저평가)")
        elif peg <= 1.8:
            vscore += 0.3
            reason_parts.append(f"PEG {peg:.2f} (적정 성장)")
        elif peg > 2.5:
            vscore -= 0.3
            reason_parts.append(f"PEG {peg:.2f} (성장 대비 고평가)")
    elif pe is not None: 
        # PEG가 없을 때만 PE로 보수적 평가
        if pe < 15:
            vscore += 0.4
            reason_parts.append(f"PE {pe:.1f} (전통적 가치주)")
        elif pe > 40:
            vscore -= 0.2
            reason_parts.append(f"PE {pe:.1f} (절대적 고평가)")

    # 🚀 2. 월가 애널리스트 컨센서스 평가 (Revision 반영 효과)
    if rec_mean is not None:
        if rec_mean <= 2.0: # 1.0 ~ 2.0은 Strong Buy ~ Buy
            vscore += 0.4
            reason_parts.append(f"월가 투자의견 {rec_mean:.1f} (강력매수)")
        elif rec_mean >= 3.0: # 3.0 이상은 Hold ~ Sell
            vscore -= 0.3
            reason_parts.append(f"월가 투자의견 {rec_mean:.1f} (매수보류/매도)")

    # 🚀 3. 실적 성장(Earnings Growth) 모멘텀
    if eg is not None:
        if eg > 0.40: # YOY 40% 이상 이익 성장
            vscore += 0.3
            reason_parts.append(f"이익성장률 폭발적 (+{eg*100:.0f}%)")
        elif eg < 0.0:
            vscore -= 0.5
            reason_parts.append(f"이익성장률 역성장 ({eg*100:.0f}%)")

    # 🚀 4. 목표가 괴리율 (Upside)
    upside = 0.0
    if price is not None and target_price is not None and price > 0:
        upside = (target_price - price) / price
        if upside >= 0.15: # 상승 여력 15% 이상
            vscore += 0.2
            reason_parts.append(f"상승여력 {upside*100:.0f}%")
        elif upside < 0.0:
            vscore -= 0.3
            reason_parts.append(f"목표가 하회 (다운사이드)")

    # 최종 점수 정규화 (-1.0 ~ 1.0)
    vscore = max(-1.0, min(1.0, vscore))

    # 🚨 가짜 바닥(Value Trap) 최종 방어막
    if vscore >= 0.5 and (eg is not None and eg < 0) and (rg is not None and rg < 0):
        vscore = 0.0
        reason_parts.append("Value Trap 필터링 (매출/이익 동시 역성장)")

    reason = " | ".join(reason_parts) if reason_parts else "데이터 부족 (Neutral)"

    return {
        "pe_used": pe,
        "value_score": vscore,
        "value_label": "growth_value",
        "reason": reason,
        "upside": upside,
        "earnings_growth": eg,
        "revenue_growth": rg
    }

# ✅ main.py 호환 래퍼
def value_score(snapshot: Dict[str, Any]) -> float:
    try:
        out = compute_fair_value_snapshot(snapshot)
        return float(out.get("value_score", 0.0))
    except Exception:
        return 0.0