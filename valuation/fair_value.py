# src/valuation/fair_value.py
from __future__ import annotations

from typing import Any, Dict, Optional

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

# 주요 섹터별 대략적인 평균 PE (yfinance 섹터 맵핑용 Fallback)
SECTOR_PE_BASELINE = {
    "Technology": 35.0,
    "Healthcare": 25.0,
    "Financial Services": 15.0,
    "Consumer Cyclical": 22.0,
    "Communication Services": 20.0,
    "Industrials": 20.0,
}

def compute_fair_value_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    PEG, 월가 투자의견, EPS Revision(실적 추정치 상향), 상대적 저평가를 입체적으로 평가합니다.
    """
    pe = _to_float(snapshot.get("forward_pe")) or _to_float(snapshot.get("trailing_pe"))
    peg = _to_float(snapshot.get("peg"))
    price = _to_float(snapshot.get("price"))
    target_price = _to_float(snapshot.get("target_price"))
    eg = _to_float(snapshot.get("earnings_growth"))
    rg = _to_float(snapshot.get("revenue_growth"))
    rec_mean = _to_float(snapshot.get("recommendation_mean"))
    
    t_eps = _to_float(snapshot.get("trailing_eps"))
    f_eps = _to_float(snapshot.get("forward_eps"))
    sector = snapshot.get("sector", "Unknown")

    vscore = 0.0
    reason_parts = []

    # 🚀 1. EPS Revision (실적 추정치 상향 모멘텀 - 가장 강력한 팩트)
    if t_eps is not None and f_eps is not None and t_eps > 0:
        eps_est_growth = (f_eps - t_eps) / t_eps
        if eps_est_growth >= 0.20:
            vscore += 0.5
            reason_parts.append(f"EPS 추정치 강력상향 (+{eps_est_growth*100:.0f}%)")
        elif eps_est_growth > 0:
            vscore += 0.2
            reason_parts.append(f"EPS 추정치 상향 (+{eps_est_growth*100:.0f}%)")
        elif eps_est_growth < -0.10:
            vscore -= 0.4
            reason_parts.append(f"EPS 추정치 하향 ({eps_est_growth*100:.0f}%)")

    # 🚀 2. 동종업계 상대 평가 (Relative Valuation) 및 PEG
    if peg is not None and peg > 0:
        if peg <= 1.2:
            vscore += 0.4
            reason_parts.append(f"PEG {peg:.2f} (초고속 성장 저평가)")
        elif peg > 2.5:
            vscore -= 0.3
            reason_parts.append(f"PEG {peg:.2f} (성장 대비 고평가)")
    elif pe is not None: 
        # 절대적 기준이 아닌 섹터 평균을 바탕으로 상대평가
        sector_pe = SECTOR_PE_BASELINE.get(sector, 20.0)
        if pe < sector_pe * 0.8: # 섹터 평균 대비 20% 저평가
            vscore += 0.3
            reason_parts.append(f"상대적 저평가 (PE {pe:.1f} < 섹터평균 {sector_pe})")
        elif pe > sector_pe * 1.5: # 섹터 평균 대비 50% 고평가
            vscore -= 0.3
            reason_parts.append(f"상대적 고평가 (PE {pe:.1f} > 섹터평균 {sector_pe})")

    # 🚀 3. 월가 애널리스트 컨센서스 평가
    if rec_mean is not None:
        if rec_mean <= 2.0:
            vscore += 0.3
            reason_parts.append(f"월가 투자의견 {rec_mean:.1f} (강력매수)")
        elif rec_mean >= 3.0:
            vscore -= 0.3
            reason_parts.append(f"월가 투자의견 {rec_mean:.1f} (보류/매도)")

    # 🚀 4. 목표가 괴리율 (Upside)
    upside = 0.0
    if price is not None and target_price is not None and price > 0:
        upside = (target_price - price) / price
        if upside >= 0.15:
            vscore += 0.2
            reason_parts.append(f"상승여력 {upside*100:.0f}%")

    # 최종 점수 정규화 (-1.0 ~ 1.0)
    vscore = max(-1.0, min(1.0, vscore))

    # 🚨 가짜 바닥(Value Trap) 방어막 (매출/이익 역성장인데 점수가 높을 때)
    if vscore >= 0.4 and (eg is not None and eg < 0) and (rg is not None and rg < 0):
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

def value_score(snapshot: Dict[str, Any]) -> float:
    try:
        out = compute_fair_value_snapshot(snapshot)
        return float(out.get("value_score", 0.0))
    except Exception:
        return 0.0