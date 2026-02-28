from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _pct(x: float) -> str:
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "N/A"


def _f(x, nd: int = 2) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "N/A"


def _i(x) -> str:
    try:
        return str(int(x))
    except Exception:
        return "0"


def fmt_news(
    *,
    tickers: str,
    title: str,
    score: float,
    event_type: str = "",
    sentiment: str = "",
    conf: float = 0.0,
    link: str = "",
) -> str:
    return (
        "📰 [뉴스 감지]\n"
        f"종목: {tickers}\n"
        f"제목: {title}\n"
        f"요약: 점수={_f(score,2)} / 유형={event_type or '-'} / 감정={sentiment or '-'} / 신뢰도={_f(conf,2)}\n"
        f"링크: {link}"
    )


def fmt_start(
    *,
    watchlist: list[str],
    tick_seconds: int,
    execute_orders: str,
    ai_gate_enabled: bool,
    decision_enabled: bool,
    decision_override: bool,
) -> str:
    return (
        "🚀 [봇 시작]\n"
        f"관심종목: {', '.join(watchlist)}\n"
        f"틱 주기: {tick_seconds}초\n"
        f"실주문 실행(EXECUTE_ORDERS): {execute_orders}\n"
        f"AI 게이트: {'ON' if ai_gate_enabled else 'OFF'}\n"
        f"의사결정 에이전트: {'ON' if decision_enabled else 'OFF'}\n"
        f"에이전트가 주문을 덮어씀: {'ON' if decision_override else 'OFF'}"
    )

def fmt_order_submitted(
    ticker: str,
    side: str,
    qty: int,
    order_no: str,
    price: float,
    total: float,
    conf: float,
    ta_label: str,
    reason: str,
) -> str:
    # 1. 차트 추세(TA) 한글화 및 아이콘화
    ta_map = {
        "strong_bullish": "🔥 매우 강한 상승",
        "bullish": "📈 상승",
        "neutral": "⚖️ 중립",
        "bearish": "📉 하락",
        "strong_bearish": "❄️ 매우 강한 하락"
    }
    kr_ta = ta_map.get(ta_label.lower(), ta_label)
    
    # 2. 종합 점수에 따른 상태 평가
    if total >= 0.4: score_eval = "최상 🌟"
    elif total >= 0.15: score_eval = "우수 🟢"
    elif total > -0.15: score_eval = "보통 🟡"
    elif total > -0.4: score_eval = "주의 🟠"
    else: score_eval = "위험 🔴"
    
    # 3. 개발용 암호문 사유(reason)를 사용자 친화적인 한글로 단순화
    simple_reason = "💡 일반 매매 조건 충족" # 기본값
    if "FORCE_SELL" in reason: simple_reason = "⏰ 장 마감 전 전량 강제 매도"
    elif "STOP2 hit" in reason or "SL1 cut" in reason: simple_reason = "🛡️ 손절매(Stop Loss) 안전장치 발동"
    elif "TP1 hit" in reason or "TP2 hit" in reason: simple_reason = "💰 익절매(Take Profit) 수익 실현"
    elif "TRAIL" in reason: simple_reason = "🏃 트레일링 스탑 (고점 대비 하락으로 이익 보존)"
    elif "DECISION_OVERRIDE" in reason: simple_reason = "🤖 AI 에이전트 강제 매매 개입"
    elif "AI_REDUCED" in reason: simple_reason = "🤖 AI가 위험 감지하여 수량 축소 매매"
    elif "CHASE_SOFT" in reason: simple_reason = "⚠️ 단기 급등 감지되어 보수적으로 진입"
    elif "INVERSE" in reason: simple_reason = "📉 하락장 방어용 인버스 매수"
    elif "COST_BLOCK" in reason: simple_reason = "💸 수수료 대비 수익이 낮아 보류됨"

    s = "🔴 매도" if side.upper() == "SELL" else "🟢 매수"
    
    # 텔레그램 메시지 조립
    lines = [
        f"{s} 주문 접수: {ticker}",
        f"━━━━━━━━━━━━━━━━━━",
        f"📝 내역: {qty}주 (예상가 ${price:.2f})",
        f"📊 봇 평가: {score_eval} (점수 {total:.2f} / 확신도 {int(conf * 100)}%)",
        f"📈 차트 추세: {kr_ta}",
        f"🎯 매매 사유: {simple_reason}",
        f"🔑 주문 번호: {order_no}"
    ]
    return "\n".join(lines)


# (선택 사항) 모의투자(Dry Run)도 똑같이 예쁘게 보고 싶으시다면 아래 함수도 교체하세요.
def fmt_dry_run(
    ticker: str,
    side: str,
    qty: int,
    price: float,
    total: float,
    conf: float,
    ta_label: str,
    reason: str,
) -> str:
    # 위의 fmt_order_submitted와 동일한 매핑 로직을 재사용하거나 간단히 구성
    ta_map = {
        "strong_bullish": "🔥 매우 강한 상승", "bullish": "📈 상승",
        "neutral": "⚖️ 중립", "bearish": "📉 하락", "strong_bearish": "❄️ 매우 강한 하락"
    }
    kr_ta = ta_map.get(ta_label.lower(), ta_label)
    
    simple_reason = "💡 일반 매매 조건 충족"
    if "FORCE_SELL" in reason: simple_reason = "⏰ 장 마감 강제 매도"
    elif "STOP" in reason or "SL" in reason: simple_reason = "🛡️ 손절매 발동"
    elif "TP" in reason or "TRAIL" in reason: simple_reason = "💰 익절매 발동"
    
    s = "🔴 매도" if side.upper() == "SELL" else "🟢 매수"
    
    lines = [
        f"🧪 [모의 훈련] {s}: {ticker}",
        f"📝 내역: {qty}주 (예상가 ${price:.2f})",
        f"📊 평가 점수: {total:.2f} / 📈 추세: {kr_ta}",
        f"🎯 사유 요약: {simple_reason}",
        f"※ 모의투자이므로 실제 주문은 들어가지 않았습니다."
    ]
    return "\n".join(lines)


def fmt_label_summary(msg: str) -> str:
    # 기존 label 메시지가 이미 요약 문자열이면 앞에 이모지/헤더만 붙여줌
    return "🏷️ [라벨링 결과]\n" + msg


def fmt_perf_summary(msg: str) -> str:
    return "📊 [성과 리포트]\n" + msg