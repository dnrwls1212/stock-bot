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
    *,
    ticker: str,
    side: str,
    qty: int,
    order_no: Optional[str],
    price: float,
    total: float,
    conf: float,
    ta_label: str,
    reason: str,
) -> str:
    return (
        "✅ [주문 접수]\n"
        f"종목: {ticker}\n"
        f"구분: {'매수' if side=='BUY' else '매도'}\n"
        f"수량: {_i(qty)}\n"
        f"주문번호: {order_no or '-'}\n"
        f"참고가격: {_f(price,2)}\n"
        f"시그널: total={_f(total,2)} / 뉴스신뢰={_f(conf,2)} / TA={ta_label}\n"
        f"사유: {reason}"
    )


def fmt_dry_run(
    *,
    ticker: str,
    side: str,
    qty: int,
    price: float,
    total: float,
    conf: float,
    ta_label: str,
    reason: str,
) -> str:
    return (
        "🧪 [모의주문(드라이런)]\n"
        f"종목: {ticker}\n"
        f"구분: {'매수' if side=='BUY' else '매도'}\n"
        f"수량: {_i(qty)}\n"
        f"참고가격: {_f(price,2)}\n"
        f"시그널: total={_f(total,2)} / 뉴스신뢰={_f(conf,2)} / TA={ta_label}\n"
        f"사유: {reason}"
    )


def fmt_label_summary(msg: str) -> str:
    # 기존 label 메시지가 이미 요약 문자열이면 앞에 이모지/헤더만 붙여줌
    return "🏷️ [라벨링 결과]\n" + msg


def fmt_perf_summary(msg: str) -> str:
    return "📊 [성과 리포트]\n" + msg