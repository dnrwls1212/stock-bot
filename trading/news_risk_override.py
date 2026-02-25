# src/trading/news_risk_override.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from zoneinfo import ZoneInfo

from src.trading.news_store import NewsStore, NewsEvent
from src.trading.dynamic_news_rules import _event_strength  # reuse helper

KST = ZoneInfo("Asia/Seoul")


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)).strip())
    except Exception:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)).strip())
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _parse_iso_kst(ts_kst: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts_kst)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=KST)
        return dt
    except Exception:
        return None


def _age_minutes(now_kst: datetime, ts_kst: str) -> Optional[float]:
    dt = _parse_iso_kst(ts_kst)
    if dt is None:
        return None
    return max(0.0, (now_kst - dt).total_seconds() / 60.0)


@dataclass
class NewsRiskDecision:
    block_buy: bool
    force_sell: bool
    sell_frac: float        # force_sell일 때, 보유수량의 몇 %를 팔지
    tighten_sell: bool      # SELL 임계치를 더 쉽게 만들지(덜 음수로)
    loosen_buy: bool        # BUY 임계치를 더 쉽게 만들지(낮춤)
    reason: str


class NewsRiskOverride:
    """
    단기 뉴스 트레이딩용 '포지션-뉴스 충돌' 안전장치.

    기본 개념:
    - strongest recent event(짧은 window) 하나를 대표로 본다.
    - 현재 포지션이 Long인데, strongest가 strong negative면:
        block_buy=True, tighten_sell=True, (아주 강하면) force_sell=True
    - 현재 포지션이 Flat인데, strong negative면:
        block_buy=True
    - strongest가 strong positive면:
        loosen_buy=True (옵션), 그리고 불필요한 SELL을 막기 위해 tighten_sell=False 유지

    설정:
    - NEWS_RISK_WINDOW_MIN: 최근 몇 분을 볼지
    - NEWS_RISK_STRONG_CUT: 강한 뉴스 판정 기준(abs(score)*conf)
    - NEWS_RISK_FORCE_CUT: 강제 청산 판정 기준
    - NEWS_RISK_FORCE_SELL_FRAC: 강제 청산 비율(0.25 = 25%)
    - NEWS_RISK_TIGHTEN_SELL: sell_th를 얼마나 올릴지 (예: +0.10이면 더 빨리 SELL)
    - NEWS_RISK_BLOCK_BUY_MIN: negative면 항상 buy 차단할 최소 강도
    """

    def __init__(self) -> None:
        self.window_min = _env_int("NEWS_RISK_WINDOW_MIN", 90)          # 90분
        self.strong_cut = _env_float("NEWS_RISK_STRONG_CUT", 0.55)      # strong 판정
        self.force_cut = _env_float("NEWS_RISK_FORCE_CUT", 0.85)        # 강제 청산 판정

        self.force_sell_frac = _env_float("NEWS_RISK_FORCE_SELL_FRAC", 0.33)  # 33% 청산
        self.tighten_sell_delta = _env_float("NEWS_RISK_TIGHTEN_SELL", 0.10)  # sell_th += 0.10 (덜 음수)
        self.loosen_buy_delta = _env_float("NEWS_RISK_LOOSEN_BUY", 0.05)      # buy_th -= 0.05
        self.block_buy_min = _env_float("NEWS_RISK_BLOCK_BUY_MIN", 0.45)      # 이 이상 negative면 buy 차단

        # 특정 타입은 단기 리스크 영향이 크다(부정일 때)
        self.bad_types = {
            "regulation", "sec", "lawsuit", "product"
        }

    def evaluate(
        self,
        *,
        news_store: NewsStore,
        ticker: str,
        now_kst: datetime,
        pos_qty: float,
        max_scan: int = 25,
    ) -> Tuple[NewsRiskDecision, Optional[float], Optional[float]]:
        """
        returns:
          (decision, buy_th_delta, sell_th_delta)

        buy_th_delta: base buy_th에 더할 값 (음수면 buy를 쉽게)
        sell_th_delta: base sell_th에 더할 값 (양수면 sell을 쉽게)
        """
        events = news_store.get_recent_events(ticker, days=2, limit=max_scan)
        recent = []
        for e in events:
            age = _age_minutes(now_kst, e.ts_kst)
            if age is None:
                continue
            if age <= float(self.window_min):
                recent.append((e, age))

        if not recent:
            return (
                NewsRiskDecision(False, False, 0.0, False, False, "risk:none(no_recent)"),
                0.0,
                0.0,
            )

        # strongest event
        best_e = None
        best_s = -1.0
        best_age = 0.0
        for e, age in recent:
            s = _event_strength(e)
            if s > best_s:
                best_s = s
                best_e = e
                best_age = age

        assert best_e is not None
        sent = _norm(best_e.sentiment)
        et = _norm(best_e.event_type)
        s = float(best_s)

        long_pos = (float(pos_qty) or 0.0) > 0.0

        block_buy = False
        force_sell = False
        tighten_sell = False
        loosen_buy = False
        sell_frac = 0.0

        buy_delta = 0.0
        sell_delta = 0.0

        # 부정 강뉴스: long이면 방어
        if sent in ("negative", "bearish"):
            if s >= float(self.block_buy_min):
                block_buy = True

            # 리스크 큰 타입이면 더 빠르게 SELL 유도
            if et in self.bad_types and s >= float(self.strong_cut):
                tighten_sell = True
                sell_delta += float(self.tighten_sell_delta)

            # long 포지션이면 강제 청산 판단
            if long_pos and s >= float(self.force_cut):
                force_sell = True
                sell_frac = _clamp(float(self.force_sell_frac), 0.05, 1.0)
                tighten_sell = True
                sell_delta += float(self.tighten_sell_delta)

        # 강한 긍정뉴스: (옵션) buy를 조금 더 쉽게
        if sent in ("positive", "bullish"):
            if s >= float(self.strong_cut):
                loosen_buy = True
                buy_delta -= float(self.loosen_buy_delta)

        # 결과 reason
        reason = f"risk:best={et}/{sent} age={best_age:.0f}m str={s:.2f} long={int(long_pos)}"
        if block_buy:
            reason += " block_buy"
        if tighten_sell:
            reason += f" tighten_sell(+{sell_delta:.2f})"
        if loosen_buy:
            reason += f" loosen_buy({buy_delta:.2f})"
        if force_sell:
            reason += f" FORCE_SELL(frac={sell_frac:.2f})"

        return (
            NewsRiskDecision(
                block_buy=block_buy,
                force_sell=force_sell,
                sell_frac=sell_frac,
                tighten_sell=tighten_sell,
                loosen_buy=loosen_buy,
                reason=reason,
            ),
            buy_delta,
            sell_delta,
        )