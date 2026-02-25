# src/trading/dynamic_news_rules.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

from src.trading.news_store import NewsEvent, NewsStore

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


def _get_ts_kst(e: Any) -> str:
    # NewsEvent or dict 모두 지원
    if isinstance(e, dict):
        return str(e.get("ts_kst", "") or "")
    return str(getattr(e, "ts_kst", "") or "")


def _get_event_type(e: Any) -> str:
    if isinstance(e, dict):
        return str(e.get("event_type", "other") or "other")
    return str(getattr(e, "event_type", "other") or "other")


def _get_sentiment(e: Any) -> str:
    if isinstance(e, dict):
        return str(e.get("sentiment", "neutral") or "neutral")
    return str(getattr(e, "sentiment", "neutral") or "neutral")


def _event_strength(e: Any) -> float:
    """
    이벤트 강도(단기용): abs(event_score) * confidence
    - NewsEvent / dict 모두 지원
    """
    try:
        if isinstance(e, dict):
            s = float(e.get("event_score", 0.0) or 0.0)
        else:
            s = float(getattr(e, "event_score", 0.0) or 0.0)
    except Exception:
        s = 0.0

    try:
        if isinstance(e, dict):
            c = float(e.get("confidence", 0.55) or 0.55)
        else:
            c = float(getattr(e, "confidence", 0.55) or 0.55)
    except Exception:
        c = 0.55

    return abs(s) * _clamp(c, 0.0, 1.0)


@dataclass
class DynamicThresholdResult:
    buy_th: float
    sell_th: float
    conf_th: float
    confirm_ticks: int
    strength_boost: float
    reason: str


class DynamicNewsRules:
    """
    단기 뉴스 트레이딩 최적화 규칙:
    - 최근 N분 내 "가장 강한 이벤트 타입"을 찾고 임계치 조정
    - 30분 내 강한 뉴스 2개 이상이면 confirm_ticks=1 (fast-track)
    - (선택) strength_boost로 decide_signal의 strength를 약간 밀어줄 수 있음
      -> main.py에서 sig.strength에 더해주는 방식으로 사용
    """

    def __init__(self) -> None:
        # window
        self.window_min = _env_int("NEWS_DYN_WINDOW_MIN", 120)         # 2시간
        self.strong_window_min = _env_int("NEWS_DYN_STRONG_MIN", 30)   # 30분
        self.strong_cut = _env_float("NEWS_DYN_STRONG_CUT", 0.60)      # abs(score)*conf >= 0.60

        # threshold moves (단기용)
        self.easy_buy = _env_float("NEWS_DYN_EASY_BUY", 0.10)          # 좋은 타입이면 buy_th 0.10 낮춤
        self.easy_sell = _env_float("NEWS_DYN_EASY_SELL", 0.10)        # 나쁜 타입이면 sell_th를 덜 음수로(더 빨리 sell)
        self.easy_conf = _env_float("NEWS_DYN_EASY_CONF", 0.05)        # conf_th 낮춤

        self.hard_buy = _env_float("NEWS_DYN_HARD_BUY", 0.08)          # 잡음 타입이면 buy_th 올림
        self.hard_conf = _env_float("NEWS_DYN_HARD_CONF", 0.05)        # conf_th 올림

        # fast-track
        self.fast_confirm_ticks = _env_int("NEWS_DYN_FAST_CONFIRM_TICKS", 1)
        self.fast_strength_boost = _env_float("NEWS_DYN_FAST_STRENGTH_BOOST", 0.08)

        # clamps
        self.buy_min = _env_float("NEWS_DYN_BUY_MIN", 0.40)
        self.buy_max = _env_float("NEWS_DYN_BUY_MAX", 0.95)
        self.sell_min = _env_float("NEWS_DYN_SELL_MIN", -0.95)
        self.sell_max = _env_float("NEWS_DYN_SELL_MAX", -0.40)
        self.conf_min = _env_float("NEWS_DYN_CONF_MIN", 0.45)
        self.conf_max = _env_float("NEWS_DYN_CONF_MAX", 0.80)

    def apply(
        self,
        *,
        news_store: NewsStore,
        ticker: str,
        now_kst: datetime,
        base_buy_th: float,
        base_sell_th: float,
        base_conf_th: float,
        base_confirm_ticks: int,
        max_scan: int = 30,
    ) -> DynamicThresholdResult:
        # 최근 이벤트 확보
        # ⚠️ NewsStore.get_recent_events 기본값이 as_dict=True라 dict가 올 수 있음.
        events = news_store.get_recent_events(ticker, days=2, limit=max_scan, as_dict=True)

        # window 필터
        recent: List[Tuple[Any, float]] = []
        strong_recent: List[Tuple[Any, float]] = []

        for e in events:
            age = _age_minutes(now_kst, _get_ts_kst(e))
            if age is None:
                continue
            if age <= float(self.window_min):
                recent.append((e, age))
            if age <= float(self.strong_window_min):
                strong_recent.append((e, age))

        # default: 그대로
        buy_th = float(base_buy_th)
        sell_th = float(base_sell_th)
        conf_th = float(base_conf_th)
        confirm_ticks = int(base_confirm_ticks)
        strength_boost = 0.0

        if not recent:
            return DynamicThresholdResult(
                buy_th=_clamp(buy_th, self.buy_min, self.buy_max),
                sell_th=_clamp(sell_th, self.sell_min, self.sell_max),
                conf_th=_clamp(conf_th, self.conf_min, self.conf_max),
                confirm_ticks=confirm_ticks,
                strength_boost=strength_boost,
                reason="dyn:none(no_recent_news)",
            )

        # "가장 강한" 이벤트 선택
        best: Optional[Tuple[Any, float]] = None
        best_score = -1.0
        for e, age in recent:
            s = _event_strength(e)
            if s > best_score:
                best_score = s
                best = (e, age)

        assert best is not None
        e_best, age_best = best
        et = _norm(_get_event_type(e_best))
        sent = _norm(_get_sentiment(e_best))

        # 타입 분류
        positive_fast = {"guidance", "earnings", "contract", "mna", "regulation"}  # regulation은 케이스 바이 케이스지만 단기 반응이 큼
        negative_fast = {"regulation", "sec", "lawsuit", "product"}
        noisy = {"macro", "other", "partnership"}

        # 단기 로직:
        # - 좋은 타입 + positive면 BUY 문턱을 낮춤 / conf도 낮춤 (진입 빨리)
        # - 나쁜 타입 + negative면 SELL 문턱을 "덜 음수로" 올림 (더 빨리 SELL)
        # - noisy면 문턱을 올려서 잡음 제거
        why_parts = [f"best={et}/{sent} age={age_best:.0f}m str={best_score:.2f}"]

        if et in positive_fast and sent in ("positive", "bullish"):
            buy_th -= self.easy_buy
            conf_th -= self.easy_conf
            why_parts.append(f"easy_buy(-{self.easy_buy:.2f}) easy_conf(-{self.easy_conf:.2f})")

        if et in negative_fast and sent in ("negative", "bearish"):
            # sell_th는 음수. -0.70 -> -0.60 처럼 "더 빨리" 셀 조건을 만들려면 절댓값을 줄여야 함(덜 음수)
            sell_th += self.easy_sell
            conf_th -= self.easy_conf
            why_parts.append(f"easy_sell(+{self.easy_sell:.2f}) easy_conf(-{self.easy_conf:.2f})")

        if et in noisy:
            buy_th += self.hard_buy
            conf_th += self.hard_conf
            # noisy는 오히려 섣부른 SELL도 피하고 싶으면 sell_th를 더 음수로(엄격) 할 수도 있음.
            # 여기서는 기본 유지.
            why_parts.append(f"hard_buy(+{self.hard_buy:.2f}) hard_conf(+{self.hard_conf:.2f})")

        # fast-track: 30분 내 강한 뉴스가 2개 이상이면 confirm_ticks=1
        strong_hits = 0
        for e, _age in strong_recent:
            if _event_strength(e) >= float(self.strong_cut):
                strong_hits += 1

        if strong_hits >= 2:
            confirm_ticks = min(confirm_ticks, int(self.fast_confirm_ticks))
            strength_boost = float(self.fast_strength_boost)
            why_parts.append(f"fast_track strong_hits={strong_hits} confirm_ticks={confirm_ticks} boost=+{strength_boost:.2f}")

        return DynamicThresholdResult(
            buy_th=_clamp(buy_th, self.buy_min, self.buy_max),
            sell_th=_clamp(sell_th, self.sell_min, self.sell_max),
            conf_th=_clamp(conf_th, self.conf_min, self.conf_max),
            confirm_ticks=confirm_ticks,
            strength_boost=strength_boost,
            reason="dyn:" + " | ".join(why_parts),
        )