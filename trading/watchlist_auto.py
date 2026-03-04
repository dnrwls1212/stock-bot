# src/trading/watchlist_auto.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from .universe_builder import normalize_ticker, is_valid_ticker, _to_float, _safe_download_history

# 🚨 [추가] 가치평가 및 차트 분석 모듈 임포트
from src.valuation.market_data import fetch_snapshot
from src.valuation.fair_value import compute_fair_value_snapshot
from src.ta.indicators import fetch_daily_ta, ta_score

def _now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))

def _read_lines(path: str) -> List[str]:
    if not path or not os.path.exists(path): return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = normalize_ticker(line)
            if t: out.append(t)
    return out

def _unique_keep_order(xs: List[str]) -> List[str]:
    seen = set(); out = []
    for x in xs:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def _avg_dollar_volume(df: pd.DataFrame) -> float:
    if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns: return 0.0
    dv = (df["Close"] * df["Volume"]).dropna()
    return _to_float(dv.mean(), 0.0) if not dv.empty else 0.0

def _last_close(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or "Close" not in df.columns: return None
    s = df["Close"].dropna()
    return _to_float(s.iloc[-1], None) if not s.empty else None

def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty: return 0.0
    for c in ("High", "Low", "Close"):
        if c not in df.columns: return 0.0
    high, low, close = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=max(3, period // 2)).mean()
    last_atr, last_close = atr.dropna(), close.dropna()
    if last_atr.empty or last_close.empty: return 0.0
    a, c = _to_float(last_atr.iloc[-1], 0.0), _to_float(last_close.iloc[-1], 0.0)
    return float(a / c) if c > 0 else 0.0

def _load_events(events_path: str) -> List[Dict[str, Any]]:
    if not events_path or not os.path.exists(events_path): return []
    out: List[Dict[str, Any]] = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line.strip())
                if isinstance(obj, dict): out.append(obj)
            except Exception: continue
    return out

def _news_score_from_events(ticker: str, events: List[Dict[str, Any]], lookback_hours: int, now_kst: datetime) -> float:
    if not events: return 0.0
    t, since, score_sum, n = ticker.upper(), now_kst - timedelta(hours=max(1, int(lookback_hours))), 0.0, 0
    for e in events:
        try:
            ts = e.get("ts") or e.get("ts_kst") or ""
            if not ts: continue
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None: dt = dt.replace(tzinfo=ZoneInfo("Asia/Seoul"))
            if dt < since: continue
            assigned = e.get("assigned") or []
            if t not in [str(x).upper() for x in assigned]: continue
            score_sum += _to_float(e.get("event_score", 0.0), 0.0)
            n += 1
        except Exception: continue
    return float(np.tanh(score_sum / 4.0)) + min(0.35, n * 0.05)

def _minmax_scale(values: List[float]) -> List[float]:
    if not values: return []
    vmin, vmax = min(values), max(values)
    return [0.5 for _ in values] if vmax - vmin < 1e-12 else [(v - vmin) / (vmax - vmin) for v in values]


@dataclass
class WatchlistAutoResult:
    ok: bool
    out_path: str
    picked: List[str]
    reason: str = ""
    scored: Optional[List[Dict[str, Any]]] = None


def build_watchlist_v1(
    *,
    universe_path: str = "data/universe.txt",
    out_path: str = "data/watchlist_auto.txt",
    top_n: int = 12, # 🚨 [수정] 20개 -> 12개로 축소 (통신 부하 최소화)
    min_price: float = 10.0, # 🚨 [수정] 5.0 -> 10.0달러 (잡주 필터링)
    min_dollar_vol: float = 15_000_000.0, # 🚨 [수정] 5백만 -> 1,500만 달러 (유동성 컷 상향)
    news_lookback_hours: int = 24,
    w_atr: float = 0.45,
    w_liq: float = 0.35,
    w_news: float = 0.20,
    events_path: str = "data/events.jsonl",
    price_min: Optional[float] = None,
    dv_min: Optional[float] = None,
    topk: Optional[int] = None,
    **_ignored: Any,
) -> WatchlistAutoResult:

    if price_min is not None: min_price = float(price_min)
    if dv_min is not None: min_dollar_vol = float(dv_min)
    if topk is not None: top_n = int(topk)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    tickers = _read_lines(universe_path)
    tickers = _unique_keep_order([t for t in [normalize_ticker(t) for t in tickers] if is_valid_ticker(t)])

    if not tickers: return WatchlistAutoResult(ok=False, out_path=out_path, picked=[], reason="universe empty", scored=[])

    now_kst = _now_kst()
    events = _load_events(events_path)

    rows: List[Dict[str, Any]] = []
    atr_list, liq_list, news_list = [], [], []

    print("[WATCHLIST] 펀더멘탈 및 기술적 지표 심층 분석 중... (최대 5분 소요될 수 있습니다)")
    for t in tickers:
        df = _safe_download_history(t, period="3mo", interval="1d")
        if df is None or df.empty: continue
        last = _last_close(df)
        if last is None or float(last) < float(min_price): continue
        adv = _avg_dollar_volume(df)
        if float(adv) < float(min_dollar_vol): continue

        atrp = _atr_pct(df, period=14)
        nscore = _news_score_from_events(ticker=t, events=events, lookback_hours=news_lookback_hours, now_kst=now_kst)

        # 🚨 [핵심 추가] 각 티커별 가치평가(vscore)와 차트 낙폭(tscore) 긁어오기
        vscore = 0.0
        tscore = 0.0
        try:
            snap = fetch_snapshot(t)
            snap["price"] = float(last)
            fv = compute_fair_value_snapshot(snap)
            vscore = float(fv.get("value_score", 0.0))
        except Exception: pass

        try:
            daily = fetch_daily_ta(t)
            ta_out = ta_score(daily)
            tscore = float(ta_out.get("ta_score", 0.0))
        except Exception: pass

        atr_list.append(float(atrp))
        liq_list.append(float(adv))
        news_list.append(float(nscore))

        rows.append({
            "ticker": t, "price": float(last), "atr_pct": float(atrp), "avg_dollar_vol": float(adv),
            "news_score": float(nscore), "vscore": vscore, "tscore": tscore
        })

    if not rows: return WatchlistAutoResult(ok=False, out_path=out_path, picked=[], reason="no candidates", scored=[])

    atr_s, liq_s = _minmax_scale(atr_list), _minmax_scale(liq_list)
    news_s = _minmax_scale([(v + 1.0) / 2.0 for v in news_list])
    wsum = max(1e-9, float(w_atr) + float(w_liq) + float(w_news))

    for i, r in enumerate(rows):
        # 1. 모멘텀 스코어 (기존: 유동성 + 변동성 + 뉴스)
        mom_score = (float(w_atr) * atr_s[i] + float(w_liq) * liq_s[i] + float(w_news) * news_s[i]) / wsum
        r["momentum_score"] = float(mom_score)

        # 2. 🚨 가치투자 바닥 줍기 스코어 (Value Dip Score)
        # 내재가치(vscore)가 높은데, 차트가 과매도(음수 tscore)일 경우 가산점 부여
        dip_bonus = abs(r["tscore"]) if r["tscore"] < -0.1 else 0.0
        r["value_dip_score"] = float(r["vscore"]) + dip_bonus

    # =========================================================
    # 🚨 투트랙(Two-Track) 와치리스트 선발 (정예 멤버 12개 압축)
    # =========================================================
    half_n = max(1, int(top_n) // 2)
    picked_set = set()
    picked_list = []

    # 🚀 트랙 1: 모멘텀 상위 선발 (단, 하락 추세인 허수 종목 제외)
    rows_mom = sorted(rows, key=lambda x: x.get("momentum_score", 0.0), reverse=True)
    mom_count = 0
    for r in rows_mom:
        # 하락 중이어서 변동성이 커진 종목은 봇이 어차피 안 사므로 제외 (tscore >= -0.1 이상만 허용)
        if r["tscore"] >= -0.1:
            picked_set.add(r["ticker"])
            picked_list.append(r["ticker"])
            mom_count += 1
        if mom_count >= half_n:
            break

    # 🚀 트랙 2: 가치투자(Value Dip) 상위 선발 (기준 강화)
    rows_val = sorted(rows, key=lambda x: x.get("value_dip_score", 0.0), reverse=True)
    for r in rows_val:
        if r["ticker"] not in picked_set:
            # vscore가 최소 0.5 이상(매우 훌륭한 펀더멘털)이면서 차트가 눌림목인 종목만 줍줍 후보로 인정
            if r["vscore"] >= 0.5: 
                picked_set.add(r["ticker"])
                picked_list.append(r["ticker"])
            if len(picked_list) >= top_n:
                break

    # 만약 조건을 만족하는 종목이 부족해서 top_n(12개)을 못 채우면, 남은 자리는 추세 무관 모멘텀 상위로 채움
    if len(picked_list) < top_n:
        for r in rows_mom:
            if r["ticker"] not in picked_set:
                picked_set.add(r["ticker"])
                picked_list.append(r["ticker"])
                if len(picked_list) >= top_n:
                    break

    with open(out_path, "w", encoding="utf-8") as f:
        for t in picked_list:
            f.write(t + "\n")

    return WatchlistAutoResult(ok=True, out_path=out_path, picked=picked_list, reason="", scored=rows[:200])