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


def _now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def _read_lines(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = normalize_ticker(line)
            if t:
                out.append(t)
    return out


def _unique_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _avg_dollar_volume(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    if "Close" not in df.columns or "Volume" not in df.columns:
        return 0.0
    dv = (df["Close"] * df["Volume"]).dropna()
    if dv.empty:
        return 0.0
    return _to_float(dv.mean(), 0.0)


def _last_close(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    s = df["Close"].dropna()
    if s.empty:
        return None
    return _to_float(s.iloc[-1], None)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty:
        return 0.0
    for c in ("High", "Low", "Close"):
        if c not in df.columns:
            return 0.0

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=period, min_periods=max(3, period // 2)).mean()
    last_atr = atr.dropna()
    last_close = close.dropna()
    if last_atr.empty or last_close.empty:
        return 0.0

    a = _to_float(last_atr.iloc[-1], 0.0)
    c = _to_float(last_close.iloc[-1], 0.0)
    if c <= 0:
        return 0.0
    return float(a / c)


def _load_events(events_path: str) -> List[Dict[str, Any]]:
    if not events_path or not os.path.exists(events_path):
        return []
    out: List[Dict[str, Any]] = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def _news_score_from_events(
    ticker: str,
    events: List[Dict[str, Any]],
    lookback_hours: int,
    now_kst: datetime,
) -> float:
    if not events:
        return 0.0

    t = ticker.upper()
    since = now_kst - timedelta(hours=max(1, int(lookback_hours)))
    score_sum = 0.0
    n = 0

    for e in events:
        try:
            ts = e.get("ts") or e.get("ts_kst") or ""
            if not ts:
                continue
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo("Asia/Seoul"))
            if dt < since:
                continue

            assigned = e.get("assigned") or []
            if not isinstance(assigned, list):
                continue
            assigned_u = [str(x).upper() for x in assigned]
            if t not in assigned_u:
                continue

            es = _to_float(e.get("event_score", 0.0), 0.0)
            score_sum += es
            n += 1
        except Exception:
            continue

    base = float(np.tanh(score_sum / 4.0))  # -1~1
    bonus = min(0.35, n * 0.05)
    return base + bonus


def _minmax_scale(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


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
    top_n: int = 12,
    min_price: float = 5.0,
    min_dollar_vol: float = 5_000_000.0,
    news_lookback_hours: int = 24,
    w_atr: float = 0.45,
    w_liq: float = 0.35,
    w_news: float = 0.20,
    events_path: str = "data/events.jsonl",
    # ---- 호환용 alias (main.py가 다른 이름을 쓰는 경우 대비) ----
    price_min: Optional[float] = None,
    dv_min: Optional[float] = None,
    topk: Optional[int] = None,
    **_ignored: Any,
) -> WatchlistAutoResult:
    """
    Universe에서 후보를 읽고 점수 기반 top_n 선정.

    ✅ ETF 포함/제외 없음: 점수 높으면 그대로 포함.
    ✅ main.py가 price_min 같은 키워드로 호출해도 자동 매핑.
    """
    # --- alias 매핑 ---
    if price_min is not None:
        min_price = float(price_min)
    if dv_min is not None:
        min_dollar_vol = float(dv_min)
    if topk is not None:
        top_n = int(topk)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    tickers = _read_lines(universe_path)
    tickers = [normalize_ticker(t) for t in tickers]
    tickers = [t for t in tickers if is_valid_ticker(t)]
    tickers = _unique_keep_order(tickers)

    if not tickers:
        return WatchlistAutoResult(
            ok=False,
            out_path=out_path,
            picked=[],
            reason=f"universe empty or not found: {universe_path}",
            scored=[],
        )

    now_kst = _now_kst()
    events = _load_events(events_path)

    rows: List[Dict[str, Any]] = []
    atr_list: List[float] = []
    liq_list: List[float] = []
    news_list: List[float] = []

    for t in tickers:
        df = _safe_download_history(t, period="3mo", interval="1d")
        if df is None or df.empty:
            continue

        last = _last_close(df)
        if last is None or float(last) < float(min_price):
            continue

        adv = _avg_dollar_volume(df)
        if float(adv) < float(min_dollar_vol):
            continue

        atrp = _atr_pct(df, period=14)
        nscore = _news_score_from_events(
            ticker=t,
            events=events,
            lookback_hours=news_lookback_hours,
            now_kst=now_kst,
        )

        atr_list.append(float(atrp))
        liq_list.append(float(adv))
        news_list.append(float(nscore))

        rows.append(
            {
                "ticker": t,
                "price": float(last),
                "atr_pct": float(atrp),
                "avg_dollar_vol": float(adv),
                "news_score": float(nscore),
            }
        )

    if not rows:
        return WatchlistAutoResult(
            ok=False,
            out_path=out_path,
            picked=[],
            reason="no candidates after filters (data fetch failed?)",
            scored=[],
        )

    atr_s = _minmax_scale(atr_list)
    liq_s = _minmax_scale(liq_list)
    news_01 = [(v + 1.0) / 2.0 for v in news_list]
    news_s = _minmax_scale(news_01)

    wsum = float(w_atr) + float(w_liq) + float(w_news)
    if wsum <= 1e-9:
        wsum = 1.0

    for i, r in enumerate(rows):
        score = (float(w_atr) * atr_s[i] + float(w_liq) * liq_s[i] + float(w_news) * news_s[i]) / wsum
        r["score"] = float(score)

    rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    picked = [r["ticker"] for r in rows[: max(1, int(top_n))]]

    with open(out_path, "w", encoding="utf-8") as f:
        for t in picked:
            f.write(t + "\n")

    return WatchlistAutoResult(
        ok=True,
        out_path=out_path,
        picked=picked,
        reason="",
        scored=rows[:200],
    )