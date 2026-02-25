# src/eval/price_history.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

from src.utils.yf_silent import safe_download

KST = ZoneInfo("Asia/Seoul")
UTC = ZoneInfo("UTC")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_iso(ts: str) -> datetime:
    """
    ISO8601 문자열 -> datetime
    예) "2026-02-21T08:58:46+09:00" 또는 "2026-02-21T08:58:46"
    """
    return datetime.fromisoformat(ts)


def to_utc(dt: datetime) -> datetime:
    """
    tz-aware면 UTC 변환.
    tz-naive면 KST로 가정 후 UTC 변환.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(UTC)


def parse_kst_iso_to_utc(ts_kst_iso: str) -> datetime:
    """
    ✅ decision_labeler.py가 import 하는 이름(필수).
    KST ISO 문자열을 UTC datetime으로 변환.

    - 입력이 tz-aware(+09:00 포함)면 그대로 UTC로 변환
    - 입력이 tz-naive면 KST로 가정하고 UTC로 변환
    """
    dt = parse_iso(ts_kst_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(UTC)


def _yf_symbol(ticker: str) -> str:
    return (ticker or "").upper().strip()


def parse_horizon_to_timedelta(horizon: str) -> timedelta:
    h = (horizon or "").strip().lower()
    if not h:
        raise ValueError("empty horizon")

    if h.endswith("m"):
        n = int(h[:-1])
        return timedelta(minutes=n)
    if h.endswith("h"):
        n = int(h[:-1])
        return timedelta(hours=n)
    if h.endswith("d"):
        n = int(h[:-1])
        return timedelta(days=n)

    raise ValueError(f"unsupported horizon: {horizon}")


def horizon_to_timedelta(horizon: str) -> timedelta:
    """
    ✅ decision_labeler.py가 import 하는 이름(필수).
    """
    return parse_horizon_to_timedelta(horizon)


def decide_interval_for_horizon(horizon: str) -> str:
    """
    horizon에 맞춰 yfinance interval 선택(보수적)
    - 1h~12h: 1m
    - 12h 초과: 5m
    - d 단위: 15m
    """
    h = (horizon or "").strip().lower()
    if not h:
        return "15m"

    if h.endswith("m"):
        return "1m"

    if h.endswith("h"):
        try:
            n = int(h[:-1])
        except Exception:
            n = 1
        return "1m" if n <= 12 else "5m"

    if h.endswith("d"):
        return "15m"

    return "15m"


def decide_max_lookback_days(interval: str) -> int:
    """
    yfinance intraday 제한 고려(보수적으로)
    """
    itv = (interval or "").strip().lower()
    if itv == "1m":
        return 10
    if itv in ("2m", "5m", "15m", "30m"):
        return 30
    if itv in ("60m", "90m", "1h"):
        return 180
    return 365


@dataclass
class PricePoint:
    ts_utc: datetime
    price: float


class PriceHistoryCache:
    def __init__(self, cache_dir: str = "data/price_cache") -> None:
        self.cache_dir = cache_dir
        _ensure_dir(cache_dir)

    def _cache_path(self, ticker: str, interval: str) -> str:
        t = _yf_symbol(ticker)
        return os.path.join(self.cache_dir, f"{t}_{interval}.parquet")

    def load_cached(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        p = self._cache_path(ticker, interval)
        if not os.path.exists(p):
            return None
        try:
            df = pd.read_parquet(p)
            if df is None or df.empty:
                return None
            if "Close" not in df.columns:
                return None
            return df
        except Exception:
            return None

    def save_cached(self, ticker: str, interval: str, df: pd.DataFrame) -> None:
        p = self._cache_path(ticker, interval)
        try:
            df.to_parquet(p, index=True)
        except Exception:
            try:
                df.to_csv(p.replace(".parquet", ".csv"))
            except Exception:
                pass

    def _ensure_utc_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize(UTC)
            else:
                df.index = df.index.tz_convert(UTC)
        except Exception:
            pass
        return df

    def get_window(
        self,
        ticker: str,
        *,
        ts_from_utc: datetime,
        ts_to_utc: datetime,
        interval: str = "15m",
        max_lookback_days: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        sym = _yf_symbol(ticker)
        if not sym:
            return None

        end = ts_to_utc
        start = ts_from_utc

        if max_lookback_days is None:
            max_lookback_days = decide_max_lookback_days(interval)

        lb = timedelta(days=max_lookback_days)
        if end - start > lb:
            start = end - lb

        cached = self.load_cached(sym, interval)
        need_fetch = True

        if cached is not None and not cached.empty:
            cached = self._ensure_utc_index(cached)
            cmin = cached.index.min()
            cmax = cached.index.max()
            if cmin <= start and cmax >= end:
                need_fetch = False

        df_new: Optional[pd.DataFrame] = None
        if need_fetch:
            r = safe_download(
                tickers=sym,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            df_new = r.get("df") if r.get("ok") else None

            if df_new is not None and not df_new.empty:
                df_new = self._ensure_utc_index(df_new)

                if isinstance(df_new.columns, pd.MultiIndex):
                    df_new.columns = [c[0] for c in df_new.columns]

                if "Close" not in df_new.columns:
                    df_new = None

        if cached is None or cached.empty:
            merged = df_new
        else:
            if df_new is None or df_new.empty:
                merged = cached
            else:
                merged = pd.concat([cached, df_new]).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]

        if merged is None or merged.empty:
            return None

        merged = self._ensure_utc_index(merged)
        self.save_cached(sym, interval, merged)

        out = merged.loc[(merged.index >= start) & (merged.index <= end)]
        if out is None or out.empty:
            return None
        return out

    # ----------------------------
    # ✅ DecisionLabeler 호환 API
    # ----------------------------
    def fetch_history(
        self,
        ticker: str,
        *,
        start_utc: datetime,
        end_utc: datetime,
        interval: str = "15m",
        max_lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        decision_labeler.py가 호출하는 함수.
        내부적으로 get_window를 사용해서 dataframe을 반환.
        실패 시 빈 DF 반환(라벨러는 empty면 스킵 처리).
        """
        start_utc = to_utc(start_utc)
        end_utc = to_utc(end_utc)

        df = self.get_window(
            ticker,
            ts_from_utc=start_utc,
            ts_to_utc=end_utc,
            interval=interval,
            max_lookback_days=max_lookback_days,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return self._ensure_utc_index(df).sort_index()

    def pick_price_at_or_after(self, df: pd.DataFrame, ts_utc: datetime) -> Optional[PricePoint]:
        """
        decision_labeler.py가 호출하는 함수.
        df에서 ts_utc 시각 '이후(>=)' 첫 번째 bar의 Close를 반환.
        """
        if df is None or df.empty or "Close" not in df.columns:
            return None

        ts_utc = to_utc(ts_utc)
        df = self._ensure_utc_index(df).sort_index()

        try:
            idx = df.index
            # searchsorted: ts_utc 이상인 첫 위치
            pos = idx.searchsorted(ts_utc, side="left")
            if pos is None or int(pos) >= len(df):
                return None
            row = df.iloc[int(pos)]
            px = float(row["Close"])
            if px <= 0:
                return None
            return PricePoint(ts_utc=idx[int(pos)].to_pydatetime(), price=px)
        except Exception:
            return None

    # ----------------------------
    # 기존 helper (단일 horizon 라벨링용)
    # ----------------------------
    def get_price_at(
        self,
        ticker: str,
        *,
        ts_utc: datetime,
        horizon: timedelta,
        interval: str,
    ) -> Optional[Tuple[float, float]]:
        start = ts_utc - timedelta(minutes=10)
        end = ts_utc + horizon + timedelta(minutes=20)

        df = self.get_window(ticker, ts_from_utc=start, ts_to_utc=end, interval=interval)
        if df is None or df.empty:
            return None

        df = df.sort_index()

        try:
            entry_idx = df.index.get_indexer([ts_utc], method="nearest")[0]
            entry = float(df.iloc[entry_idx]["Close"])
        except Exception:
            return None

        future_ts = ts_utc + horizon
        try:
            future_idx = df.index.get_indexer([future_ts], method="nearest")[0]
            future = float(df.iloc[future_idx]["Close"])
        except Exception:
            return None

        return entry, future


def label_return_for_horizon(
    *,
    decision_ts_kst: str,
    ticker: str,
    horizon: str,
    cache: Optional[PriceHistoryCache] = None,
) -> Optional[Dict[str, float]]:
    if cache is None:
        cache = PriceHistoryCache()

    dt_utc = parse_kst_iso_to_utc(decision_ts_kst)
    td = horizon_to_timedelta(horizon)
    interval = decide_interval_for_horizon(horizon)

    got = cache.get_price_at(ticker, ts_utc=dt_utc, horizon=td, interval=interval)
    if got is None:
        return None

    entry, future = got
    if entry <= 0:
        return None

    ret = (future / entry) - 1.0
    return {"entry": float(entry), "future": float(future), "ret": float(ret)}