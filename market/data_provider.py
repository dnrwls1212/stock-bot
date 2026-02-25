from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import time

from src.valuation.market_data import fetch_snapshot
from src.ta.indicators import fetch_daily_ta
from src.ta.intraday import fetch_intraday_ta


@dataclass
class MarketData:
    ticker: str
    snapshot: Dict[str, Any]
    daily_ta: Dict[str, Any]
    intraday_ta: Dict[str, Any]


class DataProvider:
    def get_market_data(self, ticker: str) -> MarketData:
        raise NotImplementedError


class DelayedProvider(DataProvider):
    """
    지연(폴링) 데이터: yfinance 기반.
    - snapshot: 매 tick마다 갱신(가격)
    - daily_ta: TTL 캐시
    - intraday_ta: interval에 맞는 TTL 캐시
    """

    def __init__(
        self,
        daily_lookback: str = "9mo",
        daily_ttl_sec: int = 1800,          # 30분
        intraday_interval: str = "5m",
        intraday_ttl_sec: Optional[int] = None,
    ):
        self.daily_lookback = daily_lookback
        self.daily_ttl_sec = int(daily_ttl_sec)
        self.intraday_interval = (intraday_interval or "5m").strip()

        if intraday_ttl_sec is None:
            intraday_ttl_sec = 120 if self.intraday_interval == "5m" else 40
        self.intraday_ttl_sec = int(intraday_ttl_sec)

        # caches: ticker -> (ts, data)
        self._daily_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._intra_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def _get_daily(self, ticker: str) -> Dict[str, Any]:
        now = time.time()
        key = ticker.upper()

        cached = self._daily_cache.get(key)
        if cached:
            ts, data = cached
            if now - ts < self.daily_ttl_sec:
                return data

        data = fetch_daily_ta(key, lookback=self.daily_lookback)
        self._daily_cache[key] = (now, data)
        return data

    def _get_intraday(self, ticker: str) -> Dict[str, Any]:
        now = time.time()
        key = ticker.upper()

        cached = self._intra_cache.get(key)
        if cached:
            ts, data = cached
            if now - ts < self.intraday_ttl_sec:
                return data

        data = fetch_intraday_ta(key, interval=self.intraday_interval)
        self._intra_cache[key] = (now, data)
        return data

    def get_market_data(self, ticker: str) -> MarketData:
        t = (ticker or "").strip().upper()
        snap = fetch_snapshot(t)
        daily = self._get_daily(t)
        intra = self._get_intraday(t)
        return MarketData(ticker=t, snapshot=snap, daily_ta=daily, intraday_ta=intra)


class RealTimeProvider(DataProvider):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RealTimeProvider는 추후 KIS WebSocket 실시간 연결 단계에서 구현합니다.")
