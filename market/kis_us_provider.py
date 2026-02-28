from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.broker.kis_client import KisClient


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _dump_jsonl(path: str, record: Dict[str, Any]) -> None:
    _ensure_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


@dataclass
class UsQuote:
    ticker: str
    price: float
    raw: Dict[str, Any]


class KisUsQuoteProvider:
    def __init__(self, kis: KisClient, tr_id_quote: Optional[str] = None) -> None:
        self.kis = kis
        self.tr_id_quote = tr_id_quote or os.environ.get("KIS_TRID_US_QUOTE", "").strip() or "HHDFS00000300"
        self.debug_dump = os.environ.get("KIS_DEBUG_DUMP", "0").strip() == "1"
        self.debug_dump_path = os.environ.get("KIS_DEBUG_DUMP_PATH", "data/kis_quote_raw.jsonl").strip()

    def get_quote(self, ticker: str, exchange: Optional[str] = None) -> UsQuote:
        excg = exchange or self.kis.cfg.exchange
        path = "/uapi/overseas-price/v1/quotations/price"
        params = {"AUTH": "", "EXCD": excg, "SYMB": ticker}

        j = self.kis.request("GET", path, tr_id=self.tr_id_quote, params=params)

        if self.debug_dump:
            _dump_jsonl(self.debug_dump_path, {"ts_utc": _utcnow_iso(), "ticker": ticker, "tr_id": self.tr_id_quote, "response": j})

        out = j.get("output") or j.get("output1") or j.get("output2") or j
        candidates = [out.get("last"), out.get("last_price"), out.get("ovrs_prpr"), out.get("stck_prpr"), out.get("prpr"), out.get("tdd_clpr")]

        price: Optional[float] = None
        for c in candidates:
            try:
                if c is not None:
                    price = float(c)
                    break
            except Exception: continue

        if price is None: raise RuntimeError(f"cannot parse quote price: {j}")
        return UsQuote(ticker=ticker, price=price, raw=j)

    # 🚨 [실전 API] 실시간 해외속보 조회
    def get_breaking_news(self) -> list:
        if self.kis.cfg.paper: return []
        path = "/uapi/overseas-price/v1/quotations/brknews-title"
        tr_id = "FHKST01011801"
        try:
            res = self.kis.request("GET", path, tr_id=tr_id, params={})
            if isinstance(res, dict) and str(res.get("rt_cd", "")) == "0":
                return res.get("output") or []
        except Exception: pass
        return []

    # 🚨 [실전 API] 거래량 급증 종목 탐색
    def get_volume_surge_tickers(self, excd: str = "NAS") -> list:
        if self.kis.cfg.paper: return []
        path = "/uapi/overseas-stock/v1/ranking/volume-surge"
        tr_id = "HHDFS76270000"
        params = {"KEYB": "", "AUTH": "", "EXCD": excd, "MIXN": "0", "VOL_RANG": "2"}
        tickers = []
        try:
            res = self.kis.request("GET", path, tr_id=tr_id, params=params)
            if isinstance(res, dict) and str(res.get("rt_cd", "")) == "0":
                for item in (res.get("output") or []):
                    if item.get("symb"): tickers.append(item.get("symb"))
        except Exception: pass
        return tickers

    # 🚨 [실전 API] 가격 급등 종목 탐색
    def get_price_fluct_tickers(self, excd: str = "NAS") -> list:
        if self.kis.cfg.paper: return []
        path = "/uapi/overseas-stock/v1/ranking/price-fluct"
        tr_id = "HHDFS76260000"
        params = {"KEYB": "", "AUTH": "", "EXCD": excd, "GUBN": "1", "MIXN": "0", "VOL_RANG": "2"}
        tickers = []
        try:
            res = self.kis.request("GET", path, tr_id=tr_id, params=params)
            if isinstance(res, dict) and str(res.get("rt_cd", "")) == "0":
                for item in (res.get("output") or []):
                    if item.get("symb"): tickers.append(item.get("symb"))
        except Exception: pass
        return tickers