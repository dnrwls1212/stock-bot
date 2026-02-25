# src/market/kis_us_provider.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.broker.kis_client import KisClient


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


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
    """
    KIS 해외주식 현재가(quote) 조회 전용 Provider

    NOTE:
      - KIS 문서/샘플에서 해외주식 시세 엔드포인트는 /uapi/overseas-price/... 형태를 흔히 사용.
      - TR_ID는 계정/환경(모의/실전)에 따라 다를 수 있어, 기본값은 env로 오버라이드 가능하게 설계.
      - 첫 단계는 "raw 응답을 남기고" 파싱을 맞추는게 안전함.
    """

    def __init__(self, kis: KisClient, tr_id_quote: Optional[str] = None) -> None:
        self.kis = kis

        # 환경변수로 교체 가능
        # 예: KIS_TRID_US_QUOTE=HHDFS00000300
        self.tr_id_quote = tr_id_quote or os.environ.get("KIS_TRID_US_QUOTE", "").strip() or "HHDFS00000300"

        # raw 응답 덤프 옵션
        self.debug_dump = os.environ.get("KIS_DEBUG_DUMP", "0").strip() == "1"
        self.debug_dump_path = os.environ.get("KIS_DEBUG_DUMP_PATH", "data/kis_quote_raw.jsonl").strip()

    def get_quote(self, ticker: str, exchange: Optional[str] = None) -> UsQuote:
        excg = exchange or self.kis.cfg.exchange

        # 해외주식 현재가(예시)
        path = "/uapi/overseas-price/v1/quotations/price"
        params = {
            "AUTH": "",
            "EXCD": excg,
            "SYMB": ticker,
        }

        j = self.kis.request("GET", path, tr_id=self.tr_id_quote, params=params)

        # ✅ raw 응답 저장(파싱/필드명/TR_ID 확인용)
        if self.debug_dump:
            _dump_jsonl(
                self.debug_dump_path,
                {
                    "ts_utc": _utcnow_iso(),
                    "ticker": ticker,
                    "exchange": excg,
                    "tr_id": self.tr_id_quote,
                    "path": path,
                    "params": params,
                    "response": j,
                },
            )

        # 응답 포맷은 계정/문서에 따라 output 키가 다를 수 있어 최대한 방어적으로 파싱
        out = j.get("output") or j.get("output1") or j.get("output2") or j

        # 흔히 현재가는 "last" 류 필드 / "ovrs_prpr" 류 필드 등으로 옴
        candidates = [
            out.get("last"),
            out.get("last_price"),
            out.get("ovrs_prpr"),
            out.get("stck_prpr"),
            out.get("prpr"),
            out.get("tdd_clpr"),
        ]

        price: Optional[float] = None
        for c in candidates:
            try:
                if c is None:
                    continue
                price = float(c)
                break
            except Exception:
                continue

        if price is None:
            raise RuntimeError(f"cannot parse quote price: {j}")

        return UsQuote(ticker=ticker, price=price, raw=j)
