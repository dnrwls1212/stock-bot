# src/trading/order_store.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


STATUS_SUBMITTED = "SUBMITTED"
STATUS_PARTIAL = "PARTIAL"
STATUS_FILLED = "FILLED"
STATUS_CANCELED = "CANCELED"
STATUS_REJECTED = "REJECTED"
STATUS_EXPIRED = "EXPIRED"

OPEN_STATUSES = {STATUS_SUBMITTED, STATUS_PARTIAL}
CLOSED_STATUSES = {STATUS_FILLED, STATUS_CANCELED, STATUS_REJECTED, STATUS_EXPIRED}


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_from_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


@dataclass
class PendingOrder:
    ticker: str
    side: str              # BUY / SELL
    qty: float

    status: str = STATUS_SUBMITTED
    submitted_at: str = ""

    # 네가 이전에 "kis_order_no"를 쓰던 흐름을 지원하기 위해 명시적으로 둠
    kis_order_no: Optional[str] = None

    # 클라이언트 내부용 ID
    client_order_id: Optional[str] = None

    raw: Optional[Dict[str, Any]] = None

    filled_qty: float = 0.0
    avg_fill_price: float = 0.0


class PendingOrderStore:
    """
    OrderManager가 호출하는 최소 API:
      - all()
      - upsert()
      - prune_filled()
      - purge_stale_open()
      - new_client_order_id(ticker, side)
      - has_open_order(ticker)
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._write({"orders": [], "_seq": 0})

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                return {"orders": [], "_seq": 0}
            obj.setdefault("orders", [])
            obj.setdefault("_seq", 0)
            if not isinstance(obj["orders"], list):
                obj["orders"] = []
            return obj
        except Exception:
            return {"orders": [], "_seq": 0}

    def _write(self, obj: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def all(self) -> List[Dict[str, Any]]:
        return list(self._read().get("orders", []))

    def list_orders(self) -> List[PendingOrder]:
        out: List[PendingOrder] = []
        for d in self.all():
            if not isinstance(d, dict):
                continue
            # unknown key 방어
            allowed = set(PendingOrder.__dataclass_fields__.keys())
            dd = {k: v for k, v in d.items() if k in allowed}
            try:
                out.append(PendingOrder(**dd))
            except Exception:
                continue
        return out

    def _make_key(self, d: Dict[str, Any]) -> str:
        # 1) kis_order_no > 2) client_order_id > 3) ticker/side/submitted_at
        if d.get("kis_order_no"):
            return str(d["kis_order_no"])
        if d.get("client_order_id"):
            return str(d["client_order_id"])
        return f"{d.get('ticker')}:{d.get('side')}:{d.get('submitted_at')}"

    def upsert(self, order: PendingOrder) -> None:
        data = self._read()
        orders = list(data.get("orders", []))

        od = asdict(order)
        key = self._make_key(od)

        replaced = False
        for i, o in enumerate(orders):
            if isinstance(o, dict) and self._make_key(o) == key:
                orders[i] = od
                replaced = True
                break

        if not replaced:
            orders.append(od)

        data["orders"] = orders
        self._write(data)

    def new_client_order_id(self, ticker: str = "UNK", side: str = "UNK") -> str:
        t = (ticker or "UNK").upper().strip()
        s = (side or "UNK").upper().strip()

        data = self._read()
        seq = int(data.get("_seq", 0)) + 1
        data["_seq"] = seq
        self._write(data)

        ts_ms = int(time.time() * 1000)
        pid = os.getpid()
        return f"{ts_ms}-{pid}-{seq}-{t}-{s}"

    def has_open_order(self, ticker: str) -> bool:
        t = (ticker or "").upper().strip()
        for o in self.list_orders():
            if o.ticker.upper().strip() == t and str(o.status).upper() in OPEN_STATUSES:
                return True
        return False

    def prune_filled(self, *, keep_last: int = 2000, max_age_sec: Optional[int] = None) -> int:
        """
        CLOSED 주문을 오래된 것부터 정리.
        - keep_last: 최신 N개는 무조건 유지
        - max_age_sec: CLOSED 중 age가 이 값 이상이면 삭제
        """
        keep_last = int(keep_last)
        max_age_sec = int(max_age_sec) if max_age_sec is not None else None

        data = self._read()
        orders = list(data.get("orders", []))
        if not orders:
            return 0

        if keep_last > 0 and len(orders) <= keep_last:
            return 0

        now_ts = datetime.now(timezone.utc).timestamp()

        head = orders[:-keep_last] if keep_last > 0 else orders
        tail = orders[-keep_last:] if keep_last > 0 else []

        kept: List[Dict[str, Any]] = []
        removed = 0

        for o in head:
            if not isinstance(o, dict):
                removed += 1
                continue
            st = str(o.get("status", "")).upper()
            if st in OPEN_STATUSES:
                kept.append(o)
                continue
            if st not in CLOSED_STATUSES:
                kept.append(o)
                continue

            if max_age_sec is None:
                removed += 1
                continue

            dt = _safe_from_iso(str(o.get("submitted_at", "")))
            if dt is None:
                kept.append(o)
                continue

            age = now_ts - dt.timestamp()
            if age >= max_age_sec:
                removed += 1
            else:
                kept.append(o)

        data["orders"] = kept + tail
        self._write(data)
        return removed

    def purge_stale_open(self, *, max_age_sec: int = 7200) -> int:
        """
        시장 종료 등으로 "OPEN 상태"가 너무 오래 남아있는 경우 삭제/종료 처리.
        """
        max_age_sec = int(max_age_sec)
        data = self._read()
        orders = list(data.get("orders", []))
        if not orders:
            return 0

        now_ts = datetime.now(timezone.utc).timestamp()
        kept: List[Dict[str, Any]] = []
        removed = 0

        for o in orders:
            if not isinstance(o, dict):
                removed += 1
                continue
            st = str(o.get("status", "")).upper()
            if st not in OPEN_STATUSES:
                kept.append(o)
                continue

            dt = _safe_from_iso(str(o.get("submitted_at", "")))
            if dt is None:
                kept.append(o)
                continue

            age = now_ts - dt.timestamp()
            if age >= max_age_sec:
                removed += 1
            else:
                kept.append(o)

        data["orders"] = kept
        self._write(data)
        return removed

    def create_submitted(
        self,
        *,
        ticker: str,
        side: str,
        qty: float,
        kis_order_no: Optional[str],
        client_order_id: Optional[str],
        raw: Optional[Dict[str, Any]],
    ) -> PendingOrder:
        po = PendingOrder(
            ticker=(ticker or "").upper().strip(),
            side=(side or "").upper().strip(),
            qty=float(qty),
            status=STATUS_SUBMITTED,
            submitted_at=_now_iso_utc(),
            kis_order_no=kis_order_no,
            client_order_id=client_order_id,
            raw=raw or {},
        )
        self.upsert(po)
        return po