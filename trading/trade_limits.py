# src/trading/trade_limits.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

from zoneinfo import ZoneInfo


@dataclass
class LimitState:
    last_order_ts: str = ""
    day: str = ""
    day_count: int = 0


class TradeLimitStore:
    """
    티커별:
    - 쿨다운(초)
    - 일일 주문 횟수 제한
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._write({})

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _write(self, d: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def allow(
        self,
        *,
        ticker: str,
        now_kst: datetime,
        cooldown_seconds: int,
        max_orders_per_ticker_per_day: int,
    ) -> Tuple[bool, str]:
        t = (ticker or "").upper().strip()
        cooldown_seconds = int(cooldown_seconds)
        max_orders_per_ticker_per_day = int(max_orders_per_ticker_per_day)

        d = self._read()
        st = d.get(t) or {}

        day = now_kst.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
        last_ts = st.get("last_order_ts") or ""
        st_day = st.get("day") or ""
        day_count = int(st.get("day_count") or 0)

        # day rollover
        if st_day != day:
            day_count = 0
            st_day = day

        # cooldown
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts)
                delta = (now_kst - last_dt).total_seconds()
                if delta < cooldown_seconds:
                    return False, f"cooldown {int(delta)}s < {cooldown_seconds}s"
            except Exception:
                pass

        if max_orders_per_ticker_per_day > 0 and day_count >= max_orders_per_ticker_per_day:
            return False, f"day_count {day_count} >= {max_orders_per_ticker_per_day}"

        return True, "ok"

    def update_on_order(self, ticker: str, now_kst: datetime) -> None:
        t = (ticker or "").upper().strip()
        d = self._read()
        st = d.get(t) or {}

        day = now_kst.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
        if st.get("day") != day:
            st["day"] = day
            st["day_count"] = 0

        st["day_count"] = int(st.get("day_count") or 0) + 1
        st["last_order_ts"] = now_kst.isoformat(timespec="seconds")

        d[t] = st
        self._write(d)