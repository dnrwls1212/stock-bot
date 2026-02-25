from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Optional, Any


DEFAULT_PATH = "data/positions.json"


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


@dataclass
class PositionState:
    ticker: str
    qty: float = 0.0
    avg_price: float = 0.0

    trades_today: int = 0
    last_trade_ts: Optional[str] = None
    cooldown_until_ts: Optional[str] = None

    # ✅ confirmation을 위한 연속 신호 카운터
    buy_streak: int = 0
    sell_streak: int = 0
    last_action_raw: str = "HOLD"

    # 날짜 바뀌면 trades_today / streak 리셋용
    last_reset_date: Optional[str] = None  # "YYYY-MM-DD"

    # ✅ strategy/position_manager 내부 상태 저장용 (TP/SL/Trail flags 등)
    meta: Dict[str, Any] = field(default_factory=dict)

    def reset_if_new_day(self, now: datetime) -> None:
        d = now.date().isoformat()
        if self.last_reset_date != d:
            self.trades_today = 0
            self.buy_streak = 0
            self.sell_streak = 0
            self.last_action_raw = "HOLD"
            self.last_reset_date = d

            # ✅ 하루 단위로 리셋하고 싶은 meta만 정리
            # (TP/SL flags, peak 등은 하루 지나면 의미가 약해져서 리셋 권장)
            for k in ("_tp1_done", "_tp2_done", "_sl1_done", "_peak_price"):
                if k in self.meta:
                    del self.meta[k]

    def update_streak(self, raw_action: str) -> None:
        a = (raw_action or "HOLD").upper()
        self.last_action_raw = a

        if a == "BUY":
            self.buy_streak += 1
            self.sell_streak = 0
        elif a == "SELL":
            self.sell_streak += 1
            self.buy_streak = 0
        else:
            self.buy_streak = 0
            self.sell_streak = 0


def load_state(path: str = DEFAULT_PATH) -> Dict[str, PositionState]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    out: Dict[str, PositionState] = {}
    if isinstance(data, dict):
        for t, v in data.items():
            if not isinstance(v, dict):
                continue
            meta = v.get("meta")
            if not isinstance(meta, dict):
                meta = {}

            ps = PositionState(
                ticker=str(t).upper(),
                qty=float(v.get("qty", 0.0) or 0.0),
                avg_price=float(v.get("avg_price", 0.0) or 0.0),
                trades_today=int(v.get("trades_today", 0) or 0),
                last_trade_ts=v.get("last_trade_ts"),
                cooldown_until_ts=v.get("cooldown_until_ts"),
                buy_streak=int(v.get("buy_streak", 0) or 0),
                sell_streak=int(v.get("sell_streak", 0) or 0),
                last_action_raw=str(v.get("last_action_raw", "HOLD") or "HOLD").upper(),
                last_reset_date=v.get("last_reset_date"),
                meta=meta,
            )
            out[ps.ticker] = ps
    return out


def save_state(state: Dict[str, PositionState], path: str = DEFAULT_PATH) -> None:
    _ensure_dir(path)
    payload = {t: asdict(ps) for t, ps in state.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_position(state: Dict[str, PositionState], ticker: str) -> PositionState:
    t = str(ticker).upper()
    if t not in state:
        state[t] = PositionState(ticker=t)
    return state[t]