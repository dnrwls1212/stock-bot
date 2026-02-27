# src/trading/risk.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any, Dict, Optional, Tuple

from .position_store import PositionState


# -------------------------
# 기존 기능 (그대로 유지)
# -------------------------
def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def can_trade(
    *,
    pos: PositionState,
    now: datetime,
    action: str,
    cooldown_minutes: int,
    max_trades_per_day: int,
    max_position_qty: float | None = None,
) -> Tuple[bool, str]:
    a = (action or "HOLD").upper()
    if a not in ("BUY", "SELL"):
        return True, "HOLD"

    # 기본 금지 규칙
    # 🚨 [추가/수정] 매도(SELL)는 횟수/쿨타임 제한을 무시하고 무조건 허용합니다!
    if a == "SELL":
        if float(pos.qty) <= 0:
            return False, "no position to sell"
        return True, "ok"
    if a == "BUY" and max_position_qty is not None and float(pos.qty) >= float(max_position_qty):
        return False, f"at max position (qty={pos.qty:.2f} / max={float(max_position_qty):.2f})"

    # 일일 거래 제한
    if int(pos.trades_today) >= int(max_trades_per_day):
        return False, f"max_trades_per_day reached ({pos.trades_today}/{max_trades_per_day})"

    # 쿨다운
    cd = parse_iso(pos.cooldown_until_ts)
    if cd and now < cd:
        return False, f"cooldown active until {pos.cooldown_until_ts}"

    return True, "ok"


def apply_trade_update(
    *,
    pos: PositionState,
    action: str,
    fill_qty: float,
    fill_price: float,
    now: datetime,
    cooldown_minutes: int,
) -> None:
    a = (action or "HOLD").upper()
    q = float(fill_qty)
    if q <= 0:
        return

    if a == "BUY":
        new_qty = pos.qty + q
        if new_qty > 0:
            pos.avg_price = (pos.avg_price * pos.qty + float(fill_price) * q) / new_qty
        pos.qty = new_qty

    elif a == "SELL":
        pos.qty = max(0.0, pos.qty - q)
        if pos.qty == 0:
            pos.avg_price = 0.0
    else:
        return

    pos.trades_today += 1
    pos.last_trade_ts = now.isoformat(timespec="seconds")
    pos.cooldown_until_ts = (now + timedelta(minutes=int(cooldown_minutes))).isoformat(timespec="seconds")


# -------------------------
# P0-2: 계좌 단위 리스크
# -------------------------
def _to_float(x: Any) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return 0.0


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip() in ("1", "true", "True", "YES", "yes", "y")


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)).strip())
    except Exception:
        return default


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    return (v.strip() if isinstance(v, str) and v.strip() else default)


@dataclass
class AccountSnapshot:
    ts_kst: str
    equity: float                      # 계좌 평가액(가능하면)
    cash: float                        # 현금(가능하면)
    gross_exposure: float              # 보유 평가금액 합(가능하면)
    pos_notional_by_ticker: Dict[str, float]


@dataclass
class DailyRiskState:
    day_kst: str
    start_equity: float
    last_equity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day_kst": self.day_kst,
            "start_equity": self.start_equity,
            "last_equity": self.last_equity,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DailyRiskState":
        return DailyRiskState(
            day_kst=str(d.get("day_kst") or ""),
            start_equity=_to_float(d.get("start_equity")),
            last_equity=_to_float(d.get("last_equity")),
        )


class AccountRiskManager:
    """
    - broker.inquire_present_balance() 결과를 이용해
      equity/cash/보유평가금액/티커별 비중을 계산
    - 하루 손실 한도 / 주문 금액 한도 / 비중 한도 체크
    """

    def __init__(self) -> None:
        self.enabled = _env_bool("ACCOUNT_RISK_ENABLED", True)

        self.max_daily_loss_pct = abs(_env_float("MAX_DAILY_LOSS_PCT", 0.0))
        self.max_daily_loss_abs = abs(_env_float("MAX_DAILY_LOSS_ABS", 0.0))

        self.max_order_notional_pct = abs(_env_float("MAX_ORDER_NOTIONAL_PCT", 0.0))
        self.max_order_notional_abs = abs(_env_float("MAX_ORDER_NOTIONAL_ABS", 0.0))

        self.max_position_notional_pct = abs(_env_float("MAX_POSITION_NOTIONAL_PCT", 0.0))
        self.max_gross_exposure_pct = abs(_env_float("MAX_GROSS_EXPOSURE_PCT", 0.0))

        self.state_path = _env_str("RISK_STATE_PATH", "data/risk_state.json")

        self._snapshot: Optional[AccountSnapshot] = None
        self._state: Optional[DailyRiskState] = None

    @staticmethod
    def from_env() -> "AccountRiskManager":
        return AccountRiskManager()

    # --------- state load/save ----------
    def _load_state(self) -> Optional[DailyRiskState]:
        try:
            if not os.path.exists(self.state_path):
                return None
            with open(self.state_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                return None
            return DailyRiskState.from_dict(d)
        except Exception:
            return None

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            if not self._state:
                return
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._state.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.state_path)
        except Exception:
            pass

    # --------- parsing present balance ----------
    def _parse_present_balance(self, pb: Dict[str, Any], ts_kst: str) -> AccountSnapshot:
        """
        KIS present-balance 응답은 환경/버전에 따라 키가 다를 수 있어 방어적으로 파싱.
        목표:
        - equity(가능하면)
        - cash(가능하면)
        - 보유 평가금액(gross exposure)
        - 티커별 보유 평가금액(가능하면)
        """
        # positions list
        items = pb.get("output1") or pb.get("output") or pb.get("OUTPUT1") or pb.get("OUTPUT") or []
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            items = []

        # summary dict
        summ = pb.get("output2") or pb.get("OUTPUT2") or {}
        if isinstance(summ, list) and summ:
            summ = summ[0]
        if not isinstance(summ, dict):
            summ = {}

        # equity candidates (없으면 gross_exposure + cash로 근사)
        equity = 0.0
        for k in (
            "tot_evlu_amt", "TOT_EVLU_AMT",
            "tot_asst_amt", "TOT_ASST_AMT",
            "tot_evlu_amt2", "TOT_EVLU_AMT2",
            "evlu_amt", "EVLU_AMT",
            "tot_blnc", "TOT_BLNC",
        ):
            if k in summ:
                equity = _to_float(summ.get(k))
                if equity > 0:
                    break

        cash = 0.0
        for k in (
            "dnca_tot_amt", "DNCA_TOT_AMT",
            "frcr_dnca_bal", "FRCR_DNCA_BAL",
            "cash", "CASH",
            "prvs_rcdl_excc_amt", "PRVS_RCDL_EXCC_AMT",
        ):
            if k in summ:
                cash = _to_float(summ.get(k))
                if cash != 0:
                    break

        pos_notional: Dict[str, float] = {}
        gross_exposure = 0.0

        for it in items:
            if not isinstance(it, dict):
                continue
            ticker = (it.get("PDNO") or it.get("OVRS_PDNO") or it.get("SYMB") or it.get("ticker") or "").upper().strip()
            if not ticker:
                continue

            # 보유 평가금액 후보
            n = 0.0
            for k in (
                "ovrs_stck_evlu_amt", "OVRS_STCK_EVLU_AMT",
                "evlu_amt", "EVLU_AMT",
                "stck_evlu_amt", "STCK_EVLU_AMT",
                "tot_evlu_amt", "TOT_EVLU_AMT",
            ):
                if k in it:
                    n = _to_float(it.get(k))
                    if n != 0:
                        break

            if n == 0.0:
                # fallback: qty * avg
                qty = _to_float(it.get("HLDG_QTY") or it.get("OVRS_HLDG_QTY") or it.get("qty") or 0.0)
                avg = _to_float(it.get("PCHS_AVG_PRIC") or it.get("PUR_AVG_PRIC") or it.get("avg_price") or 0.0)
                n = float(qty) * float(avg)

            if n <= 0:
                continue

            pos_notional[ticker] = float(n)
            gross_exposure += float(n)

        if equity <= 0:
            # 근사: equity ≈ cash + gross
            equity = max(0.0, cash + gross_exposure)

        return AccountSnapshot(
            ts_kst=ts_kst,
            equity=float(equity),
            cash=float(cash),
            gross_exposure=float(gross_exposure),
            pos_notional_by_ticker=pos_notional,
        )

    # --------- public APIs ----------
    def refresh(self, broker: Any, now_kst: datetime) -> Optional[AccountSnapshot]:
        """
        tick 시작 시 1회 호출 추천:
        - present balance 조회
        - daily state 업데이트(하루 시작 equity 저장)
        """
        if not self.enabled or broker is None:
            self._snapshot = None
            return None

        ts = now_kst.isoformat(timespec="seconds")
        try:
            pb = broker.inquire_present_balance()
        except Exception:
            self._snapshot = None
            return None

        if not isinstance(pb, dict):
            self._snapshot = None
            return None

        snap = self._parse_present_balance(pb, ts)
        self._snapshot = snap

        # daily state 업데이트
        day = str(now_kst.date().isoformat())
        st = self._state or self._load_state()
        if (st is None) or (st.day_kst != day) or (st.start_equity <= 0):
            self._state = DailyRiskState(day_kst=day, start_equity=float(snap.equity), last_equity=float(snap.equity))
            self._save_state()
        else:
            st.last_equity = float(snap.equity)
            self._state = st
            self._save_state()

        return snap

    def daily_ok(self) -> Tuple[bool, str]:
        """
        하루 손실 한도 체크
        """
        if not self.enabled:
            return True, "disabled"
        if not self._snapshot or not self._state:
            return True, "no_snapshot"

        start_eq = float(self._state.start_equity)
        cur_eq = float(self._snapshot.equity)
        if start_eq <= 0:
            return True, "no_start_equity"

        pnl_abs = cur_eq - start_eq
        pnl_pct = pnl_abs / start_eq

        if self.max_daily_loss_pct > 0 and pnl_pct <= -self.max_daily_loss_pct:
            return False, f"daily_loss_pct_block pnl={pnl_pct:.3f} (start={start_eq:.2f} cur={cur_eq:.2f})"

        if self.max_daily_loss_abs > 0 and pnl_abs <= -self.max_daily_loss_abs:
            return False, f"daily_loss_abs_block pnl={pnl_abs:.2f} (start={start_eq:.2f} cur={cur_eq:.2f})"

        return True, f"daily_ok pnl={pnl_pct:.3f}"

    def allow_order(
        self,
        *,
        ticker: str,
        action: str,
        qty: int,
        price: float,
    ) -> Tuple[bool, str]:
        """
        주문 직전에 호출:
        - 하루 손실 한도
        - 1회 주문 금액 한도
        - 종목당 비중 한도 (주문 후)
        - 총 투자비중 한도 (주문 후)
        """
        a = (action or "HOLD").upper().strip()
        if not self.enabled or a not in ("BUY", "SELL"):
            return True, "ok"

        if qty <= 0 or price <= 0:
            return False, "bad_qty_or_price"

        # 🚨 [추가] 계좌 손실 한도에 도달했더라도 매도(손절/익절)는 탈출을 위해 무조건 허용!
        if a == "SELL":
            return True, "sell_always_allowed"

        if not self._snapshot:
            return True, "no_snapshot_skip"  # 스냅샷 없으면 안전하게 스킵이 아니라 "패스" (원하면 block으로 바꿔도 됨)

        # (1) daily loss
        ok_day, why_day = self.daily_ok()
        if not ok_day:
            return False, why_day

        eq = float(self._snapshot.equity) or 0.0
        if eq <= 0:
            return True, "no_equity_skip"

        order_notional = float(qty) * float(price)

        # (2) order notional limit
        if self.max_order_notional_pct > 0:
            lim = eq * self.max_order_notional_pct
            if order_notional > lim:
                return False, f"order_notional_pct_block {order_notional:.2f} > {lim:.2f} ({self.max_order_notional_pct:.2f}*equity)"

        if self.max_order_notional_abs > 0 and order_notional > self.max_order_notional_abs:
            return False, f"order_notional_abs_block {order_notional:.2f} > {self.max_order_notional_abs:.2f}"

        # (3) position notional limit (post-trade)
        t = (ticker or "").upper().strip()
        cur_pos = float(self._snapshot.pos_notional_by_ticker.get(t, 0.0))
        post_pos = cur_pos

        if a == "BUY":
            post_pos = cur_pos + order_notional
        elif a == "SELL":
            post_pos = max(0.0, cur_pos - order_notional)

        if self.max_position_notional_pct > 0:
            lim = eq * self.max_position_notional_pct
            if post_pos > lim:
                return False, f"position_pct_block post={post_pos:.2f} > {lim:.2f} ({self.max_position_notional_pct:.2f}*equity)"

        # (4) gross exposure limit (post-trade)
        cur_gross = float(self._snapshot.gross_exposure)
        post_gross = cur_gross
        if a == "BUY":
            post_gross = cur_gross + order_notional
        elif a == "SELL":
            post_gross = max(0.0, cur_gross - order_notional)

        if self.max_gross_exposure_pct > 0:
            lim = eq * self.max_gross_exposure_pct
            if post_gross > lim:
                return False, f"gross_pct_block post={post_gross:.2f} > {lim:.2f} ({self.max_gross_exposure_pct:.2f}*equity)"

        return True, "account_risk_ok"