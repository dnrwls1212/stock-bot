from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .order_store import (
    PendingOrderStore,
    PendingOrder,
    STATUS_SUBMITTED,
    STATUS_PARTIAL,
    STATUS_FILLED,
)
from .position_store import save_state, PositionState, get_position


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip() in ("1", "true", "True", "YES", "yes", "y")


def _to_float(x: Any) -> float:
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return 0.0


class OrderManager:
    """
    역할:
    - 주문 제출 시 pending_store에 등록
    - 주기적으로 KIS 조회로 상태 동기화
      - 실전: inquire_unfilled + inquire_ccnl 가능
      - 모의: inquire_unfilled 미지원 → inquire_ccnl 기반으로만 반영(환경에 따라 ccnl도 빈 값일 수 있음)
    - 체결되면 positions.json 업데이트
    """

    def __init__(
        self,
        *,
        store: PendingOrderStore,
        position_state: Dict[str, PositionState],
        position_state_path: str,
        ccnl_lookback_minutes: int = 15,
        stale_order_seconds: int = 180,
    ) -> None:
        self.store = store
        self.positions = position_state
        self.positions_path = position_state_path
        self.ccnl_lookback_minutes = int(ccnl_lookback_minutes)
        self.stale_order_seconds = int(stale_order_seconds)

        # 모의 present-balance가 500 나는 케이스가 있어 기본 OFF
        self.pos_sync_paper_enabled = _env_bool("POS_SYNC_PAPER_ENABLED", False)

    # -----------------------
    # 주문 등록
    # -----------------------
    def register_submitted(
        self,
        *,
        ticker: str,
        side: str,
        qty: float,
        kis_response: Dict[str, Any],
        fill_price: Optional[float] = None,
    ) -> PendingOrder:
        """
        주문 성공 직후 pending에 등록.
        (옵션) PAPER_ASSUME_IMMEDIATE_FILL=1이면 모의에서는 즉시 FILLED 처리 + positions 반영.
        """
        order_no = None
        if isinstance(kis_response, dict):
            out = kis_response.get("output") or kis_response.get("OUTPUT") or {}
            if isinstance(out, dict):
                order_no = out.get("ODNO") or out.get("ORD_NO")
            order_no = order_no or kis_response.get("ODNO") or kis_response.get("ORD_NO")

        client_order_id = self.store.new_client_order_id(ticker, side)
        po = self.store.create_submitted(
            ticker=ticker,
            side=side,
            qty=qty,
            kis_order_no=str(order_no) if order_no else None,
            client_order_id=client_order_id,
            raw=kis_response,
        )

        # 모의에서는 체결조회가 불안정할 수 있어 "즉시체결 가정" 옵션 제공
        assume_fill = _env_bool("PAPER_ASSUME_IMMEDIATE_FILL", False)
        if not assume_fill:
            return po

        # ---- 즉시 FILLED 처리 ----
        po.status = STATUS_FILLED
        po.filled_qty = float(qty)
        if fill_price is not None and float(fill_price) > 0:
            po.avg_fill_price = float(fill_price)
        self.store.upsert(po)

        # positions 업데이트
        t = str(ticker).upper().strip()
        s = str(side).upper().strip()
        px = float(fill_price) if (fill_price is not None and float(fill_price) > 0) else 0.0

        ps = get_position(self.positions, t)
        cur_qty = float(ps.qty)
        cur_avg = float(ps.avg_price or 0.0)

        if s == "BUY":
            new_qty = cur_qty + float(qty)
            if new_qty > 0 and px > 0:
                new_avg = ((cur_qty * cur_avg) + (float(qty) * px)) / new_qty
            else:
                new_avg = cur_avg
            ps.qty = float(new_qty)
            ps.avg_price = float(new_avg) if ps.qty > 0 else 0.0

        elif s == "SELL":
            new_qty = max(0.0, cur_qty - float(qty))
            ps.qty = float(new_qty)
            if ps.qty <= 0:
                ps.avg_price = 0.0

        try:
            save_state(self.positions, self.positions_path)
        except Exception as e:
            print(f"[ORDER_SYNC] positions save failed (immediate fill): {e!r}")

        return po

    # -----------------------
    # broker truth sync (present-balance)
    # -----------------------
    def sync_positions_from_broker(self, broker: Any) -> None:
        is_paper = bool(getattr(getattr(broker, "kis", None), "cfg", None) and getattr(broker.kis.cfg, "paper", False))
        if is_paper and not self.pos_sync_paper_enabled:
            return

        try:
            pb = broker.inquire_present_balance()
        except Exception as e:
            print(f"[POS_SYNC] inquire_present_balance skipped: {e!r}")
            return

        items = None
        if isinstance(pb, dict):
            # ✅ 방어선 1: API 통신 에러 시 동기화 스킵
            rt_cd = pb.get("rt_cd") or pb.get("RT_CD")
            if rt_cd is not None and str(rt_cd) != "0":
                print(f"[POS_SYNC] API error (rt_cd={rt_cd}, msg={pb.get('msg1')}). skipping sync.")
                return

            items = (
                pb.get("output1") or pb.get("output2") or pb.get("output")
                or pb.get("OUTPUT1") or pb.get("OUTPUT2") or pb.get("OUTPUT")
            )
            
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return

        broker_pos: Dict[str, Dict[str, float]] = {}
        for it in items:
            if not isinstance(it, dict):
                continue

            ticker = (it.get("PDNO") or it.get("OVRS_PDNO") or it.get("SYMB") or it.get("ticker") or "")
            ticker = str(ticker).upper().strip()
            if not ticker:
                continue

            qty = _to_float(it.get("HLDG_QTY") or it.get("OVRS_HLDG_QTY") or it.get("BAL_QTY") or it.get("qty") or 0.0)
            avg = _to_float(it.get("PCHS_AVG_PRIC") or it.get("PUR_AVG_PRIC") or it.get("AVG_PRIC") or it.get("avg_price") or 0.0)

            if qty <= 0:
                continue

            broker_pos[ticker] = {"qty": float(qty), "avg_price": float(avg)}

        # ✅ 방어선 2: 모의투자 서버 불안정으로 빈 잔고 리스트를 응답받았을 때, 내 주식 날리지 않기
        if is_paper and len(items) == 0:
            print("[POS_SYNC] Warning: API returned empty balance list. Skipping zero-out to prevent data loss.")
            return

        changed = False

        for t, v in broker_pos.items():
            ps = get_position(self.positions, t)
            new_qty = float(v["qty"])
            new_avg = float(v["avg_price"]) if v["avg_price"] > 0 else float(ps.avg_price or 0.0)

            if abs(float(ps.qty) - new_qty) > 1e-9:
                ps.qty = new_qty
                changed = True
            if new_qty > 0 and new_avg > 0 and abs(float(ps.avg_price or 0.0) - new_avg) > 1e-9:
                ps.avg_price = new_avg
                changed = True

        for t, ps in list(self.positions.items()):
            t_u = str(t).upper()
            if t_u not in broker_pos and float(ps.qty) > 0:
                ps.qty = 0.0
                ps.avg_price = 0.0
                changed = True

        if changed:
            try:
                save_state(self.positions, self.positions_path)
                print(f"[POS_SYNC] synced positions from broker. held={len(broker_pos)}")
            except Exception as e:
                print(f"[POS_SYNC] positions save failed: {e!r}")

    # -----------------------
    # 주문/체결 동기화
    # -----------------------
    def sync_once(self, broker: Any) -> None:
        is_paper = bool(getattr(getattr(broker, "kis", None), "cfg", None) and getattr(broker.kis.cfg, "paper", False))

        # (A) 미체결: 실전만
        if not is_paper:
            try:
                unfilled = broker.inquire_unfilled()
            except Exception as e:
                print(f"[ORDER_SYNC] inquire_unfilled skipped: {e!r}")
                unfilled = None

            if isinstance(unfilled, dict):
                items = unfilled.get("output") or unfilled.get("output1") or unfilled.get("OUTPUT") or []
                if isinstance(items, dict):
                    items = [items]
                if isinstance(items, list):
                    self._apply_unfilled(items)

        # (B) 체결: 모의/실전
        try:
            ccnl = broker.inquire_ccnl(lookback_days=1)
        except Exception as e:
            print(f"[ORDER_SYNC] inquire_ccnl skipped: {e!r}")
            ccnl = None

        if isinstance(ccnl, dict):
            items = ccnl.get("output") or ccnl.get("output1") or ccnl.get("OUTPUT") or []
            if isinstance(items, dict):
                items = [items]
            if isinstance(items, list):
                self._apply_ccnl(items)

        # (C) STALE OPEN cleanup
        removed = self.store.purge_stale_open(max_age_sec=max(600, self.stale_order_seconds * 10))
        if removed > 0:
            print(f"[ORDER_SYNC] purged {removed} stale open orders")

    def _apply_unfilled(self, items: list[dict]) -> None:
        orders = self.store.list_orders()
        open_map: Dict[str, PendingOrder] = {}
        for o in orders:
            if o.kis_order_no:
                open_map[str(o.kis_order_no)] = o

        for it in items:
            if not isinstance(it, dict):
                continue

            odno = it.get("ODNO") or it.get("ORD_NO") or it.get("ord_no")
            if not odno:
                continue
            odno = str(odno)

            po = open_map.get(odno)
            if not po:
                continue

            filled_qty = _to_float(it.get("CCLD_QTY") or it.get("ccld_qty") or 0.0)
            rmnd_qty = _to_float(it.get("RMND_QTY") or it.get("rmnd_qty") or 0.0)

            if filled_qty > 0 and rmnd_qty > 0:
                po.status = STATUS_PARTIAL
            else:
                po.status = STATUS_SUBMITTED

            po.filled_qty = max(float(po.filled_qty or 0.0), float(filled_qty))
            po.raw = it
            self.store.upsert(po)

    def _apply_ccnl(self, items: list[dict]) -> None:
        orders = self.store.list_orders()
        ord_map: Dict[str, PendingOrder] = {}
        for o in orders:
            if o.kis_order_no:
                ord_map[str(o.kis_order_no)] = o

        changed_positions = False

        for it in items:
            if not isinstance(it, dict):
                continue

            odno = it.get("ODNO") or it.get("ORD_NO") or it.get("ord_no")
            if odno:
                odno = str(odno)

            ticker = (it.get("PDNO") or it.get("SYMB") or it.get("ticker") or "").upper().strip()
            side = (it.get("SLL_BUY_DVSN_CD") or it.get("side") or "").upper().strip()

            if side in ("01", "BUY"):
                side = "BUY"
            elif side in ("02", "SELL"):
                side = "SELL"

            qty = _to_float(it.get("CCLD_QTY") or it.get("qty") or it.get("ORD_QTY") or 0.0)
            px = _to_float(it.get("CCLD_UNPR") or it.get("avg_prc") or it.get("price") or 0.0)

            if not ticker or qty <= 0:
                continue

            if odno and odno in ord_map:
                po = ord_map[odno]
                po.status = STATUS_FILLED
                po.filled_qty = max(float(po.filled_qty or 0.0), float(qty))
                if px > 0:
                    po.avg_fill_price = px
                po.raw = it
                self.store.upsert(po)

            ps = get_position(self.positions, ticker)
            cur_qty = float(ps.qty)
            cur_avg = float(ps.avg_price or 0.0)

            if side == "BUY":
                new_qty = cur_qty + qty
                new_avg = ((cur_qty * cur_avg) + (qty * px)) / new_qty if (new_qty > 0 and px > 0) else cur_avg
                ps.qty = float(new_qty)
                ps.avg_price = float(new_avg) if ps.qty > 0 else 0.0
                changed_positions = True

            elif side == "SELL":
                new_qty = max(0.0, cur_qty - qty)
                ps.qty = float(new_qty)
                if ps.qty <= 0:
                    ps.avg_price = 0.0
                changed_positions = True

        if changed_positions:
            try:
                save_state(self.positions, self.positions_path)
            except Exception as e:
                print(f"[ORDER_SYNC] positions save failed: {e!r}")