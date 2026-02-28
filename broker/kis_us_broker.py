from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from .kis_client import KisClient


@dataclass
class BrokerOrderResult:
    ok: bool
    order_no: Optional[str]
    raw: Dict[str, Any]


class KisUsBroker:
    def __init__(self, kis: KisClient) -> None:
        self.kis = kis

    # -------------------------
    # Orders
    # -------------------------
    def buy_market(self, symbol: str, qty: int, *, last_price: Optional[float] = None) -> BrokerOrderResult:
        return self._order("BUY", symbol, qty, last_price=last_price)

    def sell_market(self, symbol: str, qty: int, *, last_price: Optional[float] = None) -> BrokerOrderResult:
        return self._order("SELL", symbol, qty, last_price=last_price)

    # -------------------------
    # Sync APIs
    # -------------------------
    def inquire_ccnl(self, *, lookback_days: int = 1) -> Dict[str, Any]:
        days = max(1, int(lookback_days))
        today = datetime.now(ZoneInfo("Asia/Seoul")).date()

        tr_id = "VTTS3035R" if self.kis.cfg.paper else "TTTS3035R"
        all_items: list[dict] = []
        last_resp: Dict[str, Any] = {}

        for i in range(days):
            d = today - timedelta(days=i)
            ord_dt = d.strftime("%Y%m%d")

            params: Dict[str, Any] = {
                "CANO": self.kis.cfg.account_no,
                "ACNT_PRDT_CD": self.kis.cfg.account_prdt,
                "ORD_DT": ord_dt,
                "ORD_STRT_DT": ord_dt,
                "ORD_END_DT": ord_dt,
                "PDNO": "" if self.kis.cfg.paper else "%",
                "SLL_BUY_DVSN": "00",      
                "CCLD_NCCS_DVSN": "00",    
                "OVRS_EXCG_CD": "" if self.kis.cfg.paper else (self.kis.cfg.exchange or "NASD"),
                "SORT_SQN": "DS",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            }

            resp = self.kis.request("GET", "/uapi/overseas-stock/v1/trading/inquire-ccnl", tr_id=tr_id, params=params)
            last_resp = resp if isinstance(resp, dict) else {"raw": resp}
            self._dump_order({"kind": "ccnl_response", "tr_id": tr_id, "params": params, "response": last_resp})

            if isinstance(resp, dict) and str(resp.get("rt_cd", "")) == "0":
                items = resp.get("output1") or resp.get("output") or resp.get("OUTPUT1") or resp.get("OUTPUT") or []
                if isinstance(items, dict): items = [items]
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict): all_items.append(it)

        if isinstance(last_resp, dict):
            merged = dict(last_resp)
            merged["output1"] = all_items
            return merged

        return {"rt_cd": "0", "output1": all_items}

    def inquire_present_balance(self) -> Dict[str, Any]:
        tr_id = "VTTS3012R" if self.kis.cfg.paper else "TTTS3012R"
        excg = (os.environ.get("KIS_BAL_EXCG_CD") or self.kis.cfg.exchange or "NASD").strip()
        crcy = (os.environ.get("KIS_BAL_TR_CRCY_CD") or "USD").strip()

        params: Dict[str, Any] = {
            "CANO": self.kis.cfg.account_no,
            "ACNT_PRDT_CD": self.kis.cfg.account_prdt,
            "OVRS_EXCG_CD": excg,
            "TR_CRCY_CD": crcy,
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        resp = self.kis.request("GET", "/uapi/overseas-stock/v1/trading/inquire-balance", tr_id=tr_id, params=params)
        self._dump_order({"kind": "balance_response", "tr_id": tr_id, "params": params, "response": resp})
        return resp

    # 🚨 [추가] 실전 전용 미체결 조회
    def inquire_unfilled(self, excg_cd: str = "NASD") -> list:
        if self.kis.cfg.paper:
            return []

        tr_id = "TTTS3018R"
        params: Dict[str, Any] = {
            "CANO": self.kis.cfg.account_no,
            "ACNT_PRDT_CD": self.kis.cfg.account_prdt,
            "OVRS_EXCG_CD": excg_cd,
            "SORT_SQN": "DS",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        resp = self.kis.request("GET", "/uapi/overseas-stock/v1/trading/inquire-nccs", tr_id=tr_id, params=params)
        self._dump_order({"kind": "unfilled_response", "tr_id": tr_id, "params": params, "response": resp})
        
        if isinstance(resp, dict) and str(resp.get("rt_cd", "")) == "0":
            return resp.get("output") or []
        return []

    # -------------------------
    # Internals
    # -------------------------
    def _now_kst(self) -> str:
        return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")

    def _dump_order(self, payload: Dict[str, Any]) -> None:
        path = os.environ.get("KIS_ORDER_DUMP_PATH", "data/kis_order_raw.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _fmt_price(p: float) -> str:
        return f"{float(p):.4f}"

    # 🚨 [가장 중요] 실전 매도 TR_ID 반영
    def _resolve_tr_id(self, side: str) -> str:
        if self.kis.cfg.paper:
            return "VTTT1002U" if side == "BUY" else "VTTT1001U"
        return "TTTT1002U" if side == "BUY" else "TTTT1006U"

    def _resolve_order_division(self) -> str:
        return "00"

    def _extract_order_no(self, raw: Dict[str, Any]) -> Optional[str]:
        if not isinstance(raw, dict): return None
        if raw.get("ODNO") or raw.get("odno"): return raw.get("ODNO") or raw.get("odno")
        out = raw.get("output") or raw.get("OUTPUT")
        if isinstance(out, dict): return out.get("ODNO") or out.get("odno")
        return None

    def _order(self, side: str, symbol: str, qty: int, *, last_price: Optional[float] = None) -> BrokerOrderResult:
        sym = (symbol or "").upper().strip()
        q = int(qty)
        if not sym or q <= 0: return BrokerOrderResult(False, None, {"error": "invalid symbol/qty", "symbol": sym, "qty": q})

        execute_orders = bool(int(os.environ.get("EXECUTE_ORDERS", "0") or "0"))
        if not execute_orders:
            raw = {"dry_run": True, "side": side, "symbol": sym, "qty": q, "ts_kst": self._now_kst()}
            self._dump_order({"kind": "dry_run", "raw": raw})
            return BrokerOrderResult(True, "DRY_RUN", raw)

        tr_id = self._resolve_tr_id(side)
        ord_dvsn = self._resolve_order_division()

        if ord_dvsn == "00":
            if last_price is None: return BrokerOrderResult(False, None, {"error": "limit-order requires last_price"})
            ovrs_unpr = self._fmt_price(last_price)
        else:
            ovrs_unpr = "0"

        p: Dict[str, Any] = {
            "CANO": self.kis.cfg.account_no,
            "ACNT_PRDT_CD": self.kis.cfg.account_prdt,
            "OVRS_EXCG_CD": self.kis.cfg.exchange,
            "PDNO": sym,
            "ORD_QTY": str(q),
            "OVRS_ORD_UNPR": str(ovrs_unpr),
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": str(ord_dvsn),
        }

        meta: Dict[str, Any] = {
            "ts_kst": self._now_kst(), "paper": bool(self.kis.cfg.paper), "side": side,
            "symbol": sym, "qty": q, "last_price": last_price, "tr_id": tr_id,
            "exchange": self.kis.cfg.exchange, "execute_orders": True, "ord_dvsn": ord_dvsn,
        }

        try:
            raw = self.kis.request("POST", "/uapi/overseas-stock/v1/trading/order", tr_id=tr_id, data=p, need_hashkey=True)
            ok = str(raw.get("rt_cd", "")) == "0"
            order_no = self._extract_order_no(raw)
            self._dump_order({"kind": "order_response", "meta": meta, "request": p, "response": raw})

            if not ok: print(f"[KIS_ORDER_ERR] {side} {sym} qty={q} rt_cd={raw.get('rt_cd')} msg={raw.get('msg1')}")
            return BrokerOrderResult(ok, order_no, raw)
        except Exception as e:
            payload = {"kind": "order_exception", "meta": meta, "request": p, "error": repr(e)}
            try:
                r = getattr(e, "response", None)
                if r is not None: payload["response_text"] = r.text[:2000]
            except Exception: pass
            self._dump_order(payload)
            print(f"[KIS_ORDER_EXC] {side} {sym} qty={q} err={e!r}")
            return BrokerOrderResult(False, None, {"error": repr(e)})