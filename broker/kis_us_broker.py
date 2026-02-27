# src/broker/kis_us_broker.py
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
    """
    KIS OpenAPI(REST) 기반 해외(미국) 주식 브로커.

    OrderManager가 기대하는 인터페이스:
      - buy_market / sell_market
      - inquire_ccnl (주문체결내역)  ✅ 모의/실전
      - inquire_present_balance (잔고) ✅ 모의/실전
      - inquire_unfilled (미체결)      ✅ 실전만 (모의는 미지원 케이스 많음)
    """

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
    # Sync APIs (OrderManager)
    # -------------------------
    def inquire_ccnl(self, *, lookback_days: int = 1) -> Dict[str, Any]:
        """
        해외주식 주문체결내역 조회(모의/실전).

        ✅ 네 kis_ccnl_raw.jsonl 에서 "ORD_DT is required"가 나온 걸로 봐서
        환경(특히 VTS)에서 ORD_DT 기반 조회를 요구하는 케이스가 있음.
        그래서 lookback_days 동안 날짜를 돌며 조회해서 합쳐서 반환.
        """
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

                # ✅ 핵심 수정: KIS API 에러 방지를 위해 세 가지 날짜 필드를 모두 전송
                "ORD_DT": ord_dt,
                "ORD_STRT_DT": ord_dt,
                "ORD_END_DT": ord_dt,

                # 전체조회
                # 모의는 PDNO/OVRS_EXCG_CD 공란만 허용되는 케이스가 있어 공란 처리
                "PDNO": "" if self.kis.cfg.paper else "%",
                "SLL_BUY_DVSN": "00",      # 00=전체
                "CCLD_NCCS_DVSN": "00",    # 00=전체(체결/미체결 포함)

                "OVRS_EXCG_CD": "" if self.kis.cfg.paper else (self.kis.cfg.exchange or "NASD"),
                "SORT_SQN": "DS",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            }

            resp = self.kis.request(
                "GET",
                "/uapi/overseas-stock/v1/trading/inquire-ccnl",
                tr_id=tr_id,
                params=params,
            )
            last_resp = resp if isinstance(resp, dict) else {"raw": resp}

            # ✅ 진단용 덤프 (반드시 실행되게 return 위로)
            self._dump_order({"kind": "ccnl_response", "tr_id": tr_id, "params": params, "response": last_resp})

            # 정상 응답이면 output/output1 등을 긁어서 누적
            if isinstance(resp, dict) and str(resp.get("rt_cd", "")) == "0":
                items = resp.get("output1") or resp.get("output") or resp.get("OUTPUT1") or resp.get("OUTPUT") or []
                if isinstance(items, dict):
                    items = [items]
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            all_items.append(it)

        # OrderManager가 처리하기 쉽게 output1에 합쳐서 반환
        if isinstance(last_resp, dict):
            merged = dict(last_resp)
            merged["output1"] = all_items
            return merged

        return {"rt_cd": "0", "output1": all_items}

    def inquire_present_balance(self) -> Dict[str, Any]:
        """해외주식 잔고 조회(모의/실전)."""
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

        resp = self.kis.request(
            "GET",
            "/uapi/overseas-stock/v1/trading/inquire-balance",
            tr_id=tr_id,
            params=params,
        )
        self._dump_order({"kind": "balance_response", "tr_id": tr_id, "params": params, "response": resp})
        return resp

    def inquire_unfilled(self) -> Dict[str, Any]:
        """해외주식 미체결내역(실전만)."""
        if self.kis.cfg.paper:
            raise NotImplementedError("KIS paper trading often does not support inquire-nccs (unfilled)")

        tr_id = "TTTS3018R"
        params: Dict[str, Any] = {
            "CANO": self.kis.cfg.account_no,
            "ACNT_PRDT_CD": self.kis.cfg.account_prdt,
            "OVRS_EXCG_CD": self.kis.cfg.exchange,
            "SORT_SQN": "DS",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }

        resp = self.kis.request(
            "GET",
            "/uapi/overseas-stock/v1/trading/inquire-nccs",
            tr_id=tr_id,
            params=params,
        )
        self._dump_order({"kind": "unfilled_response", "tr_id": tr_id, "params": params, "response": resp})
        return resp

    # -------------------------
    # Internals
    # -------------------------
    def _now_kst(self) -> str:
        return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")

    def _dump_order(self, payload: Dict[str, Any]) -> None:
        """주문/응답을 jsonl로 누적 저장(원인 분석용)."""
        path = os.environ.get("KIS_ORDER_DUMP_PATH", "data/kis_order_raw.jsonl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _fmt_price(p: float) -> str:
        return f"{float(p):.4f}"

    def _resolve_tr_id(self, side: str) -> str:
        if self.kis.cfg.paper:
            return "VTTT1002U" if side == "BUY" else "VTTT1001U"
        return "TTTT1002U" if side == "BUY" else "TTTT1006U"

    def _resolve_order_division(self) -> str:
        # 모의는 지정가(00) 강제 (VTS 제약 회피)
        if self.kis.cfg.paper:
            return "00"
        return (os.environ.get("KIS_US_ORD_DVSN", "01") or "01").strip()

    def _extract_order_no(self, raw: Dict[str, Any]) -> Optional[str]:
        if not isinstance(raw, dict):
            return None
        if raw.get("ODNO") or raw.get("odno"):
            return raw.get("ODNO") or raw.get("odno")
        out = raw.get("output") or raw.get("OUTPUT")
        if isinstance(out, dict):
            return out.get("ODNO") or out.get("odno")
        return None

    def _order(self, side: str, symbol: str, qty: int, *, last_price: Optional[float] = None) -> BrokerOrderResult:
        sym = (symbol or "").upper().strip()
        q = int(qty)
        if not sym or q <= 0:
            return BrokerOrderResult(False, None, {"error": "invalid symbol/qty", "symbol": sym, "qty": q})

        execute_orders = bool(int(os.environ.get("EXECUTE_ORDERS", "0") or "0"))
        if not execute_orders:
            raw = {"dry_run": True, "side": side, "symbol": sym, "qty": q, "ts_kst": self._now_kst()}
            self._dump_order({"kind": "dry_run", "raw": raw})
            return BrokerOrderResult(True, "DRY_RUN", raw)

        tr_id = self._resolve_tr_id(side)
        ord_dvsn = self._resolve_order_division()

        if ord_dvsn == "00":
            if last_price is None:
                raw = {"error": "limit-order requires last_price", "side": side, "symbol": sym, "qty": q}
                self._dump_order({"kind": "order_rejected_local", "raw": raw})
                return BrokerOrderResult(False, None, raw)
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
        if side == "SELL":
            p["SLL_TYPE"] = "00"

        meta: Dict[str, Any] = {
            "ts_kst": self._now_kst(),
            "paper": bool(self.kis.cfg.paper),
            "side": side,
            "symbol": sym,
            "qty": q,
            "last_price": last_price,
            "tr_id": tr_id,
            "exchange": self.kis.cfg.exchange,
            "execute_orders": True,
            "ord_dvsn": ord_dvsn,
        }

        try:
            raw = self.kis.request(
                method="POST",
                path="/uapi/overseas-stock/v1/trading/order",
                tr_id=tr_id,
                data=p,
                need_hashkey=True,
            )

            ok = str(raw.get("rt_cd", "")) == "0"
            order_no = self._extract_order_no(raw)

            self._dump_order({"kind": "order_response", "meta": meta, "request": p, "response": raw})

            if not ok:
                rt_cd = raw.get("rt_cd")
                msg = raw.get("msg1") or raw.get("rt_msg") or raw.get("message")
                print(f"[KIS_ORDER_ERR] {side} {sym} qty={q} rt_cd={rt_cd} msg={msg}")

            return BrokerOrderResult(ok, order_no, raw)

        except Exception as e:
            payload = {"kind": "order_exception", "meta": meta, "request": p, "error": repr(e)}
            self._dump_order(payload)

            resp_text = None
            try:
                r = getattr(e, "response", None)
                if r is not None:
                    resp_text = r.text
            except Exception:
                resp_text = None

            if resp_text:
                payload["response_text"] = resp_text[:2000]
                self._dump_order(payload)

            print(f"[KIS_ORDER_EXC] {side} {sym} qty={q} err={e!r}")
            return BrokerOrderResult(False, None, {"error": repr(e), "response_text": resp_text})