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

    # 👇👇👇 [핵심 추가] 통합증거금 완벽 대응! 계좌 자산 및 구매력 파악 기능 👇👇👇
    def get_account_summary(self) -> tuple[float, float]:
        """
        계좌의 (외화주문가능금액, 외화총자산)을 USD 기준으로 파악하여 반환합니다.
        통합증거금(원화 기반 주문)을 완벽하게 지원하기 위해 매수가능금액조회(TTTS3007R) API를 활용합니다.
        """
        if self.kis.cfg.paper:
            return 1000.0, 10000.0 # 모의투자는 현금 10%가 있다고 가정

        try:
            # 1. 잔고조회(TTTS3012R)를 통해 현재 들고있는 주식 평가액(USD) 확인
            resp_bal = self.inquire_present_balance()
            stock_usd = 0.0
            if isinstance(resp_bal, dict) and str(resp_bal.get("rt_cd", "")) == "0":
                out2 = resp_bal.get("output2")
                if isinstance(out2, list) and len(out2) > 0:
                    stock_usd = float(out2[0].get("ovrs_stck_evlu_amt") or 0.0)
                elif isinstance(out2, dict):
                    stock_usd = float(out2.get("ovrs_stck_evlu_amt") or 0.0)

            # 2. 매수가능금액조회(TTTS3007R)를 통해 '통합증거금'이 포함된 외화 구매력(USD) 확인
            tr_id_ps = "VTTS3007R" if self.kis.cfg.paper else "TTTS3007R"
            params_ps = {
                "CANO": self.kis.cfg.account_no,
                "ACNT_PRDT_CD": self.kis.cfg.account_prdt,
                "OVRS_EXCG_CD": "NASD",
                "OVRS_ORD_UNPR": "0", # 시장가 기준
                "ITEM_CD": "AAPL",    # 금액 조회를 위한 임의 티커
            }
            resp_ps = self.kis.request("GET", "/uapi/overseas-stock/v1/trading/inquire-psamount", tr_id=tr_id_ps, params=params_ps)
            
            cash_usd = 0.0
            if isinstance(resp_ps, dict) and str(resp_ps.get("rt_cd", "")) == "0":
                out_ps = resp_ps.get("output") or {}
                # frcr_ord_psbl_amt1: 통합증거금 기준 외화주문가능금액 (원화를 USD로 환산한 총알)
                # ord_psbl_frcr_amt: 일반 외화주문가능금액 (순수 달러)
                integrated_margin = float(out_ps.get("frcr_ord_psbl_amt1") or 0.0)
                normal_margin = float(out_ps.get("ord_psbl_frcr_amt") or 0.0)
                
                # 통합증거금 금액이 더 크면 그것을 사용 (원화 포함)
                cash_usd = max(integrated_margin, normal_margin)

            total_usd = cash_usd + stock_usd
            if total_usd <= 0.0:
                total_usd = cash_usd + 1.0 # 0 나누기 방지용
                
            return cash_usd, total_usd
        except Exception as e:
            print(f"[KIS_BALANCE_ERR] 통합증거금/잔고 파싱 에러: {e}")
            
        return 1000.0, 10000.0
    # 👆👆👆 ---------------------------------------------------- 👆👆👆

    # 👇👇👇 [신규 추가] 미체결 주문 취소 로직 👇👇👇
    def cancel_order(self, org_order_no: str, qty: int = 0) -> Dict[str, Any]:
        """지정된 주문 번호의 미체결 내역을 스마트 라우팅으로 취소합니다."""
        tr_id = "VTTT1004U" if self.kis.cfg.paper else "TTTT1004U"
        exchanges = [self.kis.cfg.exchange or "NASD", "NYSE", "AMEX"]
        last_resp = {}
        
        for excg in exchanges:
            params = {
                "CANO": self.kis.cfg.account_no,
                "ACNT_PRDT_CD": self.kis.cfg.account_prdt,
                "OVRS_EXCG_CD": excg,
                "ORGN_ODNO": org_order_no,
                "RVSE_CNCL_DVSN_CD": "02", # 01: 정정, 02: 취소
                "ORD_QTY": str(qty),       # 0이면 잔량 전부 취소
                "OVRS_ORD_UNPR": "0",      # 취소 시 단가는 0
                "ORD_SVR_DVSN_CD": "0"
            }
            try:
                resp = self.kis.request("POST", "/uapi/overseas-stock/v1/trading/order-rvsecncl", tr_id=tr_id, data=params, need_hashkey=True)
                last_resp = resp
                if str(resp.get("rt_cd", "")) == "0":
                    self._dump_order({"kind": "cancel_response", "tr_id": tr_id, "params": params, "response": resp})
                    return resp
            except Exception: 
                pass
        return last_resp

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
        # 👇👇 [수정] 1달러 이상은 소수점 2자리, 동전주는 4자리로 KIS 규칙에 맞춤
        if p >= 1.0:
            return f"{float(p):.2f}"
        else:
            return f"{float(p):.4f}"

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

        # 👇👇 [수정] 거래소(NASD, NYSE, AMEX) 자동 탐색 로직 (스마트 라우팅)
        exchanges_to_try = [self.kis.cfg.exchange or "NASD", "NYSE", "AMEX"]
        exchanges = []
        for e in exchanges_to_try: # 중복 제거
            if e not in exchanges: exchanges.append(e)

        last_raw = {}
        last_error = None

        for excg in exchanges:
            p: Dict[str, Any] = {
                "CANO": self.kis.cfg.account_no,
                "ACNT_PRDT_CD": self.kis.cfg.account_prdt,
                "OVRS_EXCG_CD": excg, # 여기서 NASD -> NYSE -> AMEX 순으로 찔러봅니다.
                "PDNO": sym,
                "ORD_QTY": str(q),
                "OVRS_ORD_UNPR": str(ovrs_unpr),
                "ORD_SVR_DVSN_CD": "0",
                "ORD_DVSN": str(ord_dvsn),
            }

            meta: Dict[str, Any] = {
                "ts_kst": self._now_kst(), "paper": bool(self.kis.cfg.paper), "side": side,
                "symbol": sym, "qty": q, "last_price": last_price, "tr_id": tr_id,
                "exchange": excg, "execute_orders": True, "ord_dvsn": ord_dvsn,
            }

            try:
                raw = self.kis.request("POST", "/uapi/overseas-stock/v1/trading/order", tr_id=tr_id, data=p, need_hashkey=True)
                last_raw = raw
                ok = str(raw.get("rt_cd", "")) == "0"
                
                if ok: # 성공하면 즉시 종료
                    order_no = self._extract_order_no(raw)
                    self._dump_order({"kind": "order_response", "meta": meta, "request": p, "response": raw})
                    return BrokerOrderResult(True, order_no, raw)
                else:
                    msg = str(raw.get('msg1', ''))
                    if "해당종목정보가 없습니다" in msg or "거래소코드" in msg:
                        continue # 종목이 이 거래소에 없으면 다음 거래소(NYSE 등)로 넘어가서 재시도
                    else:
                        # 돈이 부족하거나 다른 에러면 즉시 포기
                        print(f"[KIS_ORDER_ERR] {side} {sym} qty={q} excg={excg} rt_cd={raw.get('rt_cd')} msg={msg}")
                        self._dump_order({"kind": "order_response", "meta": meta, "request": p, "response": raw})
                        return BrokerOrderResult(False, None, raw)
            except Exception as e:
                last_error = e
                break

        # 모든 거래소를 다 돌았는데도 실패한 경우
        if last_error:
            payload = {"kind": "order_exception", "meta": meta, "request": p, "error": repr(last_error)}
            try:
                r = getattr(last_error, "response", None)
                if r is not None: payload["response_text"] = r.text[:2000]
            except Exception: pass
            self._dump_order(payload)
            print(f"[KIS_ORDER_EXC] {side} {sym} qty={q} err={last_error!r}")
            return BrokerOrderResult(False, None, {"error": repr(last_error)})
        
        print(f"[KIS_ORDER_ERR] {side} {sym} qty={q} rt_cd={last_raw.get('rt_cd')} msg={last_raw.get('msg1')}")
        return BrokerOrderResult(False, None, last_raw)