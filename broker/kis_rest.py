from __future__ import annotations

import os
import time
import requests
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class KISConfig:
    app_key: str
    app_secret: str
    account_no: str  # 예: "12345678-01" 형식일 수도 있고, 문서/계좌에 따라 분리됨
    account_product_code: str  # 예: "01"
    base_url: str  # 실전/모의 base url
    is_paper: bool = False


class KISRestClient:
    """
    MVP: REST로
    - 접근토큰 발급
    - 현재가 조회(국내/해외는 엔드포인트가 다름)
    - 주문(현금) (나중에 hashkey 등 추가 필요)
    """

    def __init__(self, cfg: KISConfig):
        self.cfg = cfg
        self._access_token: str = ""
        self._expires_at: float = 0.0

    def _token_valid(self) -> bool:
        return bool(self._access_token) and time.time() < (self._expires_at - 60)

    def get_access_token(self) -> str:
        """
        공식 개발자센터의 OAuth '접근토큰발급(P)'를 사용(유효기간 24h 등 안내). :contentReference[oaicite:2]{index=2}
        """
        if self._token_valid():
            return self._access_token

        url = f"{self.cfg.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.cfg.app_key,
            "appsecret": self.cfg.app_secret,
        }
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()

        token = data.get("access_token", "")
        expires_in = int(data.get("expires_in", 0))  # seconds
        if not token:
            raise RuntimeError(f"KIS token issuance failed: {data}")

        self._access_token = token
        self._expires_at = time.time() + max(0, expires_in)
        return self._access_token

    def _headers(self, tr_id: str) -> Dict[str, str]:
        token = self.get_access_token()
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.cfg.app_key,
            "appsecret": self.cfg.app_secret,
            "tr_id": tr_id,
        }

    def inquire_domestic_price(self, code6: str) -> Dict[str, Any]:
        """
        국내 주식 현재가 REST (문서: '주식현재가 시세') :contentReference[oaicite:3]{index=3}
        - code6: 6자리 종목코드(예: 005930)
        """
        url = f"{self.cfg.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 보통 J=주식(코스피/코스닥) (정확 값은 문서 확인)
            "FID_INPUT_ISCD": code6,
        }
        r = requests.get(url, headers=self._headers(tr_id="FHKST01010100"), params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def place_domestic_order_cash(
        self,
        code6: str,
        side: str,  # "BUY" or "SELL"
        qty: int,
        price: int,  # 지정가 (시장가면 별도 코드 필요)
    ) -> Dict[str, Any]:
        """
        국내 주식 주문(현금) 스켈레톤.
        실제로는 hashkey 발급/헤더 추가, TR_ID 등 계좌/환경에 따라 달라질 수 있음.
        (문서는 개발자센터 [국내주식] 주문/계좌 쪽 참조) :contentReference[oaicite:4]{index=4}
        """
        if side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")

        url = f"{self.cfg.base_url}/uapi/domestic-stock/v1/trading/order-cash"

        # TR_ID는 실전/모의, 매수/매도에 따라 다름(문서 값으로 교체 필요)
        tr_id = "TTTC0802U" if side == "BUY" else "TTTC0801U"

        # 계좌번호는 문서에서 보통 CANO(8) + ACNT_PRDT_CD(2) 분리 형태 사용
        cano = self.cfg.account_no.replace("-", "")[:8]
        prdt = self.cfg.account_product_code

        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "PDNO": code6,
            "ORD_DVSN": "00",  # 00=지정가 (문서 확인)
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": str(int(price)),
        }

        r = requests.post(url, headers=self._headers(tr_id=tr_id), json=body, timeout=20)
        r.raise_for_status()
        return r.json()
