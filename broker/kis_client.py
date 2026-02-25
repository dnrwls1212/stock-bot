# src/broker/kis_client.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass(frozen=True)
class KisConfig:
    app_key: str
    app_secret: str
    account_no: str          # CANO (앞 8자리)
    account_prdt: str        # ACNT_PRDT_CD (뒤 2자리)
    paper: bool = True       # 모의투자 여부
    exchange: str = "NASD"   # 해외 거래소 코드(예: NASD, NYSE, AMEX)
    execute_orders: bool = False  # 1이면 실주문 전송, 0이면 드라이런


class KisClient:
    """
    KIS OpenAPI 최소 클라이언트:
      - OAuth 토큰 발급/캐시
      - hashkey 발급
      - REST 요청 래퍼 (GET/POST)
    """

    def __init__(self, cfg: KisConfig, timeout: float = 10.0) -> None:
        self.cfg = cfg
        self.timeout = float(timeout)

        # 실전/모의 도메인
        self.base_url = (
            "https://openapivts.koreainvestment.com:29443"
            if cfg.paper
            else "https://openapi.koreainvestment.com:9443"
        )

        self._token: Optional[str] = None
        self._token_expire_ts: float = 0.0

    @staticmethod
    def _env_bool(key: str, default: bool = False) -> bool:
        v = os.environ.get(key)
        if v is None:
            return default
        return v.strip() in ("1", "true", "True", "YES", "yes", "y")

    @staticmethod
    def from_env(timeout: float = 10.0) -> "KisClient":
        app_key = os.environ.get("KIS_APP_KEY", "").strip()
        app_secret = os.environ.get("KIS_APP_SECRET", "").strip()
        account_no = os.environ.get("KIS_ACCOUNT_NO", "").strip()
        account_prdt = os.environ.get("KIS_ACCOUNT_PRDT", "").strip()
        paper = os.environ.get("KIS_PAPER", "1").strip() == "1"
        exchange = os.environ.get("KIS_EXCHANGE", "NASD").strip() or "NASD"
        execute_orders = KisClient._env_bool("EXECUTE_ORDERS", False)

        missing = [k for k, v in {
            "KIS_APP_KEY": app_key,
            "KIS_APP_SECRET": app_secret,
            "KIS_ACCOUNT_NO": account_no,
            "KIS_ACCOUNT_PRDT": account_prdt,
        }.items() if not v]
        if missing:
            raise RuntimeError(f"KIS env missing: {', '.join(missing)}")

        return KisClient(
            KisConfig(
                app_key=app_key,
                app_secret=app_secret,
                account_no=account_no,
                account_prdt=account_prdt,
                paper=paper,
                exchange=exchange,
                execute_orders=execute_orders,
            ),
            timeout=timeout,
        )

    def _token_valid(self) -> bool:
        return bool(self._token) and (time.time() + 30.0) < self._token_expire_ts

    def get_access_token(self) -> str:
        """OAuth 접근토큰 발급(tokenP). 만료 전이면 캐시 사용."""
        if self._token_valid():
            return self._token  # type: ignore[return-value]

        url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.cfg.app_key,
            "appsecret": self.cfg.app_secret,
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()

        token = j.get("access_token")
        expires_in = float(j.get("expires_in", 0) or 0)
        if not token:
            raise RuntimeError(f"tokenP failed: {j}")

        self._token = str(token)
        self._token_expire_ts = time.time() + max(0.0, expires_in)
        return self._token

    def make_headers(self, tr_id: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        token = self.get_access_token()
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.cfg.app_key,
            "appsecret": self.cfg.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        if extra:
            headers.update(extra)
        return headers

    def hashkey(self, data: Dict[str, Any]) -> str:
        """주문/POST에 필요한 hashkey 발급."""
        url = f"{self.base_url}/uapi/hashkey"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "appkey": self.cfg.app_key,
            "appsecret": self.cfg.app_secret,
        }
        r = requests.post(url, headers=headers, json=data, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        hk = j.get("HASH")
        if not hk:
            raise RuntimeError(f"hashkey failed: {j}")
        return str(hk)

    def request(
        self,
        method: str,
        path: str,
        tr_id: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        need_hashkey: bool = False,
    ) -> Dict[str, Any]:
        """KIS REST 호출 래퍼"""
        url = f"{self.base_url}{path}"
        headers = self.make_headers(tr_id)

        if need_hashkey and data is not None:
            headers["hashkey"] = self.hashkey(data)

        m = method.upper().strip()
        if m == "GET":
            r = requests.get(url, headers=headers, params=params, timeout=self.timeout)
        elif m == "POST":
            r = requests.post(url, headers=headers, params=params, json=data, timeout=self.timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")

        r.raise_for_status()
        return r.json()