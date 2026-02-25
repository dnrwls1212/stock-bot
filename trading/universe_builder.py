# src/trading/universe_builder.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class UniverseBuildResult:
    ok: bool
    out_path: str
    n_input: int
    n_scored: int
    n_valid: int
    target_n: int
    reason: str = ""
    dropped_sample: Optional[List[Dict[str, Any]]] = None


def _to_float(x: Any, default: float = 0.0) -> float:
    """
    pandas/numpy scalar, 단일 원소 Series, 단일 원소 ndarray 모두 안전하게 float로 변환.
    FutureWarning/TypeError 방지용.
    """
    try:
        return float(x)
    except TypeError:
        try:
            return float(np.asarray(x).item())
        except Exception:
            return float(default)
    except Exception:
        return float(default)


def normalize_ticker(t: str) -> str:
    return (t or "").strip().upper()


def is_valid_ticker(t: str) -> bool:
    t = normalize_ticker(t)
    if not t:
        return False
    # 아주 기본적인 필터만: 공백/슬래시/백슬래시 제거
    if any(ch in t for ch in (" ", "\\", "/")):
        return False
    return True


def _read_lines(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _safe_download_history(
    ticker: str,
    *,
    period: str = "3mo",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    yfinance history를 안전하게 가져온다.
    - watchlist_auto에서 interval 키워드를 넘기기 때문에 interval 지원 필수
    - 실패/빈 데이터면 빈 DF 반환
    """
    t = normalize_ticker(ticker)
    if not t:
        return pd.DataFrame()

    try:
        df = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=auto_adjust)
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _dollar_volume(df: pd.DataFrame) -> Optional[float]:
    """
    평균 달러거래대금(ADV$) = mean(Close * Volume)
    """
    if df is None or df.empty:
        return None
    if "Close" not in df.columns or "Volume" not in df.columns:
        return None
    dv = df["Close"] * df["Volume"]
    if dv is None or len(dv) == 0:
        return None
    return _to_float(dv.mean(), default=None)  # type: ignore[arg-type]


def _last_close(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    if "Close" not in df.columns:
        return None
    try:
        return _to_float(df["Close"].iloc[-1], default=None)  # type: ignore[arg-type]
    except Exception:
        return None


def _score_ticker(
    ticker: str,
    *,
    period: str,
    min_price: float,
    min_dollar_vol: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    스코어링(단순): 유동성(ADV$) 중심으로 점수화
    - 반환: (score, last_price, adv, drop_reason)
    """
    df = _safe_download_history(ticker, period=period, interval="1d", auto_adjust=True)
    if df.empty:
        return None, None, None, "no_history"

    px = _last_close(df)
    if px is None:
        return None, None, None, "no_price"

    adv = _dollar_volume(df)
    if adv is None:
        return None, float(px), None, "no_dvol"

    if float(px) < float(min_price):
        return None, float(px), float(adv), f"min_price<{min_price}"

    if float(adv) < float(min_dollar_vol):
        return None, float(px), float(adv), f"min_dvol<{min_dollar_vol}"

    # 점수: ADV$ 로그 스케일(너무 큰 값 폭주 방지)
    score = float(np.log10(max(1.0, float(adv))))
    return score, float(px), float(adv), None


def build_universe(
    *,
    out_path: str = "data/universe.txt",
    core_path: str = "data/universe_core.txt",
    period: str = "3mo",
    min_price: float = 5.0,
    min_dollar_vol: float = 5_000_000.0,
    target_n: int = 200,
    max_universe: int = 1200,
    pool_path: Optional[str] = None,
    seed_path: Optional[str] = None,
    # ----- alias (과거 main/다른 모듈이 넘기던 이름 흡수) -----
    price_min: Optional[float] = None,
    min_dv: Optional[float] = None,
    max_n: Optional[int] = None,
    # 다른 버전 키워드가 넘어와도 죽지 않게 흡수
    **_ignored: Any,
) -> UniverseBuildResult:
    """
    Universe 후보를 만들어 out_path에 저장.
    - core_path: 기본 seed 티커 파일
    - pool_path: 추가 후보 티커 파일(옵션)
    - seed_path: 과거 호환용(core_path와 동일 취급)
    - 데이터 없는 티커는 drop하고 계속 진행(유연)
    """

    # alias 적용
    if seed_path and (not core_path or core_path == "data/universe_core.txt"):
        core_path = seed_path
    if price_min is not None:
        min_price = float(price_min)
    if min_dv is not None:
        min_dollar_vol = float(min_dv)
    if max_n is not None:
        target_n = int(max_n)

    # 입력 풀 구성
    core = [normalize_ticker(x) for x in _read_lines(core_path)]
    pool = [normalize_ticker(x) for x in _read_lines(pool_path)] if pool_path else []
    tickers = [t for t in (core + pool) if is_valid_ticker(t)]

    # 중복 제거(순서 유지)
    seen = set()
    uniq: List[str] = []
    for t in tickers:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    tickers = uniq

    if len(tickers) == 0:
        return UniverseBuildResult(
            ok=False,
            out_path=out_path,
            n_input=0,
            n_scored=0,
            n_valid=0,
            target_n=int(target_n),
            reason=f"seed={core_path}",
        )

    scored: List[Tuple[str, float, float, float]] = []
    dropped: List[Dict[str, Any]] = []

    for t in tickers:
        score, px, adv, drop_reason = _score_ticker(
            t, period=period, min_price=float(min_price), min_dollar_vol=float(min_dollar_vol)
        )
        if score is None:
            dropped.append({"ticker": t, "reason": drop_reason or "dropped", "price": px, "adv": adv})
            continue
        scored.append((t, float(score), float(px), float(adv)))

        # 상한 초과 방지
        if len(scored) >= int(max_universe):
            break

    # 점수 순 정렬(내림차순)
    scored.sort(key=lambda x: x[1], reverse=True)

    # 목표 개수만큼 선택
    picked = [t for (t, _s, _px, _adv) in scored[: int(target_n)]]

    _ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in picked:
            f.write(t + "\n")

    ok = len(picked) > 0
    reason = "" if ok else f"seed={core_path}"

    # DEBUG 샘플 출력용(원하면 main에서 찍을 수 있음)
    dropped_sample = dropped[:5] if dropped else None

    return UniverseBuildResult(
        ok=ok,
        out_path=out_path,
        n_input=len(tickers),
        n_scored=len(scored),
        n_valid=len(picked),
        target_n=int(target_n),
        reason=reason,
        dropped_sample=dropped_sample,
    )