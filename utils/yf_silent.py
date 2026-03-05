from __future__ import annotations

import contextlib
import io
from typing import Any, Callable, Dict, Optional, TypeVar

import yfinance as yf

T = TypeVar("T")


def _silent_call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a function while suppressing noisy stdout/stderr (yfinance can print warnings)."""
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        return fn(*args, **kwargs)


def download_silent(**kwargs: Any):
    """yfinance.download wrapper that suppresses stdout/stderr."""
    return _silent_call(yf.download, **kwargs)


# src/utils/yf_silent.py

def safe_download(
    *,
    tickers: str,
    period: Optional[str] = None,
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = False,
    progress: bool = False,
    threads: bool = True,
    prepost: bool = False,  # 👈 [1. 추가] 프리마켓 파라미터 허용
):
    """Safe downloader with a single retry without threads."""
    try:
        df = download_silent(
            tickers=tickers,
            period=period,
            interval=interval,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            progress=progress,
            threads=threads,
            prepost=prepost,  # 👈 [2. 추가] yfinance 실제 호출 시 전달
        )
        return {"ok": True, "df": df}
    except Exception as e:
        if threads:
            try:
                df = download_silent(
                    tickers=tickers,
                    period=period,
                    interval=interval,
                    start=start,
                    end=end,
                    auto_adjust=auto_adjust,
                    progress=progress,
                    threads=False,
                    prepost=prepost,  # 👈 [3. 추가] 재시도 로직에도 전달
                )
                return {"ok": True, "df": df, "retry": "threads=False"}
            except Exception as e2:
                return {"ok": False, "error": repr(e2), "retry": "threads=False"}
        return {"ok": False, "error": repr(e)}


def ticker_info_silent(ticker: str) -> Dict[str, Any]:
    """Fetch yfinance Ticker.info/get_info while suppressing stdout/stderr."""
    t = yf.Ticker(ticker)
    # yfinance versions vary: some provide .get_info(), some use .info property
    if hasattr(t, "get_info"):
        return _silent_call(t.get_info)  # type: ignore[misc]
    # fall back to .info property access
    return _silent_call(lambda: dict(getattr(t, "info", {}) or {}))