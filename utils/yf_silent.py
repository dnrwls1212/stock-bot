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
):
    """Safe downloader with a single retry without threads.

    Returns dict:
      - ok: bool
      - df: pandas DataFrame (when ok)
      - error: str (when not ok)
      - retry: optional str
    """
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