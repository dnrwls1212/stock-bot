from __future__ import annotations

from typing import Dict, Any

from src.utils.yf_silent import ticker_info_silent


def fetch_snapshot(ticker: str) -> Dict[str, Any]:
    """
    yfinance에서 가능한 범위의 스냅샷을 가져옴.
    (무료 데이터라 일부 항목은 None일 수 있음)

    ⚠️ yfinance는 간헐적으로 stdout/stderr에 경고 문구를 출력하는 경우가 있어
    ticker_info_silent()로 감싸서 노이즈를 차단한다.
    """
    info = ticker_info_silent(ticker)

    # price
    price = info.get("regularMarketPrice") or info.get("currentPrice")

    # valuation-ish fields (availability varies)
    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    peg = info.get("pegRatio")
    market_cap = info.get("marketCap")

    # growth-ish fields (sometimes missing)
    earnings_growth = info.get("earningsGrowth")  # YOY
    revenue_growth = info.get("revenueGrowth")    # YOY

    return {
        "ticker": ticker,
        "price": price,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "peg": peg,
        "market_cap": market_cap,
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
    }