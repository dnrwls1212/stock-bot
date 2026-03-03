# src/valuation/market_data.py
from __future__ import annotations

from typing import Dict, Any
from src.utils.yf_silent import ticker_info_silent

def fetch_snapshot(ticker: str) -> Dict[str, Any]:
    """
    yfinance에서 펀더멘털 및 월가 애널리스트 컨센서스 데이터를 가져옵니다.
    """
    info = ticker_info_silent(ticker)

    # 가격 정보
    price = info.get("regularMarketPrice") or info.get("currentPrice")

    # 가치 평가 지표
    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    peg = info.get("pegRatio") # 🚀 가장 중요한 고성장주 평가 지표
    market_cap = info.get("marketCap")

    # 성장성 지표
    earnings_growth = info.get("earningsGrowth")  # YOY 이익성장률
    revenue_growth = info.get("revenueGrowth")    # YOY 매출성장률

    # 월가 컨센서스 (목표가 및 투자의견)
    target_price = info.get("targetMeanPrice") or info.get("targetMedianPrice")
    # recommendationMean: 1.0(강력매수) ~ 5.0(매도)
    recommendation_mean = info.get("recommendationMean") 
    recommendation_key = info.get("recommendationKey")

    return {
        "ticker": ticker,
        "price": price,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "peg": peg,
        "market_cap": market_cap,
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
        "target_price": target_price,
        "recommendation_mean": recommendation_mean,
        "recommendation_key": recommendation_key,
    }