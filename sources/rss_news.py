from __future__ import annotations

from typing import List, Dict, Any, Optional
import feedparser

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

# 기존 호환용 기본값
RSS_URLS = [
    YAHOO_RSS.format(ticker="NVDA"),
    YAHOO_RSS.format(ticker="ORCL"),
]


def build_rss_urls(tickers: List[str]) -> List[str]:
    urls: List[str] = []
    for t in tickers:
        t = (t or "").strip().upper()
        if not t:
            continue
        urls.append(YAHOO_RSS.format(ticker=t))
    return urls


def fetch_rss_news(limit: int = 20, rss_urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    urls = rss_urls or RSS_URLS

    items: List[Dict[str, Any]] = []
    for url in urls:
        feed = feedparser.parse(url)
        for e in feed.entries[:limit]:
            items.append(
                {
                    "title": e.get("title", "") or "",
                    "link": e.get("link", "") or "",
                    "published": e.get("published", "") or "",
                    "summary": e.get("summary", "") or "",
                    "source": url,
                }
            )
    return items
