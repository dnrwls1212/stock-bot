# src/analysis/news_filter.py
from __future__ import annotations

import re
from typing import Dict, List


# 이벤트 가능성이 높은 키워드(초기 버전)
HOT_KEYWORDS: List[str] = [
    r"\bearnings\b", r"\bguidance\b", r"\brevenue\b", r"\bEPS\b",
    r"\bacquires?\b", r"\bmerger\b", r"\bM&A\b",
    r"\bpartnership\b", r"\bcollaboration\b", r"\bcontract\b", r"\bdeal\b",
    r"\bSEC\b", r"\binvestigation\b", r"\blawsuit\b",
    r"\bexport\b", r"\bsanction\b", r"\bregulation\b",
    r"\blaunch\b", r"\brelease\b", r"\bintroduces?\b",
]

LOW_SIGNAL_PATTERNS: List[str] = [
    r"stocks to buy", r"buy and hold", r"forever", r"top \d+ stocks", r"best stocks",
    r"price target", r"opinion", r"watchlist", r"these are",
]

COMPANY_HINTS: Dict[str, List[str]] = {
    "NVDA": ["nvidia", "nvda"],
    "ORCL": ["oracle", "orcl"],
    "AVGO": ["broadcom", "avgo"],
    "AMD":  ["amd", "advanced micro devices"],
}


def _norm_text(title: str, summary: str) -> str:
    return f"{title}\n{summary}".lower()


def mentions_watchlist_company(title: str, summary: str) -> bool:
    """
    기존 함수 유지 (다른 곳에서 쓸 수 있음)
    """
    text = _norm_text(title, summary)
    for _, keys in COMPANY_HINTS.items():
        for k in keys:
            if k in text:
                return True
    return False


def mentions_ticker_company(ticker: str, title: str, summary: str) -> bool:
    """
    ticker가 watchlist에 있을 때, 해당 회사/티커가 실제로 언급되는지 체크.
    """
    if not ticker:
        return False
    t = ticker.upper().strip()
    keys = COMPANY_HINTS.get(t)
    if not keys:
        # 알 수 없는 티커면 전체 watchlist 언급이라도 있는지로 처리
        return mentions_watchlist_company(title, summary)

    text = _norm_text(title, summary)
    return any(k in text for k in keys)


def is_high_signal(title: str, summary: str) -> bool:
    """
    기존 함수 유지
    """
    text = _norm_text(title, summary)

    # 저품질/리스트형 글 먼저 제거
    for p in LOW_SIGNAL_PATTERNS:
        if re.search(p, text):
            return False

    # 이벤트성 키워드 포함 시 통과
    for k in HOT_KEYWORDS:
        if re.search(k, text, flags=re.IGNORECASE):
            return True

    return False


# ✅ 호환 레이어: main.py가 import하는 함수명 제공
def is_relevant_news(ticker: str, title: str, summary: str) -> bool:
    """
    main.py 호환용 래퍼.
    - ticker(해당 종목)가 실제로 언급되는 뉴스인지
    - 그리고 이벤트성(high signal)인지
    """
    if not mentions_ticker_company(ticker, title, summary):
        return False
    return is_high_signal(title, summary)
