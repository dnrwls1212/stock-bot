# src/analysis/news_filter.py
from __future__ import annotations

import re
from typing import Dict, List

# 🚀 기업 이벤트 + 거시경제(Macro) + 전쟁/평화 + 한국어 속보 키워드 대통합
HOT_KEYWORDS: List[str] = [
    # 1. 영문 개별 기업 이벤트
    r"\bearnings\b", r"\bguidance\b", r"\brevenue\b", r"\bEPS\b",
    r"\bacquires?\b", r"\bmerger\b", r"\bM&A\b",
    r"\bpartnership\b", r"\bcollaboration\b", r"\bcontract\b", r"\bdeal\b",
    r"\bSEC\b", r"\binvestigation\b", r"\blawsuit\b",
    r"\bexport\b", r"\bsanction\b", r"\bregulation\b",
    r"\blaunch\b", r"\brelease\b", r"\bintroduces?\b",
    
    # 2. 영문 거시경제 및 지정학적 리스크 (블랙스완 및 평화/반등용)
    r"\bwar\b", r"\bstrike\b", r"\bmissile\b", r"\battack\b", r"\bconflict\b", r"\bgeopolitical\b",
    r"\bceasefire\b", r"\btruce\b", r"\bpeace\b", r"\bnegotiation\b", r"\bresolve\b", # 🟢 휴전/협상
    r"\brebound\b", r"\brally\b", r"\boversold\b", r"\bpriced in\b", # 🟢 반등/선반영
    r"\bfed\b", r"\brate cut\b", r"\brate hike\b", r"\binflation\b", r"\bcpi\b",
    r"\bcrash\b", r"\bplunge\b", r"\bbankrupt\b", r"\bpanic\b",
    
    # 3. KIS 속보용 한국어 주요 키워드
    r"실적", r"가이던스", r"인수", r"합병", r"파트너십", r"계약", r"수주",
    r"승인", r"임상", r"소송", r"조사", r"규제", r"제재", r"수출",
    r"전쟁", r"미사일", r"공격", r"타격", r"무력", r"충돌", r"사태", r"사망",
    r"휴전", r"협상", r"평화", r"합의", r"회담", r"진전", r"완화", # 🟢 휴전/협상
    r"반등", r"회복", r"저가매수", r"선반영", r"랠리", # 🟢 반등/선반영
    r"금리", r"인플레", r"물가", r"파산", r"폭락", r"급락", r"우려", r"비상", r"쇼크"
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
    text = _norm_text(title, summary)
    for _, keys in COMPANY_HINTS.items():
        for k in keys:
            if k in text:
                return True
    return False

def mentions_ticker_company(ticker: str, title: str, summary: str) -> bool:
    if not ticker:
        return False
    t = ticker.upper().strip()
    keys = COMPANY_HINTS.get(t)
    if not keys:
        return mentions_watchlist_company(title, summary)
    text = _norm_text(title, summary)
    return any(k in text for k in keys)

def is_high_signal(title: str, summary: str) -> bool:
    text = _norm_text(title, summary)
    for p in LOW_SIGNAL_PATTERNS:
        if re.search(p, text):
            return False
    for k in HOT_KEYWORDS:
        if re.search(k, text, flags=re.IGNORECASE):
            return True
    return False

def is_relevant_news(ticker: str, title: str, summary: str) -> bool:
    if not mentions_ticker_company(ticker, title, summary):
        return False
    return is_high_signal(title, summary)