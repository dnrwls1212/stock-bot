# src/knowledge/ticker_kb.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo


def now_kst_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


@dataclass
class EvidenceItem:
    ts_kst: str
    source: str          # "news", "filing", "earnings", "manual"
    title: str
    summary: str
    link: str = ""
    sentiment: str = "neutral"
    impact: int = 0
    tags: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRecord:
    ts_kst: str
    action: str          # BUY/SELL/HOLD
    rationale: str       # LLM reasoning summary
    key_drivers: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    valuation_view: str = ""     # what changed in valuation assumptions
    confidence: float = 0.5
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TickerKB:
    ticker: str
    updated_at: str = field(default_factory=now_kst_iso)

    # 기업 방향성(사용자가 직접 or LLM이 점진 업데이트)
    thesis: str = ""                 # 한 문단 핵심
    business_summary: str = ""       # 무엇을 파는 회사인지
    moat: str = ""                   # 경쟁우위
    key_drivers: List[str] = field(default_factory=list)   # 매출/마진/수요/규제 등
    key_risks: List[str] = field(default_factory=list)

    # 밸류 가정/적정주가 관련(“모델”이 아니라 “가정 저장”이 핵심)
    valuation_method: str = "simple"   # simple / dcf / comps 등
    valuation_assumptions: Dict[str, Any] = field(default_factory=dict)
    target_price: Optional[float] = None
    fair_value_range: Optional[List[float]] = None  # [low, high]

    # 누적 근거/판단 기록
    evidence: List[EvidenceItem] = field(default_factory=list)
    decisions: List[DecisionRecord] = field(default_factory=list)

    # 메타
    tags: List[str] = field(default_factory=list)


class KBStore:
    def __init__(self, base_dir: str = "data/kb") -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, ticker: str) -> str:
        t = (ticker or "").upper().strip()
        return os.path.join(self.base_dir, f"{t}.json")

    def load(self, ticker: str) -> TickerKB:
        t = (ticker or "").upper().strip()
        p = self._path(t)
        if not os.path.exists(p):
            return TickerKB(ticker=t)

        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)

        # 역직렬화(하위 구조 포함)
        kb = TickerKB(ticker=t)
        for k, v in d.items():
            if k in ("evidence", "decisions"):
                continue
            if hasattr(kb, k):
                setattr(kb, k, v)

        ev = []
        for item in d.get("evidence", []) or []:
            if isinstance(item, dict):
                ev.append(EvidenceItem(**{
                    "ts_kst": item.get("ts_kst", now_kst_iso()),
                    "source": item.get("source", "news"),
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "link": item.get("link", ""),
                    "sentiment": item.get("sentiment", "neutral"),
                    "impact": int(item.get("impact", 0) or 0),
                    "tags": list(item.get("tags", []) or []),
                    "raw": dict(item.get("raw", {}) or {}),
                }))
        kb.evidence = ev

        decs = []
        for item in d.get("decisions", []) or []:
            if isinstance(item, dict):
                decs.append(DecisionRecord(**{
                    "ts_kst": item.get("ts_kst", now_kst_iso()),
                    "action": item.get("action", "HOLD"),
                    "rationale": item.get("rationale", ""),
                    "key_drivers": list(item.get("key_drivers", []) or []),
                    "key_risks": list(item.get("key_risks", []) or []),
                    "valuation_view": item.get("valuation_view", ""),
                    "confidence": float(item.get("confidence", 0.5) or 0.5),
                    "raw": dict(item.get("raw", {}) or {}),
                }))
        kb.decisions = decs

        return kb

    def save(self, kb: TickerKB) -> None:
        kb.updated_at = now_kst_iso()
        p = self._path(kb.ticker)
        tmp = p + ".tmp"

        d = asdict(kb)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)

    def add_evidence(
        self,
        ticker: str,
        *,
        source: str,
        title: str,
        summary: str,
        link: str = "",
        sentiment: str = "neutral",
        impact: int = 0,
        tags: Optional[List[str]] = None,
        raw: Optional[Dict[str, Any]] = None,
        max_items: int = 400,
    ) -> None:
        kb = self.load(ticker)
        kb.evidence.insert(
            0,
            EvidenceItem(
                ts_kst=now_kst_iso(),
                source=source,
                title=title,
                summary=summary,
                link=link,
                sentiment=sentiment,
                impact=int(impact),
                tags=tags or [],
                raw=raw or {},
            ),
        )
        # 최근 N개만 유지
        kb.evidence = kb.evidence[:max_items]
        self.save(kb)

    def add_decision(
        self,
        ticker: str,
        *,
        action: str,
        rationale: str,
        confidence: float,
        key_drivers: Optional[List[str]] = None,
        key_risks: Optional[List[str]] = None,
        valuation_view: str = "",
        raw: Optional[Dict[str, Any]] = None,
        max_items: int = 200,
    ) -> None:
        kb = self.load(ticker)
        kb.decisions.insert(
            0,
            DecisionRecord(
                ts_kst=now_kst_iso(),
                action=action,
                rationale=rationale,
                key_drivers=key_drivers or [],
                key_risks=key_risks or [],
                valuation_view=valuation_view,
                confidence=float(confidence),
                raw=raw or {},
            ),
        )
        kb.decisions = kb.decisions[:max_items]
        self.save(kb)