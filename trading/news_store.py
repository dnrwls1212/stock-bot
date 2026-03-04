# src/trading/news_store.py
from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List

@dataclass
class NewsEvent:
    ticker: str
    ts_kst: str
    published: str
    title: str
    summary: str
    link: str
    event_score: float
    confidence: float
    event_type: str
    sentiment: str
    impact: int
    why_it_moves: str
    raw: Dict[str, Any]

class NewsStore:
    def __init__(self, path: str, memory_dir: str = "data/news_memory"):
        self.path = path
        self.memory_dir = memory_dir
        self.events: List[Dict[str, Any]] = []
        self._new_counts: Dict[str, int] = {}
        self._load()
        os.makedirs(self.memory_dir, exist_ok=True)

    def _load(self):
        if not os.path.exists(self.path): return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.events = data.get("events", [])
                self._new_counts = data.get("new_counts", {})
        except Exception: pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"events": self.events[-2000:], "new_counts": self._new_counts}, f, ensure_ascii=False)
        except Exception: pass

    def add_event(self, ev: NewsEvent):
        self.events.append(asdict(ev))
        t = ev.ticker.upper()
        self._new_counts[t] = self._new_counts.get(t, 0) + 1
        self._save()

    def get_recent_events(self, ticker: str, days: int = 14, limit: int = 50, as_dict: bool = True) -> List[Dict[str, Any]]:
        t = ticker.upper()
        # 🚀 [버그 수정] 시간대 충돌을 막기 위해 현재 시간도 KST(Asia/Seoul)로 명확히 지정
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        
        out = []
        for e in reversed(self.events):
            if e.get("ticker", "").upper() != t: continue
            try:
                dt = datetime.fromisoformat(e["ts_kst"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("Asia/Seoul"))
                if (now - dt).total_seconds() <= days * 86400:
                    out.append(e)
            except Exception as ex: 
                pass
            if len(out) >= limit: break
        return out

    def pop_new_event_count(self, ticker: str) -> int:
        t = ticker.upper()
        c = self._new_counts.get(t, 0)
        if c > 0:
            self._new_counts[t] = 0
            self._save()
        return c

    def load_memory(self, ticker: str) -> str:
        t = ticker.upper()
        p = os.path.join(self.memory_dir, f"{t}.txt")
        if not os.path.exists(p): return ""
        try:
            with open(p, "r", encoding="utf-8") as f: return f.read()
        except Exception: return ""

    def save_memory(self, ticker: str, text: str):
        t = ticker.upper()
        p = os.path.join(self.memory_dir, f"{t}.txt")
        try:
            with open(p, "w", encoding="utf-8") as f: f.write(text)
        except Exception: pass

    def compute_signal(self, ticker: str, now_kst: datetime, half_life_hours: float = 48.0, window_days: int = 14, max_items: int = 100) -> Dict[str, Any]:
        """
        [핵심 변경] 뉴스 속성(event_type, impact)에 따라 Dynamic Half-life(다중 반감기) 적용
        """
        evs = self.get_recent_events(ticker, days=window_days, limit=max_items)
        if not evs:
            return {"ticker": ticker, "news_score": 0.0, "news_conf": 0.55, "raw_sum": 0.0, "raw_n": 0}

        score_sum, conf_sum = 0.0, 0.0
        n = 0
        raw_sum = 0.0

        # 🚀 [수정] 장기 모멘텀 및 장기 악재(리스크) 키워드 통합
        long_term_keywords = [
            # 장기 호재성
            "earning", "guidance", "m_and_a", "fda", "macro", "contract", "merger", "clinical",
            # 장기 악재성 (추가됨)
            "lawsuit", "investigation", "fraud", "downgrade", "resign", "bankrupt", "delist", "shortfall", "offering"
        ]

        for e in evs:
            ts_str = e.get("ts_kst")
            if not ts_str: continue
            try:
                dt = datetime.fromisoformat(ts_str)
                if dt.tzinfo is None:
                    from zoneinfo import ZoneInfo
                    dt = dt.replace(tzinfo=ZoneInfo("Asia/Seoul"))
                age_hours = max(0.0, (now_kst - dt).total_seconds() / 3600.0)
                
                e_type = str(e.get("event_type", "other")).lower()
                impact = int(e.get("impact", 0) or 0)
                
                # 🚀 [수정] impact가 +2(초강력 호재)이거나 -2(치명적 악재)일 때 모두 장기 기억으로 분류
                is_long_term = False
                if abs(impact) >= 2: 
                    is_long_term = True
                elif any(k in e_type for k in long_term_keywords):
                    is_long_term = True

                # 장기 이벤트(호/악재 무관)는 반감기를 대폭 늘림 (예: 48h -> 약 30일)
                eff_hl = half_life_hours * 15 if is_long_term else half_life_hours
                
                decay = 0.5 ** (age_hours / eff_hl)
                s = float(e.get("event_score", 0.0))
                c = float(e.get("confidence", 0.55))

                raw_sum += s
                score_sum += (s * decay)
                conf_sum += c
                n += 1
            except Exception: pass

        if n == 0:
            return {"ticker": ticker, "news_score": 0.0, "news_conf": 0.55, "raw_sum": 0.0, "raw_n": 0}
        
        return {
            "ticker": ticker,
            "news_score": score_sum,
            "news_conf": conf_sum / n,
            "raw_sum": raw_sum,
            "raw_n": n
        }