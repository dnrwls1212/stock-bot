# src/trading/news_store.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def _parse_ts_kst(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # e.g. 2026-02-24T00:48:45+09:00
        return datetime.fromisoformat(ts)
    except Exception:
        return None


class NewsStore:
    """
    - events: 메모리상에 적재된 뉴스 이벤트(최근 n일)
    - store_path: 영속 저장(json)
    - memory_dir: ticker별 요약 메모리 파일 저장 폴더
    """

    def __init__(self, store_path: str, memory_dir: str = "data/news_memory") -> None:
        self.store_path = store_path
        self.memory_dir = memory_dir
        self.events: Dict[str, List[NewsEvent]] = {}
        self._new_event_counter: Dict[str, int] = {}

        os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
        os.makedirs(self.memory_dir, exist_ok=True)

        self._load()

    # -------------------------
    # persistence
    # -------------------------
    def _load(self) -> None:
        if not os.path.exists(self.store_path):
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                return

            events: Dict[str, List[NewsEvent]] = {}
            for t, arr in d.items():
                if not isinstance(arr, list):
                    continue
                tmp: List[NewsEvent] = []
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    try:
                        tmp.append(
                            NewsEvent(
                                ticker=str(item.get("ticker", t)).upper(),
                                ts_kst=str(item.get("ts_kst", "")),
                                published=str(item.get("published", "")),
                                title=str(item.get("title", "")),
                                summary=str(item.get("summary", "")),
                                link=str(item.get("link", "")),
                                event_score=float(item.get("event_score", 0.0) or 0.0),
                                confidence=float(item.get("confidence", 0.55) or 0.55),
                                event_type=str(item.get("event_type", "other") or "other"),
                                sentiment=str(item.get("sentiment", "neutral") or "neutral"),
                                impact=int(item.get("impact", 0) or 0),
                                why_it_moves=str(item.get("why_it_moves", "") or ""),
                                raw=item.get("raw", {}) if isinstance(item.get("raw", {}), dict) else {},
                            )
                        )
                    except Exception:
                        continue
                if tmp:
                    events[t] = tmp

            self.events = events
        except Exception:
            # 로드 실패 시 그냥 비움
            self.events = {}

    def _save(self) -> None:
        try:
            out: Dict[str, List[Dict[str, Any]]] = {}
            for t, arr in self.events.items():
                out[t] = [e.to_dict() for e in arr]
            tmp = self.store_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.store_path)
        except Exception:
            pass

    # -------------------------
    # events
    # -------------------------
    def add_event(self, evt: NewsEvent) -> None:
        t = (evt.ticker or "").upper().strip()
        if not t:
            return

        arr = self.events.get(t)
        if arr is None:
            arr = []
            self.events[t] = arr

        # 중복 링크 방지(간단)
        link = (evt.link or "").strip()
        if link and any((x.link or "").strip() == link for x in arr[:200]):
            return

        arr.insert(0, evt)
        self._new_event_counter[t] = int(self._new_event_counter.get(t, 0)) + 1

        # 너무 오래된 이벤트 정리(기본 14일)
        self._prune(days=14)
        self._save()

    def _prune(self, days: int = 14) -> None:
        cutoff = _now_kst() - timedelta(days=max(1, int(days)))
        for t, arr in list(self.events.items()):
            kept: List[NewsEvent] = []
            for e in arr:
                dt = _parse_ts_kst(e.ts_kst) or _parse_ts_kst(e.published)
                if dt is None:
                    kept.append(e)
                    continue
                if dt >= cutoff:
                    kept.append(e)
            self.events[t] = kept[:2000]  # 상한
            if not self.events[t]:
                self.events.pop(t, None)

    def pop_new_event_count(self, ticker: str) -> int:
        t = (ticker or "").upper().strip()
        n = int(self._new_event_counter.get(t, 0))
        self._new_event_counter[t] = 0
        return n

    def get_recent_events(
        self,
        ticker: str,
        *,
        days: int = 7,
        limit: int = 50,
        as_dict: bool = True,
    ) -> List[Any]:
        """
        ✅ 기본값 as_dict=True
        - main/ai_gate/decision 등에서 dict를 기대하는 코드와 호환되게 함.
        """
        t = (ticker or "").upper().strip()
        arr = self.events.get(t, []) or []
        cutoff = _now_kst() - timedelta(days=max(1, int(days)))

        out: List[NewsEvent] = []
        for e in arr:
            dt = _parse_ts_kst(e.ts_kst) or _parse_ts_kst(e.published)
            if dt is None or dt >= cutoff:
                out.append(e)
            if len(out) >= max(1, int(limit)):
                break

        if as_dict:
            return [x.to_dict() for x in out]
        return out

    def compute_signal(
        self,
        *,
        ticker: str,
        now_kst: datetime,
        half_life_hours: float = 24.0,
        window_days: int = 7,
        max_items: int = 200,
    ) -> Dict[str, Any]:
        """
        뉴스 이벤트 점수를 time-decay로 합산하여 불합리한 누적 편향을 제거한 news_score 반환
        """
        t = (ticker or "").upper().strip()
        arr = self.events.get(t, []) or []
        cutoff = now_kst - timedelta(days=max(1, int(window_days)))

        half_life = max(0.1, float(half_life_hours))
        lam = 0.69314718056 / half_life  # ln(2)/half_life

        decayed_sum = 0.0
        weight_sum = 0.0
        max_abs_score = 0.0
        max_score_sign = 1.0
        conf_sum = 0.0
        n = 0

        for e in arr[: max(1, int(max_items))]:
            dt = _parse_ts_kst(e.ts_kst) or _parse_ts_kst(e.published)
            if dt is not None and dt < cutoff:
                continue

            age_h = 0.0
            if dt is not None:
                age_h = max(0.0, (now_kst - dt).total_seconds() / 3600.0)

            decay = pow(2.718281828, -lam * age_h)
            score = float(e.event_score) * decay

            decayed_sum += score
            weight_sum += decay
            conf_sum += float(e.confidence)
            n += 1

            # 가장 강력했던(임팩트가 큰) 뉴스의 점수 추적
            if abs(score) > max_abs_score:
                max_abs_score = abs(score)
                max_score_sign = 1.0 if score >= 0 else -1.0

        conf = (conf_sum / n) if n > 0 else 0.55
        
        # 🚨 [수정] 단순 합산이 아닌 "가중 평균"과 "최대 임팩트"를 조합
        avg_score = (decayed_sum / weight_sum) if weight_sum > 0 else 0.0
        
        if n > 0:
            # 특급 뉴스의 비중을 70%, 전체 평균 분위기를 30%로 반영하여 공정성 확보
            news_score = (max_abs_score * max_score_sign * 0.7) + (avg_score * 0.3)
        else:
            news_score = 0.0

        news_score = max(-2.0, min(2.0, news_score))

        return {
            "ticker": t,
            "raw_sum": float(decayed_sum), # legacy 코드 호환용
            "raw_n": int(n),
            "news_score": float(news_score),
            "news_conf": float(conf),
        }

    # -------------------------
    # memory summaries
    # -------------------------
    def _mem_path(self, ticker: str) -> str:
        t = (ticker or "").upper().strip()
        return os.path.join(self.memory_dir, f"{t}.txt")

    def save_memory(self, ticker: str, summary: Any) -> None:
        """
        ✅ summary가 dict여도 저장 가능하게 처리
        - dict/list -> JSON string으로 저장
        - 그 외 -> str(summary)
        """
        t = (ticker or "").upper().strip()
        if not t:
            return

        if isinstance(summary, str):
            text = summary
        else:
            try:
                text = json.dumps(summary, ensure_ascii=False, indent=2)
            except Exception:
                text = str(summary)

        os.makedirs(self.memory_dir, exist_ok=True)
        with open(self._mem_path(t), "w", encoding="utf-8") as f:
            f.write(text)

    def load_memory(self, ticker: str) -> str:
        t = (ticker or "").upper().strip()
        p = self._mem_path(t)
        if not os.path.exists(p):
            return ""
        try:
            return open(p, "r", encoding="utf-8").read()
        except Exception:
            return ""