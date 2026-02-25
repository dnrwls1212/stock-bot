# src/report/performance_report.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def _ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir_for_file(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_action(a: Any) -> str:
    s = str(a or "").upper().strip()
    if s in ("BUY", "SELL", "HOLD"):
        return s
    return "HOLD"


def _ret_for_action(action: str, entry: float, future: float) -> float:
    """
    BUY: future/entry - 1
    SELL: entry/future - 1  (가격 하락이 수익)
    HOLD: 0
    """
    if entry <= 0 or future <= 0:
        return 0.0
    if action == "BUY":
        return (future / entry) - 1.0
    if action == "SELL":
        return (entry / future) - 1.0
    return 0.0


@dataclass
class ReportConfig:
    labeled_path: str = "data/decisions_labeled.jsonl"
    out_path: str = "data/perf_report.json"
    horizons: List[str] = None  # ["1h","1d"]
    min_abs_ret_success: float = 0.002  # 성공 기준(절대수익률 0.2% 이상 + ret>0)
    include_hold_in_counts: bool = False  # HOLD까지 win_rate 분모에 넣을지(기본 False)
    only_conf_ge: float = 0.0  # decision confidence 필터(0이면 무시)

    def __post_init__(self) -> None:
        if self.horizons is None:
            self.horizons = ["1h", "1d"]


class PerformanceReporter:
    """
    decisions_labeled.jsonl을 읽어서
    - Raw signal(action)
    - Plan(action)
    - Decision(action)
    성과를 horizon별로 비교 리포트 생성
    """

    POLICIES = ("raw_signal", "plan", "decision")

    def __init__(self, cfg: ReportConfig) -> None:
        self.cfg = cfg

    def build(self) -> Dict[str, Any]:
        rows = list(_read_jsonl(self.cfg.labeled_path))

        report: Dict[str, Any] = {
            "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
            "labeled_path": self.cfg.labeled_path,
            "horizons": self.cfg.horizons,
            "min_abs_ret_success": self.cfg.min_abs_ret_success,
            "only_conf_ge": self.cfg.only_conf_ge,
            "n_rows": len(rows),
            "by_horizon": {h: {} for h in self.cfg.horizons},
            "by_ticker_horizon": {},
        }

        # 집계 함수
        def init_stat() -> Dict[str, Any]:
            return {"n": 0, "n_eval": 0, "win_rate": 0.0, "avg_ret": 0.0}

        # ticker set
        tickers = set()
        for r in rows:
            t = (r.get("ticker") or "").upper().strip()
            if t:
                tickers.add(t)

        for t in sorted(tickers):
            report["by_ticker_horizon"][t] = {h: {p: init_stat() for p in self.POLICIES} for h in self.cfg.horizons}

        # horizon/정책별 리스트(평균 계산용)
        bucket: Dict[Tuple[str, str], List[float]] = {}  # (horizon, policy)->rets
        wins: Dict[Tuple[str, str], int] = {}
        evals: Dict[Tuple[str, str], int] = {}
        counts: Dict[Tuple[str, str], int] = {}

        def add(h: str, p: str, ret: float, success: bool, counted: bool) -> None:
            key = (h, p)
            counts[key] = counts.get(key, 0) + (1 if counted else 0)
            if counted:
                evals[key] = evals.get(key, 0) + 1
                bucket.setdefault(key, []).append(ret)
                wins[key] = wins.get(key, 0) + (1 if success else 0)

        # row loop
        for r in rows:
            ticker = (r.get("ticker") or "").upper().strip()
            decision = r.get("decision") or {}
            dconf = _safe_float(decision.get("confidence", 0.0), 0.0)
            if self.cfg.only_conf_ge > 0 and dconf < self.cfg.only_conf_ge:
                continue

            snap = r.get("snapshot") or {}
            raw_action = _normalize_action(((snap.get("signal") or {}).get("raw_action")))

            plan_action = _normalize_action(((snap.get("plan") or {}).get("action")))
            decision_action = _normalize_action((decision.get("action")))

            labels = r.get("labels") or {}
            for h in self.cfg.horizons:
                lh = labels.get(h) or {}
                if not lh.get("ok"):
                    continue

                entry = _safe_float(lh.get("entry_price"), 0.0)
                future = _safe_float(lh.get("future_price"), 0.0)
                if entry <= 0 or future <= 0:
                    continue

                for policy, action in (
                    ("raw_signal", raw_action),
                    ("plan", plan_action),
                    ("decision", decision_action),
                ):
                    ret = _ret_for_action(action, entry, future)
                    success = (abs(ret) >= self.cfg.min_abs_ret_success) and (ret > 0.0)

                    counted = True
                    if (not self.cfg.include_hold_in_counts) and action == "HOLD":
                        counted = False

                    add(h, policy, ret, success, counted)

                    # ticker stats
                    if ticker in report["by_ticker_horizon"]:
                        ts = report["by_ticker_horizon"][ticker][h][policy]
                        if counted:
                            ts["n"] += 1
                            ts["n_eval"] += 1
                            # 일단 누적(나중에 finalize에서 avg/win 계산)
                            ts.setdefault("_rets", []).append(ret)
                            ts.setdefault("_wins", 0)
                            ts["_wins"] += (1 if success else 0)

        # finalize overall
        for h in self.cfg.horizons:
            report["by_horizon"][h] = {p: init_stat() for p in self.POLICIES}
            for p in self.POLICIES:
                key = (h, p)
                n_eval = evals.get(key, 0)
                rets = bucket.get(key, [])
                w = wins.get(key, 0)
                stat = report["by_horizon"][h][p]
                stat["n"] = counts.get(key, 0)
                stat["n_eval"] = n_eval
                stat["avg_ret"] = float(sum(rets) / n_eval) if n_eval > 0 else 0.0
                stat["win_rate"] = float(w / n_eval) if n_eval > 0 else 0.0

        # finalize ticker stats
        for ticker, hm in report["by_ticker_horizon"].items():
            for h in self.cfg.horizons:
                for p in self.POLICIES:
                    ts = hm[h][p]
                    rets = ts.pop("_rets", [])
                    w = ts.pop("_wins", 0)
                    n_eval = int(ts.get("n_eval", 0) or 0)
                    ts["avg_ret"] = float(sum(rets) / n_eval) if n_eval > 0 else 0.0
                    ts["win_rate"] = float(w / n_eval) if n_eval > 0 else 0.0

        return report

    def save(self, report: Dict[str, Any]) -> None:
        _write_json(self.cfg.out_path, report)

    @staticmethod
    def format_telegram_summary(report: Dict[str, Any], horizon: str = "1h") -> str:
        """
        텔레그램 1~2줄 요약용
        """
        bh = (report.get("by_horizon") or {}).get(horizon) or {}
        if not bh:
            return f"[PERF] horizon={horizon} no data"

        def fmt(p: str) -> str:
            s = bh.get(p) or {}
            return f"{p}:n={s.get('n_eval',0)} win={s.get('win_rate',0):.2f} avg={s.get('avg_ret',0):.4f}"

        return (
            f"[PERF {horizon}] "
            + " | ".join([fmt("raw_signal"), fmt("plan"), fmt("decision")])
        )