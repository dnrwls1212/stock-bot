# src/eval/decision_labeler.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from zoneinfo import ZoneInfo

from .price_history import (
    PriceHistoryCache,
    decide_interval_for_horizon,
    horizon_to_timedelta,
    parse_kst_iso_to_utc,
)

KST = ZoneInfo("Asia/Seoul")
UTC = ZoneInfo("UTC")


def _ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


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


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


def _action_sign(action: str) -> int:
    a = (action or "").upper().strip()
    if a == "BUY":
        return +1
    if a == "SELL":
        return -1
    return 0


def _compute_return(action: str, entry: float, future: float) -> float:
    """
    BUY: (future/entry - 1)
    SELL: (entry/future - 1)  (즉 가격이 내려가면 수익 +)
    HOLD: 0 (평가 제외 가능)
    """
    if entry <= 0 or future <= 0:
        return 0.0
    a = (action or "").upper().strip()
    if a == "BUY":
        return (future / entry) - 1.0
    if a == "SELL":
        return (entry / future) - 1.0
    return 0.0


@dataclass
class LabelConfig:
    horizons: List[str]
    min_abs_ret_for_success: float = 0.002  # 0.2% 이상이면 성공으로 보는 기본(너 스타일에 맞게 조절)
    require_confidence: Optional[float] = None  # 예: 0.6 이상 decision만 평가


class DecisionLabeler:
    """
    decisions.jsonl을 읽고,
    각 decision 시점(entry)과 horizon 후의 가격(future)을 조회해 라벨링한다.

    출력:
      - decisions_labeled.jsonl: 각 row에 labels dict 추가
      - decisions_summary.json: horizon별/티커별 성과 요약
    """

    def __init__(self, *, cache_dir: str = "data/price_cache") -> None:
        self.cache = PriceHistoryCache(cache_dir=cache_dir)

    def label_file(
        self,
        *,
        input_path: str,
        output_labeled_path: str,
        output_summary_path: str,
        config: LabelConfig,
        overwrite: bool = True,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        if overwrite and os.path.exists(output_labeled_path):
            os.remove(output_labeled_path)

        rows = list(_read_jsonl(input_path))
        if max_rows is not None:
            rows = rows[: int(max_rows)]

        labeled_rows: List[Dict[str, Any]] = []
        # 요약용 축적
        stats: Dict[str, Any] = {
            "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
            "input_path": input_path,
            "n_total": len(rows),
            "horizons": config.horizons,
            "by_horizon": {h: {"n": 0, "n_eval": 0, "avg_ret": 0.0, "win_rate": 0.0} for h in config.horizons},
            "by_ticker_horizon": {},  # ticker -> horizon -> stats
        }

        # 히스토리 조회 효율 위해: (ticker, interval)별 최소/최대 범위 계산 후 한번에 fetch
        # 하지만 horizon/interval이 다를 수 있어 2단계로 구현:
        # 1) row들을 horizon별로 그룹 -> 해당 horizon에 필요한 interval로 ticker별 fetch 범위
        grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}  # horizon -> ticker -> rows
        for r in rows:
            ticker = (r.get("ticker") or "").upper().strip()
            decision = r.get("decision") or {}
            action = str(decision.get("action", "HOLD")).upper()
            conf = _safe_float(decision.get("confidence", 0.5), 0.5)

            if config.require_confidence is not None and conf < config.require_confidence:
                continue
            if action not in ("BUY", "SELL", "HOLD"):
                continue

            # entry_ts: snapshot.ts_kst 우선, 없으면 r["ts"]
            snap = r.get("snapshot") or {}
            entry_ts = snap.get("ts_kst") or r.get("ts")
            if not entry_ts:
                continue

            for h in config.horizons:
                grouped.setdefault(h, {}).setdefault(ticker, []).append(r)

        # horizon별로 fetch
        price_maps: Dict[Tuple[str, str], pd.DataFrame] = {}  # (ticker, horizon)->df
        for h in config.horizons:
            interval = decide_interval_for_horizon(h)
            delta = horizon_to_timedelta(h)

            for ticker, rs in grouped.get(h, {}).items():
                # range 계산
                times_utc = []
                for r in rs:
                    snap = r.get("snapshot") or {}
                    entry_ts = snap.get("ts_kst") or r.get("ts")
                    try:
                        entry_utc = parse_kst_iso_to_utc(str(entry_ts))
                    except Exception:
                        continue
                    times_utc.append(entry_utc)
                    times_utc.append(entry_utc + delta)

                if not times_utc:
                    continue

                start_utc = min(times_utc) - timedelta(hours=2)
                end_utc = max(times_utc) + timedelta(hours=2)

                df = self.cache.fetch_history(
                    ticker,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    interval=interval,
                )
                price_maps[(ticker, h)] = df

        # 라벨링
        for r in rows:
            ticker = (r.get("ticker") or "").upper().strip()
            decision = r.get("decision") or {}
            action = str(decision.get("action", "HOLD")).upper()
            conf = _safe_float(decision.get("confidence", 0.5), 0.5)

            if config.require_confidence is not None and conf < config.require_confidence:
                continue

            snap = r.get("snapshot") or {}
            entry_ts = snap.get("ts_kst") or r.get("ts")
            if not entry_ts:
                continue

            # entry price: snapshot.price를 우선 사용. 없으면 yfinance에서 entry_ts 이후 첫 bar로 잡기
            entry_price = snap.get("price")
            entry_price_f = _safe_float(entry_price, 0.0)

            try:
                entry_utc = parse_kst_iso_to_utc(str(entry_ts))
            except Exception:
                continue

            labels: Dict[str, Any] = {}

            for h in config.horizons:
                df = price_maps.get((ticker, h))
                if df is None or df.empty:
                    labels[h] = {"ok": False, "reason": "no_price_history"}
                    continue

                # entry price fallback if snapshot missing
                if entry_price_f <= 0:
                    p0 = self.cache.pick_price_at_or_after(df, entry_utc)
                    if p0 is None:
                        labels[h] = {"ok": False, "reason": "no_entry_price"}
                        continue
                    entry_price_f = float(p0.price)

                future_utc = entry_utc + horizon_to_timedelta(h)
                p1 = self.cache.pick_price_at_or_after(df, future_utc)
                if p1 is None:
                    labels[h] = {"ok": False, "reason": "no_future_price"}
                    continue

                ret = _compute_return(action, entry_price_f, float(p1.price))
                success = (abs(ret) >= float(config.min_abs_ret_for_success)) and (ret > 0)

                labels[h] = {
                    "ok": True,
                    "entry_ts_utc": entry_utc.isoformat(),
                    "entry_price": float(entry_price_f),
                    "future_ts_utc": p1.ts_utc.isoformat(),
                    "future_price": float(p1.price),
                    "ret": float(ret),
                    "success": bool(success),
                }

                # stats update (HOLD 제외하고 평가)
                if action in ("BUY", "SELL"):
                    stats["by_horizon"][h]["n"] += 1
                    if labels[h]["ok"]:
                        stats["by_horizon"][h]["n_eval"] += 1

                        # ticker/horizon stats
                        th = stats["by_ticker_horizon"].setdefault(ticker, {}).setdefault(
                            h, {"n": 0, "n_eval": 0, "avg_ret": 0.0, "win_rate": 0.0}
                        )
                        th["n"] += 1
                        th["n_eval"] += 1

            # 라벨이 하나라도 있으면 저장
            if labels:
                out = dict(r)
                out["labels"] = labels
                labeled_rows.append(out)
                _append_jsonl(output_labeled_path, out)

        # stats finalize
        def finalize_stat(s: Dict[str, Any], items: List[float], wins: int, n_eval: int) -> None:
            if n_eval <= 0:
                s["avg_ret"] = 0.0
                s["win_rate"] = 0.0
            else:
                s["avg_ret"] = float(sum(items) / n_eval)
                s["win_rate"] = float(wins / n_eval)

        # compute per horizon aggregates by scanning labeled file (simple & reliable)
        labeled_scan = list(_read_jsonl(output_labeled_path))
        for h in config.horizons:
            rets = []
            wins = 0
            n_eval = 0
            for r in labeled_scan:
                decision = r.get("decision") or {}
                action = str(decision.get("action", "HOLD")).upper()
                if action not in ("BUY", "SELL"):
                    continue
                lh = (r.get("labels") or {}).get(h) or {}
                if not lh.get("ok"):
                    continue
                n_eval += 1
                ret = _safe_float(lh.get("ret"), 0.0)
                rets.append(ret)
                if bool(lh.get("success")):
                    wins += 1
            stats["by_horizon"][h]["n_eval"] = n_eval
            finalize_stat(stats["by_horizon"][h], rets, wins, n_eval)

        # per ticker/horizon
        for ticker, hm in stats["by_ticker_horizon"].items():
            for h in config.horizons:
                rets = []
                wins = 0
                n_eval = 0
                for r in labeled_scan:
                    if (r.get("ticker") or "").upper().strip() != ticker:
                        continue
                    decision = r.get("decision") or {}
                    action = str(decision.get("action", "HOLD")).upper()
                    if action not in ("BUY", "SELL"):
                        continue
                    lh = (r.get("labels") or {}).get(h) or {}
                    if not lh.get("ok"):
                        continue
                    n_eval += 1
                    ret = _safe_float(lh.get("ret"), 0.0)
                    rets.append(ret)
                    if bool(lh.get("success")):
                        wins += 1
                s = hm.setdefault(h, {"n": 0, "n_eval": 0, "avg_ret": 0.0, "win_rate": 0.0})
                s["n_eval"] = n_eval
                finalize_stat(s, rets, wins, n_eval)

        _write_json(output_summary_path, stats)
        return stats