# src/eval/auto_labeler.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from zoneinfo import ZoneInfo

from src.eval.decision_labeler import DecisionLabeler, LabelConfig
from src.eval.decision_labeler import _read_jsonl, _append_jsonl  # reuse
from src.eval.decision_labeler import _write_json  # reuse

KST = ZoneInfo("Asia/Seoul")


def _ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


@dataclass
class AutoLabelSettings:
    input_path: str = "data/decisions.jsonl"
    labeled_path: str = "data/decisions_labeled.jsonl"
    summary_path: str = "data/decisions_summary.json"
    cursor_path: str = "data/label_cursor.json"
    cache_dir: str = "data/price_cache"

    horizons: List[str] = None  # ["1h","1d","3d","7d"]
    min_abs_ret: float = 0.002
    min_conf: float = 0.0  # 0이면 필터 없음

    every_minutes: int = 30
    telegram_enabled: bool = False

    def __post_init__(self) -> None:
        if self.horizons is None:
            self.horizons = ["1h", "1d", "3d", "7d"]


class DecisionAutoLabeler:
    """
    - main loop에서 주기적으로 호출
    - decisions.jsonl에서 '새로 추가된 라인만' 라벨링하여 decisions_labeled.jsonl에 append
    - cursor로 "몇 번째 라인까지 처리했는지" 기록
    """

    def __init__(self, settings: AutoLabelSettings) -> None:
        self.s = settings
        self.labeler = DecisionLabeler(cache_dir=self.s.cache_dir)
        self._last_run_kst: Optional[datetime] = None

        _ensure_dir_for_file(self.s.cursor_path)
        if not os.path.exists(self.s.cursor_path):
            self._save_cursor({"last_line": 0, "updated_at_kst": datetime.now(KST).isoformat(timespec="seconds")})

    def _load_cursor(self) -> Dict[str, Any]:
        try:
            with open(self.s.cursor_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                return {"last_line": 0}
            d.setdefault("last_line", 0)
            return d
        except Exception:
            return {"last_line": 0}

    def _save_cursor(self, d: Dict[str, Any]) -> None:
        _write_json(self.s.cursor_path, d)

    def _should_run(self, now_kst: datetime) -> bool:
        if self._last_run_kst is None:
            return True
        return (now_kst - self._last_run_kst) >= timedelta(minutes=max(1, int(self.s.every_minutes)))

    def run_if_due(self, now_kst: datetime) -> Dict[str, Any]:
        """
        returns: {"ran": bool, "n_new": int, "summary_updated": bool, "summary": dict|None}
        """
        if not self._should_run(now_kst):
            return {"ran": False, "n_new": 0, "summary_updated": False, "summary": None}

        self._last_run_kst = now_kst

        cur = self._load_cursor()
        last_line = int(cur.get("last_line", 0) or 0)

        if not os.path.exists(self.s.input_path):
            return {"ran": True, "n_new": 0, "summary_updated": False, "summary": None}

        # decisions.jsonl을 line 기준으로 증분 읽기
        new_rows: List[Dict[str, Any]] = []
        line_no = 0
        with open(self.s.input_path, "r", encoding="utf-8") as f:
            for line in f:
                line_no += 1
                if line_no <= last_line:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        new_rows.append(obj)
                except Exception:
                    continue

        if not new_rows:
            # cursor는 최신 라인으로 올려두기
            self._save_cursor({"last_line": line_no, "updated_at_kst": now_kst.isoformat(timespec="seconds")})
            return {"ran": True, "n_new": 0, "summary_updated": False, "summary": None}

        # new_rows만 임시 파일에 써서 labeler가 처리하도록
        tmp_in = "data/_tmp_new_decisions.jsonl"
        _ensure_dir_for_file(tmp_in)
        if os.path.exists(tmp_in):
            os.remove(tmp_in)
        for r in new_rows:
            _append_jsonl(tmp_in, r)

        cfg = LabelConfig(
            horizons=self.s.horizons,
            min_abs_ret_for_success=float(self.s.min_abs_ret),
            require_confidence=(float(self.s.min_conf) if self.s.min_conf > 0 else None),
        )

        # labeler는 overwrite=True면 파일을 지우니, 여기서는 tmp만 overwrite하고
        # labeled_path는 append 모드로 처리:
        # -> label_file 결과물을 tmp_labeled로 만들고, 그걸 labeled_path에 append
        tmp_labeled = "data/_tmp_labeled_out.jsonl"
        tmp_summary = "data/_tmp_summary.json"
        if os.path.exists(tmp_labeled):
            os.remove(tmp_labeled)

        stats = self.labeler.label_file(
            input_path=tmp_in,
            output_labeled_path=tmp_labeled,
            output_summary_path=tmp_summary,
            config=cfg,
            overwrite=True,
            max_rows=None,
        )

        # append labeled results
        if os.path.exists(tmp_labeled):
            with open(tmp_labeled, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        _append_jsonl(self.s.labeled_path, json.loads(line))

        # summary는 전체 기준이 아니라 'new only'라서, 여기서는 "전체 labeled 파일로 재요약"을 수행
        # -> 비용 줄이려면 every_minutes를 길게 잡으면 됨
        summary = self._rebuild_summary(cfg)

        # cursor 업데이트
        self._save_cursor({"last_line": line_no, "updated_at_kst": now_kst.isoformat(timespec="seconds")})

        # cleanup
        for p in (tmp_in, tmp_labeled, tmp_summary):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        return {"ran": True, "n_new": len(new_rows), "summary_updated": True, "summary": summary}

    def _rebuild_summary(self, cfg: LabelConfig) -> Dict[str, Any]:
        """
        labeled_path 전체를 기반으로 horizon별 통계 재생성 후 summary_path 저장
        """
        from src.eval.decision_labeler import _safe_float  # local import to avoid cycles

        labeled_rows = list(_read_jsonl(self.s.labeled_path))
        summary: Dict[str, Any] = {
            "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
            "input_labeled_path": self.s.labeled_path,
            "n_labeled": len(labeled_rows),
            "horizons": cfg.horizons,
            "by_horizon": {},
            "by_ticker_horizon": {},
        }

        for h in cfg.horizons:
            rets = []
            wins = 0
            n_eval = 0
            for r in labeled_rows:
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

            avg_ret = float(sum(rets) / n_eval) if n_eval > 0 else 0.0
            win_rate = float(wins / n_eval) if n_eval > 0 else 0.0
            summary["by_horizon"][h] = {"n_eval": n_eval, "avg_ret": avg_ret, "win_rate": win_rate}

        # per ticker
        for r in labeled_rows:
            ticker = (r.get("ticker") or "").upper().strip()
            if not ticker:
                continue
            summary["by_ticker_horizon"].setdefault(ticker, {})
        for ticker in summary["by_ticker_horizon"].keys():
            for h in cfg.horizons:
                rets = []
                wins = 0
                n_eval = 0
                for r in labeled_rows:
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
                    ret = float(lh.get("ret", 0.0))
                    rets.append(ret)
                    if bool(lh.get("success")):
                        wins += 1
                avg_ret = float(sum(rets) / n_eval) if n_eval > 0 else 0.0
                win_rate = float(wins / n_eval) if n_eval > 0 else 0.0
                summary["by_ticker_horizon"][ticker][h] = {"n_eval": n_eval, "avg_ret": avg_ret, "win_rate": win_rate}

        _write_json(self.s.summary_path, summary)
        return summary