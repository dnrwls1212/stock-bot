# src/report/auto_reporter.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from zoneinfo import ZoneInfo

from src.report.performance_report import PerformanceReporter, ReportConfig

KST = ZoneInfo("Asia/Seoul")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)).strip())
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)).strip())
    except Exception:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip() in ("1", "true", "True", "YES", "yes", "y")


@dataclass
class AutoReportSettings:
    labeled_path: str = "data/decisions_labeled.jsonl"
    report_path: str = "data/perf_report.json"

    horizons: str = "1h,1d"  # 단기 개발 기본
    min_abs_ret: float = 0.002
    only_conf_ge: float = 0.0
    include_hold: bool = False

    every_minutes: int = 30
    telegram_enabled: bool = True

    telegram_horizon: str = "1h"  # 요약은 1h로 빠르게
    also_send_1d: bool = True


class AutoReporter:
    """
    main loop에서 주기적으로 호출.
    또는 labeler가 summary_updated 되었을 때만 호출해도 됨.
    """

    def __init__(self, s: AutoReportSettings) -> None:
        self.s = s
        self._last_run_kst: Optional[datetime] = None

    def _should_run(self, now_kst: datetime) -> bool:
        if self._last_run_kst is None:
            return True
        return (now_kst - self._last_run_kst) >= timedelta(minutes=max(1, int(self.s.every_minutes)))

    def run_if_due(self, now_kst: datetime) -> Dict[str, Any]:
        if not self._should_run(now_kst):
            return {"ran": False, "report": None}

        self._last_run_kst = now_kst
        horizons = [h.strip() for h in (self.s.horizons or "").split(",") if h.strip()]

        cfg = ReportConfig(
            labeled_path=self.s.labeled_path,
            out_path=self.s.report_path,
            horizons=horizons,
            min_abs_ret_success=float(self.s.min_abs_ret),
            include_hold_in_counts=bool(self.s.include_hold),
            only_conf_ge=float(self.s.only_conf_ge),
        )
        rep = PerformanceReporter(cfg)
        report = rep.build()
        rep.save(report)

        return {"ran": True, "report": report}

    @staticmethod
    def from_env() -> "AutoReporter":
        s = AutoReportSettings(
            labeled_path=os.environ.get("DECISION_LABELED_PATH", "data/decisions_labeled.jsonl"),
            report_path=os.environ.get("PERF_REPORT_PATH", "data/perf_report.json"),
            horizons=os.environ.get("PERF_HORIZONS", "1h,1d"),
            min_abs_ret=_env_float("PERF_MIN_ABS_RET", 0.002),
            only_conf_ge=_env_float("PERF_ONLY_CONF_GE", 0.0),
            include_hold=_env_bool("PERF_INCLUDE_HOLD", False),
            every_minutes=_env_int("PERF_EVERY_MIN", 30),
            telegram_enabled=_env_bool("PERF_TELEGRAM", True),
            telegram_horizon=os.environ.get("PERF_TELEGRAM_HORIZON", "1h").strip() or "1h",
            also_send_1d=_env_bool("PERF_TELEGRAM_ALSO_1D", True),
        )
        return AutoReporter(s)