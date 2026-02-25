# src/utils/market_hours.py
from __future__ import annotations

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo


def is_us_regular_market_open(now_kst: datetime) -> bool:
    """
    KST -> ET 변환 후
    미국 정규장 09:30~16:00 ET (월~금) 여부 체크
    - 휴장일(미국 공휴일)은 여기서는 반영 안 함 (추후 캘린더로 고도화 가능)
    """
    et = now_kst.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    t = et.time()
    return (t >= dtime(9, 30)) and (t <= dtime(16, 0))