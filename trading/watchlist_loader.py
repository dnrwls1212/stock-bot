# src/trading/watchlist_loader.py
from __future__ import annotations

import os
from typing import List


def _read_lines(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            t = s.split(",")[0].strip().upper()
            if t and t not in out:
                out.append(t)
    return out


def load_watchlist() -> List[str]:
    """
    우선순위:
    1) WATCHLIST_AUTO_PATH (기본: data/watchlist_auto.txt)
    2) WATCHLIST_PATH      (기본: data/watchlist.txt)
    3) 기본값 NVDA/ORCL/AVGO/AMD
    """
    auto_path = os.environ.get("WATCHLIST_AUTO_PATH", "data/watchlist_auto.txt")
    base_path = os.environ.get("WATCHLIST_PATH", "data/watchlist.txt")

    wl = _read_lines(auto_path)
    if wl:
        return wl

    wl = _read_lines(base_path)
    if wl:
        return wl

    return ["NVDA", "ORCL", "AVGO", "AMD"]