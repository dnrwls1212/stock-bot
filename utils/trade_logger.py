from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


@dataclass
class TradeLogger:
    """
    trades.jsonl: 한 줄 = 한 '체결(가상/실체결)' 기록
    """
    path: str = "data/trades.jsonl"
    enabled: bool = True

    def log(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        _ensure_dir(self.path)
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
