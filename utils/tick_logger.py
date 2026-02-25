from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


@dataclass
class TickLogger:
    """
    JSONL(한 줄 = 한 tick record) 로 저장.
    - append-only
    - pandas로 분석하기 쉬움
    """
    path: str = "data/ticks.jsonl"
    enabled: bool = True

    def log(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        _ensure_dir(self.path)

        # 안정적으로 datetime/예외 타입 직렬화
        line = json.dumps(record, ensure_ascii=False, default=_json_default)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
