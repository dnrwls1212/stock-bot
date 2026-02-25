# src/utils/dedupe.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Set

DEFAULT_SEEN_PATH = os.path.join("data", "seen_links.txt")


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_seen(path: str = DEFAULT_SEEN_PATH) -> Set[str]:
    """
    간단한 dedupe store:
    - 파일 한 줄에 링크 하나
    - 실행 중엔 set으로 보관
    """
    p = Path(path)
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def mark_seen(link: str, path: str = DEFAULT_SEEN_PATH) -> None:
    """
    link를 seen 파일에 append.
    (in-memory set 업데이트는 main에서 처리)
    """
    link = (link or "").strip()
    if not link:
        return
    _ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(link + "\n")
