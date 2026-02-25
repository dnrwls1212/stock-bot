# src/trading/signals.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Signal:
    action: str           # BUY/SELL/HOLD
    strength: float       # 0~1 정도로 쓰는 값(너 기존 로직 유지)
    reason: str


def decide_signal(
    *,
    total_score: float,
    confidence: float,
    ta_label: str,
    buy_th: float,
    sell_th: float,
    conf_th: float,
) -> Signal:
    """
    - total_score가 buy_th 이상이면 BUY
    - total_score가 sell_th 이하이면 SELL
    - confidence가 conf_th 미만이면 HOLD(필터)
    - strength는 total_score의 크기 기반(0~1로 normalize)
    """
    c = _clamp(float(confidence), 0.0, 1.0)
    ts = float(total_score)

    # 기본: conf 필터
    if c < float(conf_th):
        strength = min(1.0, abs(ts))
        return Signal("HOLD", strength, f"conf<{conf_th:.2f} (conf={c:.2f})")

    # BUY/SELL
    if ts >= float(buy_th):
        strength = min(1.0, (ts - buy_th) / max(1e-9, (1.0 - buy_th)))
        return Signal("BUY", float(strength), f"ts>={buy_th:.2f} (ts={ts:.2f}) ta={ta_label}")

    if ts <= float(sell_th):
        # sell_th는 음수. ts가 더 내려갈수록 strength↑
        strength = min(1.0, (sell_th - ts) / max(1e-9, (sell_th - (-1.0))))
        return Signal("SELL", float(strength), f"ts<={sell_th:.2f} (ts={ts:.2f}) ta={ta_label}")

    strength = min(1.0, abs(ts))
    return Signal("HOLD", float(strength), f"between(th={sell_th:.2f}~{buy_th:.2f}) ts={ts:.2f} ta={ta_label}")