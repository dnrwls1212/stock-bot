# src/trading/scalp_signal.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional, Tuple
import math


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class ScalpOut:
    score: float        # [-1, 1]  (buy +, sell -)
    label: str          # "scalp_mr_buy" / "scalp_mr_sell" / "scalp_hold"
    reason: str
    ready: bool         # warmup 완료 여부


class ScalpSignalEngine:
    """
    초단기 '평균회귀(mean-reversion)' 스캘프 신호:
    - 최근 N틱 가격으로 EMA(추정) + 표준편차 기반으로
      "너무 위면 SELL, 너무 아래면 BUY" 형태의 스코어를 만든다.
    - total_score에 섞어서 모의 단타 거래를 자주 발생시키는 목적.
    """

    def __init__(
        self,
        window: int = 60,
        warmup: int = 25,
        k: float = 0.9,
        min_move_pct: float = 0.0006,
        cooldown_ticks: int = 2,
    ) -> None:
        self.window = max(10, int(window))
        self.warmup = max(5, int(warmup))
        self.k = float(k)
        self.min_move_pct = float(min_move_pct)
        self.cooldown_ticks = max(0, int(cooldown_ticks))

        self._prices: Dict[str, Deque[float]] = {}
        self._last_px: Dict[str, float] = {}
        self._cooldown_left: Dict[str, int] = {}

    def update(self, ticker: str, price: float) -> ScalpOut:
        t = ticker.upper().strip()
        px = float(price)

        dq = self._prices.get(t)
        if dq is None:
            dq = deque(maxlen=self.window)
            self._prices[t] = dq

        # cooldown tick down
        if t not in self._cooldown_left:
            self._cooldown_left[t] = 0
        if self._cooldown_left[t] > 0:
            self._cooldown_left[t] -= 1

        # min movement gate (노이즈 억제)
        last = self._last_px.get(t)
        self._last_px[t] = px
        if last is not None:
            mov = abs(px - last) / max(1e-9, last)
            if mov < self.min_move_pct:
                dq.append(px)
                ready = len(dq) >= self.warmup
                return ScalpOut(
                    score=0.0,
                    label="scalp_hold",
                    reason=f"min_move<{self.min_move_pct:.4f} (mov={mov:.4f})",
                    ready=ready,
                )

        dq.append(px)

        # warmup
        if len(dq) < self.warmup:
            return ScalpOut(
                score=0.0,
                label="scalp_hold",
                reason=f"warmup {len(dq)}/{self.warmup}",
                ready=False,
            )

        # basic stats
        arr = list(dq)
        mean = sum(arr) / len(arr)

        # 표준편차 (population)
        var = sum((x - mean) ** 2 for x in arr) / max(1, len(arr))
        std = math.sqrt(var)
        if std <= 1e-9:
            return ScalpOut(
                score=0.0,
                label="scalp_hold",
                reason="std~0",
                ready=True,
            )

        # deviation (z)
        z = (px - mean) / std

        # mean reversion: 위로 과열(z+)이면 sell(-), 아래로 과매도(z-)면 buy(+)
        # tanh로 [-1,1]로 압축 + k로 민감도 조절
        raw = -math.tanh(z / max(1e-6, self.k))
        score = _clamp(raw, -1.0, 1.0)

        # 너무 자주 뒤집는 걸 막기 위한 cooldown
        if self._cooldown_left[t] > 0:
            return ScalpOut(
                score=0.0,
                label="scalp_hold",
                reason=f"cooldown({self._cooldown_left[t]}) z={z:.2f}",
                ready=True,
            )

        label = "scalp_hold"
        if score >= 0.15:
            label = "scalp_mr_buy"
            self._cooldown_left[t] = self.cooldown_ticks
        elif score <= -0.15:
            label = "scalp_mr_sell"
            self._cooldown_left[t] = self.cooldown_ticks

        return ScalpOut(
            score=float(score),
            label=label,
            reason=f"mr z={z:.2f} mean={mean:.2f} std={std:.4f} px={px:.2f}",
            ready=True,
        )

    def get_stats(self, ticker: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns (mean, std, last_price, pct_from_mean).
        None if not ready.
        """
        t = ticker.upper().strip()
        dq = self._prices.get(t)
        if not dq or len(dq) < self.warmup:
            return None
        arr = list(dq)
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / max(1, len(arr))
        std = math.sqrt(var)
        last = arr[-1]
        pct = (last - mean) / max(1e-9, mean)
        return (float(mean), float(std), float(last), float(pct))