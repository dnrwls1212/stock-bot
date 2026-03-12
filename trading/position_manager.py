from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# 👇 [수정] PositionInfo가 아니라 원래 있던 PositionState로 올바르게 임포트
from .position_store import PositionState

@dataclass(frozen=True)
class PositionPlan:
    action: str          # BUY | SELL | HOLD
    qty: float           # 주문 수량
    reason: str

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _tier_value(s: float, t1: float, t2: float, t3: float, v1: float, v2: float, v3: float, default: float):
    s = _clip01(s)
    if s >= t3: return v3
    if s >= t2: return v2
    if s >= t1: return v1
    return default

def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None: return default
    return v.strip() in ("1", "true", "True", "YES", "yes", "y")

def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None: return default
    try: return float(str(v).strip())
    except Exception: return default

def _mget(pos: PositionState, key: str, default: Any) -> Any:
    if not isinstance(pos.meta, dict): pos.meta = {}
    return pos.meta.get(key, default)

def _mset(pos: PositionState, key: str, value: Any) -> None:
    if not isinstance(pos.meta, dict): pos.meta = {}
    pos.meta[key] = value

def compute_position_plan(
    *,
    pos: PositionState,
    raw_action: str,
    strength: float,
    price: float,

    # confirmation
    confirm_ticks: int,
    fast_track_strength: float,

    # soft stop / tp 
    stop_loss_1: float,
    stop_loss_2: float,
    take_profit_1: float,
    stop_sell_frac: float,
    tp_sell_frac: float,

    # sizing / caps
    base_qty: float,
    max_position_qty: float,

    # buy tiers
    buy_t1: float, buy_t2: float, buy_t3: float,
    buy_m1: float, buy_m2: float, buy_m3: float,

    # sell tiers
    sell_t1: float, sell_t2: float, sell_t3: float,
    sell_f1: float, sell_f2: float, sell_f3: float,
    
    # 👇 [추가된 부분] D-Day 타이트 방어용 오버라이드 인자
    trail_dd_pct_override: Optional[float] = None,
    trail_activate_pct_override: Optional[float] = None,
) -> PositionPlan:
    a = (raw_action or "HOLD").upper()
    s = _clip01(strength)
    px = float(price)

    qty_now = float(pos.qty)
    avg = float(pos.avg_price) if getattr(pos, "avg_price", 0.0) else 0.0

    tp1_pct = _env_float("TP1_PCT", float(take_profit_1) if take_profit_1 else 0.05)
    tp2_pct = _env_float("TP2_PCT", 0.10)
    tp1_frac = _env_float("TP1_FRAC", float(tp_sell_frac) if tp_sell_frac else 0.50)
    tp2_frac = _env_float("TP2_FRAC", 1.0)  

    sl1_frac = _env_float("SL1_FRAC", float(stop_sell_frac) if stop_sell_frac else 0.50)
    
    trail_enabled = _env_bool("TRAIL_ENABLED", True)
    
    # 👇 [수정된 부분] 오버라이드 값이 있으면 최우선 적용 (D-Day 방어용)
    trail_activate_pct = trail_activate_pct_override if trail_activate_pct_override is not None else _env_float("TRAIL_ACTIVATE_PCT", 0.050)
    trail_dd_pct = trail_dd_pct_override if trail_dd_pct_override is not None else _env_float("TRAIL_DD_PCT", 0.030)
    trail_sell_frac = _env_float("TRAIL_SELL_FRAC", 1.00)

    # 포지션이 없으면 과거 상태 초기화
    if qty_now <= 0:
        _mset(pos, "_tp1_done", False)
        _mset(pos, "_tp2_done", False)
        _mset(pos, "_sl1_done", False)
        _mset(pos, "_peak_price", 0.0)

    tp1_done = bool(_mget(pos, "_tp1_done", False))
    tp2_done = bool(_mget(pos, "_tp2_done", False))
    sl1_done = bool(_mget(pos, "_sl1_done", False))
    peak_price = float(_mget(pos, "_peak_price", 0.0) or 0.0)

    # (1) Protect / Take profit (가장 최우선 순위)
    if qty_now > 0 and avg > 0:
        pnl = (px - avg) / avg

        # 최고점 기록 갱신 (트레일링 스탑용)
        if peak_price <= 0:
            peak_price = px
            _mset(pos, "_peak_price", peak_price)
        elif px > peak_price:
            peak_price = px
            _mset(pos, "_peak_price", peak_price)

        # 1. 하드 스탑 (완전 폭락, 전량 손절)
        if pnl <= float(stop_loss_2):
            _mset(pos, "_tp1_done", False)
            _mset(pos, "_tp2_done", False)
            _mset(pos, "_sl1_done", False)
            _mset(pos, "_peak_price", 0.0)
            return PositionPlan("SELL", qty_now, f"STOP2 hit pnl={pnl:.4f} <= {stop_loss_2}")

        # 2. 1차 손절 (비중 덜어내기)
        if (not sl1_done) and pnl <= float(stop_loss_1):
            frac = _clip01(float(sl1_frac))
            q = min(qty_now, qty_now * frac)
            if q > 0:
                _mset(pos, "_sl1_done", True)
                return PositionPlan("SELL", q, f"SL1 cut pnl={pnl:.4f} <= {stop_loss_1} frac={frac:.2f} (one-shot)")

        # 3. 2차 익절 (최종 목표가 도달)
        if (not tp2_done) and pnl >= float(tp2_pct):
            frac = _clip01(float(tp2_frac))
            q = min(qty_now, qty_now * frac)
            if q > 0:
                _mset(pos, "_tp2_done", True)
                return PositionPlan("SELL", q, f"TP2 hit pnl={pnl:.4f} >= {tp2_pct:.4f} frac={frac:.2f} (one-shot)")

        # 4. 1차 익절 (5% 도달 시 안전하게 반익절 등)
        if (not tp1_done) and pnl >= float(tp1_pct):
            frac = _clip01(float(tp1_frac))
            q = min(qty_now, qty_now * frac)
            if q > 0:
                _mset(pos, "_tp1_done", True)
                return PositionPlan("SELL", q, f"TP1 hit pnl={pnl:.4f} >= {tp1_pct:.4f} frac={frac:.2f} (one-shot)")

        # 5. 트레일링 스탑 (수익 구간에서 추세가 꺾일 때 도망치기)
        if trail_enabled and peak_price > 0 and pnl >= float(trail_activate_pct):
            dd = (peak_price - px) / peak_price
            if dd >= float(trail_dd_pct):
                frac = _clip01(float(trail_sell_frac))
                q = min(qty_now, qty_now * frac)
                if q > 0:
                    return PositionPlan("SELL", q, f"TRAIL hit pnl={pnl:.4f} peak={peak_price:.2f} dd={dd:.4f} >= {trail_dd_pct:.4f} frac={frac:.2f}")

    # (2) Confirmation gate (일반 매매 신호 검증)
    if a in ("BUY", "SELL") and s >= float(fast_track_strength):
        confirmed_action = a
        conf_reason = f"fast-track strength={s:.2f} >= {fast_track_strength}"
    else:
        ct = max(1, int(confirm_ticks))
        if a == "BUY":
            confirmed_action = "BUY" if pos.buy_streak >= ct else "HOLD"
            conf_reason = f"confirm BUY streak={pos.buy_streak}/{ct}"
        elif a == "SELL":
            confirmed_action = "SELL" if pos.sell_streak >= ct else "HOLD"
            conf_reason = f"confirm SELL streak={pos.sell_streak}/{ct}"
        else:
            confirmed_action = "HOLD"
            conf_reason = "raw HOLD"

    # (3) Sizing (수량 조절)
    if confirmed_action == "BUY":
        cap = max(0.0, float(max_position_qty) - qty_now)
        if cap <= 0:
            return PositionPlan("HOLD", 0.0, f"{conf_reason} | at max position cap")

        mult = _tier_value(s, buy_t1, buy_t2, buy_t3, buy_m1, buy_m2, buy_m3, 0.0)
        q = min(cap, float(base_qty) * float(mult))
        if q <= 0:
            return PositionPlan("HOLD", 0.0, f"{conf_reason} | BUY too weak for sizing")
        return PositionPlan("BUY", q, f"{conf_reason} | BUY sized base={base_qty}*{mult} cap={cap:.2f}")

    if confirmed_action == "SELL":
        if qty_now <= 0:
            return PositionPlan("HOLD", 0.0, f"{conf_reason} | no position")
        frac = _tier_value(s, sell_t1, sell_t2, sell_t3, sell_f1, sell_f2, sell_f3, 0.0)
        q = min(qty_now, qty_now * float(frac))
        if q <= 0:
            return PositionPlan("HOLD", 0.0, f"{conf_reason} | SELL too weak for sizing")
        return PositionPlan("SELL", q, f"{conf_reason} | SELL sized qty={qty_now:.2f}*{frac}")

    return PositionPlan("HOLD", 0.0, f"{conf_reason} | final HOLD")