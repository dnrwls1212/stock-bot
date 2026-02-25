from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

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
    if s >= t3:
        return v3
    if s >= t2:
        return v2
    if s >= t1:
        return v1
    return default


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip() in ("1", "true", "True", "YES", "yes", "y")


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _mget(pos: PositionState, key: str, default: Any) -> Any:
    if not isinstance(pos.meta, dict):
        pos.meta = {}
    return pos.meta.get(key, default)


def _mset(pos: PositionState, key: str, value: Any) -> None:
    if not isinstance(pos.meta, dict):
        pos.meta = {}
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

    # soft stop / tp (legacy)
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
) -> PositionPlan:
    a = (raw_action or "HOLD").upper()
    s = _clip01(strength)
    px = float(price)

    qty_now = float(pos.qty)
    avg = float(pos.avg_price) if getattr(pos, "avg_price", 0.0) else 0.0

    # --- scalp tuning knobs (weak defaults) ---
    tp1_pct = _env_float("TP1_PCT", 0.0035)   # +0.35%
    tp2_pct = _env_float("TP2_PCT", 0.0065)   # +0.65%
    tp1_frac = _env_float("TP1_FRAC", 0.30)   # 30%
    tp2_frac = _env_float("TP2_FRAC", 0.30)   # +30%

    sl1_frac = _env_float("SL1_FRAC", 0.70)   # SL1에서 70% 축소
    sl1_use_env = _env_bool("SL1_USE_ENV", True)

    trail_enabled = _env_bool("TRAIL_ENABLED", True)
    trail_activate_pct = _env_float("TRAIL_ACTIVATE_PCT", 0.0045)  # +0.45% 이상일 때만
    trail_dd_pct = _env_float("TRAIL_DD_PCT", 0.0025)              # 고점대비 -0.25%
    trail_sell_frac = _env_float("TRAIL_SELL_FRAC", 1.00)          # 기본 전량

    strong_skip_tp2 = _env_bool("STRONG_SKIP_TP2", True)
    strong_tp2_mult = _env_float("STRONG_TP2_MULT", 1.50)

    # flags
    if qty_now <= 0:
        _mset(pos, "_tp1_done", False)
        _mset(pos, "_tp2_done", False)
        _mset(pos, "_sl1_done", False)
        _mset(pos, "_peak_price", 0.0)

    tp1_done = bool(_mget(pos, "_tp1_done", False))
    tp2_done = bool(_mget(pos, "_tp2_done", False))
    sl1_done = bool(_mget(pos, "_sl1_done", False))
    peak_price = float(_mget(pos, "_peak_price", 0.0) or 0.0)

    # (1) protect / take profit first
    if qty_now > 0 and avg > 0:
        pnl = (px - avg) / avg

        # update peak
        if peak_price <= 0:
            peak_price = px
            _mset(pos, "_peak_price", peak_price)
        elif px > peak_price:
            peak_price = px
            _mset(pos, "_peak_price", peak_price)

        # hard stop
        if pnl <= float(stop_loss_2):
            _mset(pos, "_tp1_done", False)
            _mset(pos, "_tp2_done", False)
            _mset(pos, "_sl1_done", False)
            _mset(pos, "_peak_price", 0.0)
            return PositionPlan("SELL", qty_now, f"STOP2 hit pnl={pnl:.4f} <= {stop_loss_2}")

        # SL1 one-shot
        if (not sl1_done) and pnl <= float(stop_loss_1):
            frac = float(sl1_frac) if sl1_use_env else float(stop_sell_frac)
            frac = _clip01(frac)
            q = min(qty_now, qty_now * frac)
            if q > 0:
                _mset(pos, "_sl1_done", True)
                return PositionPlan("SELL", q, f"SL1 cut pnl={pnl:.4f} <= {stop_loss_1} frac={frac:.2f} (one-shot)")

        # TP1 one-shot
        if (not tp1_done) and pnl >= float(tp1_pct):
            frac = _clip01(float(tp1_frac))
            q = min(qty_now, qty_now * frac)
            if q > 0:
                _mset(pos, "_tp1_done", True)
                return PositionPlan("SELL", q, f"TP1 hit pnl={pnl:.4f} >= {tp1_pct} frac={frac:.2f} (one-shot)")

        # TP2 one-shot
        if not tp2_done:
            tp2_eff = float(tp2_pct)
            if strong_skip_tp2 and (a == "BUY" and s >= float(fast_track_strength)):
                tp2_eff = float(tp2_pct) * float(strong_tp2_mult)

            if pnl >= tp2_eff:
                frac = _clip01(float(tp2_frac))
                q = min(qty_now, qty_now * frac)
                if q > 0:
                    _mset(pos, "_tp2_done", True)
                    return PositionPlan("SELL", q, f"TP2 hit pnl={pnl:.4f} >= {tp2_eff:.4f} frac={frac:.2f} (one-shot)")

        # trailing (profit zone only)
        if trail_enabled and peak_price > 0 and pnl >= float(trail_activate_pct):
            dd = (peak_price - px) / peak_price
            if dd >= float(trail_dd_pct):
                frac = _clip01(float(trail_sell_frac))
                q = min(qty_now, qty_now * frac)
                if q > 0:
                    return PositionPlan("SELL", q, f"TRAIL hit pnl={pnl:.4f} peak={peak_price:.2f} dd={dd:.4f} >= {trail_dd_pct:.4f} frac={frac:.2f}")

        # legacy TP fallback (rare)
        if pnl >= float(take_profit_1) and not (a == "BUY" and s >= float(fast_track_strength)):
            q = min(qty_now, qty_now * float(tp_sell_frac))
            if q > 0:
                return PositionPlan("SELL", q, f"LEGACY_TP1 pnl={pnl:.4f} >= {take_profit_1} frac={tp_sell_frac:.2f}")

    # (2) confirmation gate
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

    # (3) sizing
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