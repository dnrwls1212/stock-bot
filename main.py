# src/main.py
from __future__ import annotations

import os
import time
import json
import signal
import math
from datetime import datetime, timedelta, time as dtime
from typing import Any, Dict, List, Optional, Set

from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from collections import deque

# rss / news
from src.sources.rss_news import build_rss_urls, fetch_rss_news
from src.analysis import news_filter
from src.analysis.local_llm_event import analyze_news_local_ollama

# scoring / valuation / TA
from src.strategy.scoring import event_score
from src.valuation.market_data import fetch_snapshot
from src.valuation.fair_value import compute_fair_value_snapshot
from src.ta.indicators import fetch_daily_ta, ta_score

# dedupe
from src.utils.dedupe import load_seen, mark_seen

# telegram
from src.notify.telegram import send_telegram_message
from src.notify.telegram_format import (
    fmt_news,
    fmt_start,
    fmt_order_submitted,
    fmt_dry_run,
    fmt_label_summary,
    fmt_perf_summary,
)

# KIS
from src.broker.kis_client import KisClient
from src.market.kis_us_provider import KisUsQuoteProvider
from src.broker.kis_us_broker import KisUsBroker

# trading engine
from src.trading.signals import decide_signal
from src.trading.position_store import load_state, save_state, get_position
from src.trading.risk import can_trade, AccountRiskManager
from src.trading.position_manager import compute_position_plan

# order/filled management
from src.trading.order_store import PendingOrderStore
from src.trading.order_manager import OrderManager
from src.trading.trade_limits import TradeLimitStore

# market hours
from src.utils.market_hours import is_us_regular_market_open

# trade logging
from src.utils.trade_logger import TradeLogger

# news memory
from src.trading.news_store import NewsStore, NewsEvent
from src.analysis.news_memory import build_news_memory_summary_local_ollama

# AI gate
from src.trading.ai_gate import ai_gate_check_local_ollama

from src.eval.auto_labeler import DecisionAutoLabeler, AutoLabelSettings
from src.report.auto_reporter import AutoReporter
from src.report.performance_report import PerformanceReporter

from src.trading.dynamic_news_rules import DynamicNewsRules
from src.trading.news_risk_override import NewsRiskOverride

# Ollama unified client
from src.utils.ollama_client import ollama_generate, try_parse_json

# Scalp + Regime
from src.trading.scalp_signal import ScalpSignalEngine
from src.trading.regime_filter import RegimeFilter

# Watchlist auto
from src.trading.watchlist_auto import build_watchlist_v1
from src.trading.watchlist_loader import load_watchlist

# ✅ Universe builder
from src.trading.universe_builder import build_universe

load_dotenv()

# ✅ WATCHLIST는 main() 시작 후 자동선정/로드 결과로 확정
WATCHLIST: List[str] = []

LOCK_PATH = "data/run.lock"
EVENT_LOG_PATH = "data/events.jsonl"
DECISION_LOG_PATH = os.environ.get("DECISION_LOG_PATH", "data/decisions.jsonl")

POS_PATH = os.environ.get("POS_PATH", "data/positions.json")
PENDING_PATH = os.environ.get("PENDING_PATH", "data/pending_orders.json")


# -----------------------------
# env helpers
# -----------------------------
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


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


# -----------------------------
# lock helpers
# -----------------------------
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def takeover_lock() -> None:
    os.makedirs("data", exist_ok=True)

    old_pid: Optional[int] = None
    if os.path.exists(LOCK_PATH):
        try:
            old_pid = int(open(LOCK_PATH, "r", encoding="utf-8").read().strip())
        except Exception:
            old_pid = None

    if old_pid and old_pid != os.getpid() and _pid_alive(old_pid):
        try:
            os.kill(old_pid, signal.SIGTERM)
        except Exception:
            pass

        for _ in range(15):  # ~3s
            if not _pid_alive(old_pid):
                break
            time.sleep(0.2)

        if _pid_alive(old_pid):
            try:
                os.kill(old_pid, signal.SIGKILL)
            except Exception:
                pass

    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def release_lock() -> None:
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except Exception:
        pass


# -----------------------------
# notifier
# -----------------------------
class Notifier:
    def __init__(self) -> None:
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    def send(self, text: str) -> None:
        try:
            send_telegram_message(self.bot_token, self.chat_id, text)
        except Exception as e:
            print(f"[WARN] telegram send failed: {e}")


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------------
# news helpers
# -----------------------------
def _mentions_ticker(ticker: str, title: str, summary: str) -> bool:
    t = (ticker or "").upper().strip()
    text = f"{title}\n{summary}".lower()

    hints = getattr(news_filter, "COMPANY_HINTS", {})
    keys = hints.get(t)

    if not keys:
        return t.lower() in text

    return any(k in text for k in keys)


def _candidate_tickers(title: str, summary: str, watchlist: List[str]) -> List[str]:
    out: List[str] = []
    for t in watchlist:
        if _mentions_ticker(t, title, summary):
            out.append(t)
    return out


def _is_high_signal(title: str, summary: str) -> bool:
    return bool(news_filter.is_high_signal(title, summary))


# -----------------------------
# math helpers
# -----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _total_score(news_s: float, val_s: float, ta_s: float) -> float:
    w_news = _env_float("W_NEWS", 0.55)
    w_val = _env_float("W_VAL", 0.55)
    w_daily = _env_float("W_DAILY", 0.60)
    denom = (w_news + w_val + w_daily) or 1.0
    return (w_news * news_s + w_val * val_s + w_daily * ta_s) / denom


def _to_int_qty(x: float) -> int:
    try:
        q = int(math.floor(float(x)))
        return max(0, q)
    except Exception:
        return 0


def _us_session_avoid_window(now_kst: datetime, start_min: int, end_min: int) -> bool:
    """
    미국 정규장(09:30~16:00 ET) 기준으로
    - 장 시작 후 start_min 분
    - 장 마감 전 end_min 분
    회피 구간이면 True.
    """
    try:
        ny = now_kst.astimezone(ZoneInfo("America/New_York"))
        t = ny.time()

        open_t = dtime(9, 30)
        close_t = dtime(16, 0)

        if not (open_t <= t <= close_t):
            return False

        cur_min = t.hour * 60 + t.minute
        open_min = open_t.hour * 60 + open_t.minute
        close_min = close_t.hour * 60 + close_t.minute

        if cur_min < open_min + max(0, int(start_min)):
            return True
        if cur_min > close_min - max(0, int(end_min)):
            return True
        return False
    except Exception:
        return False


# -----------------------------
# ✅ KB (Ticker Knowledge Base) - main.py inline implementation
# -----------------------------
def _now_kst_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def _kb_dir() -> str:
    return os.environ.get("KB_DIR", "data/kb")


def _kb_path(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    return os.path.join(_kb_dir(), f"{t}.json")


def kb_load(ticker: str) -> Dict[str, Any]:
    os.makedirs(_kb_dir(), exist_ok=True)
    p = _kb_path(ticker)
    if not os.path.exists(p):
        return {
            "ticker": (ticker or "").upper().strip(),
            "updated_at": _now_kst_iso(),
            "thesis": "",
            "business_summary": "",
            "moat": "",
            "key_drivers": [],
            "key_risks": [],
            "valuation_method": "simple",
            "valuation_assumptions": {},
            "target_price": None,
            "fair_value_range": None,
            "evidence": [],
            "decisions": [],
            "tags": [],
        }
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError("kb not dict")
        d.setdefault("ticker", (ticker or "").upper().strip())
        d.setdefault("updated_at", _now_kst_iso())
        d.setdefault("thesis", "")
        d.setdefault("business_summary", "")
        d.setdefault("moat", "")
        d.setdefault("key_drivers", [])
        d.setdefault("key_risks", [])
        d.setdefault("valuation_method", "simple")
        d.setdefault("valuation_assumptions", {})
        d.setdefault("target_price", None)
        d.setdefault("fair_value_range", None)
        d.setdefault("evidence", [])
        d.setdefault("decisions", [])
        d.setdefault("tags", [])
        return d
    except Exception:
        return {
            "ticker": (ticker or "").upper().strip(),
            "updated_at": _now_kst_iso(),
            "thesis": "",
            "business_summary": "",
            "moat": "",
            "key_drivers": [],
            "key_risks": [],
            "valuation_method": "simple",
            "valuation_assumptions": {},
            "target_price": None,
            "fair_value_range": None,
            "evidence": [],
            "decisions": [],
            "tags": [],
        }


def kb_save(kb: Dict[str, Any]) -> None:
    os.makedirs(_kb_dir(), exist_ok=True)
    kb["updated_at"] = _now_kst_iso()
    t = (kb.get("ticker") or "").upper().strip()
    p = _kb_path(t)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def kb_add_evidence(
    ticker: str,
    *,
    source: str,
    title: str,
    summary: str,
    link: str = "",
    sentiment: str = "neutral",
    impact: int = 0,
    tags: Optional[List[str]] = None,
    raw: Optional[Dict[str, Any]] = None,
    max_items: int = 400,
) -> None:
    kb = kb_load(ticker)
    ev = kb.get("evidence", [])
    if not isinstance(ev, list):
        ev = []
    ev.insert(
        0,
        {
            "ts_kst": _now_kst_iso(),
            "source": source,
            "title": title,
            "summary": summary,
            "link": link,
            "sentiment": sentiment,
            "impact": int(impact),
            "tags": tags or [],
            "raw": raw or {},
        },
    )
    kb["evidence"] = ev[:max_items]
    kb_save(kb)


def kb_add_decision(
    ticker: str,
    *,
    action: str,
    confidence: float,
    rationale: str,
    key_drivers: Optional[List[str]] = None,
    key_risks: Optional[List[str]] = None,
    valuation_view: str = "",
    counterfactuals: Optional[List[str]] = None,
    next_checks: Optional[List[str]] = None,
    raw: Optional[Dict[str, Any]] = None,
    max_items: int = 200,
) -> None:
    kb = kb_load(ticker)
    decs = kb.get("decisions", [])
    if not isinstance(decs, list):
        decs = []
    decs.insert(
        0,
        {
            "ts_kst": _now_kst_iso(),
            "action": action,
            "confidence": float(confidence),
            "rationale": rationale,
            "key_drivers": key_drivers or [],
            "key_risks": key_risks or [],
            "valuation_view": valuation_view,
            "counterfactuals": counterfactuals or [],
            "next_checks": next_checks or [],
            "raw": raw or {},
        },
    )
    kb["decisions"] = decs[:max_items]
    kb_save(kb)


# -----------------------------
# ✅ Decision Agent (Ollama JSON)
# -----------------------------
def build_decision_prompt(
    *,
    ticker: str,
    kb: Dict[str, Any],
    snapshot: Dict[str, Any],
    recent_news_events: List[Dict[str, Any]],
) -> str:
    kb_light = {
        "thesis": kb.get("thesis", ""),
        "business_summary": kb.get("business_summary", ""),
        "moat": kb.get("moat", ""),
        "key_drivers": kb.get("key_drivers", []) or [],
        "key_risks": kb.get("key_risks", []) or [],
        "valuation_method": kb.get("valuation_method", "simple"),
        "valuation_assumptions": kb.get("valuation_assumptions", {}) or {},
        "target_price": kb.get("target_price", None),
        "fair_value_range": kb.get("fair_value_range", None),
        "tags": kb.get("tags", []) or [],
        "recent_decisions": (kb.get("decisions", []) or [])[:5],
        "recent_evidence": (kb.get("evidence", []) or [])[:12],
    }

    schema = {
        "action": "BUY|SELL|HOLD",
        "confidence": 0.0,
        "rationale": "Korean, concise, factual",
        "key_drivers": ["..."],
        "key_risks": ["..."],
        "valuation_view": "what assumption matters / changed",
        "counterfactuals": ["what would make you wrong next 1-4 weeks"],
        "next_checks": ["what to check next"],
        "position_plan": {"prefer_qty": 0, "time_horizon": "swing_days|swing_weeks|long_months"},
    }

    return f"""
너는 '누적 지식 기반' 투자 의사결정 에이전트다.
목표: 단기/스윙 수익을 내되, 기업 방향성과 근거 축적을 최우선으로 한다.
규칙:
- 제공된 KB/스냅샷/최근 이벤트만 근거로 사용. 추측 금지.
- 모르면 모른다고 말하고 next_checks에 확인할 항목을 적는다.
- 출력은 반드시 JSON 하나만. 다른 문장/설명/코드블럭 금지.
- action은 BUY/SELL/HOLD 중 하나.

[TICKER]
{ticker}

[KB]
{json.dumps(kb_light, ensure_ascii=False)}

[SNAPSHOT]
{json.dumps(snapshot, ensure_ascii=False)}

[RECENT_NEWS_EVENTS]
{json.dumps(recent_news_events[:10], ensure_ascii=False)}

[OUTPUT_JSON_SCHEMA]
{json.dumps(schema, ensure_ascii=False)}
""".strip()


def parse_decision(text: str) -> Dict[str, Any]:
    d = try_parse_json(text) or {}
    if not isinstance(d, dict):
        d = {}

    d.setdefault("action", "HOLD")
    d.setdefault("confidence", 0.5)
    d.setdefault("rationale", "")
    d.setdefault("key_drivers", [])
    d.setdefault("key_risks", [])
    d.setdefault("valuation_view", "")
    d.setdefault("counterfactuals", [])
    d.setdefault("next_checks", [])
    d.setdefault("position_plan", {"prefer_qty": 0, "time_horizon": "swing_days"})

    try:
        d["action"] = str(d.get("action", "HOLD")).upper().strip()
        if d["action"] not in ("BUY", "SELL", "HOLD"):
            d["action"] = "HOLD"
    except Exception:
        d["action"] = "HOLD"

    try:
        d["confidence"] = float(d.get("confidence", 0.5))
        d["confidence"] = max(0.0, min(1.0, d["confidence"]))
    except Exception:
        d["confidence"] = 0.5

    return d


# -----------------------------
# ✅ Universe + watchlist auto (on start / refresh)
# -----------------------------
def _maybe_build_universe_and_watchlist_once(notifier: Notifier) -> Optional[List[str]]:
    """
    - UNIVERSE_AUTO_ON_START=1 이면 universe.txt 생성/갱신
    - WATCHLIST_AUTO_ON_START=1 이면 universe.txt 기반으로 watchlist 자동선정
    - build_universe / build_watchlist_v1 이 dict 또는 dataclass를 반환해도 모두 호환되게 처리
    """
    universe_on = _env_bool("UNIVERSE_AUTO_ON_START", True)
    watch_on = _env_bool("WATCHLIST_AUTO_ON_START", False)

    universe_core_path = _env_str("UNIVERSE_CORE_PATH", "data/universe_core.txt")
    universe_path = _env_str("UNIVERSE_PATH", "data/universe.txt")

    # -----------------------
    # 1) Universe build
    # -----------------------
    if universe_on:
        try:
            # ✅ universe_builder.py의 현재 시그니처 기준 파라미터로만 전달
            # (core_path/out_path/period/min_price/min_dollar_vol/target_n/max_universe)
            u = build_universe(
                out_path=universe_path,
                core_path=universe_core_path,
                period=_env_str("UNIVERSE_PERIOD", "3mo"),
                min_price=_env_float("UNIVERSE_MIN_PRICE", 5.0),
                min_dollar_vol=_env_float("UNIVERSE_MIN_DVOL", 5_000_000.0),
                target_n=_env_int("UNIVERSE_TARGET_N", 200),
                max_universe=_env_int("UNIVERSE_MAX", 1200),
                # 아래는 "옛 main"에서 넘기던 events_path/trades_path 같은 키워드가 와도
                # universe_builder에서 **_ignored**로 흡수하도록 설계(호환성)
            )

            # ✅ dict / dataclass 모두 호환
            ok = False
            reason = "unknown"
            outp = universe_path
            n_valid = None
            n_input = None

            if isinstance(u, dict):
                ok = bool(u.get("ok", False))
                reason = str(u.get("reason", "unknown"))
                outp = str(u.get("out_path", universe_path) or universe_path)
                n_valid = u.get("n_valid", None)
                n_input = u.get("n_input", None)
            else:
                ok = bool(getattr(u, "ok", False))
                reason = str(getattr(u, "reason", "unknown"))
                outp = str(getattr(u, "out_path", universe_path) or universe_path)
                n_valid = getattr(u, "n_valid", None)
                n_input = getattr(u, "n_input", None)

            if ok:
                msg = f"🌍 [Universe 생성 완료]\n- out: {outp}"
                if n_valid is not None:
                    msg += f"\n- valid: {n_valid}"
                if n_input is not None:
                    msg += f"\n- input: {n_input}"
                notifier.send(msg)
                print(f"[UNIVERSE] ok out={outp} n_valid={n_valid} n_input={n_input}")
            else:
                notifier.send(f"⚠️ [Universe 생성 실패]\n- reason: {reason}\n- core: {universe_core_path}")
                print(f"[UNIVERSE] failed reason={reason}")

        except Exception as e:
            notifier.send(f"⚠️ [Universe 생성 에러]\nerr={e!r}")
            print(f"[UNIVERSE] error {e!r}")

    if not watch_on:
        return None

    # -----------------------
    # 2) Watchlist auto build
    # -----------------------
    out_path = _env_str("WATCHLIST_AUTO_PATH", "data/watchlist_auto.txt")
    top_n = _env_int("WATCHLIST_TOP_N", 12)

    try:
        # ✅ watchlist_auto.py의 build_watchlist_v1 시그니처에 맞춰 전달
        # (universe_path/out_path/top_n/min_price/min_dollar_vol/news_lookback_hours 등)
        r = build_watchlist_v1(
            universe_path=universe_path,
            out_path=out_path,
            top_n=top_n,
            min_price=_env_float("WATCHLIST_MIN_PRICE", 5.0),
            min_dollar_vol=_env_float("WATCHLIST_MIN_DVOL", 5_000_000.0),
            news_lookback_hours=_env_int("WATCHLIST_NEWS_H", 24),
            w_atr=_env_float("WATCHLIST_W_ATR", 0.45),
            w_liq=_env_float("WATCHLIST_W_LIQ", 0.35),
            w_news=_env_float("WATCHLIST_W_NEWS", 0.20),
            period=_env_str("WATCHLIST_PERIOD", "3mo"),
            interval=_env_str("WATCHLIST_INTERVAL", "1d"),
        )

        ok = False
        reason = "unknown"
        picked: List[str] = []
        outp = out_path

        if isinstance(r, dict):
            ok = bool(r.get("ok", False))
            reason = str(r.get("reason", "unknown"))
            picked = list(r.get("picked", []) or [])
            outp = str(r.get("out_path", out_path) or out_path)
        else:
            ok = bool(getattr(r, "ok", False))
            reason = str(getattr(r, "reason", "unknown"))
            picked = list(getattr(r, "picked", []) or [])
            outp = str(getattr(r, "out_path", out_path) or out_path)

        # ✅ ETF 포함해도 OK (너가 요청)
        if ok and picked:
            notifier.send(f"📌 [자동 종목선정 완료]\n선정: {', '.join(picked)}\n저장: {outp}")
            print(f"[WATCHLIST_AUTO] ok picked={picked} out={outp}")
            return picked

        notifier.send(f"⚠️ [자동 종목선정 실패] 기존/기본 WATCHLIST 사용\nreason={reason}\nuniverse={universe_path}")
        print(f"[WATCHLIST_AUTO] failed reason={reason} universe={universe_path}")
        return None

    except Exception as e:
        notifier.send(f"⚠️ [자동 종목선정 에러] 기존/기본 WATCHLIST 사용\nerr={e!r}")
        print(f"[WATCHLIST_AUTO] error {e!r}")
        return None


# -----------------------------
# main
# -----------------------------
def main() -> None:
    global WATCHLIST

    takeover_lock()

    notifier = Notifier()
    trade_logger = TradeLogger(path=os.environ.get("TRADES_PATH", "data/trades.jsonl"), enabled=True)

    # ✅ 시작 시 universe+watchlist 자동 갱신(옵션)
    _maybe_build_universe_and_watchlist_once(notifier)

    # ✅ 최종 watchlist 로드 (auto 파일 우선)
    WATCHLIST = load_watchlist()
    print(f"[WATCHLIST] loaded: {WATCHLIST}")

    tick_seconds = _env_int("TICK_SECONDS", 60)

    # trading params
    cooldown_minutes = _env_int("COOLDOWN_MINUTES", 15)
    max_trades_per_day = _env_int("MAX_TRADES_PER_DAY", 6)
    max_position_qty = _env_int("MAX_POSITION_QTY", 10)
    base_qty = _env_int("BASE_QTY", 1)

    confirm_ticks = _env_int("CONFIRM_TICKS", 2)
    fast_track_strength = _env_float("FAST_TRACK_STRENGTH", 0.90)

    stop_loss_1 = _env_float("STOP_LOSS_1", -0.03)
    stop_loss_2 = _env_float("STOP_LOSS_2", -0.06)
    take_profit_1 = _env_float("TAKE_PROFIT_1", 0.04)
    stop_sell_frac = _env_float("STOP_SELL_FRAC", 0.50)
    tp_sell_frac = _env_float("TP_SELL_FRAC", 0.33)

    buy_th = _env_float("BUY_TH", 0.70)
    sell_th = _env_float("SELL_TH", -0.70)
    conf_th = _env_float("CONF_TH", 0.55)

    # strength tiers
    buy_t1 = _env_float("BUY_T1", 0.15)
    buy_t2 = _env_float("BUY_T2", 0.40)
    buy_t3 = _env_float("BUY_T3", 0.70)
    buy_m1 = _env_float("BUY_M1", 1.0)
    buy_m2 = _env_float("BUY_M2", 2.0)
    buy_m3 = _env_float("BUY_M3", 3.0)

    sell_t1 = _env_float("SELL_T1", 0.20)
    sell_t2 = _env_float("SELL_T2", 0.50)
    sell_t3 = _env_float("SELL_T3", 0.80)
    sell_f1 = _env_float("SELL_F1", 0.25)
    sell_f2 = _env_float("SELL_F2", 0.50)
    sell_f3 = _env_float("SELL_F3", 1.00)

    allow_outside_market = _env_bool("ALLOW_OUTSIDE_MARKET", False)

    # ✅ PAPER SCALP MODE (모의 단타/기능테스트용)
    paper_scalp_mode = _env_bool("PAPER_SCALP_MODE", False)
    scalp_enabled = _env_bool("SCALP_ENABLED", False) or paper_scalp_mode

    scalp_weight = _env_float("SCALP_WEIGHT", 0.55)
    scalp_engine: Optional[ScalpSignalEngine] = None
    if scalp_enabled:
        scalp_engine = ScalpSignalEngine(
            window=_env_int("SCALP_WINDOW", 60),
            warmup=_env_int("SCALP_WARMUP", 25),
            k=_env_float("SCALP_K", 0.90),
            min_move_pct=_env_float("SCALP_MIN_MOV_PCT", 0.0006),
            cooldown_ticks=_env_int("SCALP_COOLDOWN_TICKS", 2),
        )

    # 모의 단타 프리셋(옵션)
    if paper_scalp_mode:
        tick_seconds = min(tick_seconds, 10)
        cooldown_minutes = min(cooldown_minutes, 1)
        max_trades_per_day = max(max_trades_per_day, 200)
        confirm_ticks = 1
        conf_th = min(conf_th, 0.0)

        buy_th = min(buy_th, 0.12)
        sell_th = max(sell_th, -0.12)

        stop_loss_1 = max(stop_loss_1, -0.004)  # -0.4%
        stop_loss_2 = max(stop_loss_2, -0.010)  # -1.0%
        take_profit_1 = max(take_profit_1, 0.006)  # +0.6%

    # =====================
    # ✅ Profit-oriented guards (Regime / Chase / Cost)
    # =====================
    chase_ban_enabled = _env_bool("CHASE_BAN_ENABLED", False)
    chase_ban_pct = _env_float("CHASE_BAN_PCT", 0.0025)
    chase_ban_z = _env_float("CHASE_BAN_Z", 1.20)
    chase_ban_after_spike_pct = _env_float("CHASE_BAN_AFTER_SPIKE_PCT", 0.006)
    chase_ban_spike_window = _env_int("CHASE_BAN_SPIKE_WINDOW", 12)
    _recent_px: Dict[str, deque] = {}

    cost_gate_enabled = _env_bool("COST_GATE_ENABLED", False)
    cost_fee_bps = _env_float("COST_FEE_BPS", 1.0)
    cost_spread_bps = _env_float("COST_SPREAD_BPS", 3.0)
    cost_slip_bps = _env_float("COST_SLIPPAGE_BPS", 2.0)
    edge_per_score = _env_float("EDGE_PER_SCORE", 0.008)
    edge_min_mult = _env_float("EDGE_MIN_MULT", 1.1)

    # Regime (옵션)
    regime_enabled = _env_bool("REGIME_ENABLED", False)
    regime_symbol = _env_str("REGIME_SYMBOL", "QQQ")
    regime_risk_off = _env_float("REGIME_RISK_OFF", -0.20)
    regime_buy_block = _env_bool("REGIME_BUY_BLOCK", True)
    regime_th_mult = _env_float("REGIME_TH_MULT", 1.35)
    regime_scalp_w_mult = _env_float("REGIME_SCALP_WEIGHT_MULT", 0.70)
    regime_size_mult = _env_float("REGIME_SIZE_MULT", 0.60)

    regime_engine: Optional[RegimeFilter] = None
    if regime_enabled:
        regime_engine = RegimeFilter(regime_symbol)

    # =====================
    # ✅ Light guards (weak mode): A/B/C
    # =====================
    session_guard_enabled = _env_bool("SESSION_GUARD_ENABLED", True)
    session_guard_start_min = _env_int("SESSION_GUARD_START_MIN", 3)
    session_guard_end_min = _env_int("SESSION_GUARD_END_MIN", 3)

    whipsaw_guard_enabled = _env_bool("WHIPSAW_GUARD_ENABLED", True)
    whipsaw_cooldown_ticks = _env_int("WHIPSAW_COOLDOWN_TICKS", 1)

    # ✅ watchlist 기반 상태 dict 초기화 (반드시 WATCHLIST 확정 후)
    _whipsaw_last_action: Dict[str, str] = {t: "HOLD" for t in WATCHLIST}
    _whipsaw_last_tick: Dict[str, int] = {t: -10**9 for t in WATCHLIST}
    _global_tick = 0

    news_dup_penalty_enabled = _env_bool("NEWS_DUP_PENALTY_ENABLED", True)
    news_dup_mult = _env_float("NEWS_DUP_MULT", 0.70)

    # AI gate env
    ai_gate_enabled = _env_bool("AI_GATE_ENABLED", True)
    ai_gate_model = os.environ.get("AI_GATE_MODEL", "qwen2.5:7b-instruct")
    ai_gate_min_conf = _env_float("AI_GATE_MIN_CONF", 0.55)

    # Decision Agent env
    decision_enabled = _env_bool("DECISION_AGENT_ENABLED", True)
    decision_model = os.environ.get("DECISION_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    decision_min_conf = _env_float("DECISION_MIN_CONF", 0.60)
    decision_every_ticks = _env_int("DECISION_EVERY_TICKS", 1)
    decision_compare_only = not _env_bool("DECISION_OVERRIDE_TRADING", False)
    decision_tick_counter: Dict[str, int] = {t: 0 for t in WATCHLIST}

    # watchlist 자동 갱신(옵션)
    watchlist_refresh_min = _env_int("WATCHLIST_AUTO_REFRESH_MIN", 0)
    next_watchlist_refresh: Optional[datetime] = None
    if watchlist_refresh_min > 0:
        next_watchlist_refresh = datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(minutes=max(1, watchlist_refresh_min))

    # dedupe
    seen: Set[str] = load_seen()

    # rss urls
    rss_urls = build_rss_urls(WATCHLIST)

    # positions
    positions = load_state(POS_PATH)

    # pending orders + order manager
    pending_store = PendingOrderStore(PENDING_PATH)
    order_mgr = OrderManager(
        store=pending_store,
        position_state=positions,
        position_state_path=POS_PATH,
        ccnl_lookback_minutes=_env_int("CCNL_LOOKBACK_MIN", 15),
        stale_order_seconds=_env_int("STALE_ORDER_SEC", 180),
    )

    # NewsStore + memory
    NEWS_STORE_PATH = os.environ.get("NEWS_STORE_PATH", "data/news_store.json")
    NEWS_MEMORY_DIR = os.environ.get("NEWS_MEMORY_DIR", "data/news_memory")
    news_store = NewsStore(NEWS_STORE_PATH, memory_dir=NEWS_MEMORY_DIR)

    news_half_life_h = _env_float("NEWS_HALF_LIFE_H", 24.0)
    news_window_days = _env_int("NEWS_WINDOW_DAYS", 7)
    news_mem_max_items = _env_int("NEWS_MEM_MAX_ITEMS", 20)
    news_mem_update_every = _env_int("NEWS_MEM_UPDATE_EVERY", 3)
    news_mem_model = os.environ.get("NEWS_MEM_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct"))

    dyn_rules = DynamicNewsRules()
    risk_override = NewsRiskOverride()

    # per-ticker limits
    LIMITS_PATH = os.environ.get("LIMITS_PATH", "data/limits.json")
    limit_store = TradeLimitStore(LIMITS_PATH)
    ticker_cooldown_sec = _env_int("TICKER_COOLDOWN_SEC", 300)
    max_orders_per_ticker_day = _env_int("MAX_ORDERS_PER_TICKER_DAY", 2)

    # Account-level risk manager (P0-2)
    acc_risk = AccountRiskManager.from_env()

    # KIS
    kis_enabled = _env_bool("KIS_ENABLED", True)
    quote_provider: Optional[KisUsQuoteProvider] = None
    broker: Optional[KisUsBroker] = None

    if kis_enabled:
        try:
            kis = KisClient.from_env()
            quote_provider = KisUsQuoteProvider(kis)
            broker = KisUsBroker(kis)
            notifier.send(f"[INIT] KIS enabled (paper={'1' if kis.cfg.paper else '0'})")
        except Exception as e:
            quote_provider = None
            broker = None
            notifier.send(f"[INIT] KIS disabled (init error): {e}")

    notifier.send(
        fmt_start(
            watchlist=WATCHLIST,
            tick_seconds=tick_seconds,
            execute_orders=os.environ.get("EXECUTE_ORDERS", "0"),
            ai_gate_enabled=ai_gate_enabled,
            decision_enabled=decision_enabled,
            decision_override=(not decision_compare_only),
        )
    )

    auto_labeler = DecisionAutoLabeler(
        AutoLabelSettings(
            input_path=DECISION_LOG_PATH,
            labeled_path=os.environ.get("DECISION_LABELED_PATH", "data/decisions_labeled.jsonl"),
            summary_path=os.environ.get("DECISION_SUMMARY_PATH", "data/decisions_summary.json"),
            cursor_path=os.environ.get("LABEL_CURSOR_PATH", "data/label_cursor.json"),
            cache_dir=os.environ.get("PRICE_CACHE_DIR", "data/price_cache"),
            horizons=[h.strip() for h in os.environ.get("LABEL_HORIZONS", "1h,1d,3d,7d").split(",") if h.strip()],
            min_abs_ret=float(os.environ.get("LABEL_MIN_ABS_RET", "0.002")),
            min_conf=float(os.environ.get("LABEL_MIN_CONF", "0.0")),
            every_minutes=int(os.environ.get("LABEL_EVERY_MIN", "30")),
            telegram_enabled=bool(int(os.environ.get("LABEL_TELEGRAM", "0"))),
        )
    )

    auto_reporter = AutoReporter.from_env()

    pos_sync_on_start = _env_bool("POS_SYNC_ON_START", True)
    pos_sync_every_ticks = _env_int("POS_SYNC_EVERY_TICKS", 5)
    _pos_sync_tick = 0

    try:
        while True:
            now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
            _global_tick += 1

            loop_ts = now_kst.isoformat(timespec="seconds")
            market_open = is_us_regular_market_open(now_kst)

            # ✅ watchlist 자동 갱신(옵션): universe → watchlist 순으로 갱신
            if watchlist_refresh_min > 0 and next_watchlist_refresh is not None and now_kst >= next_watchlist_refresh:
                _maybe_build_universe_and_watchlist_once(notifier)
                WATCHLIST = load_watchlist()

                # watchlist 바뀌면 관련 상태도 갱신
                rss_urls = build_rss_urls(WATCHLIST)
                decision_tick_counter = {t: 0 for t in WATCHLIST}
                _whipsaw_last_action = {t: "HOLD" for t in WATCHLIST}
                _whipsaw_last_tick = {t: -10**9 for t in WATCHLIST}

                notifier.send(f"🔄 [watchlist 갱신] {', '.join(WATCHLIST)}")
                print(f"[WATCHLIST] refreshed: {WATCHLIST}")
                next_watchlist_refresh = now_kst + timedelta(minutes=watchlist_refresh_min)

            # ✅ 0) 체결/미체결 동기화
            if broker is not None:
                try:
                    order_mgr.sync_once(broker)
                except Exception as e:
                    print(f"[ORDER_SYNC_ERR] {e!r}")

            # ✅ Account risk snapshot refresh (tick당 1회)
            if broker is not None:
                try:
                    acc_risk.refresh(broker, now_kst)
                except Exception as e:
                    print(f"[ACC_RISK_REFRESH_ERR] {e!r}")

            # ✅ broker truth 기준 포지션 동기화 (qty/avg_price 교정)
            if kis_enabled and broker is not None:
                _pos_sync_tick += 1
                if (pos_sync_on_start and _pos_sync_tick == 1) or (
                    pos_sync_every_ticks > 0 and _pos_sync_tick % pos_sync_every_ticks == 0
                ):
                    order_mgr.sync_positions_from_broker(broker)

            # ✅ 장 마감이면 pending stale 자동 정리
            if not market_open:
                try:
                    removed = pending_store.purge_stale_open(max_age_sec=_env_int("PENDING_PURGE_SEC", 7200))
                    if removed > 0:
                        print(f"[ORDER_CLEANUP] removed {removed} stale open orders")
                except Exception as e:
                    print(f"[ORDER_CLEANUP_ERR] {e!r}")

            # ---------- 1) RSS 뉴스 수집 ----------
            try:
                news_items = fetch_rss_news(rss_urls=rss_urls)
            except Exception as e:
                notifier.send(f"[ERR] fetch_rss_news failed: {e}")
                news_items = []

            analyzed_links = 0
            skipped_seen = 0
            skipped_low_signal = 0
            skipped_no_candidate = 0
            llm_fail = 0

            for item in news_items:
                link = (item.get("link") or "").strip()
                title = (item.get("title") or "").strip()
                summary = (item.get("summary") or "").strip()
                published = (item.get("published") or "").strip()

                if not link:
                    continue
                if link in seen:
                    skipped_seen += 1
                    continue

                if not _is_high_signal(title, summary):
                    skipped_low_signal += 1
                    continue

                candidates = _candidate_tickers(title, summary, WATCHLIST)
                if not candidates:
                    skipped_no_candidate += 1
                    continue

                try:
                    evt = analyze_news_local_ollama(
                        title=title,
                        summary=summary,
                        link=link,
                        published=published,
                        watchlist=WATCHLIST,
                    )
                except Exception:
                    llm_fail += 1
                    continue

                try:
                    escore = float(event_score(evt))
                except Exception:
                    escore = 0.0

                try:
                    econf = float(evt.get("confidence", 0.55))
                except Exception:
                    econf = 0.55

                llm_tickers = evt.get("tickers") if isinstance(evt, dict) else None
                if isinstance(llm_tickers, list):
                    assigned = [t for t in llm_tickers if str(t).upper() in WATCHLIST]
                else:
                    assigned = []

                if not assigned:
                    assigned = candidates

                analyzed_links += 1

                _append_jsonl(
                    EVENT_LOG_PATH,
                    {
                        "ts": loop_ts,
                        "link": link,
                        "title": title,
                        "published": published,
                        "assigned": assigned,
                        "event_type": evt.get("event_type"),
                        "sentiment": evt.get("sentiment"),
                        "impact": evt.get("impact"),
                        "confidence": evt.get("confidence"),
                        "trade_horizon": evt.get("trade_horizon"),
                        "why_it_moves": evt.get("why_it_moves"),
                        "event_score": escore,
                    },
                )

                notifier.send(
                    fmt_news(
                        tickers=",".join(assigned),
                        title=title,
                        score=float(escore),
                        event_type=str(evt.get("event_type", "") or ""),
                        sentiment=str(evt.get("sentiment", "") or ""),
                        conf=float(evt.get("confidence", 0.0) or 0.0),
                        link=link,
                    )
                )

                for t in assigned:
                    try:
                        news_store.add_event(
                            NewsEvent(
                                ticker=t,
                                ts_kst=loop_ts,
                                published=published,
                                title=title,
                                summary=summary,
                                link=link,
                                event_score=float(escore),
                                confidence=float(econf),
                                event_type=str(evt.get("event_type", "other")),
                                sentiment=str(evt.get("sentiment", "neutral")),
                                impact=int(evt.get("impact", 0) or 0),
                                why_it_moves=str(evt.get("why_it_moves", "") or ""),
                                raw=evt if isinstance(evt, dict) else {},
                            )
                        )
                    except Exception as e:
                        print(f"[NEWS_STORE_ERR] {t} add_event failed: {e!r}")

                    try:
                        kb_add_evidence(
                            t,
                            source="news",
                            title=title,
                            summary=str(evt.get("why_it_moves", "") or summary),
                            link=link,
                            sentiment=str(evt.get("sentiment", "neutral")),
                            impact=int(evt.get("impact", 0) or 0),
                            tags=[str(evt.get("event_type", "other"))],
                            raw=evt if isinstance(evt, dict) else {},
                        )
                    except Exception as e:
                        print(f"[KB_EVIDENCE_ERR] {t}: {e!r}")

                seen.add(link)
                try:
                    mark_seen(link)
                except Exception:
                    pass

            # ✅ ticker별 memory 갱신
            for t in WATCHLIST:
                try:
                    new_n = news_store.pop_new_event_count(t)
                    if new_n >= news_mem_update_every:
                        recent = news_store.get_recent_events(t, days=news_window_days, limit=news_mem_max_items)
                        if recent:
                            mem = build_news_memory_summary_local_ollama(
                                ticker=t,
                                events=recent,
                                model=news_mem_model,
                            )
                            news_store.save_memory(t, mem)
                            print(f"[NEWS_MEM] updated {t} (new_events={new_n}, items={len(recent)})")
                except Exception as e:
                    print(f"[NEWS_MEM_ERR] {t}: {e!r}")

            # ✅ Market regime (risk-on/off)
            regime = None
            buy_block = False

            try:
                regime = regime_engine.get() if regime_engine is not None else None
            except Exception:
                regime = None

            # ✅ 기본 멀티플라이어(정의 누락 방지)
            th_mult = 1.0
            scalp_w_mult = 1.0
            size_mult = 1.0

            # ✅ regime 영향(옵션)
            if regime is not None and not paper_scalp_mode:
                try:
                    if float(getattr(regime, "score", 0.0)) <= float(regime_risk_off):
                        th_mult = float(regime_th_mult)
                        scalp_w_mult = float(regime_scalp_w_mult)
                        size_mult = float(regime_size_mult)
                        buy_block = bool(regime_buy_block)
                except Exception:
                    pass

            # 모의 단타 모드에서는 regime 완화
            if regime is not None and paper_scalp_mode:
                buy_block = False

            # ---------- 2) ticker loop ----------
            for ticker in WATCHLIST:
                pos = get_position(positions, ticker)
                pos.reset_if_new_day(now_kst)

                # snapshot
                try:
                    snap = fetch_snapshot(ticker)
                except Exception:
                    snap = {"ticker": ticker, "price": None}

                # KIS quote overwrite
                if quote_provider is not None:
                    try:
                        q = quote_provider.get_quote(ticker)
                        snap["price"] = q.price
                    except Exception:
                        pass

                price = snap.get("price")
                if price is None:
                    print(f"[TICK {loop_ts}] {ticker} price=None -> HOLD (no price)")
                    continue
                price_f = float(price)

                # recent px for spike-ban
                dq = _recent_px.get(ticker)
                if dq is None:
                    dq = deque(maxlen=max(30, chase_ban_spike_window + 5))
                    _recent_px[ticker] = dq
                dq.append(price_f)

                # scalp
                scalp_score = 0.0
                scalp_label = "scalp_hold"
                if scalp_engine is not None:
                    try:
                        so = scalp_engine.update(ticker, price_f)
                        scalp_score = float(so.score)
                        scalp_label = str(so.label)
                    except Exception:
                        scalp_score = 0.0
                        scalp_label = "scalp_err"

                # valuation
                fair_value = None
                fair_range = None
                try:
                    fv = compute_fair_value_snapshot(snap)
                    vscore = float(fv.get("value_score", 0.0))
                    fair_value = fv.get("fair_value")
                    fair_range = fv.get("fair_value_range")
                except Exception:
                    vscore = 0.0

                # TA
                try:
                    daily = fetch_daily_ta(ticker)
                    ta_out = ta_score(daily)
                    tscore = float(ta_out.get("ta_score", 0.0))
                    tlabel = str(ta_out.get("ta_label", "unknown"))
                except Exception:
                    tscore, tlabel = 0.0, "unknown"

                # news signal (decay)
                ns = news_store.compute_signal(
                    ticker=ticker,
                    now_kst=now_kst,
                    half_life_hours=news_half_life_h,
                    window_days=news_window_days,
                    max_items=200,
                )
                raw_news = float(ns.get("raw_sum", 0.0))
                cnt = int(ns.get("raw_n", 0))
                news_used = _clamp(float(ns.get("news_score", 0.0)), -2.0, 2.0)
                conf_avg = _clamp(float(ns.get("news_conf", 0.55)), 0.0, 1.0)

                # duplicate-news penalty (weak)
                if news_dup_penalty_enabled:
                    try:
                        recent_events = news_store.get_recent_events(ticker, days=1, limit=5) or []
                        titles = [str(e.get("title", "")).strip() for e in recent_events if isinstance(e, dict)]
                        titles = [t for t in titles if t]
                        if len(titles) >= 2 and titles[0] and titles[1] and titles[0] == titles[1]:
                            news_used = float(news_used) * float(news_dup_mult)
                    except Exception:
                        pass

                total_base = _total_score(news_used, vscore, tscore)

                # regime scalp weight adj
                scalp_weight_eff = _clamp(float(scalp_weight) * float(scalp_w_mult), 0.0, 1.0)

                if scalp_engine is not None:
                    w = scalp_weight_eff
                    total = (1.0 - w) * float(total_base) + w * float(scalp_score)
                    # ✅ A안: scalp_label은 decide_signal의 ta_label에 섞지 않음 (HOLD 억제 방지)
                    # tlabel = f"{tlabel}|{scalp_label}"
                else:
                    total = float(total_base)

                base_buy_th_eff = float(buy_th) * float(th_mult)
                base_sell_th_eff = float(sell_th) * float(th_mult)

                dyn_rules_obj = dyn_rules.apply(
                    news_store=news_store,
                    ticker=ticker,
                    now_kst=now_kst,
                    base_buy_th=base_buy_th_eff,
                    base_sell_th=base_sell_th_eff,
                    base_conf_th=conf_th,
                    base_confirm_ticks=confirm_ticks,
                    max_scan=30,
                )

                buy_th_eff = dyn_rules_obj.buy_th
                sell_th_eff = dyn_rules_obj.sell_th
                conf_th_eff = dyn_rules_obj.conf_th
                confirm_ticks_eff = dyn_rules_obj.confirm_ticks
                strength_boost = dyn_rules_obj.strength_boost

                # ✅ A안: HOLD로 바뀌는 이유 추적용
                block_reason = ""

                # ✅ A안: RiskOverride를 decide_signal "이전"에 적용해서 action 자체가 바뀌게 함
                # (임계치 델타가 의미 있게 동작)
                force_sell = False
                force_sell_frac = 0.0
                risk_reason = ""
                rd = None

                try:
                    rd, buy_delta, sell_delta = risk_override.evaluate(
                        news_store=news_store,
                        ticker=ticker,
                        now_kst=now_kst,
                        pos_qty=float(pos.qty),
                        max_scan=25,
                    )

                    buy_th_eff = _clamp(float(buy_th_eff) + float(buy_delta), 0.40, 0.95)
                    sell_th_eff = _clamp(float(sell_th_eff) + float(sell_delta), -0.95, -0.40)

                    force_sell = bool(rd.force_sell)
                    force_sell_frac = float(rd.sell_frac)
                    risk_reason = str(rd.reason)

                except Exception as e:
                    force_sell = False
                    force_sell_frac = 0.0
                    risk_reason = f"risk_err={e!r}"
                    rd = None

                # ✅ 이제 "최종 임계치"로 signal 결정
                sig = decide_signal(
                    total_score=total,
                    confidence=conf_avg,
                    ta_label=tlabel,
                    buy_th=buy_th_eff,
                    sell_th=sell_th_eff,
                    conf_th=conf_th_eff,
                )

                # ✅ strength boost 반영
                try:
                    sig = type(sig)(
                        sig.action,
                        min(1.0, float(sig.strength) + float(strength_boost)),
                        sig.reason + f" | {dyn_rules_obj.reason}",
                    )
                except Exception:
                    pass

                # ✅ A안: strength 바닥값(신호가 이미 BUY/SELL이면 qty=0으로 죽는 것 완화)
                strength_floor = _env_float("SIG_STRENGTH_FLOOR", 0.12)
                if sig.action in ("BUY", "SELL"):
                    try:
                        sig = type(sig)(
                            sig.action,
                            _clamp(max(float(sig.strength), float(strength_floor)), 0.0, 1.0),
                            sig.reason + f" | STR_FLOOR({strength_floor:.2f})",
                        )
                    except Exception:
                        pass

                # ✅ RiskOverride의 "block_buy"는 신호/플랜 단계에서 확실히 반영
                if rd is not None and bool(getattr(rd, "block_buy", False)) and sig.action == "BUY":
                    try:
                        sig = type(sig)("HOLD", float(sig.strength), sig.reason + " | " + risk_reason + " | BUY_BLOCKED")
                    except Exception:
                        sig.action = "HOLD"
                    block_reason = "RISK_OVERRIDE_BLOCK_BUY"

                pos.update_streak(sig.action)

                # market hours
                if (not market_open) and (not allow_outside_market) and sig.action in ("BUY", "SELL"):
                    plan_action = "HOLD"
                    plan_qty = 0
                    plan_reason = f"market closed (US regular) | raw={sig.action}"
                    if not block_reason:
                        block_reason = "MARKET_CLOSED"
                else:
                    ok_trade, why = can_trade(
                        pos=pos,
                        now=now_kst,
                        action=sig.action,
                        cooldown_minutes=cooldown_minutes,
                        max_trades_per_day=max_trades_per_day,
                        max_position_qty=float(max_position_qty),
                    )
                    if not ok_trade:
                        plan_action = "HOLD"
                        plan_qty = 0
                        plan_reason = f"risk blocked: {why} | raw={sig.action} | {sig.reason}"
                        if not block_reason:
                            block_reason = "CAN_TRADE_BLOCK"
                    else:
                        plan = compute_position_plan(
                            pos=pos,
                            raw_action=sig.action,
                            strength=sig.strength,
                            price=price_f,
                            confirm_ticks=confirm_ticks_eff,
                            fast_track_strength=fast_track_strength,
                            stop_loss_1=stop_loss_1,
                            stop_loss_2=stop_loss_2,
                            take_profit_1=take_profit_1,
                            stop_sell_frac=stop_sell_frac,
                            tp_sell_frac=tp_sell_frac,
                            base_qty=float(base_qty),
                            max_position_qty=float(max_position_qty),
                            buy_t1=buy_t1,
                            buy_t2=buy_t2,
                            buy_t3=buy_t3,
                            buy_m1=buy_m1,
                            buy_m2=buy_m2,
                            buy_m3=buy_m3,
                            sell_t1=sell_t1,
                            sell_t2=sell_t2,
                            sell_t3=sell_t3,
                            sell_f1=sell_f1,
                            sell_f2=sell_f2,
                            sell_f3=sell_f3,
                        )

                        plan_action = plan.action
                        plan_qty = _to_int_qty(plan.qty)
                        plan_reason = f"{plan.reason} | sig={sig.reason}"

                        if plan_action in ("BUY", "SELL") and plan_qty <= 0:
                            plan_action = "HOLD"
                            plan_reason = f"{plan_reason} | int_qty became 0"
                            if not block_reason:
                                block_reason = "QTY_ZERO"

                        # force sell: 포지션이 있을 때만
                        if force_sell and float(pos.qty) > 0:
                            fs_qty = int(math.floor(float(pos.qty) * force_sell_frac))
                            if fs_qty <= 0:
                                fs_qty = 1
                            plan_action = "SELL"
                            plan_qty = fs_qty
                            plan_reason = f"FORCE_SELL({force_sell_frac:.2f}) due to news risk | {risk_reason} | {plan_reason}"
                            # force sell은 block이 아니라 override라서 block_reason은 굳이 안 넣음

                # regime size mult
                if plan_action in ("BUY", "SELL") and plan_qty > 0:
                    plan_qty_eff = int(math.floor(float(plan_qty) * float(size_mult)))
                    if plan_qty_eff <= 0:
                        plan_qty_eff = 1
                    if plan_qty_eff != plan_qty:
                        plan_reason = f"REGIME_SIZE x{size_mult:.2f} ({plan_qty}->{plan_qty_eff}) | {plan_reason}"
                        plan_qty = plan_qty_eff

                # Session guard
                if (
                    session_guard_enabled
                    and market_open
                    and plan_action in ("BUY", "SELL")
                    and plan_qty > 0
                    and _us_session_avoid_window(now_kst, session_guard_start_min, session_guard_end_min)
                ):
                    plan_action = "HOLD"
                    plan_qty = 0
                    plan_reason = f"SESSION_GUARD({session_guard_start_min}/{session_guard_end_min}m) | {plan_reason}"

                # Whipsaw guard
                if whipsaw_guard_enabled and plan_action in ("BUY", "SELL") and plan_qty > 0:
                    desired = plan_action
                    last_a = _whipsaw_last_action.get(ticker, "HOLD")
                    last_tick = _whipsaw_last_tick.get(ticker, -10**9)
                    opposite = (last_a == "BUY" and desired == "SELL") or (last_a == "SELL" and desired == "BUY")

                    if opposite and (_global_tick - last_tick) <= max(1, int(whipsaw_cooldown_ticks)):
                        plan_action = "HOLD"
                        plan_qty = 0
                        plan_reason = f"WHIPSAW_BLOCK({last_a}->{desired}) | {plan_reason}"
                    else:
                        _whipsaw_last_action[ticker] = desired
                        _whipsaw_last_tick[ticker] = _global_tick

                # open pending block
                if plan_action in ("BUY", "SELL") and pending_store.has_open_order(ticker):
                    plan_action = "HOLD"
                    plan_qty = 0
                    plan_reason = f"blocked: open pending order exists for {ticker}"

                # per-ticker limit
                if plan_action in ("BUY", "SELL") and plan_qty > 0:
                    ok_lim, lim_reason = limit_store.allow(
                        ticker=ticker,
                        now_kst=now_kst,
                        cooldown_seconds=ticker_cooldown_sec,
                        max_orders_per_ticker_per_day=max_orders_per_ticker_day,
                    )
                    if not ok_lim:
                        plan_action = "HOLD"
                        plan_qty = 0
                        plan_reason = f"limit blocked: {lim_reason} | {plan_reason}"

                # -------------------------
                # ✅ Decision Agent
                # -------------------------
                decision_msg = ""
                decision_action = None
                decision_conf = None

                if decision_enabled:
                    decision_tick_counter[ticker] = decision_tick_counter.get(ticker, 0) + 1
                    if decision_tick_counter[ticker] >= max(1, decision_every_ticks):
                        decision_tick_counter[ticker] = 0
                        try:
                            kb = kb_load(ticker)

                            recent_events = news_store.get_recent_events(ticker, days=news_window_days, limit=12) or []
                            recent_light: List[Dict[str, Any]] = []
                            for e in recent_events:
                                if isinstance(e, dict):
                                    recent_light.append(
                                        {
                                            "ts_kst": e.get("ts_kst"),
                                            "title": e.get("title"),
                                            "summary": e.get("summary"),
                                            "event_type": e.get("event_type"),
                                            "sentiment": e.get("sentiment"),
                                            "impact": e.get("impact"),
                                            "why_it_moves": e.get("why_it_moves"),
                                            "link": e.get("link"),
                                            "event_score": e.get("event_score"),
                                            "confidence": e.get("confidence"),
                                        }
                                    )

                            snapshot = {
                                "ts_kst": loop_ts,
                                "price": price_f,
                                "market_open": market_open,
                                "regime": {
                                    "enabled": bool(regime_engine is not None),
                                    "score": getattr(regime, "score", None),
                                    "label": getattr(regime, "label", None),
                                },
                                "signal": {
                                    "total": total,
                                    "raw_action": sig.action,
                                    "strength": sig.strength,
                                    "reason": sig.reason,
                                },
                                "plan": {"action": plan_action, "qty": int(plan_qty), "reason": plan_reason},
                                "position": {"qty": float(pos.qty), "avg": float(pos.avg_price)},
                                "news": {"news_score": news_used, "raw_n": cnt, "raw_sum": raw_news, "conf": conf_avg},
                                "valuation": {"score": vscore, "fair_value": fair_value, "range": fair_range},
                                "ta": {"score": tscore, "label": tlabel},
                            }

                            prompt = build_decision_prompt(
                                ticker=ticker,
                                kb=kb,
                                snapshot=snapshot,
                                recent_news_events=recent_light,
                            )

                            llm_text = ollama_generate(
                                prompt=prompt,
                                model=decision_model,
                                temperature=0.2,
                                timeout=float(os.environ.get("OLLAMA_TIMEOUT", "120") or "120"),
                            )

                            decision = parse_decision(llm_text)
                            decision_action = str(decision.get("action", "HOLD")).upper()
                            decision_conf = float(decision.get("confidence", 0.5))

                            kb_add_decision(
                                ticker,
                                action=decision_action,
                                confidence=decision_conf,
                                rationale=str(decision.get("rationale", "")),
                                key_drivers=decision.get("key_drivers") or [],
                                key_risks=decision.get("key_risks") or [],
                                valuation_view=str(decision.get("valuation_view", "")),
                                counterfactuals=decision.get("counterfactuals") or [],
                                next_checks=decision.get("next_checks") or [],
                                raw=decision,
                            )

                            _append_jsonl(
                                DECISION_LOG_PATH,
                                {"ts": loop_ts, "ticker": ticker, "decision": decision, "snapshot": snapshot},
                            )

                            decision_msg = f" decision={decision_action} dconf={decision_conf:.2f}"

                            if (not decision_compare_only) and market_open:
                                if decision_conf >= decision_min_conf and decision_action in ("BUY", "SELL", "HOLD"):
                                    if decision_action == "HOLD":
                                        plan_action = "HOLD"
                                        plan_qty = 0
                                        plan_reason = f"DECISION_OVERRIDE: HOLD | {plan_reason}"
                                    else:
                                        prefer_qty = 0
                                        try:
                                            pp = decision.get("position_plan", {}) or {}
                                            prefer_qty = int(pp.get("prefer_qty", 0) or 0)
                                        except Exception:
                                            prefer_qty = 0

                                        if prefer_qty > 0:
                                            plan_qty = int(prefer_qty)

                                        plan_action = decision_action
                                        plan_reason = f"DECISION_OVERRIDE: {decision_action} conf={decision_conf:.2f} | {plan_reason}"

                        except Exception as e:
                            decision_msg = f" decision_err={e!r}"

                # -------------------------
                # ✅ AI gate
                # -------------------------
                ai_msg = ""
                if ai_gate_enabled and plan_action in ("BUY", "SELL") and plan_qty > 0:
                    try:
                        mem = news_store.load_memory(ticker)
                        recent = news_store.get_recent_events(ticker, days=news_window_days, limit=20)
                        gate = ai_gate_check_local_ollama(
                            ticker=ticker,
                            action=plan_action,
                            qty=int(plan_qty),
                            price=price_f,
                            total=total,
                            news_used=news_used,
                            val_score=vscore,
                            ta_score=tscore,
                            ta_label=tlabel,
                            signal_reason=sig.reason,
                            plan_reason=plan_reason,
                            market_open=market_open,
                            memory_summary=mem,
                            recent_events=recent,
                            model=ai_gate_model,
                            min_conf=ai_gate_min_conf,
                        )
                        if not gate.allow or gate.qty_mult <= 0.0:
                            ai_msg = f" ai_gate=VETO conf={gate.confidence:.2f} reason={gate.reason}"
                            plan_action = "HOLD"
                            plan_qty = 0
                            plan_reason = f"AI_GATED: {gate.reason} | {plan_reason}"
                        else:
                            new_qty = int(math.floor(plan_qty * gate.qty_mult))
                            if new_qty <= 0:
                                ai_msg = f" ai_gate=VETO(qty->0) conf={gate.confidence:.2f} reason={gate.reason}"
                                plan_action = "HOLD"
                                plan_qty = 0
                                plan_reason = f"AI_GATED: {gate.reason} | {plan_reason}"
                            elif new_qty != plan_qty:
                                ai_msg = f" ai_gate=REDUCE x{gate.qty_mult:.2f} conf={gate.confidence:.2f} reason={gate.reason}"
                                plan_qty = new_qty
                                plan_reason = f"AI_REDUCED: {gate.reason} | {plan_reason}"
                            else:
                                ai_msg = f" ai_gate=OK conf={gate.confidence:.2f}"
                    except Exception as e:
                        ai_msg = f" ai_gate_err={e!r}"

                # ✅ Account-level risk check
                if broker is not None and plan_action in ("BUY", "SELL") and plan_qty > 0:
                    ok_acc, why_acc = acc_risk.allow_order(
                        ticker=ticker,
                        action=plan_action,
                        qty=int(plan_qty),
                        price=float(price_f),
                    )
                    if not ok_acc:
                        plan_action = "HOLD"
                        plan_qty = 0
                        plan_reason = f"ACCOUNT_RISK_BLOCK: {why_acc} | {plan_reason}"
                        print(f"[ACC_RISK_BLOCK] {ticker} {why_acc}")

                # ==========================
                # ✅ FINAL GUARDS (Regime / Chase / Cost)
                # ==========================
                if plan_action == "BUY" and buy_block:
                    plan_action = "HOLD"
                    plan_qty = 0
                    plan_reason = f"REGIME_BLOCK (risk_off) | {plan_reason}"

                if chase_ban_enabled and plan_action == "BUY" and plan_qty > 0:
                    hot = False
                    hot_reason = ""

                    if scalp_engine is not None:
                        try:
                            st = scalp_engine.get_stats(ticker)
                        except Exception:
                            st = None
                        if st is not None:
                            mean, std, last_px, pct_from_mean = st
                            z = (last_px - mean) / max(1e-9, std)
                            if pct_from_mean >= chase_ban_pct:
                                hot = True
                                hot_reason = f"CHASE pct_from_mean={pct_from_mean:.4f} >= {chase_ban_pct:.4f}"
                            if z >= chase_ban_z:
                                hot = True
                                hot_reason = f"CHASE z={z:.2f} >= {chase_ban_z:.2f}"

                    dq2 = _recent_px.get(ticker)
                    if dq2 is not None and len(dq2) >= chase_ban_spike_window:
                        base_px = float(dq2[-chase_ban_spike_window])
                        spike = (price_f - base_px) / max(1e-9, base_px)
                        if spike >= chase_ban_after_spike_pct:
                            hot = True
                            hot_reason = f"SPIKE {spike:.4f} >= {chase_ban_after_spike_pct:.4f} (win={chase_ban_spike_window})"

                    if hot:
                        # ✅ A안: 완전 차단 대신 "보수 진입"
                        hard_block_pct = _env_float("CHASE_BAN_HARD_BLOCK_PCT", 0.012)  # 1.2% 급등이면 그때만 차단(선택)
                        do_hard_block = False

                        # spike 값이 계산된 경우에만 hard block 판단
                        try:
                            if dq2 is not None and len(dq2) >= chase_ban_spike_window:
                                base_px = float(dq2[-chase_ban_spike_window])
                                spike = (price_f - base_px) / max(1e-9, base_px)
                                if spike >= hard_block_pct:
                                    do_hard_block = True
                        except Exception:
                            do_hard_block = False

                        if do_hard_block:
                            plan_action = "HOLD"
                            plan_qty = 0
                            plan_reason = f"CHASE_HARD_BLOCK: {hot_reason} | {plan_reason}"
                            if not block_reason:
                                block_reason = "CHASE_HARD_BLOCK"
                        else:
                            # (1) 수량 축소
                            qty_mult = _env_float("CHASE_BAN_QTY_MULT", 0.30)
                            new_qty = int(math.floor(float(plan_qty) * float(qty_mult)))
                            if new_qty <= 0:
                                new_qty = 1

                            # (2) confirm 강화(한 틱 더 확인) - 실제 반영은 compute_position_plan 이전이 이상적이지만,
                            #      여기서는 실행 결과를 깔끔하게 만들기 위해 "qty 축소" 중심으로.
                            #      (confirm_ticks 강화는 아래 plan_reason에 기록만 하고,
                            #       원하면 다음 패치에서 compute_position_plan 호출 이전으로 이동 가능)
                            chase_confirm = _env_int("CHASE_BAN_CONFIRM_TICKS", 2)

                            if new_qty != plan_qty:
                                plan_reason = f"CHASE_SOFT: {hot_reason} | qty x{qty_mult:.2f} ({plan_qty}->{new_qty}) | confirm>= {chase_confirm} | {plan_reason}"
                                plan_qty = new_qty
                            else:
                                plan_reason = f"CHASE_SOFT: {hot_reason} | confirm>= {chase_confirm} | {plan_reason}"

                            if not block_reason:
                                block_reason = "CHASE_SOFT"

                if cost_gate_enabled and plan_action in ("BUY", "SELL") and plan_qty > 0:
                    total_cost_bps = float(cost_fee_bps) + float(cost_spread_bps) + float(cost_slip_bps)
                    exp_edge_pct = abs(float(total)) * float(edge_per_score)
                    exp_edge_bps = exp_edge_pct * 10000.0

                    if exp_edge_bps < total_cost_bps * float(edge_min_mult):
                        plan_action = "HOLD"
                        plan_qty = 0
                        plan_reason = (
                            f"COST_BLOCK edge={exp_edge_bps:.2f}bps < cost={total_cost_bps:.2f}bps*{edge_min_mult:.2f}"
                            f" | {plan_reason}"
                        )

                # 6) order
                order_msg = ""
                if broker is not None and plan_action in ("BUY", "SELL") and plan_qty > 0:
                    try:
                        if plan_action == "BUY":
                            res = broker.buy_market(ticker, int(plan_qty), last_price=price_f)
                        else:
                            res = broker.sell_market(ticker, int(plan_qty), last_price=price_f)

                        ok = bool(res.ok)
                        dry = bool(res.raw.get("dry_run", False))
                        order_msg = f" order_ok={ok} dry_run={dry} order_no={getattr(res, 'order_no', None)}"

                        if ok:
                            if dry:
                                trade_logger.log(
                                    {
                                        "ts": loop_ts,
                                        "ticker": ticker,
                                        "action": plan_action,
                                        "qty": int(plan_qty),
                                        "dry_run": True,
                                        "price_snapshot": price_f,
                                        "total_score": total,
                                        "confidence": conf_avg,
                                        "ta_label": tlabel,
                                        "news_used": news_used,
                                        "val_score": vscore,
                                        "ta_score": tscore,
                                        "reason": plan_reason,
                                        "market_open_us_regular": market_open,
                                        "raw_signal": {"action": sig.action, "strength": sig.strength, "reason": sig.reason},
                                        "decision_action": decision_action,
                                        "decision_conf": decision_conf,
                                    }
                                )
                                notifier.send(
                                    fmt_dry_run(
                                        ticker=ticker,
                                        side=plan_action,
                                        qty=int(plan_qty),
                                        price=float(price_f),
                                        total=float(total),
                                        conf=float(conf_avg),
                                        ta_label=str(tlabel),
                                        reason=str(plan_reason),
                                    )
                                )
                            else:
                                order_mgr.register_submitted(
                                    ticker=ticker,
                                    side=plan_action,
                                    qty=float(plan_qty),
                                    kis_response=res.raw,
                                    fill_price=price,
                                )
                                limit_store.update_on_order(ticker, now_kst)

                                trade_logger.log(
                                    {
                                        "ts": loop_ts,
                                        "ticker": ticker,
                                        "action": plan_action,
                                        "qty": int(plan_qty),
                                        "dry_run": False,
                                        "order_no": getattr(res, "order_no", None),
                                        "total_score": total,
                                        "confidence": conf_avg,
                                        "ta_label": tlabel,
                                        "news_used": news_used,
                                        "val_score": vscore,
                                        "ta_score": tscore,
                                        "reason": plan_reason,
                                        "market_open_us_regular": market_open,
                                        "raw_signal": {"action": sig.action, "strength": sig.strength, "reason": sig.reason},
                                        "decision_action": decision_action,
                                        "decision_conf": decision_conf,
                                    }
                                )
                                notifier.send(
                                    fmt_order_submitted(
                                        ticker=ticker,
                                        side=plan_action,
                                        qty=int(plan_qty),
                                        order_no=str(getattr(res, "order_no", None) or ""),
                                        price=float(price_f),
                                        total=float(total),
                                        conf=float(conf_avg),
                                        ta_label=str(tlabel),
                                        reason=str(plan_reason),
                                    )
                                )

                    except Exception as e:
                        resp_text = None
                        try:
                            r = getattr(e, "response", None)
                            if r is not None:
                                resp_text = r.text
                        except Exception:
                            resp_text = None

                        order_msg = f" order_err={e!r}"
                        if resp_text:
                            order_msg += f" resp={resp_text[:500]}"

                # 7) console
                reg_msg = ""
                if regime is not None:
                    try:
                        reg_msg = f" regime={getattr(regime, 'label', None)}({float(getattr(regime, 'score', 0.0)):.2f})"
                    except Exception:
                        reg_msg = ""

                print(
                    f"[TICK {loop_ts}] {ticker} price={price_f} "
                    f"total={total:.2f} conf={conf_avg:.2f} "
                    f"news={news_used:.2f}(raw={raw_news:.2f},n={cnt}) "
                    f"val={vscore:.2f} ta={tscore:.2f}({tlabel}) "
                    f"raw_sig={sig.action} strength={sig.strength:.3f} "
                    f"pos_qty={pos.qty:.0f} pos_avg={pos.avg_price:.2f} "
                    f"plan={plan_action} qty={plan_qty} market_open={market_open}"
                    f"{reg_msg}"
                    f"{decision_msg}"
                    f"{ai_msg}"
                    f"{order_msg}"
                    f"{' block=' + block_reason if block_reason else ''}"
                    f" plan_reason={plan_reason[:120]}"
                )

            # positions save
            try:
                save_state(positions, POS_PATH)
            except Exception as e:
                print(f"[WARN] save positions failed: {e!r}")

            print(
                f"[TICK_SUM {loop_ts}] analyzed_links={analyzed_links} "
                f"skip_seen={skipped_seen} skip_low_signal={skipped_low_signal} "
                f"skip_no_candidate={skipped_no_candidate} llm_fail={llm_fail}"
            )

            # auto labeling
            try:
                lab = auto_labeler.run_if_due(now_kst)
                if lab.get("ran") and lab.get("summary_updated"):
                    s = lab.get("summary") or {}
                    h = "1d"
                    bh = (s.get("by_horizon") or {}).get(h) or {}
                    msg = (
                        f"[LABEL] +{lab.get('n_new',0)} new | {h} "
                        f"n={bh.get('n_eval',0)} win={bh.get('win_rate',0):.2f} avg={bh.get('avg_ret',0):.4f}"
                    )
                    print(msg)
                    if auto_labeler.s.telegram_enabled:
                        notifier.send(fmt_label_summary(msg))
            except Exception as e:
                print(f"[LABEL_ERR] {e!r}")

            # perf reporter
            try:
                rep_run = auto_reporter.run_if_due(now_kst)
                if rep_run.get("ran"):
                    report = rep_run.get("report") or {}
                    msg_1h = PerformanceReporter.format_telegram_summary(
                        report, horizon=os.environ.get("PERF_TELEGRAM_HORIZON", "1h")
                    )
                    print(msg_1h)
                    if auto_reporter.s.telegram_enabled:
                        notifier.send(fmt_perf_summary(msg_1h))

                    if auto_reporter.s.also_send_1d:
                        msg_1d = PerformanceReporter.format_telegram_summary(report, horizon="1d")
                        print(msg_1d)
                        if auto_reporter.s.telegram_enabled:
                            notifier.send(fmt_perf_summary(msg_1d))

            except Exception as e:
                print(f"[PERF_ERR] {e!r}")

            time.sleep(max(1, tick_seconds))

    finally:
        release_lock()


if __name__ == "__main__":
    main()