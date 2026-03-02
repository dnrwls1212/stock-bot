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
    fmt_news, fmt_start, fmt_order_submitted, fmt_dry_run, fmt_label_summary, fmt_perf_summary,
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
from src.trading.order_store import PendingOrderStore
from src.trading.order_manager import OrderManager
from src.trading.trade_limits import TradeLimitStore

from src.utils.market_hours import is_us_regular_market_open
from src.utils.trade_logger import TradeLogger
from src.trading.news_store import NewsStore, NewsEvent
from src.analysis.news_memory import build_news_memory_summary_local_ollama
from src.trading.ai_gate import ai_gate_check_local_ollama
from src.eval.auto_labeler import DecisionAutoLabeler, AutoLabelSettings
from src.report.auto_reporter import AutoReporter
from src.report.performance_report import PerformanceReporter
from src.trading.dynamic_news_rules import DynamicNewsRules
from src.trading.news_risk_override import NewsRiskOverride
from src.utils.ollama_client import ollama_generate, try_parse_json
from src.trading.scalp_signal import ScalpSignalEngine
from src.trading.regime_filter import RegimeFilter
from src.trading.watchlist_auto import build_watchlist_v1
from src.trading.watchlist_loader import load_watchlist
from src.trading.universe_builder import build_universe

load_dotenv()

WATCHLIST: List[str] = []
LOCK_PATH = "data/run.lock"
EVENT_LOG_PATH = "data/events.jsonl"
DECISION_LOG_PATH = os.environ.get("DECISION_LOG_PATH", "data/decisions.jsonl")
POS_PATH = os.environ.get("POS_PATH", "data/positions.json")
PENDING_PATH = os.environ.get("PENDING_PATH", "data/pending_orders.json")

def _env_int(key: str, default: int) -> int:
    try: return int(os.environ.get(key, str(default)).strip())
    except Exception: return default

def _env_float(key: str, default: float) -> float:
    try: return float(os.environ.get(key, str(default)).strip())
    except Exception: return default

def _env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None: return default
    return v.strip() in ("1", "true", "True", "YES", "yes", "y")

def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    if v is None: return default
    s = str(v).strip()
    return s if s else default

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError: return False
    except PermissionError: return True

def takeover_lock() -> None:
    os.makedirs("data", exist_ok=True)
    old_pid: Optional[int] = None
    if os.path.exists(LOCK_PATH):
        try: old_pid = int(open(LOCK_PATH, "r", encoding="utf-8").read().strip())
        except Exception: old_pid = None
    if old_pid and old_pid != os.getpid() and _pid_alive(old_pid):
        try: os.kill(old_pid, signal.SIGTERM)
        except Exception: pass
        for _ in range(15):
            if not _pid_alive(old_pid): break
            time.sleep(0.2)
        if _pid_alive(old_pid):
            try: os.kill(old_pid, signal.SIGKILL)
            except Exception: pass
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

def release_lock() -> None:
    try:
        if os.path.exists(LOCK_PATH): os.remove(LOCK_PATH)
    except Exception: pass

class Notifier:
    def __init__(self) -> None:
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    def send(self, text: str) -> None:
        try: send_telegram_message(self.bot_token, self.chat_id, text)
        except Exception as e: print(f"[WARN] telegram send failed: {e}")

def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _mentions_ticker(ticker: str, title: str, summary: str) -> bool:
    t = (ticker or "").upper().strip()
    text = f"{title}\n{summary}".lower()
    hints = getattr(news_filter, "COMPANY_HINTS", {})
    keys = hints.get(t)
    if not keys: return t.lower() in text
    return any(k in text for k in keys)

def _candidate_tickers(title: str, summary: str, watchlist: List[str]) -> List[str]:
    out: List[str] = []
    for t in watchlist:
        if _mentions_ticker(t, title, summary): out.append(t)
    return out

def _is_high_signal(title: str, summary: str) -> bool:
    return bool(news_filter.is_high_signal(title, summary))

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _total_score(news_s: float, val_s: float, ta_s: float) -> float:
    w_news = _env_float("W_NEWS", 0.50)
    w_val = _env_float("W_VAL", 0.40)
    w_daily = _env_float("W_DAILY", 0.40)
    denom = (w_news + w_val + w_daily) or 1.0
    return (w_news * news_s + w_val * val_s + w_daily * ta_s) / denom

def _to_int_qty(x: float) -> int:
    try:
        q = int(math.floor(float(x)))
        return max(0, q)
    except Exception: return 0

def _us_session_avoid_window(now_kst: datetime, start_min: int, end_min: int) -> bool:
    try:
        ny = now_kst.astimezone(ZoneInfo("America/New_York"))
        t = ny.time()
        open_t = dtime(9, 30)
        close_t = dtime(16, 0)
        if not (open_t <= t <= close_t): return False
        cur_min = t.hour * 60 + t.minute
        open_min = open_t.hour * 60 + open_t.minute
        close_min = close_t.hour * 60 + close_t.minute
        if cur_min < open_min + max(0, int(start_min)): return True
        if cur_min > close_min - max(0, int(end_min)): return True
        return False
    except Exception: return False

def _now_kst_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
def _kb_dir() -> str: return os.environ.get("KB_DIR", "data/kb")
def _kb_path(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    return os.path.join(_kb_dir(), f"{t}.json")

def kb_load(ticker: str) -> Dict[str, Any]:
    os.makedirs(_kb_dir(), exist_ok=True)
    p = _kb_path(ticker)
    default_kb = {"ticker": (ticker or "").upper().strip(), "updated_at": _now_kst_iso(), "thesis": "", "business_summary": "", "moat": "", "key_drivers": [], "key_risks": [], "valuation_method": "simple", "valuation_assumptions": {}, "target_price": None, "fair_value_range": None, "evidence": [], "decisions": [], "tags": []}
    if not os.path.exists(p): return default_kb
    try:
        with open(p, "r", encoding="utf-8") as f: d = json.load(f)
        if not isinstance(d, dict): raise ValueError("kb not dict")
        for k, v in default_kb.items(): d.setdefault(k, v)
        return d
    except Exception: return default_kb

def kb_save(kb: Dict[str, Any]) -> None:
    os.makedirs(_kb_dir(), exist_ok=True)
    kb["updated_at"] = _now_kst_iso()
    t = (kb.get("ticker") or "").upper().strip()
    p = _kb_path(t)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def kb_add_evidence(ticker: str, *, source: str, title: str, summary: str, link: str = "", sentiment: str = "neutral", impact: int = 0, tags: Optional[List[str]] = None, raw: Optional[Dict[str, Any]] = None, max_items: int = 400) -> None:
    kb = kb_load(ticker)
    ev = kb.get("evidence", [])
    if not isinstance(ev, list): ev = []
    ev.insert(0, {"ts_kst": _now_kst_iso(), "source": source, "title": title, "summary": summary, "link": link, "sentiment": sentiment, "impact": int(impact), "tags": tags or [], "raw": raw or {}})
    kb["evidence"] = ev[:max_items]
    kb_save(kb)

def kb_add_decision(ticker: str, *, action: str, confidence: float, rationale: str, key_drivers: Optional[List[str]] = None, key_risks: Optional[List[str]] = None, valuation_view: str = "", counterfactuals: Optional[List[str]] = None, next_checks: Optional[List[str]] = None, raw: Optional[Dict[str, Any]] = None, max_items: int = 200) -> None:
    kb = kb_load(ticker)
    decs = kb.get("decisions", [])
    if not isinstance(decs, list): decs = []
    decs.insert(0, {"ts_kst": _now_kst_iso(), "action": action, "confidence": float(confidence), "rationale": rationale, "key_drivers": key_drivers or [], "key_risks": key_risks or [], "valuation_view": valuation_view, "counterfactuals": counterfactuals or [], "next_checks": next_checks or [], "raw": raw or {}})
    kb["decisions"] = decs[:max_items]
    kb_save(kb)

def build_decision_prompt(*, ticker: str, kb: Dict[str, Any], snapshot: Dict[str, Any], recent_news_events: List[Dict[str, Any]]) -> str:
    kb_light = {
        "thesis": kb.get("thesis", ""), "business_summary": kb.get("business_summary", ""), "moat": kb.get("moat", ""),
        "key_drivers": kb.get("key_drivers", []) or [], "key_risks": kb.get("key_risks", []) or [], "valuation_method": kb.get("valuation_method", "simple"),
        "valuation_assumptions": kb.get("valuation_assumptions", {}) or {}, "target_price": kb.get("target_price", None), "fair_value_range": kb.get("fair_value_range", None),
        "tags": kb.get("tags", []) or [], "recent_decisions": (kb.get("decisions", []) or [])[:5], "recent_evidence": (kb.get("evidence", []) or [])[:12],
    }
    schema = {"action": "BUY|SELL|HOLD", "confidence": 0.0, "rationale": "Korean, concise, factual", "key_drivers": ["..."], "key_risks": ["..."], "valuation_view": "what assumption matters / changed", "counterfactuals": ["what would make you wrong next 1-4 weeks"], "next_checks": ["what to check next"], "position_plan": {"prefer_qty": 0, "time_horizon": "swing_days|swing_weeks|long_months"}}
    return f"""너는 '누적 지식 기반' 투자 의사결정 에이전트다. 목표: 단기/스윙 수익을 내되, 기업 방향성과 근거 축적을 최우선으로 한다. 규칙: - 제공된 KB/스냅샷/최근 이벤트만 근거로 사용. - 출력은 반드시 JSON 하나만. \n[TICKER]\n{ticker}\n[KB]\n{json.dumps(kb_light, ensure_ascii=False)}\n[SNAPSHOT]\n{json.dumps(snapshot, ensure_ascii=False)}\n[RECENT_NEWS_EVENTS]\n{json.dumps(recent_news_events[:10], ensure_ascii=False)}\n[OUTPUT_JSON_SCHEMA]\n{json.dumps(schema, ensure_ascii=False)}""".strip()

def parse_decision(text: str) -> Dict[str, Any]:
    d = try_parse_json(text) or {}
    if not isinstance(d, dict): d = {}
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
        if d["action"] not in ("BUY", "SELL", "HOLD"): d["action"] = "HOLD"
    except Exception: d["action"] = "HOLD"
    try:
        d["confidence"] = float(d.get("confidence", 0.5))
        d["confidence"] = max(0.0, min(1.0, d["confidence"]))
    except Exception: d["confidence"] = 0.5
    return d

def _maybe_build_universe_and_watchlist_once(notifier: Notifier) -> Optional[List[str]]:
    universe_on = _env_bool("UNIVERSE_AUTO_ON_START", True)
    watch_on = _env_bool("WATCHLIST_AUTO_ON_START", False)
    universe_core_path = _env_str("UNIVERSE_CORE_PATH", "data/universe_core.txt")
    universe_path = _env_str("UNIVERSE_PATH", "data/universe.txt")

    if universe_on:
        try:
            u = build_universe(
                out_path=universe_path, core_path=universe_core_path, period=_env_str("UNIVERSE_PERIOD", "3mo"),
                min_price=_env_float("UNIVERSE_MIN_PRICE", 5.0), min_dollar_vol=_env_float("UNIVERSE_MIN_DVOL", 5_000_000.0),
                target_n=_env_int("UNIVERSE_TARGET_N", 200), max_universe=_env_int("UNIVERSE_MAX", 1200),
            )
            ok, reason, outp, n_valid, n_input = False, "unknown", universe_path, None, None
            if isinstance(u, dict):
                ok, reason, outp, n_valid, n_input = bool(u.get("ok", False)), str(u.get("reason", "unknown")), str(u.get("out_path", universe_path) or universe_path), u.get("n_valid", None), u.get("n_input", None)
            else:
                ok, reason, outp, n_valid, n_input = bool(getattr(u, "ok", False)), str(getattr(u, "reason", "unknown")), str(getattr(u, "out_path", universe_path) or universe_path), getattr(u, "n_valid", None), getattr(u, "n_input", None)
            if ok:
                msg = f"🌍 [Universe 생성 완료]\n- out: {outp}"
                if n_valid is not None: msg += f"\n- valid: {n_valid}"
                notifier.send(msg)
            else: notifier.send(f"⚠️ [Universe 생성 실패]\n- reason: {reason}\n- core: {universe_core_path}")
        except Exception as e: notifier.send(f"⚠️ [Universe 생성 에러]\nerr={e!r}")

    if not watch_on: return None
    out_path = _env_str("WATCHLIST_AUTO_PATH", "data/watchlist_auto.txt")
    top_n = _env_int("WATCHLIST_TOP_N", 20)

    try:
        r = build_watchlist_v1(
            universe_path=universe_path, out_path=out_path, top_n=top_n, min_price=_env_float("WATCHLIST_MIN_PRICE", 5.0),
            min_dollar_vol=_env_float("WATCHLIST_MIN_DVOL", 5_000_000.0), news_lookback_hours=_env_int("WATCHLIST_NEWS_H", 24),
            w_atr=_env_float("WATCHLIST_W_ATR", 0.45), w_liq=_env_float("WATCHLIST_W_LIQ", 0.35), w_news=_env_float("WATCHLIST_W_NEWS", 0.20),
            period=_env_str("WATCHLIST_PERIOD", "3mo"), interval=_env_str("WATCHLIST_INTERVAL", "1d"),
        )
        ok, reason, picked, outp = False, "unknown", [], out_path
        if isinstance(r, dict):
            ok, reason, picked, outp = bool(r.get("ok", False)), str(r.get("reason", "unknown")), list(r.get("picked", []) or []), str(r.get("out_path", out_path) or out_path)
        else:
            ok, reason, picked, outp = bool(getattr(r, "ok", False)), str(getattr(r, "reason", "unknown")), list(getattr(r, "picked", []) or []), str(getattr(r, "out_path", out_path) or out_path)
        if ok and picked:
            notifier.send(f"📌 [자동 종목선정 완료]\n선정: {', '.join(picked)}\n저장: {outp}")
            return picked
        notifier.send(f"⚠️ [자동 종목선정 실패] 기존/기본 WATCHLIST 사용\nreason={reason}")
        return None
    except Exception as e:
        notifier.send(f"⚠️ [자동 종목선정 에러] 기존 WATCHLIST 사용\nerr={e!r}")
        return None

# =====================================================================
# 🚨 [신규 추가] 장 시작 전 AI 전략 브리핑 생성 함수
# =====================================================================
def _generate_and_send_briefing(watchlist: List[str], news_store: NewsStore, model: str, notifier: Notifier, session_name: str):
    notifier.send(f"📊 [AI 분석 중] {session_name} 대비 전략 브리핑을 작성하고 있습니다...")
    lines = []
    # 각 종목별로 가장 파급력이 컸던 최근 3일치 뉴스 2개씩만 추출 (LLM Context 오버플로우 방지)
    for t in watchlist:
        evs = news_store.get_recent_events(t, days=3, limit=2)
        if evs:
            lines.append(f"[{t}]")
            for e in evs:
                if isinstance(e, dict) and e.get('title'):
                    lines.append(f"- {e.get('title')} (AI점수: {e.get('event_score', 0):.2f})")
    
    if not lines:
        notifier.send(f"📢 [{session_name} 브리핑]\n수집된 주요 호재/악재 뉴스가 없습니다. 보수적 관망을 권장합니다.")
        return

    prompt = (
        f"너는 월스트리트 수석 퀀트 애널리스트야. 곧 {session_name}이 시작돼.\n"
        "우리가 감시 중인 종목들에 대해 수집된 아래의 최신 뉴스를 바탕으로, "
        "오늘 장에서 어떤 종목이 크게 상승할 모멘텀을 가졌는지(매수 추천), "
        "어떤 종목을 조심해야 할지(리스크 경고) 3~4문단으로 아주 간결하고 명확하게 브리핑해줘.\n\n"
        + "\n".join(lines)
    )

    try:
        res = ollama_generate(prompt=prompt, model=model, temperature=0.3, timeout=180)
        notifier.send(f"🎯 [AI {session_name} 전략 보고서]\n\n{res}")
    except Exception as e:
        notifier.send(f"⚠️ 브리핑 생성 실패: {e}")

# =====================================================================
# 🚨 [신규 추가] AI 매크로(시장 전체) 리스크 진단 함수
# =====================================================================
def _evaluate_macro_risk(news_store: NewsStore, model: str) -> tuple[int, str]:
    # 시장 대장주인 SPY, QQQ의 최근 3일치 뉴스를 집중 분석
    events = news_store.get_recent_events("SPY", days=3, limit=10) + news_store.get_recent_events("QQQ", days=3, limit=10)
    
    if not events:
        return 1, "최근 거시경제 주요 뉴스가 부족하여 기본 경계(Level 1) 상태를 유지합니다."
        
    lines = []
    for e in events:
        if isinstance(e, dict) and e.get("title"):
            lines.append(f"- {e.get('title')}")
            
    prompt = (
        "너는 월스트리트 수석 거시경제(Macro) 리스크 애널리스트야.\n"
        "최근 3일간의 시장 지수(SPY, QQQ) 뉴스를 바탕으로 현재 시장의 '시스템적 리스크 레벨'을 0에서 3까지 평가해줘.\n"
        "[Risk Level 기준]\n"
        "0: 평온한 시장 또는 강세장\n"
        "1: 일반적인 조정 또는 경계감 (기본)\n"
        "2: 뚜렷한 악재 발생 (예: 금리 인상 쇼크, 큰 지정학적 긴장, 무역 분쟁)\n"
        "3: 극도 위험 / 시스템적 위기 (예: 전쟁 발발, 전염병 확산, 대형 금융위기)\n\n"
        "[최근 뉴스]\n" + "\n".join(lines[:15]) + "\n\n"
        "반드시 아래 JSON 형식으로만 응답해 (다른 말은 절대 금지):\n"
        "{\"risk_level\": 0, \"reason\": \"한국어로 아주 간결한 평가 이유 1문장\"}"
    )
    
    try:
        from src.utils.ollama_client import ollama_generate, try_parse_json
        res_text = ollama_generate(prompt=prompt, model=model, temperature=0.1, timeout=120)
        res_json = try_parse_json(res_text)
        
        if isinstance(res_json, dict):
            level = int(res_json.get("risk_level", 1))
            reason = str(res_json.get("reason", "분석 완료"))
            level = max(0, min(3, level)) # 0~3 사이 보정
            return level, reason
    except Exception as e:
        print(f"[MACRO_RISK_ERR] {e}")
        
    return 1, "AI 분석 에러로 인해 기본 방어선(Level 1)을 유지합니다."

# -----------------------------
# main
# -----------------------------
def main() -> None:
    global WATCHLIST
    takeover_lock()

    notifier = Notifier()
    trade_logger = TradeLogger(path=os.environ.get("TRADES_PATH", "data/trades.jsonl"), enabled=True)

    _maybe_build_universe_and_watchlist_once(notifier)
    WATCHLIST = load_watchlist()
    
    inverse_env = os.environ.get("INVERSE_TICKERS", "SQQQ,SOXS,PSQ")
    inverse_tickers = [t.strip().upper() for t in inverse_env.split(",") if t.strip()]
    for inv_t in inverse_tickers:
        if inv_t not in WATCHLIST:
            WATCHLIST.append(inv_t)
            
    print(f"[WATCHLIST] loaded (with inverse): {WATCHLIST}")

    tick_seconds = _env_int("TICK_SECONDS", 60)
    cooldown_minutes = _env_int("COOLDOWN_MINUTES", 15)
    max_trades_per_day = _env_int("MAX_TRADES_PER_DAY", 15)
    max_position_qty = _env_int("MAX_POSITION_QTY", 500)
    base_qty = _env_int("BASE_QTY", 1)
    confirm_ticks = _env_int("CONFIRM_TICKS", 1)
    fast_track_strength = _env_float("FAST_TRACK_STRENGTH", 0.90)

    stop_loss_1 = _env_float("STOP_LOSS_1", -0.03)
    stop_loss_2 = _env_float("STOP_LOSS_2", -0.05)
    take_profit_1 = _env_float("TAKE_PROFIT_1", 0.05)
    stop_sell_frac = _env_float("STOP_SELL_FRAC", 0.50)
    tp_sell_frac = _env_float("TP_SELL_FRAC", 0.50)

    buy_th = _env_float("BUY_TH", 0.35)
    sell_th = _env_float("SELL_TH", -0.40)
    conf_th = _env_float("CONF_TH", 0.55)

    buy_t1, buy_t2, buy_t3 = _env_float("BUY_T1", 0.15), _env_float("BUY_T2", 0.40), _env_float("BUY_T3", 0.70)
    buy_m1, buy_m2, buy_m3 = _env_float("BUY_M1", 1.0), _env_float("BUY_M2", 2.0), _env_float("BUY_M3", 3.0)
    sell_t1, sell_t2, sell_t3 = _env_float("SELL_T1", 0.20), _env_float("SELL_T2", 0.50), _env_float("SELL_T3", 0.80)
    sell_f1, sell_f2, sell_f3 = _env_float("SELL_F1", 0.25), _env_float("SELL_F2", 0.50), _env_float("SELL_F3", 1.00)

    chase_ban_enabled = _env_bool("CHASE_BAN_ENABLED", True)
    chase_ban_pct = _env_float("CHASE_BAN_PCT", 0.015)
    chase_ban_z = _env_float("CHASE_BAN_Z", 1.20)
    chase_ban_after_spike_pct = _env_float("CHASE_BAN_AFTER_SPIKE_PCT", 0.025)
    chase_ban_spike_window = _env_int("CHASE_BAN_SPIKE_WINDOW", 12)
    _recent_px: Dict[str, deque] = {}

    cost_gate_enabled = _env_bool("COST_GATE_ENABLED", True)
    cost_fee_bps = _env_float("COST_FEE_BPS", 25.0)
    cost_spread_bps = _env_float("COST_SPREAD_BPS", 5.0)
    cost_slip_bps = _env_float("COST_SLIPPAGE_BPS", 3.0)
    edge_per_score = _env_float("EDGE_PER_SCORE", 0.015)
    edge_min_mult = _env_float("EDGE_MIN_MULT", 1.1)

    regime_enabled = _env_bool("REGIME_ENABLED", True)
    regime_symbol = _env_str("REGIME_SYMBOL", "QQQ")
    regime_risk_off = _env_float("REGIME_RISK_OFF", -0.50)
    regime_buy_block = _env_bool("REGIME_BUY_BLOCK", True)
    regime_th_mult = _env_float("REGIME_TH_MULT", 1.50)
    regime_scalp_w_mult = _env_float("REGIME_SCALP_WEIGHT_MULT", 0.70)
    regime_size_mult = _env_float("REGIME_SIZE_MULT", 0.30)

    regime_engine: Optional[RegimeFilter] = None
    if regime_enabled: regime_engine = RegimeFilter(regime_symbol)

    session_guard_enabled = _env_bool("SESSION_GUARD_ENABLED", True)
    session_guard_start_min = _env_int("SESSION_GUARD_START_MIN", 3)
    session_guard_end_min = _env_int("SESSION_GUARD_END_MIN", 3)

    whipsaw_guard_enabled = _env_bool("WHIPSAW_GUARD_ENABLED", True)
    whipsaw_cooldown_ticks = _env_int("WHIPSAW_COOLDOWN_TICKS", 5)

    _whipsaw_last_action: Dict[str, str] = {t: "HOLD" for t in WATCHLIST}
    _whipsaw_last_tick: Dict[str, int] = {t: -10**9 for t in WATCHLIST}
    _global_tick = 0

    news_dup_penalty_enabled = _env_bool("NEWS_DUP_PENALTY_ENABLED", True)
    news_dup_mult = _env_float("NEWS_DUP_MULT", 0.70)

    ai_gate_enabled = _env_bool("AI_GATE_ENABLED", True)
    ai_gate_model = os.environ.get("AI_GATE_MODEL", "qwen2.5:14b-instruct")
    ai_gate_min_conf = _env_float("AI_GATE_MIN_CONF", 0.60)

    decision_enabled = _env_bool("DECISION_AGENT_ENABLED", True)
    decision_model = os.environ.get("DECISION_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5:14b-instruct"))
    decision_min_conf = _env_float("DECISION_MIN_CONF", 0.60)
    decision_every_ticks = _env_int("DECISION_EVERY_TICKS", 60)
    decision_compare_only = not _env_bool("DECISION_OVERRIDE_TRADING", False)
    decision_tick_counter: Dict[str, int] = {t: 0 for t in WATCHLIST}

    watchlist_refresh_min = _env_int("WATCHLIST_AUTO_REFRESH_MIN", 0)
    next_watchlist_refresh: Optional[datetime] = None
    if watchlist_refresh_min > 0:
        next_watchlist_refresh = datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(minutes=max(1, watchlist_refresh_min))

    seen: Set[str] = load_seen()
    rss_urls = build_rss_urls(WATCHLIST)
    positions = load_state(POS_PATH)

    for t, pos_info in positions.items():
        if float(pos_info.qty) > 0 and t not in WATCHLIST:
            WATCHLIST.append(t)
            print(f"📌 보유 종목 강제 감시 추가: {t}")

    pending_store = PendingOrderStore(PENDING_PATH)
    order_mgr = OrderManager(
        store=pending_store, position_state=positions, position_state_path=POS_PATH,
        ccnl_lookback_minutes=_env_int("CCNL_LOOKBACK_MIN", 15), stale_order_seconds=_env_int("STALE_ORDER_SEC", 180),
    )

    NEWS_STORE_PATH = os.environ.get("NEWS_STORE_PATH", "data/news_store.json")
    NEWS_MEMORY_DIR = os.environ.get("NEWS_MEMORY_DIR", "data/news_memory")
    news_store = NewsStore(NEWS_STORE_PATH, memory_dir=NEWS_MEMORY_DIR)

    news_half_life_h = _env_float("NEWS_HALF_LIFE_H", 48.0)
    news_window_days = _env_int("NEWS_WINDOW_DAYS", 14)
    news_mem_max_items = _env_int("NEWS_MEM_MAX_ITEMS", 20)
    news_mem_update_every = _env_int("NEWS_MEM_UPDATE_EVERY", 3)
    news_mem_model = os.environ.get("NEWS_MEM_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5:14b-instruct"))

    dyn_rules = DynamicNewsRules()
    risk_override = NewsRiskOverride()

    LIMITS_PATH = os.environ.get("LIMITS_PATH", "data/limits.json")
    limit_store = TradeLimitStore(LIMITS_PATH)
    ticker_cooldown_sec = _env_int("TICKER_COOLDOWN_SEC", 300)
    max_orders_per_ticker_day = _env_int("MAX_ORDERS_PER_TICKER_DAY", 3)

    acc_risk = AccountRiskManager.from_env()

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

    # =====================================================================
    # 🚨 [신규 추가] 시간대별 모드 및 브리핑 상태 변수 초기화
    # =====================================================================
    briefing_enabled = _env_bool("BRIEFING_ENABLED", True)
    briefing_pre_et = _env_str("BRIEFING_PRE_ET", "03:50")
    briefing_reg_et = _env_str("BRIEFING_REG_ET", "09:20")
    trade_start_et = _env_str("TRADE_START_ET", "04:00")
    trade_end_et = _env_str("TRADE_END_ET", "16:00")
    
    last_pre_brief_date = ""
    last_reg_brief_date = ""

    notifier.send(
        fmt_start(
            watchlist=WATCHLIST, tick_seconds=tick_seconds, execute_orders=os.environ.get("EXECUTE_ORDERS", "0"),
            ai_gate_enabled=ai_gate_enabled, decision_enabled=decision_enabled, decision_override=(not decision_compare_only),
        ) + f"\n\n🕒 [운영 모드]\n전투(매매) 모드: {trade_start_et} ~ {trade_end_et} (뉴욕시간)\n그 외 시간은 리서치 모드(주문 차단)로 동작합니다."
    )

    auto_labeler = DecisionAutoLabeler(
        AutoLabelSettings(
            input_path=DECISION_LOG_PATH, labeled_path=os.environ.get("DECISION_LABELED_PATH", "data/decisions_labeled.jsonl"),
            summary_path=os.environ.get("DECISION_SUMMARY_PATH", "data/decisions_summary.json"),
            cursor_path=os.environ.get("LABEL_CURSOR_PATH", "data/label_cursor.json"),
            cache_dir=os.environ.get("PRICE_CACHE_DIR", "data/price_cache"),
            horizons=[h.strip() for h in os.environ.get("LABEL_HORIZONS", "1h,1d").split(",") if h.strip()],
            min_abs_ret=float(os.environ.get("LABEL_MIN_ABS_RET", "0.002")), min_conf=float(os.environ.get("LABEL_MIN_CONF", "0.0")),
            every_minutes=int(os.environ.get("LABEL_EVERY_MIN", "15")), telegram_enabled=bool(int(os.environ.get("LABEL_TELEGRAM", "1"))),
        )
    )
    auto_reporter = AutoReporter.from_env()

    pos_sync_on_start = _env_bool("POS_SYNC_ON_START", True)
    pos_sync_every_ticks = _env_int("POS_SYNC_EVERY_TICKS", 15)
    _pos_sync_tick = 0
    bg_idx = 0  
    last_regime_noti_time = None

    last_macro_eval_time = None
    macro_risk_level = 1
    macro_risk_reason = "초기화 대기중"

    try:
        while True:
            now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
            _global_tick += 1
            loop_ts = now_kst.isoformat(timespec="seconds")
            market_open = is_us_regular_market_open(now_kst)

            # =====================================================================
            # 🚨 [신규 추가] 서머타임 자동대응 뉴욕 시간(ET) 판별
            # =====================================================================
            ny_time = now_kst.astimezone(ZoneInfo("America/New_York"))
            ny_hm = ny_time.strftime("%H:%M")
            ny_date = ny_time.strftime("%Y-%m-%d")

            # 1. AI 전략 브리핑 발송 (설정된 시간에 하루 1회씩)
            if briefing_enabled:
                if briefing_pre_et <= ny_hm < trade_start_et and last_pre_brief_date != ny_date:
                    _generate_and_send_briefing(WATCHLIST, news_store, ai_gate_model, notifier, "프리마켓")
                    last_pre_brief_date = ny_date
                
                if briefing_reg_et <= ny_hm < "09:30" and last_reg_brief_date != ny_date:
                    _generate_and_send_briefing(WATCHLIST, news_store, ai_gate_model, notifier, "정규장")
                    last_reg_brief_date = ny_date

            # 2. 거래 모드(Combat Mode) 판별 (월~금 평일에만 전투 모드 켜짐)
            # ny_time.weekday()는 0(월)~4(금), 5(토), 6(일)을 반환합니다.
            is_combat_mode = (trade_start_et <= ny_hm < trade_end_et) and (ny_time.weekday() < 5)
            # =====================================================================

            if watchlist_refresh_min > 0 and next_watchlist_refresh is not None and now_kst >= next_watchlist_refresh:
                _maybe_build_universe_and_watchlist_once(notifier)
                WATCHLIST = load_watchlist()
                for inv_t in inverse_tickers:
                    if inv_t not in WATCHLIST: WATCHLIST.append(inv_t)
                rss_urls = build_rss_urls(WATCHLIST)
                decision_tick_counter = {t: 0 for t in WATCHLIST}
                _whipsaw_last_action = {t: "HOLD" for t in WATCHLIST}
                _whipsaw_last_tick = {t: -10**9 for t in WATCHLIST}
                notifier.send(f"🔄 [watchlist 갱신] {', '.join(WATCHLIST)}")
                next_watchlist_refresh = now_kst + timedelta(minutes=watchlist_refresh_min)

            if broker is not None:
                try: order_mgr.sync_once(broker)
                except Exception: pass
                try: acc_risk.refresh(broker, now_kst)
                except Exception: pass

            if kis_enabled and broker is not None:
                _pos_sync_tick += 1
                if (pos_sync_on_start and _pos_sync_tick == 1) or (pos_sync_every_ticks > 0 and _pos_sync_tick % pos_sync_every_ticks == 0):
                    order_mgr.sync_positions_from_broker(broker)

            if not market_open:
                try: pending_store.purge_stale_open(max_age_sec=_env_int("PENDING_PURGE_SEC", 7200))
                except Exception: pass

            sell_all_before_close = _env_bool("SELL_ALL_BEFORE_CLOSE", False)
            sell_all_minutes = _env_int("SELL_ALL_BEFORE_CLOSE_MINUTES", 10)
            
            is_force_close_time = False
            if market_open:
                cur_min = ny_time.hour * 60 + ny_time.minute
                close_min = 16 * 60  
                if cur_min >= (close_min - sell_all_minutes) and cur_min < close_min:
                    is_force_close_time = True

            current_rss_tickers = list(WATCHLIST)
            
            # 🚨 [추가] 리스크 평가를 위해 SPY, QQQ는 무조건 1분마다 뉴스 수집 대상에 포함
            for mt in ["SPY", "QQQ"]:
                if mt not in current_rss_tickers:
                    current_rss_tickers.append(mt)

            if quote_provider is not None and not quote_provider.kis.cfg.paper:
                try:
                    hot_tickers = quote_provider.get_volume_surge_tickers("NAS") + quote_provider.get_price_fluct_tickers("NAS")
                    hot_tickers = list(set(hot_tickers)) 
                    for ht in hot_tickers[:5]:
                        if ht not in current_rss_tickers: current_rss_tickers.append(ht)
                except Exception: pass

            universe_path = _env_str("UNIVERSE_PATH", "data/universe.txt")
            try:
                if os.path.exists(universe_path):
                    with open(universe_path, "r", encoding="utf-8") as f:
                        universe_all = [line.strip().upper() for line in f if line.strip()]
                    bg_count, start_idx = 0, bg_idx
                    while bg_count < 2 and (bg_idx - start_idx) < len(universe_all):
                        bg_t = universe_all[bg_idx % len(universe_all)]
                        bg_idx += 1
                        if bg_t not in current_rss_tickers:
                            current_rss_tickers.append(bg_t)
                            bg_count += 1
            except Exception: pass

            loop_rss_urls = build_rss_urls(current_rss_tickers)
            try: news_items = fetch_rss_news(limit=20, rss_urls=loop_rss_urls)
            except Exception: news_items = []

            if quote_provider is not None and not quote_provider.kis.cfg.paper:
                try:
                    kis_news_raw = quote_provider.get_breaking_news()
                    for kn in kis_news_raw:
                        title = kn.get("hts_pbnt_titl_cntt", "")
                        if title:
                            news_items.insert(0, { "title": title, "summary": "한국투자증권 실시간 속보", "link": f"kis_news_{kn.get('cntt_usiq_srno', '')}", "published": f"{kn.get('data_dt', '')}{kn.get('data_tm', '')}" })
                except Exception: pass

            # 👇 [추가] 이번 1분 동안 새로운 매크로(지수) 뉴스가 들어왔는지 체크하는 깃발
            macro_news_added = False 

            analyzed_links, skipped_seen, skipped_low_signal, skipped_no_candidate, llm_fail = 0, 0, 0, 0, 0

            for item in news_items:
                link, title, summary, published = (item.get("link") or "").strip(), (item.get("title") or "").strip(), (item.get("summary") or "").strip(), (item.get("published") or "").strip()
                if not link: continue
                if link in seen:
                    skipped_seen += 1; continue
                if not _is_high_signal(title, summary):
                    skipped_low_signal += 1; continue

                candidates = _candidate_tickers(title, summary, current_rss_tickers)
                if not candidates:
                    skipped_no_candidate += 1; continue

                try: evt = analyze_news_local_ollama(title=title, summary=summary, link=link, published=published, watchlist=current_rss_tickers)
                except Exception: llm_fail += 1; continue

                try: escore = float(event_score(evt))
                except Exception: escore = 0.0
                try: econf = float(evt.get("confidence", 0.55))
                except Exception: econf = 0.55

                llm_tickers = evt.get("tickers") if isinstance(evt, dict) else None
                if isinstance(llm_tickers, list): assigned = [t for t in llm_tickers if str(t).upper() in current_rss_tickers]
                else: assigned = []
                if not assigned: assigned = candidates

                for t in assigned:
                    t_upper = str(t).upper()
                    
                    # 👇 [추가] 방금 들어온 뉴스가 시장 전체(SPY, QQQ) 뉴스라면 깃발을 번쩍 듭니다!
                    if t_upper in ["SPY", "QQQ"]:
                        macro_news_added = True

                    if t_upper not in WATCHLIST and (t_upper in universe_all or t_upper in current_rss_tickers):
                        is_urgent = (escore >= 0.8) or (int(evt.get("impact", 0)) >= 2)
                        if is_urgent:
                            max_wl_size = 20
                            if len(WATCHLIST) >= max_wl_size:
                                removable = [wt for wt in WATCHLIST if float(get_position(positions, wt).qty) == 0 and wt not in inverse_tickers]
                                if removable:
                                    kicked_out = removable[0] 
                                    WATCHLIST.remove(kicked_out)
                                    if kicked_out in current_rss_tickers: current_rss_tickers.remove(kicked_out)
                            WATCHLIST.append(t_upper)
                            if t_upper not in current_rss_tickers: current_rss_tickers.append(t_upper)
                            notifier.send(f"🚨 [속보 스와핑 발동!]\n🔥 AI가 {t_upper}의 강력한 호재 감지!\n📊 점수: {escore:.2f} / 파급력: {evt.get('impact', 0)}\n⚔️ 봇 투입!")

                analyzed_links += 1
                _append_jsonl(EVENT_LOG_PATH, {"ts": loop_ts, "link": link, "title": title, "published": published, "assigned": assigned, "event_type": evt.get("event_type"), "sentiment": evt.get("sentiment"), "impact": evt.get("impact"), "confidence": evt.get("confidence"), "trade_horizon": evt.get("trade_horizon"), "why_it_moves": evt.get("why_it_moves"), "event_score": escore})
                kr_title = str(evt.get("kr_title", "")).strip()
                display_title = kr_title if kr_title else title
                notifier.send(fmt_news(tickers=",".join(assigned), title=display_title, score=float(escore), event_type=str(evt.get("event_type", "") or ""), sentiment=str(evt.get("sentiment", "") or ""), conf=float(evt.get("confidence", 0.0) or 0.0), link=link))

                for t in assigned:
                    try: news_store.add_event(NewsEvent(ticker=t, ts_kst=loop_ts, published=published, title=title, summary=summary, link=link, event_score=float(escore), confidence=float(econf), event_type=str(evt.get("event_type", "other")), sentiment=str(evt.get("sentiment", "neutral")), impact=int(evt.get("impact", 0) or 0), why_it_moves=str(evt.get("why_it_moves", "") or ""), raw=evt if isinstance(evt, dict) else {}))
                    except Exception: pass
                    try: kb_add_evidence(t, source="news", title=title, summary=str(evt.get("why_it_moves", "") or summary), link=link, sentiment=str(evt.get("sentiment", "neutral")), impact=int(evt.get("impact", 0) or 0), tags=[str(evt.get("event_type", "other"))], raw=evt if isinstance(evt, dict) else {})
                    except Exception: pass
                seen.add(link)
                try: mark_seen(link)
                except Exception: pass

            # ==================================================
            # 🚨 [이벤트 기반 즉각 대응] 새 매크로 뉴스가 들어온 그 순간(1분 이내)에만 평가!
            # ==================================================
            # 처음 시작할 때(None) 또는 새로운 SPY/QQQ 뉴스가 수집되었을 때만 AI 평가 진행
            if last_macro_eval_time is None or macro_news_added:
                new_macro_level, new_macro_reason = _evaluate_macro_risk(news_store, ai_gate_model)
                
                # 처음 시작이거나, 위험 레벨이 이전과 '다르게' 변경되었을 때만 텔레그램 알림! (스팸 방지)
                if last_macro_eval_time is None or new_macro_level != macro_risk_level:
                    if new_macro_level >= 2:
                        notifier.send(f"🚨 [매크로 리스크 비상 경보!]\n새로운 주요 뉴스 감지! 위험 단계가 Risk Level {new_macro_level}로 변경되었습니다!\n진단: {new_macro_reason}")
                    else:
                        notifier.send(f"🌍 [AI 매크로 리스크 진단]\n위험 단계: Risk Level {new_macro_level}\n진단: {new_macro_reason}")
                elif macro_news_added:
                    # 위험 단계는 안 변했지만, AI가 새 뉴스를 읽고 "이상 없음" 판정을 내렸다는 내부 로그
                    print(f"[MACRO] 새 지수 뉴스 확인됨 -> 기존 위험단계 유지 (Lv {new_macro_level})")
                
                macro_risk_level = new_macro_level
                macro_risk_reason = new_macro_reason
                last_macro_eval_time = now_kst
            # ==================================================

            for t in WATCHLIST:
                try:
                    new_n = news_store.pop_new_event_count(t)
                    if new_n >= news_mem_update_every:
                        recent = news_store.get_recent_events(t, days=news_window_days, limit=news_mem_max_items)
                        if recent:
                            mem = build_news_memory_summary_local_ollama(ticker=t, events=recent, model=news_mem_model)
                            news_store.save_memory(t, mem)
                except Exception: pass

            regime = None
            buy_block = False
            try: regime = regime_engine.get() if regime_engine is not None else None
            except Exception: regime = None

            th_mult, scalp_w_mult, size_mult = 1.0, 1.0, 1.0
            if regime is not None:
                try:
                    if float(getattr(regime, "score", 0.0)) <= float(regime_risk_off):
                        th_mult, scalp_w_mult, size_mult, buy_block = float(regime_th_mult), float(regime_scalp_w_mult), float(regime_size_mult), bool(regime_buy_block)
                except Exception: pass

            # ==================================================
            # 🚨 [수정] AI 매크로 연동 '가변형 서킷브레이커'
            # ==================================================
            if quote_provider is not None and not quote_provider.kis.cfg.paper:
                try:
                    qqq_quote = quote_provider.get_quote(regime_symbol) 
                    qqq_out = qqq_quote.raw.get("output") or qqq_quote.raw
                    qqq_rate = float(qqq_out.get("rate", 0.0))
                    
                    # AI가 평가한 Risk Level에 따라 폭락 트리거 기준을 동적으로 조정!
                    if macro_risk_level >= 3:
                        cb_threshold = -0.3 # 전시 상황 (극도 민감): -0.3%만 빠져도 즉시 방어
                    elif macro_risk_level == 2:
                        cb_threshold = -0.8 # 악재 상황 (경계): -0.8% 하락 시 방어
                    else:
                        cb_threshold = -1.5 # 평온한 시장 (기본): -1.5% 대폭락 시에만 방어
                    
                    if qqq_rate <= cb_threshold:
                        buy_block = True
                        th_mult = float(regime_th_mult) if regime_enabled else 1.50
                        
                        if regime is None:
                            class TempRegime: pass
                            regime = TempRegime()
                        
                        regime.score = -1.0
                        regime.label = f"BLACK_SWAN(Lv{macro_risk_level})"
                        regime.analysis = f"🚨 매크로 위험(Lv{macro_risk_level})! 방어선({cb_threshold}%) 붕괴 감지 (현재 {qqq_rate:.2f}%) - 매수 전면 차단 및 인버스 가동"
                except Exception as e:
                    pass
            # ==================================================

            # 🚨 [복구] 15분마다 시장 판단(Regime) 텔레그램 알림 전송 (전투 모드일 때만)
            if regime is not None and is_combat_mode:
                if last_regime_noti_time is None or (now_kst - last_regime_noti_time).total_seconds() >= 15 * 60:
                    r_score = float(getattr(regime, "score", 0.0))
                    r_label = str(getattr(regime, "label", "unknown"))
                    r_ai_text = str(getattr(regime, "reason", getattr(regime, "analysis", "분석 내용 없음")))
                    if r_score <= float(regime_risk_off): kr_label, inv_status = "📉 하락장 (Risk-Off)", "일반 종목 매수 차단 / 인버스 매수 허용"
                    elif r_score >= 0.1: kr_label, inv_status = "📈 상승장 (Risk-On)", "일반 종목 적극 매수"
                    else: kr_label, inv_status = "⚖️ 횡보/중립장 (Neutral)", "기본 매매 진행"
                    notifier.send(f"🤖 [AI 시장 판단]\n상태: {kr_label} ({r_label})\n점수: {r_score:.2f}\n분석: {r_ai_text}\n봇 대응: {inv_status}")
                    last_regime_noti_time = now_kst

            # ---------- 2) ticker loop ----------
            for ticker in WATCHLIST:
                pos = get_position(positions, ticker)
                pos.reset_if_new_day(now_kst)
                
                is_inverse = ticker in inverse_tickers

                try: snap = fetch_snapshot(ticker)
                except Exception: snap = {"ticker": ticker, "price": None}

                if quote_provider is not None:
                    try: snap["price"] = quote_provider.get_quote(ticker).price
                    except Exception: pass

                price = snap.get("price")
                if price is None: continue
                price_f = float(price)

                dq = _recent_px.get(ticker)
                if dq is None:
                    dq = deque(maxlen=max(30, chase_ban_spike_window + 5))
                    _recent_px[ticker] = dq
                dq.append(price_f)

                fair_value, fair_range, vscore = None, None, 0.0
                if not is_inverse: 
                    try:
                        fv = compute_fair_value_snapshot(snap)
                        vscore = float(fv.get("value_score", 0.0))
                        fair_value, fair_range = fv.get("fair_value"), fv.get("fair_value_range")
                    except Exception: pass

                tscore, tlabel = 0.0, "unknown"
                try:
                    daily = fetch_daily_ta(ticker)
                    ta_out = ta_score(daily)
                    tscore, tlabel = float(ta_out.get("ta_score", 0.0)), str(ta_out.get("ta_label", "unknown"))
                except Exception: pass

                raw_news, cnt, news_used, conf_avg = 0.0, 0, 0.0, 0.55
                if not is_inverse:
                    ns = news_store.compute_signal(ticker=ticker, now_kst=now_kst, half_life_hours=news_half_life_h, window_days=news_window_days, max_items=200)
                    raw_news, cnt = float(ns.get("raw_sum", 0.0)), int(ns.get("raw_n", 0))
                    news_used = _clamp(float(ns.get("news_score", 0.0)), -2.0, 2.0)
                    conf_avg = _clamp(float(ns.get("news_conf", 0.55)), 0.0, 1.0)
                    if news_dup_penalty_enabled:
                        try:
                            recent_events = news_store.get_recent_events(ticker, days=1, limit=5) or []
                            titles = [t for t in [str(e.get("title", "")).strip() for e in recent_events if isinstance(e, dict)] if t]
                            if len(titles) >= 2 and titles[0] == titles[1]: news_used *= float(news_dup_mult)
                        except Exception: pass

                is_value_dip = False 
                if is_inverse:
                    if buy_block and regime is not None:
                        total_base = float(tscore) + abs(float(getattr(regime, "score", 0.0)))
                    else:
                        total_base = -1.0 
                else:
                    total_base = _total_score(news_used, vscore, tscore)
                    if vscore >= 0.60 and tscore <= -0.30:
                        total_base = float(buy_th) + 0.15 
                        tlabel = f"ValueDip({tlabel})"    
                        is_value_dip = True               

                total = float(total_base)

                base_buy_th_eff = float(buy_th) * float(th_mult)
                base_sell_th_eff = float(sell_th) * float(th_mult)

                dyn_rules_obj = dyn_rules.apply(news_store=news_store, ticker=ticker, now_kst=now_kst, base_buy_th=base_buy_th_eff, base_sell_th=base_sell_th_eff, base_conf_th=conf_th, base_confirm_ticks=confirm_ticks, max_scan=30)
                buy_th_eff, sell_th_eff, conf_th_eff, confirm_ticks_eff, strength_boost = dyn_rules_obj.buy_th, dyn_rules_obj.sell_th, dyn_rules_obj.conf_th, dyn_rules_obj.confirm_ticks, dyn_rules_obj.strength_boost

                block_reason, force_sell, force_sell_frac, risk_reason, rd = "", False, 0.0, "", None
                if not is_inverse:
                    try:
                        rd, buy_delta, sell_delta = risk_override.evaluate(news_store=news_store, ticker=ticker, now_kst=now_kst, pos_qty=float(pos.qty), max_scan=25)
                        buy_th_eff = _clamp(float(buy_th_eff) + float(buy_delta), 0.05, 0.95)
                        sell_th_eff = _clamp(float(sell_th_eff) + float(sell_delta), -0.95, -0.05)
                        force_sell, force_sell_frac, risk_reason = bool(rd.force_sell), float(rd.sell_frac), str(rd.reason)
                    except Exception as e: risk_reason = f"risk_err={e!r}"

                sig = decide_signal(total_score=total, confidence=conf_avg, ta_label=tlabel, buy_th=buy_th_eff, sell_th=sell_th_eff, conf_th=conf_th_eff)

                if buy_block and is_inverse and sig.action != "SELL":
                    try: sig = type(sig)("BUY", 1.0, f"INVERSE_FORCED_BUY (risk_off) | was {sig.action}")
                    except Exception: pass

                try: sig = type(sig)(sig.action, min(1.0, float(sig.strength) + float(strength_boost)), sig.reason + f" | {dyn_rules_obj.reason}")
                except Exception: pass

                strength_floor = _env_float("SIG_STRENGTH_FLOOR", 0.12)
                if sig.action in ("BUY", "SELL"):
                    try: sig = type(sig)(sig.action, _clamp(max(float(sig.strength), float(strength_floor)), 0.0, 1.0), sig.reason + f" | STR_FLOOR({strength_floor:.2f})")
                    except Exception: pass

                if rd is not None and bool(getattr(rd, "block_buy", False)) and sig.action == "BUY":
                    try: sig = type(sig)("HOLD", float(sig.strength), sig.reason + " | " + risk_reason + " | BUY_BLOCKED")
                    except Exception: sig.action = "HOLD"
                    block_reason = "RISK_OVERRIDE_BLOCK_BUY"

                pos.update_streak(sig.action)

                if is_inverse:
                    eff_sl1 = _env_float("INV_STOP_LOSS_1", -0.020)
                    eff_sl2 = _env_float("INV_STOP_LOSS_2", -0.030)
                    eff_tp1 = _env_float("INV_TAKE_PROFIT_1", 0.030)
                else:
                    eff_sl1, eff_sl2, eff_tp1 = stop_loss_1, stop_loss_2, take_profit_1

                # =====================================================================
                # 🚨 [신규 추가] 전투 모드(is_combat_mode)가 아닐 경우 주문 전면 차단 (리서치 모드)
                # =====================================================================
                if not is_combat_mode and sig.action in ("BUY", "SELL"):
                    plan_action, plan_qty, plan_reason = "HOLD", 0, f"RESEARCH_MODE (Trade window {trade_start_et}-{trade_end_et} ET) | raw={sig.action}"
                    if not block_reason: block_reason = "RESEARCH_MODE"
                else:
                    plan = compute_position_plan(
                        pos=pos, raw_action=sig.action, strength=sig.strength, price=price_f, confirm_ticks=confirm_ticks_eff,
                        fast_track_strength=fast_track_strength, 
                        stop_loss_1=eff_sl1, stop_loss_2=eff_sl2, take_profit_1=eff_tp1, 
                        stop_sell_frac=stop_sell_frac, tp_sell_frac=tp_sell_frac, base_qty=float(base_qty), max_position_qty=float(max_position_qty),
                        buy_t1=buy_t1, buy_t2=buy_t2, buy_t3=buy_t3, buy_m1=buy_m1, buy_m2=buy_m2, buy_m3=buy_m3,
                        sell_t1=sell_t1, sell_t2=sell_t2, sell_t3=sell_t3, sell_f1=sell_f1, sell_f2=sell_f2, sell_f3=sell_f3,
                    )
                    plan_action, plan_qty, plan_reason = plan.action, _to_int_qty(plan.qty), f"{plan.reason} | sig={sig.reason}"

                    if plan_action in ("BUY", "SELL") and plan_qty <= 0:
                        plan_action, plan_reason = "HOLD", f"{plan_reason} | int_qty became 0"
                        if not block_reason: block_reason = "QTY_ZERO"

                    if force_sell and float(pos.qty) > 0:
                        fs_qty = int(math.floor(float(pos.qty) * force_sell_frac))
                        plan_action, plan_qty, plan_reason = "SELL", max(fs_qty, 1), f"FORCE_SELL({force_sell_frac:.2f}) due to news risk | {risk_reason} | {plan_reason}"

                    if plan_action == "BUY":
                        ok_trade, why = can_trade(pos=pos, now=now_kst, action=plan_action, cooldown_minutes=cooldown_minutes, max_trades_per_day=max_trades_per_day, max_position_qty=float(max_position_qty))
                        if not ok_trade:
                            plan_action, plan_qty, plan_reason = "HOLD", 0, f"risk blocked: {why} | {plan_reason}"
                            if not block_reason: block_reason = "CAN_TRADE_BLOCK"

                if plan_action in ("BUY", "SELL") and plan_qty > 0:
                    plan_qty_eff = int(math.floor(float(plan_qty) * float(size_mult)))
                    if plan_qty_eff <= 0: plan_qty_eff = 1
                    if plan_qty_eff != plan_qty:
                        plan_reason = f"REGIME_SIZE x{size_mult:.2f} ({plan_qty}->{plan_qty_eff}) | {plan_reason}"
                        plan_qty = plan_qty_eff

                if session_guard_enabled and market_open and plan_action in ("BUY", "SELL") and plan_qty > 0 and _us_session_avoid_window(now_kst, session_guard_start_min, session_guard_end_min):
                    plan_action, plan_qty, plan_reason = "HOLD", 0, f"SESSION_GUARD({session_guard_start_min}/{session_guard_end_min}m) | {plan_reason}"

                if whipsaw_guard_enabled and plan_action in ("BUY", "SELL") and plan_qty > 0:
                    desired = plan_action
                    last_a = _whipsaw_last_action.get(ticker, "HOLD")
                    last_tick = _whipsaw_last_tick.get(ticker, -10**9)
                    opposite = (last_a == "BUY" and desired == "SELL") or (last_a == "SELL" and desired == "BUY")
                    if opposite and (_global_tick - last_tick) <= max(1, int(whipsaw_cooldown_ticks)):
                        plan_action, plan_qty, plan_reason = "HOLD", 0, f"WHIPSAW_BLOCK({last_a}->{desired}) | {plan_reason}"
                    else:
                        _whipsaw_last_action[ticker], _whipsaw_last_tick[ticker] = desired, _global_tick

                if plan_action in ("BUY", "SELL"):
                    has_open = pending_store.has_open_order(ticker)
                    if broker is not None and not broker.kis.cfg.paper:
                        try:
                            real_open_orders = broker.inquire_unfilled(excg_cd="NASD") 
                            for order in real_open_orders:
                                if order.get("pdno") == ticker and int(order.get("nccs_qty", 0)) > 0:
                                    has_open = True; break
                        except Exception: pass

                    if has_open:
                        plan_action, plan_qty, plan_reason = "HOLD", 0, f"blocked: open pending order exists for {ticker} (Real Server Confirmed)"

                if plan_action == "BUY" and plan_qty > 0:
                    ok_lim, lim_reason = limit_store.allow(ticker=ticker, now_kst=now_kst, cooldown_seconds=ticker_cooldown_sec, max_orders_per_ticker_per_day=max_orders_per_ticker_day)
                    if not ok_lim: plan_action, plan_qty, plan_reason = "HOLD", 0, f"limit blocked: {lim_reason} | {plan_reason}"

                decision_msg, decision_action, decision_conf = "", None, None
                
                if decision_enabled and not is_inverse:
                    decision_tick_counter[ticker] = decision_tick_counter.get(ticker, 0) + 1
                    if decision_tick_counter[ticker] >= max(1, decision_every_ticks):
                        decision_tick_counter[ticker] = 0
                        try:
                            kb = kb_load(ticker)
                            recent_events = news_store.get_recent_events(ticker, days=news_window_days, limit=12) or []
                            recent_light = [{"ts_kst": e.get("ts_kst"), "title": e.get("title"), "summary": e.get("summary"), "event_type": e.get("event_type"), "sentiment": e.get("sentiment"), "impact": e.get("impact"), "why_it_moves": e.get("why_it_moves"), "link": e.get("link"), "event_score": e.get("event_score"), "confidence": e.get("confidence")} for e in recent_events if isinstance(e, dict)]
                            snapshot = {"ts_kst": loop_ts, "price": price_f, "market_open": market_open, "regime": {"enabled": bool(regime_engine is not None), "score": getattr(regime, "score", None), "label": getattr(regime, "label", None)}, "signal": {"total": total, "raw_action": sig.action, "strength": sig.strength, "reason": sig.reason}, "plan": {"action": plan_action, "qty": int(plan_qty), "reason": plan_reason}, "position": {"qty": float(pos.qty), "avg": float(pos.avg_price)}, "news": {"news_score": news_used, "raw_n": cnt, "raw_sum": raw_news, "conf": conf_avg}, "valuation": {"score": vscore, "fair_value": fair_value, "range": fair_range}, "ta": {"score": tscore, "label": tlabel}}
                            prompt = build_decision_prompt(ticker=ticker, kb=kb, snapshot=snapshot, recent_news_events=recent_light)
                            llm_text = ollama_generate(prompt=prompt, model=decision_model, temperature=0.2, timeout=float(os.environ.get("OLLAMA_TIMEOUT", "120") or "120"))
                            decision = parse_decision(llm_text)
                            decision_action, decision_conf = str(decision.get("action", "HOLD")).upper(), float(decision.get("confidence", 0.5))

                            kb_add_decision(ticker, action=decision_action, confidence=decision_conf, rationale=str(decision.get("rationale", "")), key_drivers=decision.get("key_drivers") or [], key_risks=decision.get("key_risks") or [], valuation_view=str(decision.get("valuation_view", "")), counterfactuals=decision.get("counterfactuals") or [], next_checks=decision.get("next_checks") or [], raw=decision)
                            _append_jsonl(DECISION_LOG_PATH, {"ts": loop_ts, "ticker": ticker, "decision": decision, "snapshot": snapshot})
                            decision_msg = f" decision={decision_action} dconf={decision_conf:.2f}"

                            if (not decision_compare_only) and is_combat_mode and decision_conf >= decision_min_conf and decision_action in ("BUY", "SELL", "HOLD"):
                                if decision_action == "HOLD": plan_action, plan_qty, plan_reason = "HOLD", 0, f"DECISION_OVERRIDE: HOLD | {plan_reason}"
                                else:
                                    prefer_qty = int(decision.get("position_plan", {}).get("prefer_qty", 0) or 0)
                                    if prefer_qty > 0: plan_qty = prefer_qty
                                    plan_action, plan_reason = decision_action, f"DECISION_OVERRIDE: {decision_action} conf={decision_conf:.2f} | {plan_reason}"
                        except Exception as e: decision_msg = f" decision_err={e!r}"

                ai_msg = ""
                if ai_gate_enabled and not is_inverse and not is_value_dip and plan_action in ("BUY", "SELL") and plan_qty > 0:
                    try:
                        gate = ai_gate_check_local_ollama(ticker=ticker, action=plan_action, qty=int(plan_qty), price=price_f, total=total, news_used=news_used, val_score=vscore, ta_score=tscore, ta_label=tlabel, signal_reason=sig.reason, plan_reason=plan_reason, market_open=market_open, memory_summary=news_store.load_memory(ticker), recent_events=news_store.get_recent_events(ticker, days=news_window_days, limit=20), model=ai_gate_model, min_conf=ai_gate_min_conf)
                        if not gate.allow or gate.qty_mult <= 0.0:
                            ai_msg, plan_action, plan_qty, plan_reason = f" ai_gate=VETO conf={gate.confidence:.2f} reason={gate.reason}", "HOLD", 0, f"AI_GATED: {gate.reason} | {plan_reason}"
                        else:
                            new_qty = int(math.floor(plan_qty * gate.qty_mult))
                            if new_qty <= 0: ai_msg, plan_action, plan_qty, plan_reason = f" ai_gate=VETO(qty->0) conf={gate.confidence:.2f} reason={gate.reason}", "HOLD", 0, f"AI_GATED: {gate.reason} | {plan_reason}"
                            elif new_qty != plan_qty: ai_msg, plan_qty, plan_reason = f" ai_gate=REDUCE x{gate.qty_mult:.2f} conf={gate.confidence:.2f} reason={gate.reason}", new_qty, f"AI_REDUCED: {gate.reason} | {plan_reason}"
                            else: ai_msg = f" ai_gate=OK conf={gate.confidence:.2f}"
                    except Exception as e: ai_msg = f" ai_gate_err={e!r}"

                if broker is not None and plan_action in ("BUY", "SELL") and plan_qty > 0:
                    ok_acc, why_acc = acc_risk.allow_order(ticker=ticker, action=plan_action, qty=int(plan_qty), price=float(price_f))
                    if not ok_acc:
                        plan_action, plan_qty, plan_reason = "HOLD", 0, f"ACCOUNT_RISK_BLOCK: {why_acc} | {plan_reason}"
                        print(f"[ACC_RISK_BLOCK] {ticker} {why_acc}")

                if plan_action == "BUY":
                    if buy_block: 
                        if is_inverse: plan_reason = f"INVERSE_ALLOWED (risk_off) | {plan_reason}"
                        else:
                            plan_action, plan_qty, plan_reason = "HOLD", 0, f"REGIME_BLOCK (risk_off) | {plan_reason}"
                            if not block_reason: block_reason = "REGIME_BLOCK"
                    else: 
                        if is_inverse:
                            plan_action, plan_qty, plan_reason = "HOLD", 0, f"INVERSE_BLOCKED (risk_on) | {plan_reason}"
                            if not block_reason: block_reason = "INVERSE_BLOCK"

                if chase_ban_enabled and plan_action == "BUY" and plan_qty > 0:
                    hot, hot_reason = False, ""
                    if scalp_engine is not None:
                        pass # 스캘핑 로직 생략
                    dq2 = _recent_px.get(ticker)
                    if dq2 is not None and len(dq2) >= chase_ban_spike_window:
                        base_px = float(dq2[-chase_ban_spike_window])
                        spike = (price_f - base_px) / max(1e-9, base_px)
                        if spike >= chase_ban_after_spike_pct: hot, hot_reason = True, f"SPIKE {spike:.4f} >= {chase_ban_after_spike_pct:.4f} (win={chase_ban_spike_window})"

                    if hot:
                        hard_block_pct = _env_float("CHASE_BAN_HARD_BLOCK_PCT", 0.012) 
                        do_hard_block = False
                        try:
                            if dq2 is not None and len(dq2) >= chase_ban_spike_window:
                                base_px = float(dq2[-chase_ban_spike_window])
                                spike = (price_f - base_px) / max(1e-9, base_px)
                                if spike >= hard_block_pct: do_hard_block = True
                        except Exception: pass

                        if do_hard_block:
                            plan_action, plan_qty, plan_reason = "HOLD", 0, f"CHASE_HARD_BLOCK: {hot_reason} | {plan_reason}"
                            if not block_reason: block_reason = "CHASE_HARD_BLOCK"
                        else:
                            new_qty = max(int(math.floor(float(plan_qty) * _env_float("CHASE_BAN_QTY_MULT", 0.30))), 1)
                            chase_confirm = _env_int("CHASE_BAN_CONFIRM_TICKS", 2)
                            if new_qty != plan_qty: plan_qty, plan_reason = new_qty, f"CHASE_SOFT: {hot_reason} | qty ({plan_qty}->{new_qty}) | confirm>= {chase_confirm} | {plan_reason}"
                            else: plan_reason = f"CHASE_SOFT: {hot_reason} | confirm>= {chase_confirm} | {plan_reason}"
                            if not block_reason: block_reason = "CHASE_SOFT"

                if cost_gate_enabled and plan_action == "BUY" and plan_qty > 0:
                    total_cost_bps = float(cost_fee_bps) + float(cost_spread_bps) + float(cost_slip_bps)
                    exp_edge_bps = abs(float(total)) * float(edge_per_score) * 10000.0
                    if exp_edge_bps < total_cost_bps * float(edge_min_mult):
                        plan_action, plan_qty, plan_reason = "HOLD", 0, f"COST_BLOCK edge={exp_edge_bps:.2f}bps < cost={total_cost_bps:.2f}bps*{edge_min_mult:.2f} | {plan_reason}"

                if is_force_close_time and float(pos.qty) > 0:
                    if sell_all_before_close or is_inverse:
                        if not pending_store.has_open_order(ticker):
                            plan_action, plan_qty = "SELL", int(float(pos.qty))
                            plan_reason = f"FORCE_SELL_CLOSE: {'인버스 강제 당일청산' if is_inverse else '장 마감 매도 옵션 작동'}"
                        else:
                            plan_action, plan_qty, plan_reason = "HOLD", 0, f"FORCE_SELL_CLOSE_WAIT: 장 마감 강제청산 시간이나 미체결 주문 대기 중"

                order_msg = ""
                if broker is not None and plan_action in ("BUY", "SELL") and plan_qty > 0:
                    try:
                        res = broker.buy_market(ticker, int(plan_qty), last_price=price_f) if plan_action == "BUY" else broker.sell_market(ticker, int(plan_qty), last_price=price_f)
                        ok, dry = bool(res.ok), bool(res.raw.get("dry_run", False))
                        order_msg = f" order_ok={ok} dry_run={dry} order_no={getattr(res, 'order_no', None)}"
                        if ok:
                            log_data = {"ts": loop_ts, "ticker": ticker, "action": plan_action, "qty": int(plan_qty), "dry_run": dry, "price_snapshot": price_f, "total_score": total, "confidence": conf_avg, "ta_label": tlabel, "news_used": news_used, "val_score": vscore, "ta_score": tscore, "reason": plan_reason, "market_open_us_regular": market_open, "raw_signal": {"action": sig.action, "strength": sig.strength, "reason": sig.reason}, "decision_action": decision_action, "decision_conf": decision_conf}
                            if not dry:
                                log_data["order_no"] = getattr(res, "order_no", None)
                                order_mgr.register_submitted(ticker=ticker, side=plan_action, qty=float(plan_qty), kis_response=res.raw, fill_price=price)
                                limit_store.update_on_order(ticker, now_kst)
                            trade_logger.log(log_data)
                            notifier.send(fmt_dry_run(ticker=ticker, side=plan_action, qty=int(plan_qty), price=float(price_f), total=float(total), conf=float(conf_avg), ta_label=str(tlabel), reason=str(plan_reason)) if dry else fmt_order_submitted(ticker=ticker, side=plan_action, qty=int(plan_qty), order_no=str(getattr(res, "order_no", None) or ""), price=float(price_f), total=float(total), conf=float(conf_avg), ta_label=str(tlabel), reason=str(plan_reason)))
                    except Exception as e:
                        resp_text = getattr(getattr(e, "response", None), "text", None)
                        order_msg = f" order_err={e!r}" + (f" resp={resp_text[:500]}" if resp_text else "")

                reg_msg = f" regime={getattr(regime, 'label', None)}({float(getattr(regime, 'score', 0.0)):.2f})" if regime is not None else ""
                
                # 전투모드가 아니면 로그 앞부분에 [RESEARCH]를 붙여 출력
                mode_prefix = "[TICK]" if is_combat_mode else "[RESEARCH]"
                print(f"{mode_prefix} {ticker} price={price_f} total={total:.2f} conf={conf_avg:.2f} news={news_used:.2f}(raw={raw_news:.2f},n={cnt}) val={vscore:.2f} ta={tscore:.2f}({tlabel}) raw_sig={sig.action} strength={sig.strength:.3f} pos_qty={pos.qty:.0f} pos_avg={pos.avg_price:.2f} plan={plan_action} qty={plan_qty} market_open={market_open}{reg_msg}{decision_msg}{ai_msg}{order_msg}{' block=' + block_reason if block_reason else ''} plan_reason={plan_reason[:120]}")

            try: save_state(positions, POS_PATH)
            except Exception as e: print(f"[WARN] save positions failed: {e!r}")

            try:
                lab = auto_labeler.run_if_due(now_kst)
                if lab.get("ran") and lab.get("summary_updated"):
                    h, s = "1d", lab.get("summary") or {}
                    bh = (s.get("by_horizon") or {}).get(h) or {}
                    msg = f"새로 채점한 판단: {lab.get('n_new',0)}건\n⏱️ 기준 시간: {h} 뒤\n✅ 평가된 건수: {bh.get('n_eval',0)}건\n🎯 승률: {bh.get('win_rate',0)*100:.1f}%\n💰 평균 수익률: {bh.get('avg_ret',0)*100:.2f}%"
                    if auto_labeler.s.telegram_enabled: notifier.send(fmt_label_summary(msg))
            except Exception: pass

            try:
                rep_run = auto_reporter.run_if_due(now_kst)
                if rep_run.get("ran"):
                    report = rep_run.get("report") or {}
                    msg_1h = PerformanceReporter.format_telegram_summary(report, horizon=os.environ.get("PERF_TELEGRAM_HORIZON", "1h"))
                    if auto_reporter.s.telegram_enabled: notifier.send(fmt_perf_summary(msg_1h))
                    if auto_reporter.s.also_send_1d:
                        msg_1d = PerformanceReporter.format_telegram_summary(report, horizon="1d")
                        if auto_reporter.s.telegram_enabled: notifier.send(fmt_perf_summary(msg_1d))
            except Exception: pass

            time.sleep(max(1, tick_seconds))

    finally: release_lock()

if __name__ == "__main__": main()