# src/trading/ai_gate.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.utils.ollama_client import ollama_generate

@dataclass(frozen=True)
class AIGateDecision:
    allow: bool
    qty_mult: float           # 0..1 (0이면 veto)
    confidence: float         # 0..1
    reason: str
    raw: Optional[Dict[str, Any]] = None


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found. head={text[:200]!r}")
    return m.group(0)


def ai_gate_check_local_ollama(
    *,
    ticker: str,
    action: str,              # BUY/SELL
    qty: int,
    price: float,
    total: float,
    news_used: float,
    val_score: float,
    ta_score: float,
    ta_label: str,
    signal_reason: str,
    plan_reason: str,
    market_open: bool,
    memory_summary: str,
    recent_events: List[Dict[str, Any]],
    model: str = "qwen2.5:7b-instruct",
    timeout_sec: int = 120,
    min_conf: float = 0.55,
) -> AIGateDecision:
    """
    목적:
    - '수익 예측'이 아니라, 명백한 리스크/근거부족/시장상황 부적합이면 veto 또는 수량 축소
    """

    schema = {
        "allow": True,
        "qty_mult": 1.0,
        "confidence": 0.6,
        "reason": "",
    }

    compact = []
    for e in recent_events[-12:]:
        compact.append(
            {
                "title": (e.get("title", "") or "")[:160],
                "published": (e.get("published", "") or "")[:40],
                "event_score": e.get("event_score", 0.0),
                "confidence": e.get("confidence", 0.55),
                "event_type": e.get("event_type", ""),
                "sentiment": e.get("sentiment", ""),
                "why": (e.get("why_it_moves", "") or "")[:200],
            }
        )

    prompt = f"""
Return ONLY valid JSON. No markdown. No extra keys.

Schema:
{json.dumps(schema, ensure_ascii=False)}

You are a risk-gate for an automated US stock trading bot.
You do NOT try to predict profit. You only decide whether to allow the order and whether to reduce size.

Guidelines:
- If evidence is weak/contradictory, set allow=false or qty_mult between 0.0~0.5.
- If market_open is false, allow must be false.
- If action is BUY but ta_label is bearish/weak_bearish and news_used is near 0, prefer reduce or veto.
- If action is SELL but position protection logic suggests no urgency, consider reduce.
- confidence should reflect how sure you are about the gate decision.

Inputs:
ticker={ticker}
market_open={market_open}
action={action}
qty={qty}
price={price}

scores:
total={total}
news_used={news_used}
val_score={val_score}
ta_score={ta_score}
ta_label={ta_label}

signal_reason={signal_reason}
plan_reason={plan_reason}

memory_summary:
{memory_summary}

recent_events(latest last):
{json.dumps(compact, ensure_ascii=False)}
""".strip()

    raw = ollama_generate(
        prompt=prompt,
        model=model,
        temperature=0.2,
        timeout=float(timeout_sec),
    )
    obj = json.loads(_extract_json(raw))

    allow = bool(obj.get("allow", True))
    qty_mult = float(obj.get("qty_mult", 1.0))
    qty_mult = max(0.0, min(1.0, qty_mult))
    conf = float(obj.get("confidence", 0.6))
    conf = max(0.0, min(1.0, conf))
    reason = str(obj.get("reason", "")).strip()

    # 최소 신뢰도 미달이면 veto
    if conf < float(min_conf):
        allow = False
        qty_mult = 0.0
        if not reason:
            reason = f"AI gate confidence {conf:.2f} < {min_conf:.2f}"

    if not market_open:
        allow = False
        qty_mult = 0.0
        if not reason:
            reason = "market closed"

    return AIGateDecision(allow=allow, qty_mult=qty_mult, confidence=conf, reason=reason, raw=obj)