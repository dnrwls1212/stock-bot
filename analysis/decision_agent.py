# src/analysis/decision_agent.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# 너 프로젝트의 ollama 호출 유틸이 있다면 그걸 import해서 써도 됨.
# 여기서는 "외부 함수 하나만" 맞춰주면 되는 형태로 설계.
# main.py에서 call_llm(prompt: str) -> str 형태로 넘겨주면 됨.


def _clip(items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return items[:n] if len(items) > n else items


def build_decision_prompt(
    *,
    ticker: str,
    kb: Dict[str, Any],
    snapshot: Dict[str, Any],
    recent_news_events: List[Dict[str, Any]],
) -> str:
    """
    kb: TickerKB를 dict(asdict) 형태로 넣어도 되고, 필요한 필드만 추려서 넣어도 됨.
    snapshot 예:
      {
        "price": 187.9,
        "market_open": false,
        "ta": {"label":"bullish","score":0.6,"key_levels": {...}},
        "valuation": {"fair_value": 210.0, "score":0.2, "assumptions": {...}},
        "news": {"news_score": 0.2, "raw_n": 4},
        "position": {"qty":0,"avg":0.0},
        "risk": {"can_trade": true, "reason":""}
      }
    """
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
        "recent_decisions": _clip(kb.get("decisions", []) or [], 5),
        "recent_evidence": _clip(kb.get("evidence", []) or [], 12),
    }

    news_light = _clip(recent_news_events, 10)

    schema = {
        "action": "BUY|SELL|HOLD",
        "confidence": 0.0,
        "rationale": "string (Korean, concise, factual)",
        "key_drivers": ["..."],
        "key_risks": ["..."],
        "valuation_view": "what changed / what assumption matters",
        "counterfactuals": ["what would make you wrong next 1-4 weeks"],
        "next_checks": ["what data to check next (earnings, guidance, orders, macro...)"],
        "position_plan": {
            "prefer_qty": 0,
            "max_loss_pct": 0.0,
            "time_horizon": "swing_days|swing_weeks|long_months"
        }
    }

    return f"""
너는 '누적 지식 기반' 투자 의사결정 에이전트다.
목표: 단기/스윙 수익을 내되, 기업 방향성과 근거 축적을 최우선으로 한다.
규칙:
- 모르면 모른다고 말하고 next_checks에 확인할 항목을 적는다.
- 근거는 제공된 데이터에서만. 추측 금지.
- 출력은 반드시 JSON 하나만. 다른 문장/설명/코드블럭 금지.
- action은 BUY/SELL/HOLD 중 하나.

[TICKER]
{ticker}

[KB]
{json.dumps(kb_light, ensure_ascii=False)}

[SNAPSHOT]
{json.dumps(snapshot, ensure_ascii=False)}

[RECENT_NEWS_EVENTS]
{json.dumps(news_light, ensure_ascii=False)}

[OUTPUT_JSON_SCHEMA]
{json.dumps(schema, ensure_ascii=False)}
""".strip()


def parse_decision_json(text: str) -> Dict[str, Any]:
    """
    LLM이 JSON만 내보내도록 강제했지만, 혹시 앞뒤 잡텍스트가 섞일 수 있어 방어.
    """
    text = text.strip()
    # 가장 바깥 {} 영역만 추출 시도
    if not text.startswith("{"):
        i = text.find("{")
        j = text.rfind("}")
        if i >= 0 and j > i:
            text = text[i:j+1]
    return json.loads(text)