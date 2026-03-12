import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


def analyze_news_with_llm(
    title: str,
    summary: str,
    link: str,
    published: str,
    tickers_hint: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    뉴스 1건을 LLM으로 분석해서 구조화된 JSON으로 반환.
    Structured Outputs(JSON Schema)로 필드 누락/형식 깨짐을 방지.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 .env에 없습니다.")

    client = OpenAI(api_key=api_key)

    # JSON Schema: 필요한 필드만 MVP로 최소화
    schema = {
        "name": "news_analysis",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
                "event_type": {
                    "type": "string",
                    "enum": [
                        "earnings",
                        "guidance",
                        "contract_partnership",
                        "regulation_export",
                        "mna",
                        "financing_offering",
                        "product_launch",
                        "litigation_investigation",
                        "macro_sector",
                        "other",
                    ],
                },
                "sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                "impact": {"type": "integer", "minimum": -3, "maximum": 3},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "key_points": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
                "trade_horizon": {"type": "string", "enum": ["intraday", "swing_days", "swing_weeks", "long_term"]},
                "why_it_moves": {"type": "string"},
                # 👇👇👇 [신규 추가] 뉴스 속 미래 일정 추출 필드
                "upcoming_event_date": {
                    "type": "string", 
                    "description": "If the news mentions a specific FUTURE event date (e.g., keynote, product launch, FDA approval), extract it in YYYY-MM-DD format. If none, leave as empty string."
                },
                "upcoming_event_desc": {
                    "type": "string",
                    "description": "Short description of the future event. If none, leave as empty string."
                }
            },
            "required": [
                "tickers",
                "event_type",
                "sentiment",
                "impact",
                "confidence",
                "key_points",
                "trade_horizon",
                "why_it_moves",
                "upcoming_event_date", # 👈 필수 응답값으로 추가
                "upcoming_event_desc"  # 👈 필수 응답값으로 추가
            ],
        },
        "strict": True,
    }

    tickers_text = ", ".join(tickers_hint or [])
    # 👇 프롬프트에 "과거의 일은 무시하고 미래의 중요한 일정만 잡아내라"는 지시 추가
    prompt = f"""
You are an analyst for event-driven stock trading signals.
Analyze the news item and output ONLY valid JSON following the schema.

If tickers are not explicit, infer cautiously from the content (or return an empty list).
[CRITICAL] For `upcoming_event_date`, ONLY extract dates for FUTURE major events (like product launches, keynotes, earnings, conferences). Ignore past events. Use "YYYY-MM-DD" format.

Tickers hint (watchlist): {tickers_text}

NEWS:
- title: {title}
- published: {published}
- link: {link}
- summary: {summary}
""".strip()

    # Responses API 사용 (권장 최신 인터페이스)
    # Structured outputs 가이드 참고
    resp = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "json_schema": schema,
            }
        },
    )

    # SDK가 반환하는 텍스트를 파싱 (Responses는 output_text로 접근 가능)
    # 안전하게: resp.output_text 가 JSON 문자열
    import json
    data = json.loads(resp.output_text)
    return data
