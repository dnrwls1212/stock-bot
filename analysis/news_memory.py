# src/analysis/news_memory.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from src.utils.ollama_client import ollama_generate, try_parse_json


def _to_md_from_json(d: Dict[str, Any]) -> str:
    """
    LLM이 JSON을 반환했을 때도, 저장/프롬프트에 쓰기 좋은 문자열로 변환.
    """
    lines: List[str] = []
    title = str(d.get("title", "") or "").strip()
    summary = str(d.get("summary", "") or "").strip()

    if title:
        lines.append(f"# {title}")
    if summary:
        lines.append(summary)

    drivers = d.get("drivers") or d.get("key_drivers") or []
    risks = d.get("risks") or d.get("key_risks") or []
    tags = d.get("tags") or []

    if isinstance(drivers, list) and drivers:
        lines.append("\n## Drivers")
        for x in drivers[:8]:
            s = str(x).strip()
            if s:
                lines.append(f"- {s}")

    if isinstance(risks, list) and risks:
        lines.append("\n## Risks")
        for x in risks[:8]:
            s = str(x).strip()
            if s:
                lines.append(f"- {s}")

    if isinstance(tags, list) and tags:
        t = ", ".join([str(x).strip() for x in tags if str(x).strip()])
        if t:
            lines.append(f"\n## Tags\n{t}")

    # 그래도 비어있으면 통째로 JSON 덤프
    out = "\n".join(lines).strip()
    if out:
        return out
    return json.dumps(d, ensure_ascii=False, indent=2)


def build_news_memory_summary_local_ollama(
    *,
    ticker: str,
    events: List[Dict[str, Any]],
    model: str = "qwen2.5:7b-instruct",
) -> str:
    """
    ✅ 항상 str을 반환한다.
    events는 dict list로 들어오는 걸 기대(main에서 get_recent_events(as_dict=True)로 해결)
    """
    t = (ticker or "").upper().strip()

    # 너무 길면 잘라서
    light: List[Dict[str, Any]] = []
    for e in (events or [])[:30]:
        if not isinstance(e, dict):
            continue
        light.append(
            {
                "ts_kst": e.get("ts_kst"),
                "published": e.get("published"),
                "title": e.get("title"),
                "summary": e.get("summary"),
                "event_type": e.get("event_type"),
                "sentiment": e.get("sentiment"),
                "impact": e.get("impact"),
                "why_it_moves": e.get("why_it_moves"),
                "event_score": e.get("event_score"),
                "confidence": e.get("confidence"),
                "link": e.get("link"),
            }
        )

    prompt = f"""
너는 뉴스 이벤트들을 요약해서 '다음 의사결정에 재사용 가능한 메모리'를 만든다.

규칙:
- 과장 금지. 이벤트에서 확인되는 사실만.
- 출력은 (1) 짧은 요약문 (2) Drivers (3) Risks (4) Tags 로 구성.
- 한국어로 작성.
- 길이는 15줄 내외.

[TICKER]
{t}

[EVENTS]
{json.dumps(light, ensure_ascii=False)}

[OUTPUT]
문장/불릿으로만 작성해도 되고, JSON으로 작성해도 된다.
""".strip()

    text = ollama_generate(prompt=prompt, model=model, temperature=0.2)

    # JSON이면 MD로 변환해서 str로 반환
    parsed = try_parse_json(text)
    if isinstance(parsed, dict):
        return _to_md_from_json(parsed)

    return str(text or "").strip()