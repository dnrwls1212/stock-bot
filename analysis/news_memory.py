# src/analysis/news_memory.py
from __future__ import annotations
import json
from typing import Dict, Any, List
from src.utils.ollama_client import ollama_generate

def build_news_memory_summary_local_ollama(ticker: str, events: List[Dict[str, Any]], model: str = "qwen2.5:14b-instruct") -> str:
    """
    최근 뉴스를 바탕으로 종목의 핵심 팩트를 [🟢 호재/모멘텀]과 [🔴 악재/리스크]로 명확히 분리하여 누적합니다.
    """
    if not events:
        return ""

    try:
        from src.trading.news_store import NewsStore
        import os
        ns = NewsStore(os.environ.get("NEWS_STORE_PATH", "data/news_store.json"))
        existing_memory = ns.load_memory(ticker)
    except Exception:
        existing_memory = ""

    news_texts = []
    for e in events[:15]: 
        t = e.get('title', '')
        s = e.get('summary', '')
        d = e.get('published', '')
        if t or s:
            news_texts.append(f"[{d}] {t} - {s}")

    prompt = f"""너는 월스트리트 수석 데이터 아키텍트야.
목표: [{ticker}] 종목의 기존 메모리에 새로운 뉴스의 팩트를 추가하되, 반드시 '호재'와 '악재'를 엄격하게 분리하여 업데이트해.

[기존 메모리]
{existing_memory if existing_memory else "기존 데이터 없음"}

[새로 수집된 뉴스]
{json.dumps(news_texts, ensure_ascii=False)}

요청사항:
1. 새로운 뉴스에서 주가 상승 동력이 될 팩트는 [🟢 호재/상승 모멘텀] 아래에 추가해.
2. 주가 하락 요인이 될 팩트(실적 부진, 소송, 금리 인상 타격 등)는 [🔴 악재/하락 리스크] 아래에 추가해.
3. 중립적인 팩트라도 향후 리스크가 될 수 있다면 악재에 배치해.
4. 상반되는 내용이 들어오면(예: 과거엔 실적 악화였으나 이번엔 서프라이즈), 과거 낡은 팩트를 지우고 최신 내용으로 덮어써도 됨.
5. 절대 구구절절 쓰지 말고 간결한 단답형/개조식(-)으로 작성할 것.

반드시 아래 포맷을 그대로 유지해서 출력해 (다른 인사말 절대 금지):
[🟢 호재/상승 모멘텀]
- 
- 

[🔴 악재/하락 리스크]
- 
- 
"""
    try:
        res = ollama_generate(prompt=prompt, model=model, temperature=0.1, timeout=120)
        return res.strip()
    except Exception as e:
        print(f"[MEMORY_ERR] {ticker} memory build failed: {e}")
        return existing_memory