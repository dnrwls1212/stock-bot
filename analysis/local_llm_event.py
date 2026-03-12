# src/analysis/local_llm_event.py
import json
import re
from typing import Any, Dict, List, Optional

from .schema import SCHEMA_TEMPLATE
from src.utils.ollama_client import ollama_generate

# 🚀 [수정] 직관적이고 명확한 단일 키워드로 이벤트 유형 개편
ALLOWED_EVENT_TYPES = [
    "earnings", "guidance", "contract", "M&A", "lawsuit", 
    "launching", "macro", "war", "clinical", "offering", "other"
]

ALLOWED_SENTIMENT = ["bullish","bearish","neutral"]
ALLOWED_HORIZON = ["intraday","swing_days","swing_weeks","long_term"]

def _extract_json(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found. Raw output head: {text[:200]!r}")
    return m.group(0)

def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(SCHEMA_TEMPLATE)
    out.update(data or {})

    if not isinstance(out.get("tickers"), list):
        out["tickers"] = []
    out["tickers"] = [str(x).upper().strip() for x in out["tickers"] if str(x).strip()]

    if out.get("event_type") not in ALLOWED_EVENT_TYPES:
        out["event_type"] = "other"
    if out.get("sentiment") not in ALLOWED_SENTIMENT:
        out["sentiment"] = "neutral"

    try:
        out["impact"] = int(out.get("impact", 0))
    except Exception:
        out["impact"] = 0
    out["impact"] = max(-3, min(3, out["impact"]))

    try:
        out["confidence"] = float(out.get("confidence", 0.5))
    except Exception:
        out["confidence"] = 0.5
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))

    if not isinstance(out.get("key_points"), list):
        out["key_points"] = []
    out["key_points"] = [str(x).strip() for x in out["key_points"] if str(x).strip()][:5]

    if out.get("trade_horizon") not in ALLOWED_HORIZON:
        out["trade_horizon"] = "swing_days"

    out["why_it_moves"] = str(out.get("why_it_moves", "")).strip()
    return out

def analyze_news_local_ollama(
    title: str,
    summary: str,
    link: str = "",
    published: str = "",
    watchlist: Optional[List[str]] = None,
    model: str = "qwen2.5:14b-instruct", # 모델명 14b로 상향 통일
    retries: int = 2,
) -> Dict[str, Any]:

    watch = ", ".join(watchlist or [])
    schema_text = json.dumps(SCHEMA_TEMPLATE, ensure_ascii=False)

    prompt = f"""
Return ONLY valid JSON. No markdown. No extra keys. No trailing text.

IMPORTANT RULES:
- If the news does NOT clearly mention a ticker or a watchlist company as the MAIN SUBJECT, set "tickers" to [].
- Do NOT fill tickers with the watchlist by default.
- Ticker mapping rule (MAIN SUBJECT only): map company names to tickers ONLY when the article is primarily about that company (main subject).
  If it is only mentioned as a partner/supplier/customer, DO NOT include it in tickers unless the news is clearly impactful to that company.
- Company name -> ticker mapping:
  - "NVIDIA" or "Nvidia" -> "NVDA"
  - "Broadcom" -> "AVGO"
  - "Advanced Micro Devices" or "AMD" -> "AMD"
  - "Oracle" -> "ORCL"
- key_points must contain 2~5 concrete bullets (short).
- why_it_moves must be a non-empty 1~2 sentence explanation. If you truly cannot explain, set impact=0 and write why_it_moves="Unclear impact / insufficient info".
- Only set impact != 0 if key_points and why_it_moves justify it.
- If uncertain, keep impact = 0 and confidence around 0.5.
- STRICT: Only include tickers if the article is primarily about a watchlist company. If it is primarily about a different company (e.g., TSMC), set tickers=[] even if watchlist companies are mentioned.

[EVENT EXTRACTION RULE]
- If the news mentions ANY FUTURE event date (keynote, product launch, etc.), fill `upcoming_event_date` (Try YYYY-MM-DD. If unknown, use raw text like 'Next Tuesday' or 'March 15') and `upcoming_event_desc` (short summary). If none, leave them as "".

[TRANSLATION STRICT RULES for kr_title]
1. kr_title MUST be the exact Korean translation of the original English title.
2. ABSOLUTELY NO CHINESE CHARACTERS (Hanzi). Use 100% pure, natural Korean.
3. NEVER translate or transliterate proper nouns strangely. Keep company names (e.g., NVIDIA, Tesla, Apple, AMD) in original English OR use standard Korean market names (e.g., 엔비디아, 테슬라, 애플). 
4. DO NOT invent phonetic spellings like '나VIDIA' or '엔비디아스'. 

[EVENT TYPE CATEGORIZATION GUIDE]
- earnings: Quarterly/Annual earnings reports (실적발표)
- guidance: Future outlook, guidance updates (가이던스)
- contract: New contracts, partnerships, deals (계약/수주)
- M&A: Mergers and acquisitions (인수합병)
- lawsuit: Lawsuits, SEC investigations, regulations (소송/조사)
- launching: New product or service launches (제품출시)
- macro: CPI, Fed rates, inflation, broader economic data (거시경제)
- war: Geopolitical conflicts, wars, missile attacks (전쟁/지정학적 위기)
- clinical: FDA approvals, clinical trial results (임상/FDA)
- offering: Stock offerings, capital raises (유상증자)
- other: Anything else (기타)

Allowed values:
- event_type: {",".join(ALLOWED_EVENT_TYPES)}
- sentiment: {",".join(ALLOWED_SENTIMENT)}
- impact: integer -3..3
- confidence: 0.0..1.0
- trade_horizon: {",".join(ALLOWED_HORIZON)}

Schema (fill values, keep keys exactly):
{schema_text}

Watchlist tickers hint (use ONLY if clearly relevant): {watch}

NEWS:
- title: {title}
- published: {published}
- link: {link}
- summary: {summary}
""".strip()

    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = ollama_generate(
                prompt=prompt,
                model=model,              
                temperature=0.2,
                timeout=180.0,            
            )
            js = _extract_json(raw)
            data = json.loads(js)
            return _normalize(data)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Local LLM JSON parse failed after retries: {last_err}")


def analyze_news_with_local_llm(
    title: str,
    summary: str,
    link: str = "",
    published: str = "",
    watchlist: Optional[List[str]] = None,
    model: str = "qwen2.5:14b-instruct",
    retries: int = 2,
) -> Dict[str, Any]:
    return analyze_news_local_ollama(
        title=title,
        summary=summary,
        link=link,
        published=published,
        watchlist=watchlist,
        model=model,
        retries=retries,
    )