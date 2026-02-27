# src/analysis/local_llm_event.py
import json
import re
from typing import Any, Dict, List, Optional

from .schema import SCHEMA_TEMPLATE
from src.utils.ollama_client import ollama_generate

ALLOWED_EVENT_TYPES = [
    "earnings","guidance","contract_partnership","regulation_export",
    "mna","financing_offering","product_launch","litigation_investigation",
    "macro_sector","other",
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
    model: str = "qwen2.5:7b-instruct",
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
- kr_title MUST be the Korean translation of the original English title. Do not summarize, just translate it naturally into Korean.
- Only set impact != 0 if key_points and why_it_moves justify it.
- If uncertain, keep impact = 0 and confidence around 0.5.
- Heuristic: litigation/investigation usually has negative sentiment unless clearly dismissed or a favorable ruling is mentioned.
- STRICT: Only include tickers if the article is primarily about a watchlist company. If it is primarily about a different company (e.g., TSMC), set tickers=[] even if watchlist companies are mentioned.

Allowed values:
- event_type: {",".join(ALLOWED_EVENT_TYPES)}
- sentiment: {",".join(ALLOWED_SENTIMENT)}
- impact: integer -3..3
- confidence: 0..1
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
                model=model,              # ✅ still accepts per-call override
                temperature=0.2,
                timeout=180.0,            # ✅ uses OLLAMA_TIMEOUT if you pass None; here we keep explicit
            )
            js = _extract_json(raw)
            data = json.loads(js)
            return _normalize(data)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Local LLM JSON parse failed after retries: {last_err}")


# ✅ main.py 호환 래퍼
def analyze_news_with_local_llm(
    title: str,
    summary: str,
    link: str = "",
    published: str = "",
    watchlist: Optional[List[str]] = None,
    model: str = "qwen2.5:7b-instruct",
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