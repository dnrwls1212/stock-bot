# src/knowledge/kb_agent.py
from __future__ import annotations
import json
from typing import Dict, Any, List
from src.utils.ollama_client import ollama_generate, try_parse_json

def refine_ticker_kb(ticker: str, kb: Dict[str, Any], recent_evidence: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """
    주말/장 마감 시 기존 KB와 최근 뉴스를 바탕으로 핵심 투자 아이디어(thesis)와 리스크를 자동 업데이트(Rewrite)합니다.
    """
    if not recent_evidence:
        return kb # 업데이트할 뉴스가 없으면 기존 KB 유지
        
    # AI에게 전달할 기존 KB의 뼈대
    kb_light = {
        "thesis": kb.get("thesis", ""),
        "business_summary": kb.get("business_summary", ""),
        "key_drivers": kb.get("key_drivers", []),
        "key_risks": kb.get("key_risks", [])
    }
    
    # 최근 1주일 치 뉴스 추출 (최대 15개)
    news_text = json.dumps(recent_evidence[:15], ensure_ascii=False)
    
    prompt = (
        f"너는 월스트리트 수석 퀀트 애널리스트야. [{ticker}] 종목의 기존 분석 보고서(Knowledge Base)를 최신 뉴스에 맞게 전면 개정(Rewrite)해야 해.\n\n"
        f"[기존 KB]\n{json.dumps(kb_light, ensure_ascii=False)}\n\n"
        f"[최근 수집된 핵심 뉴스(Evidence)]\n{news_text}\n\n"
        "요청사항:\n"
        "1. 최근 뉴스를 반영하여 기존 'thesis(핵심 투자 아이디어)', 'business_summary(비즈니스 현황)', 'key_drivers(상승 동력)', 'key_risks(핵심 위험)'를 새롭게 작성해.\n"
        "2. 더 이상 유효하지 않은 낡은 정보는 과감히 삭제하고, 새로운 모멘텀이나 리스크를 추가해.\n"
        "3. 반드시 아래 JSON 형식으로만 응답해. (다른 설명은 절대 금지)\n"
        "{\n"
        '  "thesis": "업데이트된 2~3문장 분량의 핵심 투자 포인트",\n'
        '  "business_summary": "현재 시점의 비즈니스 현황 요약",\n'
        '  "key_drivers": ["동력1", "동력2"],\n'
        '  "key_risks": ["위험1", "위험2"]\n'
        "}"
    )
    
    try:
        # LLM 호출 (ollama_generate 활용)
        res_text = ollama_generate(prompt=prompt, model=model, temperature=0.2, timeout=120)
        updated_data = try_parse_json(res_text)
        
        if isinstance(updated_data, dict) and "thesis" in updated_data:
            kb["thesis"] = updated_data.get("thesis", kb.get("thesis"))
            kb["business_summary"] = updated_data.get("business_summary", kb.get("business_summary"))
            kb["key_drivers"] = updated_data.get("key_drivers", kb.get("key_drivers"))
            kb["key_risks"] = updated_data.get("key_risks", kb.get("key_risks"))
            return kb
    except Exception as e:
        print(f"[KB_REFINE_ERR] {ticker} KB 자동 정제 실패: {e}")
        
    return kb