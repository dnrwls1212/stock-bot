from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

# AI 호출을 위한 함수 임포트
from src.utils.ollama_client import ollama_generate, try_parse_json

@dataclass
class RegimeState:
    score: float
    label: str  # risk_on / neutral / risk_off
    reason: str = ""

def _to_float_scalar(x) -> float:
    """pandas/numpy scalar 안전 변환"""
    try:
        if isinstance(x, pd.Series):
            if len(x) == 0:
                return float("nan")
            return float(x.iloc[-1])
        return float(x)
    except Exception:
        try:
            import numpy as np
            return float(np.asarray(x).item())
        except Exception:
            return float("nan")

class RegimeFilter:
    """
    🤖 AI 기반 Market Regime 판단기 (단타/스캘핑용):
    - 15분봉 최근 차트 데이터를 텍스트로 변환하여 LLM에 전달
    - LLM이 차트 흐름과 추세를 분석하여 상승/하락장 판단
    - 매 틱마다 AI 대기시간이 걸리지 않도록 N분 간격으로 캐싱(Caching) 작동
    """

    def __init__(self, symbol: str = "QQQ"):
        self.symbol = symbol
        
        # .env 설정 가져오기 (기본값 10분마다 AI 갱신)
        try:
            self.refresh_min = int(os.environ.get("REGIME_REFRESH_MIN", "10"))
        except ValueError:
            self.refresh_min = 10
            
        self.refresh_sec = self.refresh_min * 60
        
        # AI 모델 설정 (.env의 DECISION_MODEL 활용)
        self.model = os.environ.get("DECISION_MODEL", "qwen2.5:14b-instruct")
        
        self._last_state: Optional[RegimeState] = None
        self._last_update_ts = 0.0

    def get(self) -> RegimeState:
        now = time.time()
        
        # 1. 쿨타임(캐시) 확인: 지정된 시간이 지나지 않았다면 이전 AI 판단 결과 즉시 반환 (속도 유지)
        if self._last_state is not None and (now - self._last_update_ts) < self.refresh_sec:
            return self._last_state

        # 2. yfinance를 통해 최근 데이터 가져오기 (15분봉 2일치)
        try:
            df = yf.download(self.symbol, period="2d", interval="15m", progress=False)
        except Exception as e:
            print(f"[AI_REGIME_ERR] 데이터 다운로드 실패: {e}")
            return self._fallback_state()

        if df is None or len(df) == 0:
            return self._fallback_state()

        # 3. AI에게 차트 모양을 보여주기 위해 최근 15개 캔들 추출
        tail_df = df.tail(15)
        price_history = []
        
        for idx, row in tail_df.iterrows():
            try:
                c = _to_float_scalar(row["Close"])
                v = _to_float_scalar(row["Volume"])
                t_str = idx.strftime("%m-%d %H:%M") # "02-28 09:30" 형식
                price_history.append(f"[{t_str}] 지수: {c:.2f} (거래량:{v:.0f})")
            except Exception:
                continue

        history_str = "\n".join(price_history)

        # 4. AI에게 물어볼 프롬프트 작성
        prompt = f"""
너는 퀀트 트레이딩의 시황 분석 최고 전문가(AI 에이전트)야.
현재 미국 시장 지수({self.symbol})의 최근 15분봉 흐름을 보고 현재의 장세(Regime)를 판단해줘.
나는 초단타(스캘핑) 매매 봇이므로 '현재의 짧은 추세와 모멘텀'이 가장 중요해.

[최근 15분봉 시장 흐름]
{history_str}

[분석 및 판단 기준]
1. risk_on (강세장/상승장): 
   - 뚜렷한 우상향 추세가 유지되거나, 급락 후 강한 매수세가 들어오며 반등 중일 때.
   - 점수: 0.3 ~ 1.0 사이 (강할수록 1.0)
2. neutral (횡보장/모호함): 
   - 뚜렷한 방향 없이 오르내리거나 추세를 확신하기 어려울 때. 
   - 점수: -0.29 ~ 0.29 사이
3. risk_off (약세장/하락장): 
   - 지속적인 우하향 추세이거나, 지지선을 깨고 급락 중이라 인버스(숏) 대응이 필요할 때.
   - 점수: -1.0 ~ -0.3 사이 (약할수록 -1.0)

결과는 반드시 아래 JSON 형식으로만 응답해. (부연 설명 절대 금지)
{{
    "label": "risk_on" 또는 "neutral" 또는 "risk_off",
    "score": 0.0,
    "reason": "한국어로 간략한 1~2줄 분석 이유"
}}
"""
        # 5. 로컬 Ollama AI 호출
        try:
            llm_text = ollama_generate(
                prompt=prompt, 
                model=self.model, 
                temperature=0.2, 
                timeout=180.0
            )
            
            parsed = try_parse_json(llm_text) or {}
            
            label = str(parsed.get("label", "neutral")).lower()
            score = float(parsed.get("score", 0.0))
            reason = str(parsed.get("reason", "판단 불가"))

            # 안전망: 이상한 라벨이 나오면 중립으로
            if label not in ["risk_on", "neutral", "risk_off"]:
                label = "neutral"

            # 점수 범위 제한
            score = max(-1.0, min(1.0, score))

            self._last_state = RegimeState(score=score, label=label, reason=reason)
            self._last_update_ts = now
            
            # 터미널에 AI의 판단 결과를 멋지게 출력!
            print(f"🤖 [AI 시장 판단 완료] {label} (점수: {score:.2f}) | 분석: {reason}")
            
        except Exception as e:
            print(f"[AI_REGIME_ERR] AI 호출 실패: {e}")
            if self._last_state is None:
                return self._fallback_state()
            # 에러 나면 기존 캐시 그대로 사용
            return self._last_state 

        return self._last_state

    def _fallback_state(self) -> RegimeState:
        return RegimeState(score=0.0, label="neutral", reason="데이터 부족 또는 에러로 인한 자동 중립")