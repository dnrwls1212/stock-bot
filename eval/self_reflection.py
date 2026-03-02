import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from src.utils.ollama_client import ollama_generate, try_parse_json

class SelfReflectionEngine:
    def __init__(self, trades_path="data/trades.jsonl", lessons_path="data/ai_lessons.json", model="qwen2.5:14b-instruct"):
        self.trades_path = trades_path
        self.lessons_path = lessons_path
        self.model = model

    def load_lessons(self) -> str:
        if not os.path.exists(self.lessons_path):
            return "아직 누적된 학습 데이터가 없습니다."
        try:
            with open(self.lessons_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return "\n".join(data.get("lessons", []))
        except:
            return "아직 누적된 학습 데이터가 없습니다."

    def run_reflection(self) -> dict:
        if not os.path.exists(self.trades_path):
            return {"status": "no_data"}

        # 최근 매매 기록 최대 30개 읽어오기
        trades = []
        try:
            with open(self.trades_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-30:]: # 최근 30건
                    trades.append(json.loads(line.strip()))
        except Exception as e:
            return {"status": "error", "message": str(e)}

        if len(trades) < 5:
            return {"status": "not_enough_data"} # 데이터가 적으면 아직 학습 안함

        # 기존 교훈 불러오기
        current_lessons = self.load_lessons()

        prompt = f"""너는 세계 최고의 퀀트 헤지펀드 트레이딩 봇의 '자가 학습(Self-Reflection) 엔진'이야.
아래는 최근 우리 봇이 실행한 매매(BUY/SELL) 기록들이야.

[최근 매매 기록 요약]
{json.dumps(trades[-10:], ensure_ascii=False, indent=2)}

[기존에 배운 투자 교훈]
{current_lessons}

[임무]
최근 매매 기록을 보고, 손실이 났거나 수익이 났던 패턴을 분석해서 봇이 진화할 수 있도록 '투자 교훈(Lessons)'을 업데이트해줘.
단, 절대 지켜야 할 아주 중요한 규칙이 있어.
1. "절대 사지 마라", "무조건 팔아라" 같은 극단적이고 제약이 심한 룰은 만들지 마. 봇의 매매 기회를 빼앗게 돼.
2. 손실이 났더라도 종목 자체가 문제라기보다, "당시 매크로(시장) 상황이 안 좋았다", "뉴스 점수 대비 차트가 너무 과매수 상태였다" 등 복합적이고 확률적인 원인을 분석해.
3. "이런 상황에서는 비중을 줄이자", "이런 상황에서는 뉴스 신뢰도를 평소보다 더 보수적으로(0.8 이상) 요구하자"처럼 유연하고 확률적인 대안을 제시해.

반드시 아래 JSON 형식으로만 응답해:
{{
  "reflection_analysis": "최근 매매에 대한 너의 종합적인 확률적 분석 (한국어)",
  "lessons": [
    "유연한 교훈 1 (예: 하락장에서는 단순 낙폭과대보다 강력한 실적 뉴스가 동반될 때만 진입하는 것이 승률이 높다)",
    "유연한 교훈 2"
  ]
}}"""

        try:
            res_text = ollama_generate(prompt=prompt, model=self.model, temperature=0.3, timeout=120)
            res_json = try_parse_json(res_text)
            if isinstance(res_json, dict) and "lessons" in res_json:
                with open(self.lessons_path, "w", encoding="utf-8") as f:
                    json.dump(res_json, f, ensure_ascii=False, indent=2)
                return {"status": "success", "data": res_json}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        return {"status": "fail"}