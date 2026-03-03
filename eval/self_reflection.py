import os
import json
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

    def run_reflection(self, market_condition: str, account_status: str) -> dict:
        if not os.path.exists(self.trades_path):
            return {"status": "not_enough_data"}

        trades = []
        try:
            with open(self.trades_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-30:]: # 최근 30건 가져오기
                    trades.append(json.loads(line.strip()))
        except Exception as e:
            return {"status": "error", "message": str(e)}

        # 👇 [수정] 거래가 1건만 있어도(>=1) 무조건 학습 진행!
        if len(trades) < 1:
            return {"status": "not_enough_data"} 

        current_lessons = self.load_lessons()

        # 👇 [수정] 시황과 계좌 상태를 포함하여 펀드매니저의 일기를 쓰게 하는 프롬프트
        prompt = f"""너는 세계 최고의 퀀트 헤지펀드 트레이딩 봇의 '자가 학습(Self-Reflection) 엔진'이야.
오늘 정규장이 마감되었어. 네가 스스로 판단하여 내린 매매 결정들을 복기할 시간이야.

[오늘 마감 기준 시황 (Market Condition)]
{market_condition}

[현재 계좌 상태 (Account Status)]
{account_status}

[최근 매매 기록 요약]
{json.dumps(trades[-10:], ensure_ascii=False, indent=2)}

[기존에 배운 투자 교훈]
{current_lessons}

[임무]
오늘 진행된 거래 내역과 시황, 그리고 계좌 상태(잔고 및 평가금액)를 종합적으로 분석해줘.
1. "오늘 시황은 어떠했고, 어떠한 이유로 이런 거래들을 진행하여 현재 계좌 수익률/상태가 이러하다."라는 종합 분석을 'reflection_analysis'에 아주 논리적으로 작성해.
2. 손실/수익 패턴을 파악하여 앞으로 봇이 지켜야 할 유연한 '투자 교훈(Lessons)'을 도출해.
3. "절대 사지 마라" 같은 극단적 제약은 피하고, 확률적이고 조건적인 룰(예: "시황이 하락장일 때는 확실한 실적 호재가 있을 때만 진입하자")을 제시해.

반드시 아래 JSON 형식으로만 응답해 (다른 텍스트 절대 금지):
{{
  "reflection_analysis": "오늘 시황과 매매에 대한 너의 종합적인 복기 및 분석 (한국어)",
  "lessons": [
    "상황에 맞게 적용할 수 있는 유연한 투자 교훈 1",
    "상황에 맞게 적용할 수 있는 유연한 투자 교훈 2"
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