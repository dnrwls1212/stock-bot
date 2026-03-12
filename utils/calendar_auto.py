# src/utils/calendar_auto.py
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf

def update_event_calendar_auto(watchlist: list[str]) -> list[dict]:
    path = "data/upcoming_events.json"
    existing_events = []
    
    # 1. 기존 파일 읽기 (수동으로 넣은 GTC 등 특별 이벤트 보존용)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing_events = json.load(f)
        except Exception:
            pass

    # 2. 수동 이벤트 보존: '실적'이라는 단어가 들어가지 않은 이벤트만 살림 (하루 지난 과거 이벤트는 삭제)
    now_date = datetime.now(ZoneInfo("Asia/Seoul")).date()
    valid_manual_events = []
    for e in existing_events:
        if "실적" not in e.get("desc", ""): # 실적이 아닌 특별 이벤트만
            try:
                ev_date = datetime.strptime(e["date"], "%Y-%m-%d").date()
                if ev_date >= now_date - timedelta(days=1): 
                    valid_manual_events.append(e)
            except Exception:
                pass

    # 3. Yahoo Finance로 워치리스트 종목의 다음 '실적발표일' 자동 수집
    auto_earnings = []
    print("📅 [SYSTEM] 워치리스트 종목들의 향후 실적발표일을 자동 수집합니다... (약 10~20초 소요)")
    
    for ticker in watchlist:
        try:
            tk = yf.Ticker(ticker)
            next_date_str = None
            
            # 1순위: 야후 파이낸스 Summary 화면의 Calendar 정보 가져오기 (가장 정확함)
            if hasattr(tk, 'calendar') and isinstance(tk.calendar, dict):
                earn_dates = tk.calendar.get('Earnings Date')
                if earn_dates and len(earn_dates) > 0:
                    # 야후 캘린더는 보통 datetime.date 객체의 리스트를 반환함
                    next_date = earn_dates[0]
                    if hasattr(next_date, 'strftime'):
                        next_date_str = next_date.strftime("%Y-%m-%d")
            
            # 2순위: 캘린더에 없다면 기존 방식(earnings_dates)으로 백업 확인
            if not next_date_str and hasattr(tk, 'earnings_dates') and tk.earnings_dates is not None and not tk.earnings_dates.empty:
                now_utc = pd.Timestamp.utcnow()
                if tk.earnings_dates.index.tz is None:
                    now_utc = now_utc.tz_localize(None)
                future_dates = tk.earnings_dates[tk.earnings_dates.index >= now_utc]
                if not future_dates.empty:
                    next_date_str = future_dates.index.min().strftime("%Y-%m-%d")

            # 최종적으로 미래 실적발표일이 찾아졌다면 추가
            if next_date_str:
                auto_earnings.append({
                    "ticker": ticker,
                    "date": next_date_str,
                    "desc": f"{ticker} 분기 실적발표"
                })
        except Exception as e:
            pass # 통신 에러 발생 시 조용히 넘어감

    # 4. 수동 이벤트 + 자동 실적발표 병합 및 날짜순 정렬
    final_events = valid_manual_events + auto_earnings
    final_events.sort(key=lambda x: x.get("date", "2099-12-31"))

    # 5. 파일 덮어쓰기 저장
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final_events, f, ensure_ascii=False, indent=2)
        
    return final_events