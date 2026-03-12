SCHEMA_TEMPLATE = {
    "tickers": [],
    "event_type": "other",  # earnings,guidance,contract_partnership,regulation_export,mna,financing_offering,product_launch,litigation_investigation,macro_sector,other
    "sentiment": "neutral",  # bullish,bearish,neutral
    "impact": 0,             # -3..3
    "confidence": 0.5,       # 0..1
    "key_points": [],
    "trade_horizon": "swing_days",  # intraday,swing_days,swing_weeks,long_term
    "why_it_moves": "",
    "kr_title": "주식 종목 티커를 제외한 나머지 내용은 반드시 100% 한국어로만 번역된 제목을 작성할 것 (중국어/한자 절대 포함 금지)",
    # 👇 신규 추가
    "upcoming_event_date": "",
    "upcoming_event_desc": ""
}
