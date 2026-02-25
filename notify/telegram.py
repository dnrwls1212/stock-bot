# src/notify/telegram.py
from __future__ import annotations

import os
import requests


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    r = requests.post(url, data=payload, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram send failed: {r.status_code} {r.text}")


class TelegramNotifier:
    """
    main.py 호환용 클래스.
    - from_env()로 env에서 토큰/챗ID를 읽고
    - send()로 메시지 전송
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = (bot_token or "").strip()
        self.chat_id = (chat_id or "").strip()

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        return cls(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        )

    def send(self, text: str) -> None:
        send_telegram_message(self.bot_token, self.chat_id, text)
