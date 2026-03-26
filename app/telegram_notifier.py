from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import httpx


@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool
    bot_token: Optional[str]
    chat_id: Optional[str]


async def send_telegram_message(
    cfg: TelegramConfig,
    text: str,
    *,
    timeout_s: float = 10.0,
) -> Tuple[bool, Optional[str]]:
    if not cfg.enabled:
        return False, None
    if not cfg.bot_token or not cfg.chat_id:
        return False, "telegram enabled but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID missing"

    url = f"https://api.telegram.org/bot{cfg.bot_token}/sendMessage"
    payload = {"chat_id": cfg.chat_id, "text": text}

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
            if body.get("ok") is True:
                return True, None
            return False, f"telegram returned ok={body.get('ok')} body={str(body)[:200]}"
    except Exception as exc:
        return False, str(exc)

