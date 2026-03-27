from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


DEFAULT_TOP100_SP500 = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BRK.B",
    "LLY",
    "AVGO",
    "TSLA",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "MA",
    "COST",
    "WMT",
    "PG",
    "JNJ",
    "HD",
    "ORCL",
    "BAC",
    "ABBV",
    "KO",
    "MRK",
    "CVX",
    "CRM",
    "NFLX",
    "TMO",
    "ACN",
    "MCD",
    "CSCO",
    "PEP",
    "LIN",
    "ABT",
    "AMD",
    "WFC",
    "DIS",
    "ADBE",
    "TXN",
    "DHR",
    "PM",
    "RTX",
    "QCOM",
    "IBM",
    "CAT",
    "GE",
    "SPGI",
    "BKNG",
    "GS",
    "INTU",
    "AMGN",
    "LOW",
    "BLK",
    "AMAT",
    "NOW",
    "ISRG",
    "HON",
    "UNP",
    "SYK",
    "BA",
    "NEE",
    "PGR",
    "SCHW",
    "TJX",
    "ADP",
    "DE",
    "MU",
    "COP",
    "GILD",
    "LRCX",
    "MDT",
    "MMC",
    "ELV",
    "C",
    "EG",
    "UBER",
    "FI",
    "KLAC",
    "CB",
    "SO",
    "ANET",
    "PANW",
    "SNPS",
    "MO",
    "APD",
    "TT",
    "CME",
    "CDNS",
    "UPS",
    "ICE",
    "PH",
    "AON",
    "EQIX",
    "EOG",
    "CMG",
    "MSI",
    "ROP",
    "SHW",
    "MDLZ",
]

# Backward compatibility for older imports.
DEFAULT_TOP10_SP500 = DEFAULT_TOP100_SP500
DEFAULT_TOP25_SP500 = DEFAULT_TOP100_SP500[:25]


@dataclass(frozen=True)
class AppConfig:
    stock_data_api_key: str
    openai_api_key: str
    google_api_key: str
    anthropic_api_key: str
    xai_api_key: str
    openai_model: str
    gemini_model: str
    claude_model: str
    grok_model: str
    context_mode: str
    telegram_enabled: bool
    telegram_bot_token: Optional[str]
    telegram_chat_id: Optional[str]
    alpaca_enabled: bool
    alpaca_api_key_id: Optional[str]
    alpaca_api_secret_key: Optional[str]
    alpaca_paper: bool
    alpaca_order_dollars: float
    alpaca_hold_seconds: int
    symbols: List[str]

    @staticmethod
    def from_env() -> "AppConfig":
        telegram_enabled = _parse_bool(os.getenv("TELEGRAM_ENABLED", "false"))
        telegram_bot_token = _parse_optional_str(os.getenv("TELEGRAM_BOT_TOKEN"))
        telegram_chat_id = _parse_optional_str(os.getenv("TELEGRAM_CHAT_ID"))
        telegram_enabled = bool(telegram_enabled and telegram_bot_token and telegram_chat_id)

        alpaca_enabled = _parse_bool(os.getenv("ALPACA_ENABLED", "false"))
        alpaca_api_key_id = _parse_optional_str(os.getenv("ALPACA_API_KEY_ID"))
        alpaca_api_secret_key = _parse_optional_str(os.getenv("ALPACA_API_SECRET_KEY"))
        if alpaca_enabled and (not alpaca_api_key_id or not alpaca_api_secret_key):
            raise ValueError(
                "ALPACA_ENABLED=true requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY"
            )

        alpaca_paper = _parse_bool(os.getenv("ALPACA_PAPER", "true"))
        alpaca_order_dollars = _parse_positive_float(
            os.getenv("ALPACA_ORDER_DOLLARS"),
            "ALPACA_ORDER_DOLLARS",
            default="100",
        )
        alpaca_hold_seconds = _parse_positive_int(os.getenv("ALPACA_HOLD_SECONDS", "3600"))

        return AppConfig(
            stock_data_api_key=_required("STOCK_DATA_API_KEY"),
            openai_api_key=_required("OPENAI_API_KEY"),
            google_api_key=_required("GOOGLE_API_KEY"),
            anthropic_api_key=_required("ANTHROPIC_API_KEY"),
            xai_api_key=_required("XAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest"),
            grok_model=os.getenv("GROK_MODEL", "grok-beta"),
            context_mode=_parse_context_mode(os.getenv("CONTEXT_MODE", "hybrid")),
            telegram_enabled=telegram_enabled,
            telegram_bot_token=telegram_bot_token,
            telegram_chat_id=telegram_chat_id,
            alpaca_enabled=alpaca_enabled,
            alpaca_api_key_id=alpaca_api_key_id,
            alpaca_api_secret_key=alpaca_api_secret_key,
            alpaca_paper=alpaca_paper,
            alpaca_order_dollars=alpaca_order_dollars,
            alpaca_hold_seconds=alpaca_hold_seconds,
            symbols=_parse_symbols(os.getenv("SYMBOLS")),
        )


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing environment variable: {name}")
    return value


def _parse_symbols(raw: str | None) -> List[str]:
    if not raw:
        return DEFAULT_TOP100_SP500
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _parse_context_mode(raw: str) -> str:
    mode = (raw or "hybrid").strip().lower()
    allowed = {"raw", "hybrid", "features"}
    if mode not in allowed:
        raise ValueError(f"Invalid CONTEXT_MODE={raw!r}. Allowed: raw, hybrid, features")
    return mode


def _parse_optional_str(raw: str | None) -> Optional[str]:
    value = (raw or "").strip()
    return value or None


def _parse_bool(raw: str) -> bool:
    v = (raw or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _parse_positive_float(raw: str | None, label: str, *, default: str) -> float:
    v = float((raw or default).strip())
    if v <= 0:
        raise ValueError(f"{label} must be positive")
    return v


def _parse_positive_int(raw: str | None) -> int:
    v = int((raw or "3600").strip())
    if v <= 0:
        raise ValueError("ALPACA_HOLD_SECONDS must be positive")
    return v
