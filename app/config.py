from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    stock_data_api_key: str
    """Optional second TwelveData key; tried after primary on rate/credit errors."""
    stock_data_api_key_secondary: Optional[str]
    """Optional third TwelveData key; tried after primary and secondary fail."""
    stock_data_api_key_tertiary: Optional[str]
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
    alpaca_order_min_dollars: float
    alpaca_order_max_dollars: float
    alpaca_hold_seconds: int
    alpaca_pending_closes_file: str
    symbols: List[str]
    crypto_symbols: List[str]
    consensus_min_models: int
    consensus_min_confidence_pct: int
    consensus_min_confidence_crypto_pct: int
    # ── News trading module (disabled by default) ──────────────────────────────
    news_trading_enabled: bool
    news_alpaca_news_enabled: bool
    news_polygon_api_key: Optional[str]
    news_crypto_panic_api_key: Optional[str]
    news_trade_dollars: float
    news_max_hold_minutes: int
    news_stop_loss_pct: float
    news_take_profit_pct: float
    news_min_confidence_pct: int
    news_crypto_enabled: bool
    news_equity_enabled: bool
    binance_api_key: Optional[str]
    binance_secret_key: Optional[str]
    binance_testnet: bool
    # ── News source mode: "alpaca" (default), "x_stream", or "both" ────────────
    news_source_mode: str
    x_bearer_token: Optional[str]
    news_truth_social_enabled: bool
    news_bluesky_enabled: bool
    news_bluesky_firehose_enabled: bool
    news_x_syndication_enabled: bool  # free embed endpoint — often HTTP429 from cloud IPs
    news_x_api_stream_enabled: bool  # X API v2 filtered stream (needs X_BEARER_TOKEN + access tier)
    news_x_syndication_force: bool  # run syndication even when Bluesky firehose is enabled
    news_x_monitor_enabled: bool

    @staticmethod
    def from_env() -> "AppConfig":
        telegram_enabled = _parse_bool(_required("TELEGRAM_ENABLED"))
        telegram_bot_token = _parse_optional_str(os.getenv("TELEGRAM_BOT_TOKEN"))
        telegram_chat_id = _parse_optional_str(os.getenv("TELEGRAM_CHAT_ID"))
        if telegram_enabled and (not telegram_bot_token or not telegram_chat_id):
            raise ValueError(
                "TELEGRAM_ENABLED=true requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
            )
        telegram_enabled = bool(telegram_enabled and telegram_bot_token and telegram_chat_id)

        alpaca_enabled = _parse_bool(_required("ALPACA_ENABLED"))
        alpaca_api_key_id = _parse_optional_str(os.getenv("ALPACA_API_KEY_ID"))
        alpaca_api_secret_key = _parse_optional_str(os.getenv("ALPACA_API_SECRET_KEY"))
        if alpaca_enabled and (not alpaca_api_key_id or not alpaca_api_secret_key):
            raise ValueError(
                "ALPACA_ENABLED=true requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY"
            )

        alpaca_paper = _parse_bool(_required("ALPACA_PAPER"))
        alpaca_order_min_dollars = _parse_positive_float(
            os.getenv("ALPACA_ORDER_MIN_DOLLARS"),
            "ALPACA_ORDER_MIN_DOLLARS",
        )
        alpaca_order_max_dollars = _parse_positive_float(
            os.getenv("ALPACA_ORDER_MAX_DOLLARS"),
            "ALPACA_ORDER_MAX_DOLLARS",
        )
        if alpaca_order_min_dollars > alpaca_order_max_dollars:
            raise ValueError(
                "ALPACA_ORDER_MIN_DOLLARS must be less than or equal to ALPACA_ORDER_MAX_DOLLARS"
            )
        alpaca_hold_seconds = _parse_positive_int_env(
            os.getenv("ALPACA_HOLD_SECONDS"),
            "ALPACA_HOLD_SECONDS",
        )
        alpaca_pending_closes_file = _required("ALPACA_PENDING_CLOSES_FILE")

        consensus_min_models = _parse_consensus_min_models(os.getenv("CONSENSUS_MIN_MODELS"))
        consensus_min_confidence_pct = _parse_confidence_pct(
            os.getenv("CONSENSUS_MIN_CONFIDENCE_PERCENT"),
            "CONSENSUS_MIN_CONFIDENCE_PERCENT",
        )
        consensus_min_confidence_crypto_pct = _parse_confidence_pct(
            os.getenv("CONSENSUS_MIN_CONFIDENCE_CRYPTO_PERCENT"),
            "CONSENSUS_MIN_CONFIDENCE_CRYPTO_PERCENT",
        )

        # News trading — all optional with safe defaults so existing users are not broken
        news_trading_enabled = _parse_bool_default(os.getenv("NEWS_TRADING_ENABLED"), False)
        news_alpaca_news_enabled = _parse_bool_default(os.getenv("NEWS_ALPACA_ENABLED"), True)
        news_polygon_api_key = _parse_optional_str(os.getenv("NEWS_POLYGON_API_KEY"))
        news_crypto_panic_api_key = _parse_optional_str(os.getenv("NEWS_CRYPTO_PANIC_API_KEY"))
        news_trade_dollars = _parse_positive_float_default(
            os.getenv("NEWS_TRADE_DOLLARS"), 200.0
        )
        news_max_hold_minutes = _parse_positive_int_default(
            os.getenv("NEWS_MAX_HOLD_MINUTES"), 60
        )
        news_stop_loss_pct = _parse_positive_float_default(
            os.getenv("NEWS_STOP_LOSS_PCT"), 0.8
        )
        news_take_profit_pct = _parse_positive_float_default(
            os.getenv("NEWS_TAKE_PROFIT_PCT"), 2.0
        )
        news_min_confidence_pct = _parse_confidence_pct_default(
            os.getenv("NEWS_MIN_CONFIDENCE_PCT"), 70
        )
        news_crypto_enabled = _parse_bool_default(os.getenv("NEWS_CRYPTO_ENABLED"), False)
        news_equity_enabled = _parse_bool_default(os.getenv("NEWS_EQUITY_ENABLED"), True)
        binance_api_key = _parse_optional_str(os.getenv("BINANCE_API_KEY"))
        binance_secret_key = _parse_optional_str(os.getenv("BINANCE_SECRET_KEY"))
        binance_testnet = _parse_bool_default(os.getenv("BINANCE_TESTNET"), True)

        stock_data_secondary = _parse_optional_str(os.getenv("STOCK_DATA_API_KEY_SECONDARY"))
        stock_data_tertiary = _parse_optional_str(os.getenv("STOCK_DATA_API_KEY_TERTIARY"))

        news_source_mode_raw = (os.getenv("NEWS_SOURCE_MODE") or "alpaca").strip().lower()
        if news_source_mode_raw not in ("alpaca", "x_stream", "both"):
            news_source_mode_raw = "alpaca"
        x_bearer_token = _parse_optional_str(os.getenv("X_BEARER_TOKEN"))
        news_truth_social_enabled = _parse_bool_default(
            os.getenv("NEWS_TRUTH_SOCIAL_ENABLED"), False
        )
        news_bluesky_enabled = _parse_bool_default(
            os.getenv("NEWS_BLUESKY_ENABLED"), True
        )
        news_bluesky_firehose_enabled = _parse_bool_default(
            os.getenv("NEWS_BLUESKY_FIREHOSE_ENABLED"), True
        )
        news_x_syndication_enabled = _parse_bool_default(
            os.getenv("NEWS_X_SYNDICATION_ENABLED"), False  # blocked on cloud/datacenter IPs
        )
        news_x_api_stream_enabled = _parse_bool_default(
            os.getenv("NEWS_X_API_STREAM_ENABLED"), False
        )
        news_x_syndication_force = _parse_bool_default(
            os.getenv("NEWS_X_SYNDICATION_FORCE"), False
        )
        news_x_monitor_enabled = _parse_bool_default(
            os.getenv("NEWS_X_MONITOR_ENABLED"), False  # Nitter is defunct; off by default
        )

        return AppConfig(
            stock_data_api_key=_required("STOCK_DATA_API_KEY"),
            stock_data_api_key_secondary=stock_data_secondary,
            stock_data_api_key_tertiary=stock_data_tertiary,
            openai_api_key=_required("OPENAI_API_KEY"),
            google_api_key=_required("GOOGLE_API_KEY"),
            anthropic_api_key=_required("ANTHROPIC_API_KEY"),
            xai_api_key=_required("XAI_API_KEY"),
            openai_model=_required("OPENAI_MODEL"),
            gemini_model=_required("GEMINI_MODEL"),
            claude_model=_required("CLAUDE_MODEL"),
            grok_model=_required("GROK_MODEL"),
            context_mode=_parse_context_mode(_required("CONTEXT_MODE")),
            telegram_enabled=telegram_enabled,
            telegram_bot_token=telegram_bot_token,
            telegram_chat_id=telegram_chat_id,
            alpaca_enabled=alpaca_enabled,
            alpaca_api_key_id=alpaca_api_key_id,
            alpaca_api_secret_key=alpaca_api_secret_key,
            alpaca_paper=alpaca_paper,
            alpaca_order_min_dollars=alpaca_order_min_dollars,
            alpaca_order_max_dollars=alpaca_order_max_dollars,
            alpaca_hold_seconds=alpaca_hold_seconds,
            alpaca_pending_closes_file=alpaca_pending_closes_file,
            symbols=_parse_symbols(os.getenv("SYMBOLS")),
            crypto_symbols=_parse_crypto_symbols(os.getenv("CRYPTO_SYMBOLS")),
            consensus_min_models=consensus_min_models,
            consensus_min_confidence_pct=consensus_min_confidence_pct,
            consensus_min_confidence_crypto_pct=consensus_min_confidence_crypto_pct,
            news_trading_enabled=news_trading_enabled,
            news_alpaca_news_enabled=news_alpaca_news_enabled,
            news_polygon_api_key=news_polygon_api_key,
            news_crypto_panic_api_key=news_crypto_panic_api_key,
            news_trade_dollars=news_trade_dollars,
            news_max_hold_minutes=news_max_hold_minutes,
            news_stop_loss_pct=news_stop_loss_pct,
            news_take_profit_pct=news_take_profit_pct,
            news_min_confidence_pct=news_min_confidence_pct,
            news_crypto_enabled=news_crypto_enabled,
            news_equity_enabled=news_equity_enabled,
            binance_api_key=binance_api_key,
            binance_secret_key=binance_secret_key,
            binance_testnet=binance_testnet,
            news_source_mode=news_source_mode_raw,
            x_bearer_token=x_bearer_token,
            news_truth_social_enabled=news_truth_social_enabled,
            news_bluesky_enabled=news_bluesky_enabled,
            news_bluesky_firehose_enabled=news_bluesky_firehose_enabled,
            news_x_syndication_enabled=news_x_syndication_enabled,
            news_x_api_stream_enabled=news_x_api_stream_enabled,
            news_x_syndication_force=news_x_syndication_force,
            news_x_monitor_enabled=news_x_monitor_enabled,
        )

    def twelve_data_api_keys(self) -> List[str]:
        """Primary, then optional second and third keys (same order for hourly + quotes)."""
        keys: List[str] = [self.stock_data_api_key]
        if self.stock_data_api_key_secondary:
            keys.append(self.stock_data_api_key_secondary)
        if self.stock_data_api_key_tertiary:
            keys.append(self.stock_data_api_key_tertiary)
        return keys


def _required(name: str) -> str:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        raise ValueError(f"Missing or empty environment variable: {name}")
    return str(value).strip()


def _parse_symbols(raw: str | None) -> List[str]:
    if raw is None or not str(raw).strip():
        raise ValueError(
            "SYMBOLS must be set in .env with a comma-separated list of equity tickers"
        )
    xs = [x.strip().upper() for x in str(raw).split(",") if x.strip()]
    if not xs:
        raise ValueError("SYMBOLS must contain at least one ticker")
    return xs


def _parse_crypto_symbols(raw: str | None) -> List[str]:
    if raw is None or not str(raw).strip():
        raise ValueError(
            "CRYPTO_SYMBOLS must be set in .env with a comma-separated list of pairs (e.g. BTC/USD)"
        )
    xs = [x.strip().upper() for x in str(raw).split(",") if x.strip()]
    if not xs:
        raise ValueError("CRYPTO_SYMBOLS must contain at least one pair")
    return xs


def _parse_context_mode(raw: str) -> str:
    mode = raw.strip().lower()
    allowed = {"raw", "hybrid", "features"}
    if mode not in allowed:
        raise ValueError(f"Invalid CONTEXT_MODE={raw!r}. Allowed: raw, hybrid, features")
    return mode


def _parse_optional_str(raw: str | None) -> Optional[str]:
    value = (raw or "").strip()
    return value or None


def _parse_bool(raw: str) -> bool:
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Boolean env must be true/false (or 1/0, yes/no, on/off): got {raw!r}"
    )


def _parse_positive_float(raw: str | None, label: str) -> float:
    if raw is None or not str(raw).strip():
        raise ValueError(f"Missing or empty environment variable: {label}")
    v = float(str(raw).strip())
    if v <= 0:
        raise ValueError(f"{label} must be positive")
    return v


def _parse_positive_int_env(raw: str | None, label: str) -> int:
    if raw is None or not str(raw).strip():
        raise ValueError(f"Missing or empty environment variable: {label}")
    v = int(str(raw).strip())
    if v <= 0:
        raise ValueError(f"{label} must be positive")
    return v


def _parse_consensus_min_models(raw: str | None) -> int:
    if raw is None or not str(raw).strip():
        raise ValueError("Missing or empty environment variable: CONSENSUS_MIN_MODELS")
    v = int(str(raw).strip())
    if not 1 <= v <= 4:
        raise ValueError("CONSENSUS_MIN_MODELS must be between 1 and 4 (four LLMs)")
    return v


def _parse_confidence_pct(raw: str | None, label: str) -> int:
    if raw is None or not str(raw).strip():
        raise ValueError(f"Missing or empty environment variable: {label}")
    v = int(str(raw).strip())
    if not 0 <= v <= 100:
        raise ValueError(f"{label} must be between 0 and 100")
    return v


def _parse_bool_default(raw: str | None, default: bool) -> bool:
    if raw is None or not str(raw).strip():
        return default
    try:
        return _parse_bool(raw)
    except ValueError:
        return default


def _parse_positive_float_default(raw: str | None, default: float) -> float:
    if raw is None or not str(raw).strip():
        return default
    try:
        v = float(str(raw).strip())
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


def _parse_positive_int_default(raw: str | None, default: int) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip())
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


def _parse_confidence_pct_default(raw: str | None, default: int) -> int:
    if raw is None or not str(raw).strip():
        return default
    try:
        v = int(str(raw).strip())
        return v if 0 <= v <= 100 else default
    except (ValueError, TypeError):
        return default