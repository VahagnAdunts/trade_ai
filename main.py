from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
import webbrowser

import httpx

from app.alpaca_pending import reconcile_pending_closes_on_startup
from app.config import AppConfig
from app.engine import run_analysis
from app.telegram_notifier import TelegramConfig, send_telegram_message


async def _run_cli(*, crypto: bool = False) -> None:
    config = AppConfig.from_env()
    output_path = await run_analysis(config, crypto=crypto)
    print(f"Research report: {output_path}")
    if output_path.exists():
        webbrowser.open(output_path.resolve().as_uri())


async def _run_from_telegram(config: AppConfig, tg: TelegramConfig) -> None:
    if not tg.enabled or not tg.bot_token or not tg.chat_id:
        raise SystemExit(
            "Telegram runner requires TELEGRAM_ENABLED=true, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID."
        )

    print(
        "Telegram runner started. Send /run (equities) or /run_crypto (crypto) to trigger analysis."
    )
    if config.alpaca_api_key_id and config.alpaca_api_secret_key:
        await reconcile_pending_closes_on_startup(config)
    await send_telegram_message(
        tg,
        "Tred_ai bot is online.\nCommands: /run, /run_crypto, /status, /help",
    )

    updates_url = f"https://api.telegram.org/bot{tg.bot_token}/getUpdates"
    offset = 0
    running = False

    while True:
        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                resp = await client.get(
                    updates_url,
                    params={"timeout": 30, "offset": offset},
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:
            print(f"Telegram polling error: {exc}")
            await asyncio.sleep(3)
            continue

        for item in payload.get("result", []):
            offset = max(offset, int(item.get("update_id", 0)) + 1)
            msg = item.get("message") or {}
            chat = msg.get("chat") or {}
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            # Only allow commands from configured chat.
            if str(chat.get("id")) != str(tg.chat_id):
                continue

            cmd = text.lower().split()[0]
            if cmd == "/help":
                await send_telegram_message(
                    tg,
                    "Commands:\n"
                    "/run — full equity analysis (default universe)\n"
                    "/run_crypto — crypto (USD pairs, default top 20)\n"
                    "/status — check whether a run is in progress\n"
                    "/help — this message",
                )
                continue

            if cmd == "/status":
                await send_telegram_message(
                    tg,
                    "Analysis is running."
                    if running
                    else "Idle. Send /run or /run_crypto to start.",
                )
                continue

            if cmd not in ("/run", "/run_crypto"):
                continue

            crypto_run = cmd == "/run_crypto"

            if running:
                await send_telegram_message(
                    tg,
                    "Run request received, but analysis is already running.",
                )
                continue

            running = True
            start_msg = (
                "Starting crypto analysis now..."
                if crypto_run
                else "Starting equity analysis now..."
            )
            await send_telegram_message(tg, start_msg)
            try:
                # Consensus Telegram alerts are sent from run_analysis (single message per signal).
                output_path = await run_analysis(config, crypto=crypto_run)
                report = json.loads(Path(output_path).read_text(encoding="utf-8"))
                consensus_count = len(report.get("consensus_signals", []))
                kind = "Crypto" if crypto_run else "Equity"
                await send_telegram_message(
                    tg,
                    f"{kind} analysis completed.\nConsensus signals: {consensus_count}\n"
                    f"Report: {output_path}",
                )
            except Exception as exc:
                await send_telegram_message(
                    tg,
                    f"Analysis failed: {exc}",
                )
            finally:
                running = False


def _run_serve(host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        print("Install server deps: pip install fastapi uvicorn websockets", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Open http://{host}:{port} — click “Start analysis” for live progress.\n")
    webbrowser.open(f"http://{host}:{port}/")
    uvicorn.run(
        "app.dashboard:app",
        host=host,
        port=port,
        reload=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-LLM trading consensus runner")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start local dashboard with live WebSocket progress (browser)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address for --serve (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for --serve (default: 8765)",
    )
    parser.add_argument(
        "--telegram-runner",
        action="store_true",
        help="Run long-lived Telegram command listener (/run and /run_crypto trigger analysis)",
    )
    parser.add_argument(
        "--crypto",
        action="store_true",
        help="Run crypto pipeline (default top-20 USD pairs, outputs under outputs_crypto/)",
    )
    args = parser.parse_args()

    if args.serve:
        _run_serve(args.host, args.port)
    elif args.telegram_runner:
        cfg = AppConfig.from_env()
        tg = TelegramConfig(
            enabled=cfg.telegram_enabled,
            bot_token=cfg.telegram_bot_token,
            chat_id=cfg.telegram_chat_id,
        )
        asyncio.run(_run_from_telegram(cfg, tg))
    else:
        asyncio.run(_run_cli(crypto=args.crypto))


if __name__ == "__main__":
    main()
