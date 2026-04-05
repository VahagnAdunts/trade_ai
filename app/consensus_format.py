"""Format consensus alerts for Telegram (shared by engine and CLI)."""

from __future__ import annotations

from typing import Any, Mapping


def format_consensus_telegram_message(
    symbol: str,
    consensus: Mapping[str, Any],
    per_model: Mapping[str, Mapping[str, Any] | dict],
    *,
    crypto: bool = False,
    manual_execution_note: bool = False,
) -> str:
    """
    per_model keys: chatgpt, gemini, claude, grok — each value model dump or {"error": "..."}.
    Equities and crypto use the same long/short lines; crypto adds a tag and optional manual note.
    """
    action = str(consensus.get("aligned_action") or "none").upper()
    min_conf = consensus.get("minimum_confidence", 0)
    tag = "[CRYPTO] " if crypto else ""
    lines = [f"{tag}CONSENSUS ✅ {symbol} {action} (min {min_conf}%)"]
    for key in ("chatgpt", "gemini", "claude", "grok"):
        item = per_model.get(key) or {}
        if item.get("error"):
            lines.append(f"{key}: ERR")
            continue
        lc = item.get("long_confidence", "-")
        sc = item.get("short_confidence", "-")
        side = str(item.get("action") or item.get("predicted_side") or "?").upper()
        conf = item.get("confidence", item.get("winning_confidence", "-"))
        lines.append(f"{key}: L{lc}/S{sc} -> {side} {conf}%")

    if manual_execution_note and crypto:
        lines.append("")
        lines.append("(No exchange API — trade manually on your venue.)")

    return "\n".join(lines)
