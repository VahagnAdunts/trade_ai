"""
Persist scheduled Alpaca closes to disk so a worker restart can still close positions
after ALPACA_HOLD_SECONDS. Single JSON file; entries removed after successful close.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from alpaca.common.exceptions import APIError

from app.alpaca_trading import _close_position_with_retries, _format_alpaca_error
from app.config import AppConfig

_file_lock = asyncio.Lock()
_recovery_lock = asyncio.Lock()
_scheduled_recovery_ids: set[str] = set()


def _pending_path(config: AppConfig) -> Path:
    p = Path(config.alpaca_pending_closes_file)
    return p if p.is_absolute() else Path.cwd() / p


def _parse_close_at(raw: str) -> datetime:
    s = raw.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def should_clear_stale_pending_no_position(exc: BaseException, formatted: str) -> bool:
    """True if close failed because there is nothing to close (safe to drop pending row)."""
    text = (formatted + str(exc)).lower()
    if any(
        x in text
        for x in (
            "no open",
            "no position",
            "not found",
            "does not exist",
            "nothing to close",
        )
    ):
        return True
    if isinstance(exc, APIError):
        sc = getattr(exc, "status_code", None)
        if sc in (404, 422):
            return True
    return False


def _read_all_sync(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict) and x.get("id")]
    except (json.JSONDecodeError, OSError):
        return []


def _write_all_sync(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(items, indent=2)
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


async def register_pending_close(config: AppConfig, record: Dict[str, Any]) -> None:
    """Append one pending close (call after open order succeeds)."""
    path = _pending_path(config)
    async with _file_lock:
        items = _read_all_sync(path)
        # de-dupe by id
        ids = {r["id"] for r in items}
        rid = record["id"]
        if rid in ids:
            return
        items.append(record)
        await asyncio.to_thread(_write_all_sync, path, items)
    print(
        f"[Alpaca pending] registered close id={rid} at {record.get('close_at_utc')} "
        f"symbol={record.get('symbol')}",
        flush=True,
    )


async def clear_pending_close(config: AppConfig, pending_id: str) -> None:
    """Remove pending row after close succeeds or is confirmed flat."""
    path = _pending_path(config)
    async with _file_lock:
        items = _read_all_sync(path)
        kept = [r for r in items if r.get("id") != pending_id]
        if len(kept) == len(items):
            return
        await asyncio.to_thread(_write_all_sync, path, kept)
    print(f"[Alpaca pending] cleared id={pending_id}", flush=True)


async def _close_and_finalize(
    config: AppConfig,
    item: Dict[str, Any],
) -> None:
    pid = str(item["id"])
    symbol = str(item["symbol"])
    _close_res, close_exc = await _close_position_with_retries(
        config, symbol, hold_seconds=None
    )
    if close_exc is None:
        await clear_pending_close(config, pid)
        print(f"[Alpaca pending] closed & cleared {symbol} id={pid}", flush=True)
        return
    err = _format_alpaca_error(close_exc)
    if should_clear_stale_pending_no_position(close_exc, err):
        await clear_pending_close(config, pid)
        print(
            f"[Alpaca pending] no position for {symbol}; cleared stale id={pid}",
            flush=True,
        )
    else:
        print(f"[Alpaca pending] close failed {symbol} id={pid}: {err}", flush=True)


async def _delayed_close_task(config: AppConfig, item: Dict[str, Any]) -> None:
    """Sleep until close_at, then close and clear."""
    pid = str(item.get("id", ""))
    try:
        close_at = _parse_close_at(str(item["close_at_utc"]))
        now = datetime.now(timezone.utc)
        delay = (close_at - now).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        await _close_and_finalize(config, item)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"[Alpaca pending] delayed task error: {exc}", flush=True)
    finally:
        if pid:
            _scheduled_recovery_ids.discard(pid)


async def reconcile_pending_closes_on_startup(config: AppConfig) -> None:
    """
    On startup: close overdue pendings immediately; schedule future ones.
    Safe to call when no file exists or Alpaca keys are missing.
    """
    if not (config.alpaca_api_key_id and config.alpaca_api_secret_key):
        return
    async with _recovery_lock:
        path = _pending_path(config)
        async with _file_lock:
            items = _read_all_sync(path)
        if not items:
            return
        now = datetime.now(timezone.utc)
        overdue: List[Dict[str, Any]] = []
        future: List[Dict[str, Any]] = []
        for item in items:
            try:
                close_at = _parse_close_at(str(item["close_at_utc"]))
            except (TypeError, ValueError):
                print(f"[Alpaca pending] dropping bad close_at: {item!r}", flush=True)
                continue
            if close_at <= now:
                overdue.append(item)
            else:
                future.append(item)
        if overdue:
            print(
                f"[Alpaca pending] recovering {len(overdue)} overdue close(s) from {path}",
                flush=True,
            )
        for item in overdue:
            await _close_and_finalize(config, item)
        for item in future:
            pid = str(item.get("id", ""))
            if not pid or pid in _scheduled_recovery_ids:
                continue
            _scheduled_recovery_ids.add(pid)
            asyncio.create_task(_delayed_close_task(config, item))
            print(
                f"[Alpaca pending] scheduled close for {item.get('symbol')} at "
                f"{item.get('close_at_utc')}",
                flush=True,
            )


def new_pending_record(
    *,
    symbol: str,
    hold_seconds: int,
    crypto: bool,
    side: str,
    paper: bool,
) -> tuple[str, Dict[str, Any]]:
    """Build a pending close row and return (id, record)."""
    pending_id = str(uuid.uuid4())
    close_at = datetime.now(timezone.utc) + timedelta(seconds=hold_seconds)
    return pending_id, {
        "id": pending_id,
        "symbol": symbol,
        "close_at_utc": close_at.isoformat(),
        "crypto": crypto,
        "side": side,
        "paper": paper,
    }
