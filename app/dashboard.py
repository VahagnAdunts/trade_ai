"""
Local web dashboard with live progress over WebSocket.
Run: python main.py --serve
Then open http://127.0.0.1:8765 and click "Start analysis".
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app.config import AppConfig
from app.engine import run_analysis

app = FastAPI(title="Tred AI — live run")

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tred AI — live run</title>
  <style>
    :root { --bg: #0f172a; --card: #1e293b; --text: #f1f5f9; --muted: #94a3b8; --ok: #22c55e; --err: #ef4444; --accent: #38bdf8; }
    * { box-sizing: border-box; }
    body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--text); min-height: 100vh; }
    header { padding: 1rem 1.5rem; border-bottom: 1px solid #334155; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
    h1 { font-size: 1.125rem; margin: 0; font-weight: 600; }
    main { padding: 1rem 1.5rem; max-width: 1100px; margin: 0 auto; }
    button { background: var(--accent); color: #0f172a; border: 0; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status { color: var(--muted); font-size: 0.875rem; }
    .log { background: var(--card); border-radius: 10px; padding: 1rem; margin-top: 1rem; max-height: 45vh; overflow: auto; font-family: ui-monospace, monospace; font-size: 0.8rem; line-height: 1.5; }
    .log-line { border-bottom: 1px solid #334155; padding: 0.35rem 0; word-break: break-word; }
    .log-line:last-child { border-bottom: 0; }
    .tag { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.7rem; margin-right: 0.5rem; }
    .tag-run { background: #4f46e5; }
    .tag-data { background: #0369a1; }
    .tag-model { background: #15803d; }
    .tag-consensus { background: #a16207; }
    .tag-done { background: #7c3aed; }
    .tag-err { background: var(--err); }
    section.card { background: var(--card); border-radius: 10px; padding: 1rem; margin-top: 1rem; }
    #symbolPanels { display: grid; gap: 0.75rem; }
    .sym { border: 1px solid #334155; border-radius: 8px; padding: 0.75rem; }
    .sym h3 { margin: 0 0 0.5rem; font-size: 0.95rem; }
    .sym .row { font-size: 0.8rem; color: var(--muted); margin: 0.25rem 0; }
    table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 0.5rem; }
    th, td { border: 1px solid #334155; padding: 0.35rem 0.5rem; text-align: left; }
    th { background: #334155; }
    a { color: var(--accent); }
  </style>
</head>
<body>
  <header>
    <h1>Tred AI — live analysis</h1>
    <button id="btn" type="button">Start analysis</button>
    <span class="status" id="conn">Disconnected</span>
  </header>
  <main>
    <p class="status">Connects via WebSocket to this machine only. Click <strong>Start analysis</strong> to fetch data and run all models; results stream below as they finish.</p>
    <div class="log" id="log"></div>
    <section class="card">
      <h2 style="margin:0 0 0.5rem; font-size:1rem;">Current symbol</h2>
      <div id="symbolPanels"></div>
    </section>
    <section class="card" id="reportLinks" style="display:none;">
      <h2 style="margin:0 0 0.5rem; font-size:1rem;">Reports saved</h2>
      <p id="links"></p>
    </section>
  </main>
  <script>
    const logEl = document.getElementById('log');
    const btn = document.getElementById('btn');
    const connEl = document.getElementById('conn');
    const panels = document.getElementById('symbolPanels');
    const reportLinks = document.getElementById('reportLinks');
    const linksEl = document.getElementById('links');

    function appendLog(tag, text) {
      const line = document.createElement('div');
      line.className = 'log-line';
      line.innerHTML = '<span class="tag tag-' + tag + '">' + tag + '</span>' + escapeHtml(text);
      logEl.appendChild(line);
      logEl.scrollTop = logEl.scrollHeight;
    }
    function escapeHtml(s) {
      const d = document.createElement('div');
      d.textContent = s;
      return d.innerHTML;
    }

    let ws;
    function connect() {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(proto + '//' + location.host + '/ws');
      ws.onopen = () => { connEl.textContent = 'Connected'; connEl.style.color = 'var(--ok)'; };
      ws.onclose = () => { connEl.textContent = 'Disconnected'; connEl.style.color = 'var(--muted)'; btn.disabled = false; };
      ws.onerror = () => appendLog('err', 'WebSocket error');
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        handle(msg);
      };
    }

    const symState = {};
    function ensurePanel(symbol) {
      if (symState[symbol]) return symState[symbol];
      const div = document.createElement('div');
      div.className = 'sym';
      div.id = 'panel-' + symbol;
      div.innerHTML = '<h3>' + escapeHtml(symbol) + '</h3><div class="meta"></div><table><thead><tr><th>Model</th><th>Long</th><th>Short</th><th>Chosen</th><th>Win</th></tr></thead><tbody></tbody></table>';
      panels.appendChild(div);
      symState[symbol] = { div, tbody: div.querySelector('tbody'), meta: div.querySelector('.meta') };
      return symState[symbol];
    }

    function handle(msg) {
      const t = msg.type;
      if (t === 'run_started') {
        appendLog('run', 'Run started — symbols: ' + (msg.symbols || []).join(', '));
        panels.innerHTML = '';
        Object.keys(symState).forEach(k => delete symState[k]);
        reportLinks.style.display = 'none';
        return;
      }
      if (t === 'symbol_phase' && msg.phase === 'start') {
        appendLog('run', 'Symbol: ' + msg.symbol);
        ensurePanel(msg.symbol);
        return;
      }
      if (t === 'data_loaded') {
        appendLog('data', msg.symbol + ' — ' + msg.candles_count + ' hourly candles | ' + msg.range_start + ' → ' + msg.range_end);
        const p = ensurePanel(msg.symbol);
        p.meta.textContent = 'Candles: ' + msg.candles_count;
        return;
      }
      if (t === 'models_started') {
        appendLog('run', msg.symbol + ' — models running…');
        return;
      }
      if (t === 'model_result') {
        const p = ensurePanel(msg.symbol);
        const tr = document.createElement('tr');
        if (msg.ok) {
          const d = msg.decision;
          tr.innerHTML = '<td>' + escapeHtml(d.model) + '</td><td>' + d.long_confidence + '%</td><td>' + d.short_confidence + '%</td><td>' + escapeHtml(d.action) + '</td><td>' + d.confidence + '%</td>';
          appendLog('model', msg.symbol + ' ' + d.model + ' L' + d.long_confidence + '% S' + d.short_confidence + '% → ' + d.action + ' (win ' + d.confidence + '%)');
        } else {
          tr.innerHTML = '<td>' + escapeHtml(msg.model_label) + '</td><td colspan="4" style="color:var(--err)">' + escapeHtml(msg.error || '') + '</td>';
          appendLog('err', msg.symbol + ' ' + msg.model_label + ': ' + (msg.error || ''));
        }
        p.tbody.appendChild(tr);
        return;
      }
      if (t === 'consensus') {
        const c = msg.consensus;
        appendLog('consensus', msg.symbol + ' — aligned=' + (c.aligned_action || 'none') + ' min_conf=' + c.minimum_confidence + '% passes=' + c.passes_threshold);
        return;
      }
      if (t === 'finished') {
        appendLog('done', 'Finished — report.json written.');
        reportLinks.style.display = 'block';
        linksEl.innerHTML = '<a href="/outputs/report.json" target="_blank">Open report.json (latest)</a>';
        btn.disabled = false;
        return;
      }
      if (t === 'error') {
        appendLog('err', msg.message || 'Unknown error');
        btn.disabled = false;
        return;
      }
      appendLog('run', JSON.stringify(msg));
    }

    btn.onclick = () => {
      if (!ws || ws.readyState !== WebSocket.OPEN) connect();
      const trySend = () => {
        if (ws.readyState === WebSocket.OPEN) {
          btn.disabled = true;
          logEl.innerHTML = '';
          ws.send(JSON.stringify({ cmd: 'run' }));
        } else setTimeout(trySend, 100);
      };
      trySend();
    };

    connect();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return DASHBOARD_HTML


@app.websocket("/ws")
async def websocket_run(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
    except WebSocketDisconnect:
        await websocket.close()
        return

    if payload.get("cmd") != "run":
        await websocket.send_json({"type": "error", "message": "Send {cmd: run}"})
        await websocket.close()
        return

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def runner() -> None:
        try:
            config = AppConfig.from_env()
            await run_analysis(config, on_event=on_event)
        except Exception as exc:
            await queue.put({"type": "error", "message": str(exc)})

    task = asyncio.create_task(runner())

    try:
        while True:
            if task.done() and queue.empty():
                try:
                    task.result()
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.4)
            except asyncio.TimeoutError:
                continue
            await websocket.send_json(event)
            if event.get("type") == "finished":
                break
            if event.get("type") == "error":
                break
    except WebSocketDisconnect:
        task.cancel()
    finally:
        if not task.done():
            task.cancel()


def mount_static_outputs(application: FastAPI) -> None:
    """Serve outputs/ so /outputs/report.json is available after a run."""
    from fastapi.staticfiles import StaticFiles

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    application.mount(
        "/outputs",
        StaticFiles(directory=str(out.resolve()), html=True),
        name="outputs",
    )


mount_static_outputs(app)
