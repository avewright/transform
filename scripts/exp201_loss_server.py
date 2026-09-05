#!/usr/bin/env python3
"""Live loss plot for exp201. Parses train.log, serves Chart.js.

  python3 scripts/exp201_loss_server.py
  # http://localhost:8090/
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "outputs/exp201_recurrent_64/train.log"
DEFAULT_PORT = 8090

STEP_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\] step (\d+)/(\d+) \| loss=([-\d.]+|nan|inf) \| "
    r"([\d.]+) pos/s(?: \| vram=([\d.]+)GB)?"
)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>exp201 loss</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root { color-scheme: dark; --bg:#0f1115; --panel:#171a21; --text:#e8eaed; --muted:#9aa0a6; --line:#2a2f3a; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--text); }
  header { padding:20px 24px 8px; }
  h1 { margin:0 0 6px; font-size:20px; font-weight:600; }
  .sub { color:var(--muted); font-size:12px; }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; padding:8px 24px 16px; }
  .stat { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
  .stat b { display:block; font-size:20px; font-variant-numeric:tabular-nums; }
  .stat span { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .grid { display:grid; grid-template-columns:1fr; gap:16px; padding:0 24px 24px; }
  @media (min-width:960px){ .grid.two { grid-template-columns:2fr 1fr; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px 8px; }
  .card h2 { margin:0 0 8px; font-size:13px; font-weight:600; color:var(--muted); }
  canvas { width:100% !important; max-height:420px; }
</style>
</head>
<body>
<header>
  <h1>exp201 · squares64 loss</h1>
  <div class="sub" id="sub">Auto-refreshes every 10s from train.log</div>
</header>
<div class="stats" id="stats"></div>
<div class="grid two">
  <div class="card"><h2>Loss vs step</h2><canvas id="loss"></canvas></div>
  <div class="card"><h2>Throughput (pos/s)</h2><canvas id="speed"></canvas></div>
</div>
<script>
const charts = {};
function ema(arr, alpha){
  if(!arr.length) return [];
  let y = arr[0];
  return arr.map(x => { y = alpha*x + (1-alpha)*y; return y; });
}
function line(label, data, color, extra={}){
  return Object.assign({
    label, data, borderColor: color, backgroundColor: color,
    borderWidth: 1.6, pointRadius: 0, tension: 0.15,
  }, extra);
}
function upsert(id, labels, datasets, yTitle){
  if(charts[id]){
    charts[id].data.labels = labels;
    charts[id].data.datasets = datasets;
    charts[id].update('none');
    return;
  }
  charts[id] = new Chart(document.getElementById(id), {
    type:'line',
    data:{ labels, datasets },
    options:{
      responsive:true, animation:false,
      interaction:{ mode:'index', intersect:false },
      plugins:{ legend:{ labels:{ color:'#c4c7ce', boxWidth:12 } } },
      scales:{
        x:{ title:{display:true,text:'step',color:'#9aa0a6'}, ticks:{color:'#9aa0a6',maxTicksLimit:12}, grid:{color:'#2a2f3a'} },
        y:{ title:{display:true,text:yTitle,color:'#9aa0a6'}, ticks:{color:'#9aa0a6'}, grid:{color:'#2a2f3a'} },
      },
    },
  });
}
async function refresh(){
  const r = await fetch('/api/metrics');
  const d = await r.json();
  const steps = d.steps || [];
  const last = steps[steps.length-1];
  const recent = steps.slice(-20);
  const avg = recent.length ? recent.reduce((a,s)=>a+s.loss,0)/recent.length : null;
  const lo = recent.length ? Math.min(...recent.map(s=>s.loss)) : null;
  if(last){
    const pct = (100*last.step/d.total).toFixed(1);
    document.getElementById('stats').innerHTML = [
      ['Step', last.step.toLocaleString()+' / '+d.total.toLocaleString()],
      ['Progress', pct+'%'],
      ['Loss', last.loss.toFixed(4)],
      ['EMA / last 20', avg!==null ? avg.toFixed(4) : '—'],
      ['Recent min', lo!==null ? lo.toFixed(4) : '—'],
      ['pos/s', last.pos_s.toFixed(0)],
      ['VRAM', last.vram ? last.vram.toFixed(2)+' GB' : '—'],
      ['Clock', last.t || '—'],
    ].map(([k,v])=>`<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');
    document.getElementById('sub').textContent =
      `Auto-refreshes every 10s · ${steps.length} points · log ${d.log}`;
  }
  const stride = steps.length > 400 ? Math.ceil(steps.length/400) : 1;
  const S = steps.filter((_,i)=> i%stride===0 || i===steps.length-1);
  const L = S.map(s=>s.step);
  const raw = S.map(s=>s.loss);
  upsert('loss', L, [
    line('loss', raw, '#8ab4f8', {borderWidth:1, pointRadius:0}),
    line('EMA', ema(raw, 0.08), '#81c995'),
  ], 'loss');
  upsert('speed', L, [line('pos/s', S.map(s=>s.pos_s), '#c58af9')], 'pos/s');
}
refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


def parse_log(path: Path) -> dict:
    if not path.exists():
        return {"steps": [], "total": 0, "log": str(path)}
    steps = []
    total = 0
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        loss_s = m.group(4)
        try:
            loss = float(loss_s)
        except ValueError:
            continue
        if loss != loss:  # NaN
            continue
        total = int(m.group(3))
        steps.append(
            {
                "t": m.group(1),
                "step": int(m.group(2)),
                "loss": loss,
                "pos_s": float(m.group(5)),
                "vram": float(m.group(6)) if m.group(6) else None,
            }
        )
    cut = 0
    for i in range(1, len(steps)):
        if steps[i]["step"] < steps[i - 1]["step"]:
            cut = i
    steps = steps[cut:]
    return {"steps": steps, "total": total, "log": str(path)}


class Handler(BaseHTTPRequestHandler):
    log_path: Path = DEFAULT_LOG

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/metrics"):
            payload = json.dumps(parse_log(self.log_path)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    Handler.log_path = Path(args.log)
    print(f"exp201 loss: http://0.0.0.0:{args.port}/  (log={Handler.log_path})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
