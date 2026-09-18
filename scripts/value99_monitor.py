#!/usr/bin/env python3
"""Live Value99 loss monitor.

  python3 scripts/value99_monitor.py
  # http://0.0.0.0:8092/
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "outputs/value99_50m/events.jsonl"
DEFAULT_PORT = 8092
TOTAL_STEPS = 96_677
VENDOR_CHART = Path(__file__).resolve().parent / "vendor" / "chart.umd.min.js"
CHART_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"


def parse_events(path: Path) -> dict:
    train, val, teacher = [], [], []
    loaded = None
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stage = row.get("stage")
            if stage == "loaded":
                loaded = row
            elif stage == "train" and "step" in row and "loss" in row:
                train.append({
                    "step": int(row["step"]),
                    "loss": float(row["loss"]),
                    "mse": float(row.get("mse") or 0),
                    "grad_norm": float(row["grad_norm"]) if row.get("grad_norm") is not None else None,
                    "examples": int(row.get("examples") or 0),
                    "pos_s": float(row.get("positions_per_s") or 0),
                    "lr": row.get("lr"),
                    "elapsed_s": float(row.get("elapsed_s") or 0),
                    "vram": row.get("peak_vram_gb"),
                })
            elif stage in ("validation", "initial_validation") and "mse" in row:
                val.append({
                    "step": int(row.get("step") or 0),
                    "mse": float(row["mse"]),
                    "mae": float(row.get("mae") or 0),
                    "decisive_direction": row.get("decisive_direction"),
                    "phase_mse": row.get("phase_mse") or {},
                })
            elif stage == "teacher" and "mse" in row:
                teacher.append({
                    "step": int(row.get("step") or 0),
                    "mse": float(row["mse"]),
                    "mae": float(row.get("mae") or 0),
                    "decisive_direction": row.get("decisive_direction"),
                    "phase_mse": row.get("phase_mse") or {},
                })
    last = train[-1] if train else None
    total = parse_events.total_steps
    eta_s = None
    if last and last["pos_s"] > 0:
        remain = max(total - last["step"], 0)
        eta_s = remain * 512 / last["pos_s"]
    return {
        "train": train,
        "val": val,
        "last": last,
        "last_val": val[-1] if val else None,
        "teacher": teacher,
        "last_teacher": teacher[-1] if teacher else None,
        "loaded": loaded,
        "n": len(train),
        "total_steps": total,
        "eta_s": eta_s,
        "log": str(path),
        "alive": path.exists(),
    }


parse_events.total_steps = TOTAL_STEPS


def ensure_chart() -> Path:
    if VENDOR_CHART.exists() and VENDOR_CHART.stat().st_size > 10_000:
        return VENDOR_CHART
    VENDOR_CHART.parent.mkdir(parents=True, exist_ok=True)
    from urllib.request import urlopen
    with urlopen(CHART_URL, timeout=30) as r:
        VENDOR_CHART.write_bytes(r.read())
    return VENDOR_CHART


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Value99 live</title>
<script src="vendor/chart.umd.min.js"></script>
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
  @media (min-width:960px){ .grid.two { grid-template-columns:1fr 1fr; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px 8px; }
  .card h2 { margin:0 0 8px; font-size:13px; font-weight:600; color:var(--muted); }
  canvas { width:100% !important; max-height:360px; }
</style>
</head>
<body>
<header>
  <h1>Value99 · 99M scratch · live snapshot</h1>
  <div class="sub" id="sub">Auto-refreshes every 5s</div>
</header>
<div class="stats" id="stats"></div>
<div class="grid two">
  <div class="card"><h2>Train loss</h2><canvas id="loss"></canvas></div>
  <div class="card"><h2>Train / val MSE</h2><canvas id="mse"></canvas></div>
</div>
<div class="grid two">
  <div class="card"><h2>Throughput</h2><canvas id="speed"></canvas></div>
  <div class="card"><h2>Grad norm</h2><canvas id="grad"></canvas></div>
</div>
<script>
const charts = {};
function line(label, data, color, extra={}){
  return Object.assign({label, data, borderColor:color, backgroundColor:'transparent',
    borderWidth:1.6, pointRadius:0, tension:0.15}, extra);
}
function ema(xs, a){
  let y=null; return xs.map(x => { if(x==null) return null; y = y==null?x:y*(1-a)+x*a; return y; });
}
function upsert(id, labels, datasets, yTitle){
  const el = document.getElementById(id);
  if(charts[id]){ charts[id].data.labels=labels; charts[id].data.datasets=datasets; charts[id].update('none'); return; }
  charts[id] = new Chart(el, { type:'line', data:{labels, datasets}, options:{
    responsive:true, animation:false, interaction:{mode:'index', intersect:false},
    plugins:{legend:{labels:{color:'#c4c7ce', boxWidth:12}}},
    scales:{
      x:{title:{display:true,text:'step',color:'#9aa0a6'}, ticks:{color:'#9aa0a6',maxTicksLimit:12}, grid:{color:'#2a2f3a'}},
      y:{title:{display:true,text:yTitle,color:'#9aa0a6'}, ticks:{color:'#9aa0a6'}, grid:{color:'#2a2f3a'}},
    },
  }});
}
function stat(k,v){ return `<div class="stat"><b>${v}</b><span>${k}</span></div>`; }
function fmt(n){ return n==null ? '—' : Number(n).toLocaleString(); }
function eta(s){
  if(s==null || s<0) return '—';
  const h=s/3600;
  if(h<24) return h.toFixed(1)+' h';
  return (h/24).toFixed(1)+' d';
}
async function refresh(){
  const d = await (await fetch('/api/metrics')).json();
  const steps = d.train || [];
  const last = d.last;
    const lv = d.last_val;
    const lt = d.last_teacher;
  const recent = steps.slice(-20);
  const avg = recent.length ? recent.reduce((a,s)=>a+s.loss,0)/recent.length : null;
  document.getElementById('stats').innerHTML = [
    ['Step', last ? fmt(last.step)+' / '+fmt(d.total_steps) : 'waiting'],
    ['Loss', last ? last.loss.toFixed(4) : '—'],
    ['EMA / last 20', avg!=null ? avg.toFixed(4) : '—'],
    ['Train MSE', last ? last.mse.toFixed(4) : '—'],
    ['Val MSE', lv ? lv.mse.toFixed(4) : '—'],
    ['SF WDL MSE', lt ? lt.mse.toFixed(4) : '—'],
    ['SF direction', lt && lt.decisive_direction!=null ? (100*lt.decisive_direction).toFixed(1)+'%' : '—'],
    ['Positions', last ? fmt(last.examples) : '—'],
    ['pos/s', last ? last.pos_s.toFixed(1) : '—'],
    ['Grad', last && last.grad_norm!=null ? last.grad_norm.toExponential(2) : '—'],
    ['Muon LR', last && last.lr && last.lr.muon!=null ? Number(last.lr.muon).toExponential(2) : (last && typeof last.lr==='number' ? last.lr.toExponential(2) : '—')],
    ['ETA', eta(d.eta_s)],
    ['Elapsed', last ? (last.elapsed_s/60).toFixed(1)+' min' : '—'],
  ].map(([k,v])=>stat(k,v)).join('');
  document.getElementById('sub').textContent =
    `Auto-refreshes every 5s · ${steps.length} train points · ${d.log}`;
  if(!steps.length) return;
  const stride = steps.length>400 ? Math.ceil(steps.length/400) : 1;
  const S = steps.filter((_,i)=> i%stride===0 || i===steps.length-1);
  const L = S.map(s=>s.step);
  upsert('loss', L, [
    line('loss', S.map(s=>s.loss), '#8ab4f8', {borderWidth:1}),
    line('EMA', ema(S.map(s=>s.loss), 0.08), '#81c995'),
  ], 'loss');
  const val = d.val || [];
  const teacher = d.teacher || [];
  upsert('mse', L, [
    line('train mse', S.map(s=>s.mse), '#8ab4f8'),
    line('val mse', L.map(step => {
      let best=null;
      for(const v of val){ if(v.step<=step) best=v.mse; }
      return best;
    }), '#f28b82'),
    line('sf wdl mse', L.map(step => {
      let best=null;
      for(const v of teacher){ if(v.step<=step) best=v.mse; }
      return best;
    }), '#fdd663'),
  ], 'mse');
  upsert('speed', L, [line('pos/s', S.map(s=>s.pos_s), '#c58af9')], 'pos/s');
  upsert('grad', L, [line('grad', S.map(s=>s.grad_norm), '#fdd663')], 'norm');
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    log_path: Path = DEFAULT_LOG

    def log_message(self, fmt, *args):
        pass

    def _send(self, body: bytes, content_type: str, *, cache: str = "no-store") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(HTML.encode(), "text/html; charset=utf-8")
            return
        if path in ("/vendor/chart.umd.min.js", "/chart.js"):
            self._send(ensure_chart().read_bytes(), "application/javascript; charset=utf-8",
                       cache="public, max-age=86400")
            return
        if path.startswith("/api/metrics"):
            self._send(json.dumps(parse_events(self.log_path)).encode(), "application/json")
            return
        self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = ap.parse_args()
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = (ROOT / log_path).resolve()
    Handler.log_path = log_path
    parse_events.total_steps = int(args.steps)
    try:
        ensure_chart()
    except Exception as e:
        print(f"warn: chart.js download failed ({e})", flush=True)
    print(f"value99 monitor: http://0.0.0.0:{args.port}/  (log={Handler.log_path})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
