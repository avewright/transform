#!/usr/bin/env python3
"""Live monitor for gated N=3 ChessBot SF19 Polar-NorMuon.

  python3 scripts/chessbot_sf19_monitor.py
  # http://0.0.0.0:8094/
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "outputs/chessbot_sf19_n3/events.jsonl"
DEFAULT_CFG = ROOT / "configs/chessbot_sf19_n3.json"
DEFAULT_PORT = 8094
VENDOR_CHART = Path(__file__).resolve().parent / "vendor" / "chart.umd.min.js"
CHART_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"


def total_steps(cfg_path: Path) -> int:
    if cfg_path.exists():
        try:
            return int(json.loads(cfg_path.read_text()).get("steps") or 122500)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 122500


def parse_events(path: Path, steps: int) -> dict:
    train, val, checkpoints, matches = [], [], [], []
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
            elif stage == "train" and "step" in row:
                lr = row.get("lr") or {}
                train.append({
                    "step": int(row["step"]),
                    "loss": float(row.get("loss") or 0),
                    "policy": float(row.get("policy") or 0),
                    "hard": float(row.get("hard") or 0),
                    "soft": float(row.get("soft") or 0),
                    "value": float(row.get("value") or 0),
                    "top1": float(row.get("top1") or 0),
                    "value_frac": float(row.get("value_frac") or 0),
                    "gate": float(row.get("gate") or 0),
                    "grad_norm": float(row["grad_norm"]) if row.get("grad_norm") is not None else None,
                    "examples": int(row.get("examples") or 0),
                    "pos_s": float(row.get("pos_per_s") or 0),
                    "muon_lr": float(lr["muon"]) if isinstance(lr, dict) and lr.get("muon") is not None else None,
                    "vram": row.get("peak_vram_gb"),
                })
            elif stage == "val" and "step" in row:
                val.append({
                    "step": int(row["step"]),
                    "hard": float(row.get("hard") or 0),
                    "soft": float(row.get("soft") or 0),
                    "top1": float(row.get("top1") or 0),
                    "value_hard": float(row.get("value_hard") or 0),
                    "gate": float(row.get("gate") or 0),
                    "n": int(row.get("n") or 0),
                })
            elif stage == "checkpoint":
                checkpoints.append(int(row.get("step") or 0))
            elif stage == "match":
                matches.append({
                    "step": int(row.get("step") or 0),
                    "score": row.get("score"),
                    "wins": row.get("wins"),
                    "draws": row.get("draws"),
                    "losses": row.get("losses"),
                    "verdict": row.get("verdict"),
                    "paired_ci_95": row.get("paired_ci_95"),
                    "gate": row.get("gate"),
                    "elapsed_s": row.get("elapsed_s"),
                    "n": row.get("n"),
                })
    last = train[-1] if train else None
    eta_s = None
    if last and last["pos_s"] > 0:
        remain = max(steps - last["step"], 0)
        batch = 64
        if last["step"] > 0 and last["examples"]:
            batch = max(int(round(last["examples"] / last["step"])), 1)
        eta_s = remain * batch / last["pos_s"]
    return {
        "train": train,
        "val": val,
        "last": last,
        "last_val": val[-1] if val else None,
        "loaded": loaded,
        "n": len(train),
        "total_steps": steps,
        "eta_s": eta_s,
        "log": str(path),
        "alive": path.exists(),
        "checkpoints": checkpoints[-6:],
        "matches": matches,
        "last_match": matches[-1] if matches else None,
    }


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
<title>ChessBot SF19 N=3</title>
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
  <h1>ChessBot SF19 · gated N=3 · Polar-NorMuon</h1>
  <div class="sub" id="sub">Auto-refreshes every 5s</div>
</header>
<div class="stats" id="stats"></div>
<div class="grid two">
  <div class="card"><h2>Vs published ChessBot (paired score)</h2><canvas id="match"></canvas></div>
  <div class="card"><h2>Holdout hard CE / top-1</h2><canvas id="val"></canvas></div>
</div>
<div class="grid two">
  <div class="card"><h2>Train policy CE</h2><canvas id="train"></canvas></div>
  <div class="card"><h2>Gate tanh(α)</h2><canvas id="gate"></canvas></div>
</div>
<div class="grid two">
  <div class="card"><h2>Grad / throughput</h2><canvas id="grad"></canvas></div>
  <div class="card"><h2>Match log</h2><pre id="matchlog" style="margin:0;font:12px/1.45 ui-monospace,monospace;color:#c4c7ce;white-space:pre-wrap"></pre></div>
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
function pct(x){ return x==null ? '—' : (100*x).toFixed(1)+'%'; }
async function refresh(){
  const d = await (await fetch('/api/metrics')).json();
  const steps = d.train || [];
  const last = d.last;
  const lv = d.last_val;
  const lm = d.last_match;
  const loaded = d.loaded || {};
  const ci = lm && lm.paired_ci_95;
  document.getElementById('stats').innerHTML = [
    ['Step', last ? fmt(last.step)+' / '+fmt(d.total_steps) : 'waiting'],
    ['Vs ChessBot', lm && lm.verdict ? lm.verdict : 'not evaluated'],
    ['Match score', lm && lm.score!=null ? Number(lm.score).toFixed(3) : '—'],
    ['W / D / L', lm ? `${lm.wins}/${lm.draws}/${lm.losses}` : '—'],
    ['Paired 95% CI', ci && ci[0]!=null ? `${Number(ci[0]).toFixed(2)}–${Number(ci[1]).toFixed(2)}` : '—'],
    ['Val hard', lv ? lv.hard.toFixed(3) : '—'],
    ['Val top-1', lv ? pct(lv.top1) : '—'],
    ['Gate', last ? last.gate.toExponential(2) : '—'],
    ['pos/s', last ? last.pos_s.toFixed(0) : '—'],
    ['ETA', eta(d.eta_s)],
  ].map(([k,v])=>stat(k,v)).join('');
  document.getElementById('sub').textContent =
    `Auto-refreshes every 5s · ${steps.length} train · ${(d.val||[]).length} val · ${(d.matches||[]).length} matches · depth ${loaded.effective_depth||22} · ${d.log}`;
  document.getElementById('matchlog').textContent = (d.matches||[]).slice(-8).map(m => {
    const interval = m.paired_ci_95 && m.paired_ci_95[0]!=null
      ? `[${Number(m.paired_ci_95[0]).toFixed(2)}, ${Number(m.paired_ci_95[1]).toFixed(2)}]` : '';
    return `step ${m.step}  ${m.verdict||'—'}  ${m.score!=null?Number(m.score).toFixed(3):'—'}  ${m.wins}/${m.draws}/${m.losses}  ${interval}`;
  }).join('\n') || 'No matches yet. Watcher plays 16 opening pairs vs published ChessBot every 2500 steps.';
  if(!steps.length) return;
  const stride = steps.length>400 ? Math.ceil(steps.length/400) : 1;
  const S = steps.filter((_,i)=> i%stride===0 || i===steps.length-1);
  const L = S.map(s=>s.step);
  const val = d.val || [];
  upsert('val', val.map(s=>s.step), [
    line('hard CE', val.map(s=>s.hard), '#f28b82', {pointRadius:3}),
    line('soft CE', val.map(s=>s.soft), '#8ab4f8', {pointRadius:3}),
    line('top-1', val.map(s=>s.top1), '#81c995', {pointRadius:3, yAxisID:undefined}),
  ], 'holdout');
  upsert('train', L, [
    line('hard', S.map(s=>s.hard), '#f28b82', {borderWidth:1}),
    line('soft', S.map(s=>s.soft), '#8ab4f8', {borderWidth:1}),
    line('hard EMA', ema(S.map(s=>s.hard), 0.08), '#fdd663'),
  ], 'CE');
  const matches = d.matches || [];
  if(matches.length){
    upsert('match', matches.map(s=>s.step), [
      line('score', matches.map(s=>s.score), '#81c995', {pointRadius:4}),
      line('0.50', matches.map(()=>0.5), '#9aa0a6', {borderDash:[4,4], borderWidth:1}),
    ], 'paired score');
  }
  upsert('gate', L, [line('tanh(α)', S.map(s=>s.gate), '#c58af9')], 'gate');
  upsert('grad', L, [
    line('grad', S.map(s=>s.grad_norm), '#fdd663'),
    line('pos/s', S.map(s=>s.pos_s), '#8ab4f8'),
  ], 'norm / pos/s');
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
    steps: int = 122500

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
            self._send(json.dumps(parse_events(self.log_path, self.steps)).encode(), "application/json")
            return
        self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = (ROOT / log_path).resolve()
    Handler.log_path = log_path
    Handler.steps = total_steps(Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config)
    try:
        ensure_chart()
    except Exception as e:
        print(f"warn: chart.js download failed ({e})", flush=True)
    print(f"sf19 n3 monitor: http://0.0.0.0:{args.port}/  (log={Handler.log_path})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
