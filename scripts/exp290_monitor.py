#!/usr/bin/env python3
"""Live train+val monitor for exp290.

  python3 scripts/exp290_monitor.py
  # http://127.0.0.1:8091/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_cache"))

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]
DEFAULT_TRAIN = ROOT / "outputs/exp290_chessbot_local_mix/train/train.jsonl"
DEFAULT_VAL = ROOT / "outputs/exp290_chessbot_local_mix/train/val.jsonl"
DEFAULT_CACHE = ROOT / "outputs/exp290_chessbot_local_mix/val_cache.pt"
DEFAULT_CKPT = ROOT / "outputs/exp290_chessbot_local_mix/train/latest.pt"
DEFAULT_PORT = 8091
VENDOR_CHART = Path(__file__).resolve().parent / "vendor" / "chart.umd.min.js"
CHART_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"
TOTAL_STEPS = 156_250


def parse_train(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" not in row or "loss" not in row:
            continue
        rows.append(row)
    return rows


def parse_val(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" not in row or "by_unrolls" not in row:
            continue
        rows.append(row)
    return rows


def metrics(train_path: Path, val_path: Path) -> dict:
    train = parse_train(train_path)
    val = parse_val(val_path)
    last = train[-1] if train else None
    last_val = val[-1] if val else None
    eta_s = None
    if last and last.get("pos_per_s"):
        remain = max(TOTAL_STEPS - int(last["step"]), 0)
        eta_s = remain * 32 / float(last["pos_per_s"]) if last["pos_per_s"] else None
    return {
        "train": train,
        "val": val,
        "last": last,
        "last_val": last_val,
        "n_train": len(train),
        "n_val": len(val),
        "total_steps": TOTAL_STEPS,
        "eta_s": eta_s,
        "train_log": str(train_path),
        "val_log": str(val_path),
    }


def ensure_chart() -> Path:
    if VENDOR_CHART.exists() and VENDOR_CHART.stat().st_size > 10_000:
        return VENDOR_CHART
    VENDOR_CHART.parent.mkdir(parents=True, exist_ok=True)
    from urllib.request import urlopen
    with urlopen(CHART_URL, timeout=30) as r:
        VENDOR_CHART.write_bytes(r.read())
    return VENDOR_CHART


def append_val(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({"val": {k: row[k] for k in ("step", "source") if k in row},
                      "by_unrolls": row.get("by_unrolls")}), flush=True)


def val_watch(cache_path: Path, ckpt_path: Path, val_path: Path) -> None:
    import torch
    from chess_chessbot_recurrent import wrap_published
    from experiments.exp290_chessbot_local_mix import ensure_val_cache, load_model, run_val

    device = torch.device("cpu")
    print(f"val watch: building cache {cache_path}", flush=True)
    try:
        cache = ensure_val_cache(cache_path, 48)
    except Exception as exc:
        print(f"val cache failed: {type(exc).__name__}: {exc}", flush=True)
        return
    print(f"val watch: {len(cache['fen'])} rows", flush=True)
    teacher = wrap_published(device, default_unrolls=1)
    if not val_path.exists():
        metrics = run_val(teacher, teacher, cache, device, [1, 2, 3], batch_size=8)
        append_val(val_path, {"step": 0, "source": "published", "examples": 0, **metrics})
    last_mtime = None
    while True:
        if ckpt_path.exists():
            mtime = ckpt_path.stat().st_mtime
            if mtime != last_mtime:
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    step = int(ckpt.get("step") or (ckpt.get("experiment") or {}).get("step") or 0)
                    model = load_model(ckpt_path, device)
                    metrics = run_val(model, teacher, cache, device, [1, 2, 3], batch_size=8)
                    append_val(val_path, {"step": step, "source": "latest", **metrics})
                    last_mtime = mtime
                    del model
                except Exception as exc:
                    print(f"val eval failed: {type(exc).__name__}: {exc}", flush=True)
                    last_mtime = mtime
        time.sleep(15)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>exp290 ChessBot local mix</title>
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
  <h1>exp290 · published 34.7M · even phase mix</h1>
  <div class="sub" id="sub">Auto-refreshes every 5s</div>
</header>
<div class="stats" id="stats"></div>
<div class="grid two">
  <div class="card"><h2>Train loss</h2><canvas id="train_loss"></canvas></div>
  <div class="card"><h2>Val hard CE</h2><canvas id="val_hard"></canvas></div>
</div>
<div class="grid two">
  <div class="card"><h2>Val soft / KL</h2><canvas id="val_soft"></canvas></div>
  <div class="card"><h2>Val top-1 vs teacher</h2><canvas id="val_top1"></canvas></div>
</div>
<div class="grid two">
  <div class="card"><h2>Train policy / KL</h2><canvas id="train_parts"></canvas></div>
  <div class="card"><h2>LR / depth</h2><canvas id="lr"></canvas></div>
</div>
<script>
const charts = {};
function line(label, data, color, extra={}){
  return Object.assign({label, data, borderColor:color, backgroundColor:'transparent',
    borderWidth:1.6, pointRadius:0, tension:0.15}, extra);
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
function pick(val, u, key){
  const b = val.by_unrolls && val.by_unrolls[String(u)];
  return b ? b[key] : null;
}
async function refresh(){
  const d = await (await fetch('/api/metrics')).json();
  const train = d.train || [];
  const val = d.val || [];
  const last = d.last;
  const lv = d.last_val;
  document.getElementById('stats').innerHTML = [
    ['Step', last ? fmt(last.step)+' / '+fmt(d.total_steps) : 'waiting'],
    ['Train loss', last ? Number(last.loss).toFixed(4) : '—'],
    ['Train hard', last && last.hard!=null ? Number(last.hard).toFixed(4) : '—'],
    ['Val hard N=1', lv ? (pick(lv,1,'hard')||0).toFixed(4) : 'building'],
    ['Val hard N=2', lv ? (pick(lv,2,'hard')||0).toFixed(4) : '—'],
    ['Val hard N=3', lv ? (pick(lv,3,'hard')||0).toFixed(4) : '—'],
    ['Val top1 N=2', lv && pick(lv,2,'top1')!=null ? (100*pick(lv,2,'top1')).toFixed(1)+'%' : '—'],
    ['Val KL N=1', lv && pick(lv,1,'kl')!=null ? Number(pick(lv,1,'kl')).toExponential(2) : '—'],
    ['pos/s', last ? Number(last.pos_per_s).toFixed(1) : '—'],
    ['LR', last ? Number(last.lr).toExponential(2) : '—'],
    ['ETA', eta(d.eta_s)],
  ].map(([k,v])=>stat(k,v)).join('');
  document.getElementById('sub').textContent =
    `Auto-refreshes every 5s · train ${train.length} · val ${val.length} · ${d.train_log}`;
  if(train.length){
    const stride = train.length>400 ? Math.ceil(train.length/400) : 1;
    const S = train.filter((_,i)=> i%stride===0 || i===train.length-1);
    const L = S.map(s=>s.step);
    upsert('train_loss', L, [line('loss', S.map(s=>s.loss), '#8ab4f8')], 'loss');
    upsert('train_parts', L, [
      line('policy', S.map(s=>s.policy), '#8ab4f8'),
      line('kl', S.map(s=>s.kl), '#f9ab00'),
    ], 'loss');
    upsert('lr', L, [
      line('lr', S.map(s=>s.lr), '#81c995'),
      line('depth', S.map(s=>s.depth), '#c58af9', {yAxisID:undefined}),
    ], 'lr / depth');
  }
  if(val.length){
    const L = val.map(s=>s.step);
    upsert('val_hard', L, [
      line('N=1', val.map(s=>pick(s,1,'hard')), '#8ab4f8', {pointRadius:3}),
      line('N=2', val.map(s=>pick(s,2,'hard')), '#81c995', {pointRadius:3}),
      line('N=3', val.map(s=>pick(s,3,'hard')), '#c58af9', {pointRadius:3}),
    ], 'hard CE');
    upsert('val_soft', L, [
      line('soft N=1', val.map(s=>pick(s,1,'soft')), '#8ab4f8', {pointRadius:3}),
      line('kl N=1', val.map(s=>pick(s,1,'kl')), '#f9ab00', {pointRadius:3}),
    ], 'soft / KL');
    upsert('val_top1', L, [
      line('N=1', val.map(s=>pick(s,1,'top1')), '#8ab4f8', {pointRadius:3}),
      line('N=2', val.map(s=>pick(s,2,'top1')), '#81c995', {pointRadius:3}),
      line('N=3', val.map(s=>pick(s,3,'top1')), '#c58af9', {pointRadius:3}),
    ], 'top-1');
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    train_path: Path = DEFAULT_TRAIN
    val_path: Path = DEFAULT_VAL

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
            self._send(json.dumps(metrics(self.train_path, self.val_path)).encode(), "application/json")
            return
        self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-log", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val-log", default=str(DEFAULT_VAL))
    ap.add_argument("--val-cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-val-watch", action="store_true")
    args = ap.parse_args()
    Handler.train_path = Path(args.train_log)
    Handler.val_path = Path(args.val_log)
    if not Handler.train_path.is_absolute():
        Handler.train_path = (ROOT / Handler.train_path).resolve()
    if not Handler.val_path.is_absolute():
        Handler.val_path = (ROOT / Handler.val_path).resolve()
    try:
        ensure_chart()
    except Exception as e:
        print(f"warn: chart.js download failed ({e})", flush=True)
    if not args.no_val_watch:
        threading.Thread(
            target=val_watch,
            args=(Path(args.val_cache), Path(args.ckpt), Handler.val_path),
            daemon=True,
        ).start()
    print(f"exp290 monitor: http://0.0.0.0:{args.port}/  train={Handler.train_path}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
