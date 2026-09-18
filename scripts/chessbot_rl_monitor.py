#!/usr/bin/env python3
"""Live ChessBot RL monitor. Strength verdicts, not PPO loss.

  python3 scripts/chessbot_rl_monitor.py
  # http://0.0.0.0:8093/
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "outputs/chessbot_rl_n2"
DEFAULT_LOG = DEFAULT_OUT / "events.jsonl"
DEFAULT_CFG = ROOT / "configs/chessbot_rl_n2.json"
DEFAULT_PORT = 8093
VENDOR_CHART = Path(__file__).resolve().parent / "vendor" / "chart.umd.min.js"
CHART_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"


def _load_eval_files(out: Path) -> list[dict]:
    rows = []
    if not out.exists():
        return rows
    try:
        from rl_selfplay.chessbot_eval import paired_eval
    except Exception:
        paired_eval = None
    for path in sorted(out.glob("eval_*.json")):
        try:
            match = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        name = path.stem.split("_", 1)[1]
        parts = name.split("_", 1)
        tag, label = (parts[0], parts[1] if len(parts) > 1 else "unknown")
        if paired_eval is not None and match.get("games"):
            from rl_selfplay.chessbot_eval import question_for
            ev = paired_eval(match, question=question_for(label), opponent=label)
            ev["tag"] = tag
            ev["source"] = str(path.name)
            rows.append(ev)
        else:
            rows.append(dict(tag=tag, opponent=label, source=str(path.name),
                             verdict="not_evaluated", wins=match.get("wins"),
                             draws=match.get("draws"), losses=match.get("losses")))
    return rows


def parse_events(path: Path, cfg: dict, out: Path | None = None) -> dict:
    loaded = identity = identity_depth = published = supervised = baseline = None
    collects, updates, iters, evals, guards, recurrence = [], [], [], [], [], []
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
            elif stage == "identity_n1":
                identity = row
            elif stage == "identity_train_depth":
                identity_depth = row
            elif stage == "published_parity":
                published = row
            elif stage == "supervised_parity":
                supervised = row
            elif stage == "baseline":
                baseline = row
            elif stage == "collect":
                collects.append({
                    "t": float(row.get("time") or 0),
                    "decisions": int(row.get("decisions") or 0),
                    "games": int(row.get("finished_games") or 0),
                    "live": int(row.get("live") or 0),
                    "elapsed_s": float(row.get("elapsed_s") or 0),
                })
            elif stage == "update":
                updates.append({
                    "n": int(row.get("update") or len(updates) + 1),
                    "epoch": int(row.get("epoch") or 0),
                    "policy": float(row.get("policy") or 0),
                    "value": float(row.get("value") or 0),
                    "reference_kl": float(row.get("reference_kl") or 0),
                    "behavior_kl": float(row.get("behavior_kl") or 0),
                    "entropy": float(row.get("entropy") or 0),
                    "clip_fraction": float(row.get("clip_fraction") or 0),
                    "grad_norm": float(row.get("grad_norm") or 0),
                    "gate": row.get("gate"),
                    "gate_grad": row.get("gate_grad"),
                })
            elif stage == "iteration_complete":
                diag = row.get("diagnostics") or {}
                iters.append({
                    "iteration": int(row.get("iteration") or 0),
                    "wins": int(row.get("wins") or 0),
                    "draws": int(row.get("draws") or 0),
                    "losses": int(row.get("losses") or 0),
                    "truncated": int(row.get("truncated") or 0),
                    "games": int(row.get("games") or 0),
                    "decisions": int(row.get("decisions") or 0),
                    "elapsed_s": float(row.get("elapsed_s") or 0),
                    "reference_kl": float(diag.get("reference_kl") or 0),
                    "legal_agreement": float(diag.get("legal_agreement") or 0),
                    "updates": int(row.get("updates") or 0),
                    "kl_stopped": bool(row.get("kl_stopped")),
                    "gate": row.get("gate"),
                })
            elif stage in ("development_evaluation", "confirmation_evaluation"):
                evals.append({
                    "tag": row.get("tag"),
                    "opponent": row.get("opponent"),
                    "question": row.get("question"),
                    "kind": row.get("kind") or ("confirmation" if stage.startswith("confirm") else "development"),
                    "verdict": row.get("verdict") or "not_evaluated",
                    "score": row.get("score"),
                    "paired_ci_95": row.get("paired_ci_95"),
                    "score_bounds": row.get("score_bounds"),
                    "wins": row.get("wins"),
                    "draws": row.get("draws"),
                    "losses": row.get("losses"),
                    "pair_score": row.get("pair_score"),
                    "plus_pairs": row.get("plus_pairs"),
                    "minus_pairs": row.get("minus_pairs"),
                    "tied_pairs": row.get("tied_pairs"),
                })
            elif stage == "recurrence":
                recurrence.append(row)
            elif stage in ("guard_stop", "stopped", "complete", "promoted", "promotion_rejected"):
                guards.append(row)
    disk = _load_eval_files(out or path.parent)
    if disk:
        evals = disk
    last_c = collects[-1] if collects else None
    target_dec = int(cfg.get("decisions") or 32768)
    target_iter = int(cfg.get("iterations") or 20)
    rate = None
    if last_c and last_c["elapsed_s"] > 0 and last_c["decisions"] > 0:
        rate = last_c["decisions"] / last_c["elapsed_s"]
    inspect = None
    inspect_path = (out or path.parent) / "inspect_latest.json"
    if inspect_path.exists():
        try:
            inspect = json.loads(inspect_path.read_text())
        except json.JSONDecodeError:
            inspect = None
    return {
        "loaded": loaded,
        "identity": identity,
        "identity_depth": identity_depth,
        "published_parity": published,
        "supervised_parity": supervised,
        "baseline": baseline,
        "collect": collects[-400:],
        "last_collect": last_c,
        "updates": updates[-400:],
        "last_update": updates[-1] if updates else None,
        "iters": iters,
        "last_iter": iters[-1] if iters else None,
        "evals": evals,
        "recurrence": recurrence[-20:],
        "last_recurrence": recurrence[-1] if recurrence else None,
        "guards": guards,
        "inspect": inspect,
        "target_decisions": target_dec,
        "target_iters": target_iter,
        "decisions_per_s": rate,
        "log": str(path),
        "alive": path.exists(),
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
<title>ChessBot RL eval</title>
<script src="vendor/chart.umd.min.js"></script>
<style>
  :root { color-scheme: dark; --bg:#0f1115; --panel:#171a21; --text:#e8eaed; --muted:#9aa0a6; --line:#2a2f3a; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 ui-sans-serif, system-ui, sans-serif; background:var(--bg); color:var(--text); }
  header { padding:20px 24px 8px; }
  h1 { margin:0 0 6px; font-size:20px; font-weight:600; }
  .sub { color:var(--muted); font-size:12px; }
  .bar { margin:8px 24px 0; height:8px; background:var(--line); border-radius:99px; overflow:hidden; }
  .bar > i { display:block; height:100%; background:#8ab4f8; width:0; }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; padding:16px 24px; }
  .stat { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
  .stat b { display:block; font-size:20px; font-variant-numeric:tabular-nums; }
  .stat span { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  .grid { display:grid; grid-template-columns:1fr; gap:16px; padding:0 24px 24px; }
  @media (min-width:960px){ .grid.two { grid-template-columns:1fr 1fr; } .grid.three { grid-template-columns:1fr 1fr 1fr; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px 10px; }
  .card h2 { margin:0 0 8px; font-size:13px; font-weight:600; color:var(--muted); }
  .verdict { font-size:22px; font-weight:650; text-transform:lowercase; }
  .v-stronger { color:#81c995; } .v-weaker { color:#f28b82; }
  .v-inconclusive { color:#fdd663; } .v-not_evaluated { color:#9aa0a6; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:12px; }
  th,td { text-align:left; padding:4px 6px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  canvas { width:100% !important; max-height:280px; }
  .warn { color:#fdd663; font-size:12px; }
  .fen { font-family:ui-monospace,monospace; font-size:11px; word-break:break-all; color:var(--muted); }
  .chg { color:#fdd663; }
</style>
</head>
<body>
<header>
  <h1>ChessBot RL · evaluation</h1>
  <div class="sub" id="sub">Auto-refreshes every 3s · PPO loss is not a strength signal</div>
</header>
<div class="bar"><i id="fill"></i></div>
<div class="stats" id="stats"></div>
<div class="grid three" id="questions"></div>
<div class="grid two">
  <div class="card"><h2>Collect · decisions</h2><canvas id="collect"></canvas></div>
  <div class="card"><h2>PPO (diagnostics only)</h2><canvas id="ppo"></canvas></div>
</div>
<div class="grid two">
  <div class="card"><h2>KL / gate</h2><canvas id="kl"></canvas></div>
  <div class="card"><h2>Matches</h2><div id="matches"></div></div>
</div>
<div class="grid">
  <div class="card"><h2>Board inspector · published / incumbent / N=1 / N=2</h2><div id="inspect" class="warn">not evaluated</div></div>
</div>
<script>
const charts = {};
function line(label, data, color){
  return {label, data, borderColor:color, backgroundColor:'transparent',
    borderWidth:1.6, pointRadius:0, tension:0.15};
}
function upsert(id, labels, datasets, yTitle, xTitle){
  const el = document.getElementById(id);
  if(charts[id]){ charts[id].data.labels=labels; charts[id].data.datasets=datasets; charts[id].update('none'); return; }
  charts[id] = new Chart(el, { type:'line', data:{labels, datasets}, options:{
    responsive:true, animation:false, interaction:{mode:'index', intersect:false},
    plugins:{legend:{labels:{color:'#c4c7ce', boxWidth:12}}},
    scales:{
      x:{title:{display:true,text:xTitle,color:'#9aa0a6'}, ticks:{color:'#9aa0a6',maxTicksLimit:12}, grid:{color:'#2a2f3a'}},
      y:{title:{display:true,text:yTitle,color:'#9aa0a6'}, ticks:{color:'#9aa0a6'}, grid:{color:'#2a2f3a'}},
    },
  }});
}
function stat(k,v){ return `<div class="stat"><b>${v}</b><span>${k}</span></div>`; }
function fmt(n){ return n==null ? '—' : Number(n).toLocaleString(); }
function latest(evals, pred){
  for(let i=evals.length-1;i>=0;i--){ if(pred(evals[i])) return evals[i]; }
  return null;
}
function vhtml(e){
  const v = (e && e.verdict) || 'not_evaluated';
  const ci = e && e.paired_ci_95;
  const score = e && e.score!=null ? Number(e.score).toFixed(3) : '—';
  const wdl = e && e.wins!=null ? `${e.wins}–${e.draws}–${e.losses}` : '—';
  const band = ci && ci[0]!=null ? `[${Number(ci[0]).toFixed(3)}, ${Number(ci[1]).toFixed(3)}]` : 'no paired CI';
  return `<div class="card">
    <h2>${e && e._title || ''}</h2>
    <div class="verdict v-${v}">${v.replace('_',' ')}</div>
    <div>score ${score} · ${wdl}</div>
    <div class="sub">paired 95% ${band}</div>
    <div class="sub">${e && e.opponent ? 'vs '+e.opponent+' · tag '+(e.tag||'') : 'not evaluated'}</div>
  </div>`;
}
function tops(block){
  if(!block || !block.top) return '—';
  return block.top.map(m => `${m.uci} ${(100*m.p).toFixed(1)}%`).join('<br/>');
}
async function refresh(){
  const d = await (await fetch('/api/metrics')).json();
  const L = d.loaded || {};
  const c = d.last_collect;
  const u = d.last_update;
  const it = d.last_iter;
  const r = d.last_recurrence || {};
  const g = (d.guards||[]).slice(-1)[0];
  const pct = c ? Math.min(100, 100*c.decisions/d.target_decisions) : 0;
  document.getElementById('fill').style.width = pct.toFixed(1)+'%';
  document.getElementById('stats').innerHTML = [
    ['Stage', c ? 'collect' : (u ? 'update' : (L.stage||'waiting'))],
    ['Iter', it ? it.iteration : '0'],
    ['Gate', (u && u.gate!=null) ? Number(u.gate).toExponential(2) : (r.gate!=null ? Number(r.gate).toExponential(2) : '—')],
    ['N2 Δmoves', r.move_changes!=null ? r.move_changes+' / '+(r.n||0) : '—'],
    ['Hidden RMS', r.extra_hidden_rms!=null ? Number(r.extra_hidden_rms).toExponential(2) : '—'],
    ['+lat ms', r.extra_latency_ms!=null ? Number(r.extra_latency_ms).toFixed(2) : '—'],
    ['Behavior KL', u ? u.behavior_kl.toExponential(2) : '—'],
    ['Pub agree', (d.published_parity&&d.published_parity.legal_agreement!=null) ? (100*d.published_parity.legal_agreement).toFixed(1)+'%' : '—'],
    ['Guard', g ? (g.stage+' '+(g.reason||'')) : 'ok'],
  ].map(([k,v])=>stat(k,v)).join('');
  const E = d.evals||[];
  const q = [
    Object.assign({_title:'Is it stronger? vs published / incumbent'}, latest(E, e => e.opponent==='incumbent') || latest(E, e => e.opponent==='original') || {}),
    Object.assign({_title:'Is RL responsible? vs control'}, latest(E, e => e.opponent==='control') || {}),
    Object.assign({_title:'Does recurrence help? N=2 vs N=1'}, latest(E, e => e.opponent==='self_n1') || {}),
  ];
  document.getElementById('questions').innerHTML = q.map(vhtml).join('');
  document.getElementById('matches').innerHTML = E.length ? `<table><tr><th>tag</th><th>vs</th><th>verdict</th><th>WDL</th><th>score</th><th>paired CI</th></tr>${
    E.slice(-12).reverse().map(e => `<tr><td>${e.tag||''}</td><td>${e.opponent||''}</td><td class="v-${e.verdict||'not_evaluated'}">${e.verdict||'not_evaluated'}</td><td>${e.wins}–${e.draws}–${e.losses}</td><td>${e.score==null?'—':Number(e.score).toFixed(3)}</td><td>${e.paired_ci_95&&e.paired_ci_95[0]!=null ? Number(e.paired_ci_95[0]).toFixed(3)+'–'+Number(e.paired_ci_95[1]).toFixed(3) : '—'}</td></tr>`).join('')
  }</table>` : '<div class="warn">No completed matches yet. PPO loss does not mean stronger.</div>';
  const I = d.inspect && d.inspect.boards;
  if(I && I.length){
    document.getElementById('inspect').innerHTML = I.map(b => {
      const chg = b.n2_changed_from_published ? 'chg' : '';
      return `<div style="margin-bottom:12px"><div class="fen ${chg}">${b.fen}${b.n2_changed_from_published?' · N=2 ≠ published':''}</div>
        <table><tr><th>published</th><th>incumbent</th><th>N=1</th><th>N=2</th></tr>
        <tr><td>${tops(b.published)}</td><td>${tops(b.incumbent)}</td><td>${tops(b.n1)}</td><td>${tops(b.n2)}</td></tr></table></div>`;
    }).join('');
  }
  document.getElementById('sub').textContent =
    `Auto-refreshes every 3s · PPO loss is not a strength signal · ${d.log}`;
  const C = d.collect||[];
  if(C.length){
    upsert('collect', C.map((_,i)=>i), [
      line('decisions', C.map(x=>x.decisions), '#8ab4f8'),
      line('games', C.map(x=>x.games), '#81c995'),
    ], 'count', 'collect tick');
  }
  const U = d.updates||[];
  if(U.length){
    const xs = U.map(x=>x.n);
    upsert('ppo', xs, [
      line('policy', U.map(x=>x.policy), '#8ab4f8'),
      line('value', U.map(x=>x.value), '#f28b82'),
    ], 'loss', 'update');
    const klsets = [
      line('reference', U.map(x=>x.reference_kl), '#fdd663'),
      line('behavior', U.map(x=>x.behavior_kl), '#c58af9'),
    ];
    if(U.some(x=>x.gate!=null)) klsets.push(line('gate', U.map(x=>x.gate||0), '#81c995'));
    upsert('kl', xs, klsets, 'kl / gate', 'update');
  }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    log_path: Path = DEFAULT_LOG
    out_dir: Path = DEFAULT_OUT
    cfg: dict = {}

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
            self._send(json.dumps(parse_events(self.log_path, self.cfg, self.out_dir)).encode(),
                       "application/json")
            return
        self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("-p", "--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = (ROOT / log_path).resolve()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    Handler.log_path = log_path
    Handler.out_dir = out_dir
    Handler.cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    try:
        ensure_chart()
    except Exception as e:
        print(f"warn: chart.js download failed ({e})", flush=True)
    print(f"chessbot rl monitor: http://0.0.0.0:{args.port}/  (out={Handler.out_dir})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
