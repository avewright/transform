#!/usr/bin/env python3
"""Live training-loss dashboard for exp191 (parses training.log).

  python scripts/train_dashboard.py
  # open http://<pod-ip>:7860/
"""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "outputs/exp191_400m_meta_attention/training.log"
PORT = 7860

STEP_RE = re.compile(
    r"step ([\d,]+)/([\d,]+) \| p=([\d.]+) v=([\d.]+) soft=([\d.]+) hardCE=([\d.]+)"
    r".*? \| ([\d.]+) pos/s"
)
SHALLOW_RE = re.compile(r"eval shallow top1=([\d.]+)% soft_loss=([\d.]+)")
DEEP_RE = re.compile(r"eval deep top1=([\d.]+)%")
BEST_RE = re.compile(r"new best track=([\d.]+)%")


def parse_log(path: Path) -> dict:
    if not path.exists():
        return {"steps": [], "evals": [], "total": 16000}
    text = path.read_text(errors="replace")
    # Prefer the 16k schedule run (ignore any earlier aborted schedules)
    steps = []
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        total = int(m.group(2).replace(",", ""))
        if total != 16000:
            continue
        steps.append(
            {
                "step": int(m.group(1).replace(",", "")),
                "p": float(m.group(3)),
                "v": float(m.group(4)),
                "soft": float(m.group(5)),
                "hard": float(m.group(6)),
                "pos_s": float(m.group(7)),
            }
        )
    # Drop aborted restart prefix if step counter resets
    cut = 0
    for i in range(1, len(steps)):
        if steps[i]["step"] < steps[i - 1]["step"]:
            cut = i
    steps = steps[cut:]

    evals = []
    cur = None
    idx = text.rfind("steps=16,000")
    chunk = text[idx:] if idx >= 0 else text
    for line in chunk.splitlines():
        m = re.search(r"step ([\d,]+)/16,000", line)
        if m:
            cur = int(m.group(1).replace(",", ""))
        m = SHALLOW_RE.search(line)
        if m and cur is not None:
            evals.append(
                {
                    "kind": "shallow",
                    "top1": float(m.group(1)),
                    "soft_loss": float(m.group(2)),
                    "step": cur,
                }
            )
        m = DEEP_RE.search(line)
        if m and cur is not None:
            evals.append({"kind": "deep", "top1": float(m.group(1)), "step": cur})
        m = BEST_RE.search(line)
        if m and cur is not None:
            evals.append({"kind": "best", "top1": float(m.group(1)), "step": cur})

    return {"steps": steps, "evals": evals, "total": 16000}


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>exp191 training</title>
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
  @media (min-width:960px){ .grid.two { grid-template-columns:1fr 1fr; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px 8px; }
  .card h2 { margin:0 0 8px; font-size:13px; font-weight:600; color:var(--muted); }
  canvas { width:100% !important; max-height:340px; }
</style>
</head>
<body>
<header>
  <h1>exp191 · 437M loss dashboard</h1>
  <div class="sub">Auto-refreshes every 15s from training.log · soft MultiPV + NorMuon</div>
</header>
<div class="stats" id="stats"></div>
<div class="grid">
  <div class="card"><h2>Train losses vs step</h2><canvas id="loss"></canvas></div>
</div>
<div class="grid two">
  <div class="card"><h2>Holdout top-1 (%)</h2><canvas id="eval"></canvas></div>
  <div class="card"><h2>Throughput (pos/s)</h2><canvas id="speed"></canvas></div>
</div>
<script>
const charts = {};
function stat(html){ return html; }
async function refresh(){
  const r = await fetch('/api/metrics');
  const d = await r.json();
  const steps = d.steps || [];
  const evals = d.evals || [];
  const last = steps[steps.length-1];
  const best = evals.filter(e=>e.kind==='best').at(-1);
  const el = document.getElementById('stats');
  if(last){
    const pct = (100*last.step/d.total).toFixed(1);
    el.innerHTML = [
      ['Step', last.step.toLocaleString()+' / '+d.total.toLocaleString()],
      ['Progress', pct+'%'],
      ['Policy p', last.p.toFixed(3)],
      ['Soft', last.soft.toFixed(3)],
      ['Hard CE', last.hard.toFixed(3)],
      ['Value v', last.v.toFixed(3)],
      ['pos/s', last.pos_s.toFixed(0)],
      ['Best track', best ? best.top1.toFixed(2)+'%' : '—'],
    ].map(([k,v])=>`<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');
  }
  const labels = steps.map(s=>s.step);
  const mk = (id, datasets, opts={}) => {
    const ctx = document.getElementById(id);
    if(charts[id]){ charts[id].data.labels = labels; charts[id].data.datasets = datasets; charts[id].update('none'); return; }
    charts[id] = new Chart(ctx, {
      type:'line',
      data:{ labels, datasets },
      options:{
        responsive:true, animation:false,
        interaction:{ mode:'index', intersect:false },
        plugins:{ legend:{ labels:{ color:'#c4c7ce', boxWidth:12 } } },
        scales:{
          x:{ title:{ display:true, text:'step', color:'#9aa0a6' }, ticks:{ color:'#9aa0a6', maxTicksLimit:10 }, grid:{ color:'#2a2f3a' } },
          y:{ title:{ display:true, text: opts.yTitle||'', color:'#9aa0a6' }, ticks:{ color:'#9aa0a6' }, grid:{ color:'#2a2f3a' }, beginAtZero:!!opts.zero },
        },
      },
    });
  };
  // downsample for chart if huge
  const stride = steps.length > 120 ? Math.ceil(steps.length/120) : 1;
  const S = steps.filter((_,i)=> i%stride===0 || i===steps.length-1);
  const L = S.map(s=>s.step);
  const line = (label, key, color) => ({
    label, data: S.map(s=>s[key]), borderColor: color, backgroundColor: color,
    borderWidth: 1.5, pointRadius: 0, tension: 0.15,
  });
  if(charts.loss){
    charts.loss.data.labels = L;
    charts.loss.data.datasets = [
      line('policy (p)','p','#8ab4f8'),
      line('soft','soft','#81c995'),
      line('hard CE','hard','#fdd663'),
      line('value (v)','v','#9aa0a6'),
    ];
    charts.loss.update('none');
  } else {
    charts.loss = new Chart(document.getElementById('loss'), {
      type:'line',
      data:{ labels:L, datasets:[
        line('policy (p)','p','#8ab4f8'),
        line('soft','soft','#81c995'),
        line('hard CE','hard','#fdd663'),
        line('value (v)','v','#9aa0a6'),
      ]},
      options:{
        responsive:true, animation:false,
        interaction:{ mode:'index', intersect:false },
        plugins:{ legend:{ labels:{ color:'#c4c7ce', boxWidth:12 } } },
        scales:{
          x:{ title:{display:true,text:'step',color:'#9aa0a6'}, ticks:{color:'#9aa0a6',maxTicksLimit:12}, grid:{color:'#2a2f3a'} },
          y:{ title:{display:true,text:'loss',color:'#9aa0a6'}, ticks:{color:'#9aa0a6'}, grid:{color:'#2a2f3a'} },
        },
      },
    });
  }
  const eSteps = [...new Set(evals.map(e=>e.step))].sort((a,b)=>a-b);
  const series = (kind, color) => ({
    label: kind, borderColor: color, backgroundColor: color, borderWidth: 1.5,
    pointRadius: 3, tension: 0.15,
    data: eSteps.map(s => (evals.find(e=>e.kind===kind && e.step===s)||{}).top1 ?? null),
  });
  if(charts.eval){
    charts.eval.data.labels = eSteps;
    charts.eval.data.datasets = [series('shallow','#8ab4f8'), series('deep','#81c995'), series('best','#fdd663')];
    charts.eval.update('none');
  } else {
    charts.eval = new Chart(document.getElementById('eval'), {
      type:'line',
      data:{ labels:eSteps, datasets:[series('shallow','#8ab4f8'), series('deep','#81c995'), series('best','#fdd663')] },
      options:{
        responsive:true, animation:false,
        plugins:{ legend:{ labels:{ color:'#c4c7ce', boxWidth:12 } } },
        scales:{
          x:{ title:{display:true,text:'step',color:'#9aa0a6'}, ticks:{color:'#9aa0a6'}, grid:{color:'#2a2f3a'} },
          y:{ title:{display:true,text:'top-1 %',color:'#9aa0a6'}, ticks:{color:'#9aa0a6'}, grid:{color:'#2a2f3a'} },
        },
      },
    });
  }
  if(charts.speed){
    charts.speed.data.labels = L;
    charts.speed.data.datasets = [line('pos/s','pos_s','#c58af9')];
    charts.speed.update('none');
  } else {
    charts.speed = new Chart(document.getElementById('speed'), {
      type:'line',
      data:{ labels:L, datasets:[line('pos/s','pos_s','#c58af9')] },
      options:{
        responsive:true, animation:false,
        plugins:{ legend:{ labels:{ color:'#c4c7ce', boxWidth:12 } } },
        scales:{
          x:{ title:{display:true,text:'step',color:'#9aa0a6'}, ticks:{color:'#9aa0a6',maxTicksLimit:12}, grid:{color:'#2a2f3a'} },
          y:{ title:{display:true,text:'pos/s',color:'#9aa0a6'}, ticks:{color:'#9aa0a6'}, grid:{color:'#2a2f3a'} },
        },
      },
    });
  }
}
refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
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
            payload = json.dumps(parse_log(LOG)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)


def main():
    print(f"exp191 dashboard: http://0.0.0.0:{PORT}/  (log={LOG})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
