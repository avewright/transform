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
    r"\[(\d{2}:\d{2}:\d{2})\] step (\d+)/(\d+) \| loss=([-\d.]+|nan|inf)"
    r".*? ([\d.]+) pos/s(?: \| vram=([\d.]+)GB)?"
)
MIX_RE = re.compile(r"mix s/d(?:/b)?=(\d+)/(\d+)(?:/(\d+))?")
VAL_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\] val/(\S+) hard_ce=([-\d.]+) "
    r"soft_ce=([-\d.]+) soft_temp_ce=([-\d.]+)"
    r"(?:.*? teacher_entropy=([-\d.]+) teacher_kl=([-\d.]+))?"
    r"(?:.*? wdl_ce=([-\d.]+))?"
)
ELO_RE = re.compile(r"elo@(\d+) estimate=([-\d.]+|None) rc=(\d+)")
WARM_RE = re.compile(r"WEIGHTS-ONLY WARM START \S+ steps=(\d+)")
DISJOINT_RE = re.compile(
    r"disjoint (shard_\d+): in=([\d,]+) out=([\d,]+) "
    r"internal_dups=(\d+) vs_prior=(\d+)"
)
ATTACH_RE = re.compile(
    r"attached (\d+) disjoint SF shards → n=([\d,]+) unique_hashes=([\d,]+)"
)
LIVE_DUP_RE = re.compile(r"live soft dropped ([\d,]+) internal hash dups")
EXCLUDE_RE = re.compile(
    r"exclude (\d+) prior ATTACHED shards \(([\d,]+) hashes\) seen=([\d,]+)"
)
SOFT_RE = re.compile(r"soft train=([\d,]+) val=([\d,]+) blocked=(\d+)")
DEEP_RE = re.compile(r"deep train=([\d,]+) val=([\d,]+)")
RESUME_RE = re.compile(r"FULL RESUME \S+ steps=(\d+)")
EXPOSURE_RE = re.compile(
    r"exposure deep/shallow odds=([\d.]+)x \(mix=([\d.]+) deep_n=(\d+) shallow_n=(\d+)\)"
)
DONE_RE = re.compile(r"done steps=(\d+) .* status=(\w+)")
SWA_RE = re.compile(r"(?:restored SWA eval weights n=|eval_swa\.pt n=)(\d+)")
SWA_FROM_RE = re.compile(r"SWA from step (\d+)/(\d+)")
PROBE_RE = re.compile(r"batch probe (?:OOM at bs=(\d+); retry bs=(\d+)|ok bs=(\d+))")
ROUTER_VAL_RE = re.compile(
    r"val_exp_ce=([-\d.]+)\s+val_route_ce=([-\d.]+)\s+"
    r"prior_ce=([-\d.]+)\s+oracle_ce=([-\d.]+)"
)
CLS_RE = re.compile(r"trn_cls=([-\d.]+)\s+val_cls_ce=([-\d.]+)")
ACC_RE = re.compile(r"trn_acc=([-\d.]+).*?best_acc=([-\d.]+)")
SACC_RE = re.compile(
    r"sacc ope=([-\d.]+)\s+mid=([-\d.]+)\s+end=([-\d.]+)\s+gen=([-\d.]+)\s+puz=([-\d.]+)"
)


def _i(s: str) -> int:
    return int(str(s).replace(",", ""))


def parse_log(path: Path) -> dict:
    if not path.exists():
        return {"steps": [], "vals": [], "data": {}, "log": str(path)}
    steps: list[dict] = []
    vals: list[dict] = []
    elos: list[dict] = []
    shards: list[dict] = []
    data: dict = {"phase": "idle"}
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            loss_s = m.group(4)
            try:
                loss = float(loss_s)
            except ValueError:
                continue
            if loss != loss:
                continue
            rec = {
                "t": m.group(1),
                "step": int(m.group(2)),
                "total": int(m.group(3)),
                "loss": loss,
                "pos_s": float(m.group(5)),
                "vram": float(m.group(6)) if m.group(6) else None,
            }
            mx = MIX_RE.search(line)
            if mx:
                rec["mix_s"] = int(mx.group(1))
                rec["mix_d"] = int(mx.group(2))
                rec["mix_b"] = int(mx.group(3)) if mx.group(3) else 0
            rv = ROUTER_VAL_RE.search(line)
            if rv:
                rec["val_exp_ce"] = float(rv.group(1))
                rec["val_route_ce"] = float(rv.group(2))
                rec["prior_ce"] = float(rv.group(3))
                rec["oracle_ce"] = float(rv.group(4))
            cr = CLS_RE.search(line)
            if cr:
                rec["trn_cls"] = float(cr.group(1))
                rec["val_cls_ce"] = float(cr.group(2))
            ar = ACC_RE.search(line)
            if ar:
                rec["trn_acc"] = float(ar.group(1))
                rec["val_acc"] = float(ar.group(2))
            sr = SACC_RE.search(line)
            if sr:
                rec["sacc"] = {
                    "opening": float(sr.group(1)),
                    "middlegame": float(sr.group(2)),
                    "endgame": float(sr.group(3)),
                    "puzzle": float(sr.group(5)),
                }
            if rv:
                cls_only = rec.get("val_route_ce", 1) == 0 and rec.get("prior_ce", 1) == 0
                series = (
                    (("val_cls", "val_cls_ce" if "val_cls_ce" in rec else "val_exp_ce"),)
                    if cls_only
                    else (
                        ("expected", "val_exp_ce"),
                        ("route", "val_route_ce"),
                        ("prior", "prior_ce"),
                        ("oracle", "oracle_ce"),
                    )
                )
                for split, key in series:
                    if rec.get(key) is None:
                        continue
                    vals.append(
                        {
                            "t": rec["t"],
                            "split": split,
                            "hard_ce": rec[key],
                            "soft_ce": rec[key],
                            "soft_temp_ce": rec[key],
                            "teacher_entropy": None,
                            "teacher_kl": None,
                            "wdl_ce": None,
                            "step": rec["step"],
                        }
                    )
            if steps and steps[-1]["step"] == rec["step"]:
                steps[-1] = rec
            else:
                steps.append(rec)
            data["phase"] = "training"
            data["total"] = rec["total"]
            continue
        vm = VAL_RE.search(line)
        if vm:
            vals.append(
                {
                    "t": vm.group(1),
                    "split": vm.group(2),
                    "hard_ce": float(vm.group(3)),
                    "soft_ce": float(vm.group(4)),
                    "soft_temp_ce": float(vm.group(5)),
                    "teacher_entropy": float(vm.group(6)) if vm.group(6) else None,
                    "teacher_kl": float(vm.group(7)) if vm.group(7) else None,
                    "wdl_ce": float(vm.group(8)) if vm.group(8) else None,
                    "step": steps[-1]["step"] if steps else None,
                }
            )
            continue
        em = ELO_RE.search(line)
        if em:
            est = None if em.group(2) == "None" else float(em.group(2))
            elos.append({"step": int(em.group(1)), "elo": est, "rc": int(em.group(3))})
            continue
        if "FULL RESUME" in line or "WEIGHTS-ONLY WARM START" in line:
            shards = []
            rm = RESUME_RE.search(line) or WARM_RE.search(line)
            data["resume_step"] = int(rm.group(1)) if rm else data.get("resume_step")
            data["phase"] = "starting"
            continue
        lm = LIVE_DUP_RE.search(line)
        if lm:
            data["live_internal_dups"] = _i(lm.group(1))
            continue
        em = EXCLUDE_RE.search(line)
        if em:
            data["prior_attached"] = int(em.group(1))
            data["prior_hashes"] = int(em.group(2).replace(",", ""))
            data["seen_before_attach"] = int(em.group(3).replace(",", ""))
            continue
        dm = DISJOINT_RE.search(line)
        if dm:
            shards.append(
                {
                    "name": dm.group(1),
                    "n_in": int(dm.group(2).replace(",", "")),
                    "n_out": int(dm.group(3).replace(",", "")),
                    "internal_dups": int(dm.group(4)),
                    "vs_prior": int(dm.group(5)),
                }
            )
            continue
        am = ATTACH_RE.search(line)
        if am:
            data["n_shards"] = int(am.group(1))
            data["n_rows"] = int(am.group(2).replace(",", ""))
            data["unique_hashes"] = int(am.group(3).replace(",", ""))
            continue
        sm = SOFT_RE.search(line)
        if sm:
            data["soft_train"] = int(sm.group(1).replace(",", ""))
            data["soft_val"] = int(sm.group(2).replace(",", ""))
            data["soft_blocked"] = int(sm.group(3))
            continue
        dpm = DEEP_RE.search(line)
        if dpm:
            data["deep_train"] = int(dpm.group(1).replace(",", ""))
            data["deep_val"] = int(dpm.group(2).replace(",", ""))
            continue
        xm = EXPOSURE_RE.search(line)
        if xm:
            data["odds"] = float(xm.group(1))
            data["deep_mix"] = float(xm.group(2))
            continue
        sw = SWA_RE.search(line)
        if sw:
            data["swa_n"] = int(sw.group(1))
        sf = SWA_FROM_RE.search(line)
        if sf:
            data["swa_from"] = int(sf.group(1))
            data["total"] = int(sf.group(2))
        pb = PROBE_RE.search(line)
        if pb:
            data["phase"] = "probing"
            data["batch"] = int(pb.group(3) or pb.group(2) or pb.group(1) or 0)
            if pb.group(3):
                data["phase"] = "compiled"
        if "STOP file seen" in line:
            data["phase"] = "stopping"
        dn = DONE_RE.search(line)
        if dn:
            data["phase"] = dn.group(2)
            data["done_step"] = int(dn.group(1))

    by_step = {s["step"]: s for s in steps}
    steps = [by_step[k] for k in sorted(by_step)]
    vs_prior = sum(s["vs_prior"] for s in shards)
    data["shards"] = shards
    data["disjoint"] = bool(shards) and vs_prior == 0
    data["vs_prior_total"] = vs_prior
    last = steps[-1] if steps else None
    hist = path.parent / "elo_gauntlet.jsonl"
    if hist.exists():
        for line in hist.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("step")
            if step is None:
                continue
            if any(e["step"] == step for e in elos):
                continue
            est = row.get("elo")
            if est is None:
                est = (row.get("estimate") or {}).get("estimated_elo")
            elos.append({"step": int(step), "elo": est, "rc": row.get("rc")})
    elos.sort(key=lambda e: e["step"])
    return {
        "steps": steps,
        "vals": vals,
        "elos": elos,
        "data": data,
        "total": (last or {}).get("total") or data.get("total") or 0,
        "log": str(path),
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>train loss</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root { color-scheme: dark; --bg:#0f1115; --panel:#171a21; --text:#e8eaed; --muted:#9aa0a6; --line:#2a2f3a; --ok:#81c995; --bad:#f28b82; }
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
  .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
  .badge.ok { background:#16351f; color:var(--ok); }
  .badge.bad { background:#3a1816; color:var(--bad); }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:12px; }
  th, td { text-align:right; padding:4px 6px; border-top:1px solid var(--line); }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:600; border-top:none; }
</style>
</head>
<body>
<header>
  <h1 id="title">squares64 loss</h1>
  <div class="sub" id="sub">Auto-refreshes every 10s from train.log</div>
</header>
<div class="stats" id="stats"></div>
<div class="grid two">
  <div class="card"><h2>Soft data · this attach</h2><div id="data"></div></div>
  <div class="card"><h2 id="val-title">Val (hard CE)</h2><canvas id="val"></canvas></div>
</div>
<div class="grid two">
  <div class="card"><h2>Loss vs step</h2><canvas id="loss"></canvas></div>
  <div class="card"><h2>Throughput (pos/s)</h2><canvas id="speed"></canvas></div>
</div>
<div class="grid">
  <div class="card"><h2>Elo gauntlet (UCI_Elo)</h2><canvas id="elo"></canvas></div>
</div>
<script>
const charts = {};
const fmt = n => n==null ? '—' : Number(n).toLocaleString();
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
  const el = document.getElementById(id);
  if(!el) return;
  if(charts[id]){
    charts[id].data.labels = labels;
    charts[id].data.datasets = datasets;
    charts[id].update('none');
    return;
  }
  charts[id] = new Chart(el, {
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
function stat(k,v){ return `<div class="stat"><b>${v}</b><span>${k}</span></div>`; }
async function refresh(){
  const r = await fetch('/api/metrics');
  const d = await r.json();
  const steps = d.steps || [];
  const last = steps[steps.length-1];
  const recent = steps.slice(-20);
  const avg = recent.length ? recent.reduce((a,s)=>a+s.loss,0)/recent.length : null;
  const lo = recent.length ? Math.min(...recent.map(s=>s.loss)) : null;
  const info = d.data || {};
  const total = d.total || (last && last.total) || 0;
  const vals = d.vals || [];
  const lastSoft = [...vals].reverse().find(v=>v.split==='sf19' || v.split==='soft');
  const lastDeep = [...vals].reverse().find(v=>v.split==='syzygy' || v.split==='deep');
  const lastPuzzle = [...vals].reverse().find(v=>v.split==='puzzles');
  const mix = last && last.mix_s!=null ? (100*last.mix_d/(last.mix_s+last.mix_d)).toFixed(0)+'% deep' : (info.deep_mix!=null ? (100*info.deep_mix).toFixed(0)+'% deep' : '—');
  const logName = (d.log || '').split('/').slice(-2).join('/');
  const isRouter = logName.includes('exp279');
  const clsOnly = isRouter && last && !(last.val_route_ce > 0);
  const sacc = last && last.sacc ? last.sacc : {};
  document.getElementById('stats').innerHTML = (clsOnly ? [
    ['Step', last ? last.step.toLocaleString()+' / '+Number(total).toLocaleString() : '— / '+fmt(total)],
    ['Phase', info.phase || '—'],
    ['Train cls', last && last.trn_cls!=null ? last.trn_cls.toFixed(4) : (last ? last.loss.toFixed(4) : '—')],
    ['Val cls CE', last && (last.val_cls_ce??last.val_exp_ce)!=null ? (last.val_cls_ce??last.val_exp_ce).toFixed(4) : '—'],
    ['Val acc', last && last.val_acc!=null ? last.val_acc.toFixed(3) : '—'],
    ['Opening acc', sacc.opening!=null ? sacc.opening.toFixed(2) : '—'],
    ['Middlegame acc', sacc.middlegame!=null ? sacc.middlegame.toFixed(2) : '—'],
    ['Endgame acc', sacc.endgame!=null ? sacc.endgame.toFixed(2) : '—'],
    ['Puzzle acc', sacc.puzzle!=null ? sacc.puzzle.toFixed(2) : '—'],
  ] : [
    ['Step', last ? last.step.toLocaleString()+' / '+Number(total).toLocaleString() : '— / '+fmt(total)],
    ['Phase', info.phase || '—'],
    ['Loss', last ? last.loss.toFixed(4) : '—'],
    ['EMA / last 20', avg!==null ? avg.toFixed(4) : '—'],
    ['Recent min', lo!==null ? lo.toFixed(4) : '—'],
    ['pos/s', last ? last.pos_s.toFixed(0) : '—'],
    ['VRAM', last && last.vram ? last.vram.toFixed(2)+' GB' : '—'],
    ['Val puzzles', lastPuzzle?lastPuzzle.hard_ce.toFixed(3):'—'],
    ['Val sf19 / syzygy', (lastSoft?lastSoft.hard_ce.toFixed(3):'—')+' / '+(lastDeep?lastDeep.hard_ce.toFixed(3):'—')],
    ['Teacher KL', lastSoft && lastSoft.teacher_kl!=null ? lastSoft.teacher_kl.toFixed(3) : '—'],
    ['Elo', (d.elos||[]).at(-1)?.elo!=null ? Number((d.elos||[]).at(-1).elo).toFixed(0) : '—'],
    ['Mix', mix],
    ['SWA n', fmt(info.swa_n)],
  ]).map(([k,v])=>stat(k,v)).join('');
  const shards = info.shards || [];
  const badge = info.disjoint
    ? '<span class="badge ok">disjoint · vs_prior=0</span>'
    : (shards.length ? '<span class="badge bad">overlap vs_prior='+info.vs_prior_total+'</span>'
      : (info.soft_train ? '<span class="badge ok">saved split</span>' : '<span class="badge bad">no attach parsed</span>'));
  const ids = shards.map(s=>s.name.replace('shard_',''));
  const names = ids.length ? (ids[0]+'–'+ids[ids.length-1]) : '';
  const rows = shards.map(s=>`<tr><td>${s.name}</td><td>${fmt(s.n_in)}</td><td>${fmt(s.n_out)}</td><td>${fmt(s.internal_dups)}</td><td>${s.vs_prior}</td></tr>`).join('');
  document.getElementById('data').innerHTML = isRouter
    ? `<div class="stats" style="padding:0">
        ${stat('Train cls', last && last.trn_cls!=null ? last.trn_cls.toFixed(4) : '—')}
        ${stat('Val cls CE', last && (last.val_cls_ce??last.val_exp_ce)!=null ? (last.val_cls_ce??last.val_exp_ce).toFixed(4) : '—')}
        ${stat('Val acc', last && last.val_acc!=null ? last.val_acc.toFixed(3) : '—')}
        ${stat('Train acc', last && last.trn_acc!=null ? last.trn_acc.toFixed(3) : '—')}
      </div>
      <div class="sub" style="margin-top:8px">400k/source source classifier · live 99M encode · no expert CE on this pack</div>`
    : `
    <div style="margin-bottom:8px">${badge}
      <span class="sub"> · ${fmt(info.n_shards)} new shards ${names?('('+names+')'):''}
      · exclude ${fmt(info.prior_attached)} prior ATTACHED (${fmt(info.prior_hashes)} hashes)</span></div>
    <div class="stats" style="padding:0 0 10px">
      ${stat('Soft rows', fmt(info.n_rows))}
      ${stat('Unique hashes', fmt(info.unique_hashes))}
      ${stat('Soft train / val', fmt(info.soft_train)+' / '+fmt(info.soft_val))}
      ${stat('Deep train', fmt(info.deep_train))}
      ${stat('Live dups dropped', fmt(info.live_internal_dups))}
      ${stat('Blocked val flips', fmt(info.soft_blocked))}
    </div>
    <table><thead><tr><th>shard</th><th>in</th><th>kept</th><th>internal dups</th><th>vs prior</th></tr></thead>
    <tbody>${rows || '<tr><td colspan=5>waiting for attach</td></tr>'}</tbody></table>`;
  document.getElementById('title').textContent = logName.includes('exp280')
    ? 'exp280 · full MoE (router + experts) · general Lichess'
    : logName.includes('exp279')
    ? (clsOnly ? 'exp279 · source classifier' : 'exp279 · frozen-expert MoE router')
    : logName.includes('exp278')
    ? 'exp278 · 99M Lichess 16–26 middlegame one-hot'
    : logName.includes('exp277')
    ? 'exp277 · 99M Lichess ≥26 opening one-hot'
    : logName.includes('exp276')
    ? 'exp276 · 99M Lichess <14 one-hot'
    : logName.includes('exp275')
    ? 'exp275 · 99M endgame FT'
    : logName.includes('exp274')
    ? 'exp274 · 99M syzygy FT'
    : logName.includes('exp273')
    ? 'exp273 · 99M puzzle FT'
    : logName.includes('exp271')
    ? 'exp271 · 99M → 270M KD loss' : 'squares64 loss';
  const vt = document.getElementById('val-title');
  if (vt) vt.textContent = clsOnly ? 'Val accuracy by source' : (isRouter ? 'Val CE (route / expected / prior / oracle)' : 'Val (hard CE)');
  document.getElementById('sub').textContent = isRouter && !steps.length
    ? 'Router not training yet. Scoring frozen experts on HF next-move data. Train/val curves appear after --go.'
    : `Auto-refreshes every 10s · ${steps.length} step points · resume ${fmt(info.resume_step)} · log ${d.log}`;
  const stride = steps.length > 400 ? Math.ceil(steps.length/400) : 1;
  const S = steps.filter((_,i)=> i%stride===0 || i===steps.length-1);
  const L = S.map(s=>s.step);
  const raw = S.map(s=>s.loss);
  if (clsOnly) {
    upsert('loss', L, [
      line('train cls', S.map(s=>s.trn_cls ?? s.loss), '#8ab4f8', {borderWidth:1.6, pointRadius:0}),
      line('val cls CE', S.map(s=>s.val_cls_ce ?? s.val_exp_ce), '#f9ab00', {pointRadius:2}),
    ], 'CE');
    upsert('val', L, [
      line('val acc', S.map(s=>s.val_acc), '#e8eaed', {pointRadius:2}),
      line('opening', S.map(s=>s.sacc && s.sacc.opening), '#f9ab00', {pointRadius:2}),
      line('middlegame', S.map(s=>s.sacc && s.sacc.middlegame), '#fdd663', {pointRadius:2}),
      line('endgame', S.map(s=>s.sacc && s.sacc.endgame), '#81c995', {pointRadius:2}),
      line('puzzle', S.map(s=>s.sacc && s.sacc.puzzle), '#c58af9', {pointRadius:2}),
    ], 'acc');
  } else if (isRouter) {
    upsert('loss', L, [
      line('train expected CE', raw, '#8ab4f8', {borderWidth:1.6, pointRadius:0}),
      line('val expected CE', S.map(s=>s.val_exp_ce), '#f9ab00', {pointRadius:2}),
      line('val route CE', S.map(s=>s.val_route_ce), '#81c995', {pointRadius:2}),
    ], 'CE');
  } else {
    upsert('loss', L, [
      line('loss', raw, '#8ab4f8', {borderWidth:1, pointRadius:0}),
      line('EMA', ema(raw, 0.08), '#81c995'),
    ], 'loss');
  }
  upsert('speed', L, [line('pos/s', S.map(s=>s.pos_s), '#c58af9')], 'pos/s');
  const colors = {sf19:'#8ab4f8', lichess:'#f9ab00', lichess_opening:'#f9ab00', lichess_middlegame:'#fdd663', lichess_endgame:'#81c995', puzzles:'#c58af9', syzygy:'#81c995', soft:'#8ab4f8', deep:'#f9ab00', expected:'#f9ab00', route:'#81c995', prior:'#c58af9', oracle:'#e8eaed'};
  const splits = [...new Set(vals.map(v=>v.split))];
  const valSets = splits.filter(s=>vals.some(v=>v.split===s && v.step!=null)).map(s =>
    line(s+' hard CE', vals.filter(v=>v.split===s && v.step!=null).map(v=>v.hard_ce), colors[s]||'#e8eaed', {pointRadius:2})
  );
  const sf19V = vals.filter(v=>v.split==='sf19' && v.step!=null);
  if(sf19V.some(v=>v.teacher_kl!=null)){
    valSets.push(line('sf19 teacher KL', sf19V.map(v=>v.teacher_kl), '#81c995', {pointRadius:2, borderDash:[4,3]}));
  }
  if (!clsOnly) {
    const valLabels = (vals.find(v=>v.step!=null) ? vals.filter(v=>v.split===splits[0] && v.step!=null).map(v=>v.step) : []);
    upsert('val', valLabels, valSets, 'hard CE / KL');
  }
  const elos = d.elos || [];
  if(elos.length){
    upsert('elo', elos.map(e=>e.step), [
      line('UCI_Elo', elos.map(e=>e.elo), '#fdd663', {pointRadius:4}),
    ], 'Elo');
  }
}
refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    log_path: Path = DEFAULT_LOG

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
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
