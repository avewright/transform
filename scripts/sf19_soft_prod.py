"""SF19 production helpers: quality audit, frozen eval, mixed-source generate."""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import chess
import numpy as np
import torch

from scripts.harvest_exp201_lapses import board_array_to_fen
from scripts.sf19_soft_dataset import (
    DECIDED_CP,
    FLAG_BOUNDS,
    FLAG_DECIDED,
    FLAG_SHALLOW,
    OPENINGS,
    ORIGIN_RELABEL,
    ORIGIN_SELFPLAY,
    SHALLOW_DEPTH,
    GenConfig,
    SeenDB,
    analyze_board,
    compact_key_bytes,
    compact_key_from_board,
    encode_board,
    log,
    ref_move_regret,
    row_key,
    stack_rows,
    union_kl_and_coverage,
    write_manifest,
    write_shard,
    _init_worker,
    _play_one_game,
    _W_ENGINE,
    _W_CFG,
)

DEFAULT_SUPPLY = [
    Path("outputs/hf_soft_mix/soft_cache.pt"),
    Path("outputs/hf_elo_mix/soft_cache.pt"),
    Path("outputs/autoresearch_8gb/soft_cache_200k.pt"),
]


def eval_bucket(cp: int, mate: int) -> str:
    if int(mate) != 0:
        return "mate"
    c = int(cp)
    if c <= -150:
        return "losing"
    if c >= 150:
        return "winning"
    return "equal"


def is_tactical(board: chess.Board) -> bool:
    if board.is_check():
        return True
    n_cap = n_chk = 0
    for mv in board.legal_moves:
        if board.is_capture(mv):
            n_cap += 1
        if board.gives_check(mv):
            n_chk += 1
        if n_cap >= 3 or n_chk:
            return True
    return False


def _clear_hash(engine) -> None:
    try:
        engine.configure({"Clear Hash": True})
    except Exception:
        pass


def percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
    }


def sample_existing_specs(
    caches: list[Path],
    n: int,
    rng: random.Random,
    *,
    seen: SeenDB | None = None,
    decided_frac: float = 0.15,
    stride: int = 3,
) -> list[dict]:
    """Diverse existing-board specs. Dedup before return. Skip adjacent cache rows."""
    want_phase = {0: n // 4, 2: n // 4, 1: n - 2 * (n // 4)}
    want_eval = {"losing": n // 3, "winning": n // 3, "equal": n - 2 * (n // 3)}
    decided_cap = max(1, int(n * decided_frac))
    got_phase: Counter[int] = Counter()
    got_eval: Counter[str] = Counter()
    n_decided = 0
    out: list[dict] = []
    used: set[bytes] = set()
    last_idx = -999
    for cache in caches:
        if not cache.exists() or len(out) >= n:
            continue
        data = torch.load(cache, map_location="cpu", weights_only=False)
        m = int(data["move_idx"].shape[0])
        order = list(range(0, m, max(1, stride)))
        rng.shuffle(order)
        ba = data["board_array"]
        turn = data["turn"]
        castle = data["castling"]
        ep = data["ep_square"]
        phase = data["phase"] if "phase" in data else None
        cp = data["cp"] if "cp" in data else None
        mate = data["mate"] if "mate" in data else None
        ply = data["ply"] if "ply" in data else None
        for i in order:
            if len(out) >= n:
                break
            if abs(i - last_idx) < stride:
                continue
            key = compact_key_bytes(ba[i], turn[i], castle[i], ep[i])
            if key in used or (seen is not None and seen.has(key)):
                continue
            ph = int(phase[i]) if phase is not None else 1
            if ph not in (0, 1, 2):
                ph = 1
            ev = eval_bucket(int(cp[i]) if cp is not None else 0, int(mate[i]) if mate is not None else 0)
            decided = ev == "mate" or (cp is not None and abs(int(cp[i])) >= DECIDED_CP)
            if decided and n_decided >= decided_cap and got_phase[ph] >= want_phase[ph] * 0.3:
                continue
            phase_full = got_phase[ph] >= want_phase[ph]
            eval_full = got_eval[ev] >= want_eval.get(ev, n)
            if phase_full and eval_full and len(out) > n // 2:
                continue
            try:
                fen = board_array_to_fen(ba[i].numpy(), int(turn[i]), int(castle[i]), int(ep[i]))
                board = chess.Board(fen)
            except ValueError:
                continue
            if not board.is_valid() or board.is_game_over(claim_draw=True):
                continue
            used.add(key)
            last_idx = i
            got_phase[ph] += 1
            got_eval[ev] += 1
            if decided:
                n_decided += 1
            out.append({
                "fen": fen,
                "origin": ORIGIN_RELABEL,
                "phase": ph,
                "eval_bucket": ev,
                "tactical": is_tactical(board),
                "cache": cache.name,
                "cache_idx": i,
                "ply": int(ply[i]) if ply is not None else -1,
                "key": key,
            })
        del data
    rng.shuffle(out)
    return out[:n]


def iter_existing_specs(caches: list[Path], rng: random.Random, *, seen: SeenDB | None, stride: int = 4, decided_frac: float = 0.15):
    """Load each cache once; yield unique, non-adjacent, quota-capped boards."""
    decided_n = 0
    yielded = 0
    for cache in caches:
        if not cache.exists():
            continue
        data = torch.load(cache, map_location="cpu", weights_only=False)
        m = int(data["move_idx"].shape[0])
        order = list(range(0, m, max(1, stride)))
        rng.shuffle(order)
        ba, turn, castle, ep = data["board_array"], data["turn"], data["castling"], data["ep_square"]
        phase = data["phase"] if "phase" in data else None
        cp = data["cp"] if "cp" in data else None
        mate = data["mate"] if "mate" in data else None
        ply = data["ply"] if "ply" in data else None
        last_i = -10**9
        pending_keys: list[bytes] = []
        for i in order:
            if abs(i - last_i) < stride:
                continue
            key = compact_key_bytes(ba[i], turn[i], castle[i], ep[i])
            pending_keys.append(key)
            last_i = i
        # batch seen lookup
        found = seen.has_many(pending_keys) if seen is not None and pending_keys else set()
        last_i = -10**9
        for i in order:
            if abs(i - last_i) < stride:
                continue
            key = compact_key_bytes(ba[i], turn[i], castle[i], ep[i])
            if key in found:
                continue
            last_i = i
            ev = eval_bucket(int(cp[i]) if cp is not None else 0, int(mate[i]) if mate is not None else 0)
            decided = ev == "mate" or (cp is not None and abs(int(cp[i])) >= DECIDED_CP)
            if decided:
                decided_n += 1
                if yielded and decided_n / max(yielded, 1) > decided_frac:
                    continue
            try:
                fen = board_array_to_fen(ba[i].numpy(), int(turn[i]), int(castle[i]), int(ep[i]))
                board = chess.Board(fen)
            except ValueError:
                continue
            if not board.is_valid() or board.is_game_over(claim_draw=True):
                continue
            yielded += 1
            yield {
                "fen": fen,
                "origin": ORIGIN_RELABEL,
                "phase": int(phase[i]) if phase is not None else 1,
                "eval_bucket": ev,
                "tactical": False,
                "cache": cache.name,
                "cache_idx": i,
                "ply": int(ply[i]) if ply is not None else -1,
                "key": key,
                "split": 0,
                "game_id": int(i),
            }
        del data


def sample_selfplay_starts(n: int, rng: random.Random) -> list[dict]:
    out = []
    for i in range(n):
        board = chess.Board()
        for uci in OPENINGS[i % len(OPENINGS)]:
            mv = chess.Move.from_uci(uci)
            if mv in board.legal_moves:
                board.push(mv)
        noise = rng.randint(2, 10)
        for _ in range(noise):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over(claim_draw=True):
            continue
        out.append({
            "fen": board.fen(),
            "origin": ORIGIN_SELFPLAY,
            "phase": 0 if len(board.piece_map()) >= 26 else (1 if len(board.piece_map()) >= 14 else 2),
            "eval_bucket": "unknown",
            "tactical": is_tactical(board),
            "cache": "selfplay",
            "cache_idx": -1,
            "ply": board.ply(),
            "key": compact_key_from_board(board),
        })
    return out


def _audit_pair(spec: dict) -> dict:
    assert _W_ENGINE is not None and _W_CFG is not None
    board = chess.Board(spec["fen"])
    _clear_hash(_W_ENGINE)
    cand = analyze_board(
        _W_ENGINE, board,
        nodes=_W_CFG.nodes, multipv=_W_CFG.multipv,
        tau=_W_CFG.tau, watchdog_s=_W_CFG.watchdog_s,
    )
    _clear_hash(_W_ENGINE)
    ref = analyze_board(
        _W_ENGINE, board,
        nodes=int(spec["ref_nodes"]), multipv=8,
        tau=_W_CFG.tau, watchdog_s=max(_W_CFG.watchdog_s, 30.0),
    )
    rec = {
        "fen": spec["fen"],
        "origin": spec.get("origin", ORIGIN_RELABEL),
        "phase": spec.get("phase", -1),
        "tactical": bool(spec.get("tactical")),
        "ok": cand is not None and ref is not None and int(ref.get("policy_mask", 0)) == 1,
    }
    if rec["ok"]:
        regret, missing = ref_move_regret(ref, cand)
        kl, ref_in_q, q_in_ref = union_kl_and_coverage(ref, cand)
        rec.update({
            "top1": int(cand["move_idx"]) == int(ref["move_idx"]),
            "regret": None if missing else float(regret),
            "missing": bool(missing),
            "kl": kl,
            "ref_mass_in_cand": ref_in_q,
            "missing_ref_mass": 1.0 - ref_in_q,
            "wdl_l1": float(np.abs(cand["wdl"].astype(np.float64) - ref["wdl"].astype(np.float64)).sum()),
            "cand_depth": int(cand["label_depth"]),
            "ref_depth": int(ref["label_depth"]),
            "flags": int(cand.get("flags", 0)),
            "eval_bucket": eval_bucket(int(ref["cp"]), int(ref["mate"])),
        })
    return rec


def _deep_ref(spec: dict) -> dict:
    assert _W_ENGINE is not None and _W_CFG is not None
    board = chess.Board(spec["fen"])
    _clear_hash(_W_ENGINE)
    row = analyze_board(
        _W_ENGINE, board,
        nodes=int(spec["deep_nodes"]), multipv=8,
        tau=_W_CFG.tau, watchdog_s=max(_W_CFG.watchdog_s, 60.0),
    )
    return {"fen": spec["fen"], "row": row}


def run_quality_audit(args) -> dict:
    from scripts.sf19_soft_dataset import resolve_sf, teacher_fingerprint, require_sf19

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "audit.log"
    sf = resolve_sf()
    fp = teacher_fingerprint(sf)
    require_sf19(fp)
    rng = random.Random(args.seed)
    n = int(args.audit_n)
    n_exist = int(n * 0.7)
    n_sp = n - n_exist
    caches = [Path(p) for p in (args.seed_caches or [])] or [p for p in DEFAULT_SUPPLY if p.exists()]
    exist = sample_existing_specs(caches, n_exist, rng, stride=5)
    sp = sample_selfplay_starts(n_sp + 32, rng)[:n_sp]
    specs = exist + sp
    rng.shuffle(specs)
    for s in specs:
        s["ref_nodes"] = int(args.ref_nodes)
    cfg = GenConfig(nodes=args.nodes, multipv=args.multipv, tau=args.tau, hash_mb=args.hash_mb, watchdog_s=12.0)
    workers = max(1, args.workers)
    log(f"audit n={len(specs)} exist={len(exist)} selfplay={len(sp)} cand={cfg.nodes}/{cfg.multipv} ref={args.ref_nodes}", log_path)
    ctx = get_context("spawn")
    pool = ctx.Pool(workers, initializer=_init_worker, initargs=(sf, cfg, None))
    t0 = time.time()
    recs = []
    try:
        for i, rec in enumerate(pool.imap_unordered(_audit_pair, specs, chunksize=1), 1):
            recs.append(rec)
            if i % 50 == 0:
                log(f"audit {i}/{len(specs)} {(time.time()-t0):.0f}s", log_path)
    finally:
        pool.terminate()
        pool.join()
    ok = [r for r in recs if r.get("ok")]
    flagged = [r for r in ok if int(r.get("flags") or 0) & (FLAG_SHALLOW | FLAG_BOUNDS)]
    unflagged = [r for r in ok if r not in flagged]
    report = {
        "teacher": fp,
        "n_requested": n,
        "n_ok": len(ok),
        "elapsed_s": time.time() - t0,
        "cpu_hours": workers * (time.time() - t0) / 3600.0,
        "cand": {"nodes": cfg.nodes, "multipv": cfg.multipv},
        "ref_nodes": int(args.ref_nodes),
        "top1": float(np.mean([r["top1"] for r in ok])) if ok else 0.0,
        "regret": percentiles([r["regret"] for r in ok if r.get("regret") is not None]),
        "kl_union": percentiles([r["kl"] for r in ok if math.isfinite(r.get("kl", float("nan")))]),
        "missing_ref_mass": percentiles([r["missing_ref_mass"] for r in ok]),
        "wdl_l1": percentiles([r["wdl_l1"] for r in ok]),
        "cand_depth": percentiles([r["cand_depth"] for r in ok]),
        "ref_depth": percentiles([r["ref_depth"] for r in ok]),
        "missing_move_frac": float(np.mean([r["missing"] for r in ok])) if ok else 0.0,
        "by_phase": {},
        "by_origin": {},
        "by_eval": {},
        "flagged_vs_unflagged": {},
        "source_mix": {"existing": len(exist), "selfplay": len(sp)},
    }
    def _slice(name, rows):
        if not rows:
            return {}
        return {
            "n": len(rows),
            "top1": float(np.mean([r["top1"] for r in rows])),
            "regret_p50": float(np.median([r["regret"] for r in rows if r.get("regret") is not None] or [0])),
            "missing_ref_mass_p50": float(np.median([r["missing_ref_mass"] for r in rows])),
            "kl_p50": float(np.median([r["kl"] for r in rows if math.isfinite(r.get("kl", float("nan")))] or [0])),
        }
    for ph in (0, 1, 2):
        report["by_phase"][str(ph)] = _slice(ph, [r for r in ok if r.get("phase") == ph])
    report["by_origin"]["relabel"] = _slice("r", [r for r in ok if r.get("origin") == ORIGIN_RELABEL])
    report["by_origin"]["selfplay"] = _slice("s", [r for r in ok if r.get("origin") == ORIGIN_SELFPLAY])
    for ev in ("losing", "equal", "winning", "mate"):
        report["by_eval"][ev] = _slice(ev, [r for r in ok if r.get("eval_bucket") == ev])
    report["flagged_vs_unflagged"] = {
        "flagged": _slice("f", flagged),
        "unflagged": _slice("u", unflagged),
        "unflagged_audit_n": min(64, len(unflagged)),
    }
    # 5M recheck on a smaller mixed subset
    deep_n = min(int(args.deep_n), len(ok))
    deep_specs = []
    if deep_n:
        pick = rng.sample(ok, deep_n)
        deep_pool = ctx.Pool(workers, initializer=_init_worker, initargs=(sf, cfg, None))
        try:
            jobs = [{"fen": r["fen"], "deep_nodes": int(args.deep_nodes)} for r in pick]
            deep_rows = list(deep_pool.imap_unordered(_deep_ref, jobs, chunksize=1))
        finally:
            deep_pool.terminate()
            deep_pool.join()
        by_fen = {r["fen"]: r for r in pick}
        deep_reg = []
        deep_kl = []
        for d in deep_rows:
            base = by_fen.get(d["fen"])
            row = d.get("row")
            if base is None or row is None:
                continue
            # compare 1M-ref-labeled cand fields vs 5M using stored cand vs new ref
            # we only stored metrics, not full cand row; compare 1M ref_depth vs 5M
            deep_reg.append(int(row["label_depth"]))
        report["deep_recheck"] = {
            "n": deep_n,
            "nodes": int(args.deep_nodes),
            "depth": percentiles(deep_reg),
        }
        # pair 100k cand vs 5M using a second isolated pass on the same FENs
        pair_jobs = [{**{"fen": r["fen"], "origin": r.get("origin", 0), "phase": r.get("phase", -1),
                         "tactical": r.get("tactical", False)}, "ref_nodes": int(args.deep_nodes)} for r in pick]
        pair_pool = ctx.Pool(workers, initializer=_init_worker, initargs=(sf, cfg, None))
        try:
            deep_pair = [r for r in pair_pool.imap_unordered(_audit_pair, pair_jobs, chunksize=1) if r.get("ok")]
        finally:
            pair_pool.terminate()
            pair_pool.join()
        report["deep_vs_cand"] = {
            "n": len(deep_pair),
            "top1": float(np.mean([r["top1"] for r in deep_pair])) if deep_pair else 0.0,
            "regret": percentiles([r["regret"] for r in deep_pair if r.get("regret") is not None]),
            "kl_union": percentiles([r["kl"] for r in deep_pair if math.isfinite(r.get("kl", float("nan")))]),
            "missing_ref_mass": percentiles([r["missing_ref_mass"] for r in deep_pair]),
            "wdl_l1": percentiles([r["wdl_l1"] for r in deep_pair]),
        }
        # do flags predict 1M disagreement?
        if flagged and unflagged:
            report["flag_predicts_disagreement"] = {
                "flagged_top1": report["flagged_vs_unflagged"]["flagged"].get("top1"),
                "unflagged_top1": report["flagged_vs_unflagged"]["unflagged"].get("top1"),
                "flagged_regret_p50": report["flagged_vs_unflagged"]["flagged"].get("regret_p50"),
                "unflagged_regret_p50": report["flagged_vs_unflagged"]["unflagged"].get("regret_p50"),
                "adopt_adaptive": (
                    report["flagged_vs_unflagged"]["flagged"].get("top1", 1) + 0.05
                    < report["flagged_vs_unflagged"]["unflagged"].get("top1", 0)
                ),
            }
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(json.dumps({k: report[k] for k in ("n_ok", "top1", "regret", "missing_ref_mass", "flag_predicts_disagreement") if k in report}), log_path)
    return report


def _label_one(spec: dict) -> dict:
    assert _W_ENGINE is not None and _W_CFG is not None
    board = chess.Board(spec["fen"])
    key = compact_key_from_board(board)
    rejects = {"seen_skip": 0, "analyze_fail": 0, "decided": 0}
    if key is None:
        return {"rows": [], "rejects": rejects}
    from scripts.sf19_soft_dataset import _W_SEEN as seen_db
    if seen_db is not None and seen_db.has(key):
        rejects["seen_skip"] += 1
        return {"rows": [], "rejects": rejects}
    row = analyze_board(
        _W_ENGINE, board,
        nodes=_W_CFG.nodes, multipv=_W_CFG.multipv,
        tau=_W_CFG.tau, watchdog_s=_W_CFG.watchdog_s,
    )
    if row is None or int(row["policy_mask"]) == 0:
        rejects["analyze_fail"] += 1
        return {"rows": [], "rejects": rejects}
    row["game_id"] = np.int64(spec.get("game_id", -1))
    row["ply"] = np.int16(spec.get("ply", -1))
    row["split"] = np.int8(spec.get("split", 0))
    row["origin"] = np.int8(spec.get("origin", ORIGIN_RELABEL))
    return {"rows": [row], "rejects": rejects}


def freeze_eval_set(args) -> None:
    from scripts.sf19_soft_dataset import (
        resolve_sf, teacher_fingerprint, require_sf19, run_fingerprint,
        assert_resume_compatible, inbox_state, next_shard_dir,
    )
    out = Path(args.out_dir)
    inbox = out / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    sf = resolve_sf()
    fp = teacher_fingerprint(sf)
    require_sf19(fp)
    cfg = GenConfig(nodes=args.nodes, multipv=args.multipv, tau=args.tau, hash_mb=args.hash_mb)
    run_fp = run_fingerprint(fp, cfg)
    assert_resume_compatible(out, run_fp)
    rng = random.Random(args.seed + 1)
    n = int(args.eval_n)
    caches = [Path(p) for p in (args.seed_caches or [])] or [p for p in DEFAULT_SUPPLY if p.exists()]
    specs = sample_existing_specs(caches, int(n * 0.75), rng, stride=7)
    specs += sample_selfplay_starts(int(n * 0.35), rng)
    rng.shuffle(specs)
    specs = specs[:n]
    for i, s in enumerate(specs):
        s["split"] = 1
        s["game_id"] = -(i + 1)
        s["fen"] = s["fen"]
    seen = SeenDB(out / "seen.sqlite")
    workers = max(1, args.workers)
    ctx = get_context("spawn")
    pool = ctx.Pool(workers, initializer=_init_worker, initargs=(sf, cfg, str(out / "seen.sqlite")))
    rows = []
    t0 = time.time()
    try:
        for result in pool.imap_unordered(_label_one, specs, chunksize=1):
            for row in result["rows"]:
                k = row_key(row)
                if seen.has(k):
                    continue
                row["split"] = np.int8(1)
                rows.append(row)
                if len(rows) >= n:
                    break
            if len(rows) >= n:
                break
    finally:
        pool.terminate()
        pool.join()
    if not rows:
        raise SystemExit("freeze-eval produced no rows")
    data = stack_rows(rows)
    sh = next_shard_dir(inbox)
    write_shard(data, sh, {
        "split": "eval",
        "fingerprint_id": run_fp["fingerprint_id"],
        "nodes": cfg.nodes,
        "multipv": cfg.multipv,
        "tau": cfg.tau,
        "n": len(rows),
        "max_game_id": int(data["game_id"].max()),
    })
    keys = [row_key(r) for r in rows]
    seen.add_many(keys)
    seen.mark_shard(sh.name, len(rows))
    man = {
        "method": "saved_split_v1",
        "n": len(rows),
        "shard": sh.name,
        "fingerprint_id": run_fp["fingerprint_id"],
        "keys_hex": [k.hex() for k in keys],
        "note": "Eval frozen before production. Training must honor split==1; do not resample a position hash holdout.",
    }
    (out / "eval_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    teacher_path = out / "teacher.json"
    if not teacher_path.exists():
        teacher_path.write_text(json.dumps({"teacher": fp, "config": cfg.__dict__, "run": run_fp, "fingerprint_id": run_fp["fingerprint_id"]}, indent=2), encoding="utf-8")
    write_manifest(out, {"eval_n": len(rows), "fingerprint_id": run_fp["fingerprint_id"]})
    log(f"froze eval n={len(rows)} shard={sh} {time.time()-t0:.1f}s", out / "harvest.log")


def generate_mix(args) -> None:
    """Existing boards as main supply; self-play for extra diversity. Dedup first."""
    from dataclasses import asdict as _asdict
    from scripts.sf19_soft_dataset import (
        resolve_sf, teacher_fingerprint, require_sf19, run_fingerprint,
        assert_resume_compatible, inbox_state, next_shard_dir,
    )
    out = Path(args.out_dir)
    inbox = out / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    log_path = out / "harvest.log"
    sf = resolve_sf()
    fp = teacher_fingerprint(sf)
    require_sf19(fp)
    cfg = GenConfig(
        nodes=args.nodes, play_nodes=args.play_nodes, multipv=args.multipv,
        tau=args.tau, epsilon=args.epsilon, ply_stride=max(args.ply_stride, 3),
        ply_skip_open=args.ply_skip_open, ply_cap=args.ply_cap,
        book_noise=max(args.book_noise, 4), hash_mb=args.hash_mb,
        clear_hash_every=args.clear_hash_every,
    )
    run_fp = run_fingerprint(fp, cfg)
    assert_resume_compatible(out, run_fp)
    teacher_path = out / "teacher.json"
    if not teacher_path.exists():
        teacher_path.write_text(json.dumps({"teacher": fp, "config": _asdict(cfg), "run": run_fp, "fingerprint_id": run_fp["fingerprint_id"]}, indent=2), encoding="utf-8")
    seen = SeenDB(out / "seen.sqlite")
    committed, next_game = inbox_state(inbox)
    known = seen.known_shards()
    for sh in sorted(inbox.glob("shard_*")):
        if (sh / "READY").exists() and sh.name not in known:
            seen.ingest_shard_keys(sh)
    target = int(args.target)
    if committed >= target:
        log(f"already complete {committed} >= {target}", log_path)
        return
    relabel_target = int((target - committed) * float(args.relabel_frac))
    selfplay_target = (target - committed) - relabel_target
    caches = [Path(p) for p in (args.seed_caches or [])] or [p for p in DEFAULT_SUPPLY if p.exists()]
    rng = random.Random(args.seed + 2)
    workers = max(1, args.workers)
    ctx = get_context("spawn")
    pool = ctx.Pool(workers, initializer=_init_worker, initargs=(sf, cfg, str(out / "seen.sqlite")))
    pending: list[dict] = []
    pending_keys: set[bytes] = set()
    new_rows = 0
    rejected = {"dup": 0, "seen_skip": 0, "analyze_fail": 0, "decided": 0, "stride": 0}
    t0 = time.time()
    origin_counts = Counter()
    phase_counts = Counter()
    eval_counts = Counter()

    def flush(force: bool = False) -> None:
        nonlocal pending, pending_keys
        if not pending or (len(pending) < args.shard_size and not force):
            return
        take = pending[: args.shard_size] if not force else pending
        pending = pending[len(take):] if not force else []
        keys = [row_key(r) for r in take]
        data = stack_rows(take)
        sh = next_shard_dir(inbox)
        write_shard(data, sh, {
            "fingerprint_id": run_fp["fingerprint_id"],
            "nodes": cfg.nodes,
            "multipv": cfg.multipv,
            "tau": cfg.tau,
            "max_game_id": int(data["game_id"].max()),
        })
        seen.add_many(keys)
        seen.mark_shard(sh.name, len(take))
        seen.forget_hot(keys)
        pending_keys.difference_update(keys)
        log(f"wrote {sh} n={len(take)} total={committed + new_rows:,}", log_path)
        write_manifest(out, {"fingerprint_id": run_fp["fingerprint_id"], "accepted": committed + new_rows})

    def accept(rows: list[dict]) -> None:
        nonlocal new_rows
        for row in rows:
            if committed + new_rows >= target:
                return
            k = row_key(row)
            if seen.has(k) or k in pending_keys:
                rejected["dup"] += 1
                continue
            pending_keys.add(k)
            pending.append(row)
            new_rows += 1
            origin_counts[int(row["origin"])] += 1
            phase_counts[int(row["phase"])] += 1
            eval_counts[eval_bucket(int(row["cp"]), int(row["mate"]))] += 1
        seen.remember_hot([row_key(r) for r in rows])

    try:
        batch = 256
        jobs: list[dict] = []
        relabel_done = 0
        for spec in iter_existing_specs(caches, rng, seen=seen, stride=4):
            if relabel_done >= relabel_target or committed + new_rows >= target:
                break
            if spec["key"] in pending_keys:
                rejected["seen_skip"] += 1
                continue
            jobs.append(spec)
            if len(jobs) < batch:
                continue
            for result in pool.imap_unordered(_label_one, jobs, chunksize=1):
                rejected["seen_skip"] += int(result["rejects"].get("seen_skip") or 0)
                rejected["analyze_fail"] += int(result["rejects"].get("analyze_fail") or 0)
                before = new_rows
                accept(result["rows"])
                relabel_done += new_rows - before
                flush(False)
                if committed + new_rows >= target or relabel_done >= relabel_target:
                    break
            jobs = []
            elapsed = max(time.time() - t0, 1e-6)
            rate = new_rows / elapsed
            log(
                f"mix relabel accepted={committed+new_rows:,} relabel={relabel_done:,}/{relabel_target:,} "
                f"{rate:.1f}/s cpu_h={workers * elapsed / 3600.0:.2f} "
                f"eta={(target-committed-new_rows)/max(rate,1e-9)/60:.1f}m",
                log_path,
            )
        if jobs and committed + new_rows < target and relabel_done < relabel_target:
            for result in pool.imap_unordered(_label_one, jobs, chunksize=1):
                accept(result["rows"])
                flush(False)
        # Self-play remainder
        game_i = max(next_game, 1)
        while committed + new_rows < target:
            jobs = []
            for _ in range(max(workers * 2, 8)):
                if committed + new_rows >= target:
                    break
                jobs.append({
                    "game_id": game_i,
                    "seed": args.seed + game_i * 10007,
                    "opening": list(OPENINGS[game_i % len(OPENINGS)]),
                    "book_noise": cfg.book_noise,
                    "split": 0,
                })
                game_i += 1
            if not jobs:
                break
            for result in pool.imap_unordered(_play_one_game, jobs, chunksize=1):
                rejected["stride"] += int(result["rejects"].get("stride") or 0)
                rejected["analyze_fail"] += int(result["rejects"].get("analyze_fail") or 0)
                for row in result["rows"]:
                    row["origin"] = np.int8(ORIGIN_SELFPLAY)
                    row["split"] = np.int8(0)
                accept(result["rows"])
                flush(False)
                if committed + new_rows >= target:
                    break
            elapsed = max(time.time() - t0, 1e-6)
            rate = new_rows / elapsed
            log(
                f"mix selfplay accepted={committed+new_rows:,} {rate:.1f}/s "
                f"cpu_h={workers * elapsed / 3600.0:.2f}",
                log_path,
            )
    finally:
        flush(True)
        pool.terminate()
        pool.join()
    elapsed = max(time.time() - t0, 1e-6)
    rate = new_rows / elapsed if new_rows else 0.0
    summary = {
        "accepted": committed + new_rows,
        "new_rows": new_rows,
        "rejected": rejected,
        "pos_per_s": rate,
        "elapsed_s": elapsed,
        "cpu_hours": workers * elapsed / 3600.0,
        "hours_per_1m": (1e6 / max(rate, 1e-9)) / 3600.0,
        "origin": dict(origin_counts),
        "phase": dict(phase_counts),
        "eval_bucket": dict(eval_counts),
        "caches": [str(p) for p in caches],
        "fingerprint_id": run_fp["fingerprint_id"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "sampling.json").write_text(json.dumps({
        "relabel_frac": args.relabel_frac,
        "decided_cap": 0.15,
        "stride": 4,
        "origin": dict(origin_counts),
        "phase": dict(phase_counts),
        "eval_bucket": dict(eval_counts),
    }, indent=2), encoding="utf-8")
    write_manifest(out, {"summary": summary})
    log(json.dumps(summary), log_path)


def pack_train_cache(out_dir: Path, dest: Path, *, split: int | None = 0) -> int:
    """Concatenate READY shards, optionally keeping only one split value."""
    rows_n = 0
    chunks = []
    for sh in sorted((out_dir / "inbox").glob("shard_*")):
        cache = sh / "soft_cache.pt"
        if not (sh / "READY").exists() or not cache.exists():
            continue
        d = torch.load(cache, map_location="cpu", weights_only=False)
        if split is not None and "split" in d:
            mask = d["split"] == int(split)
            if not bool(mask.any()):
                del d
                continue
            d = {k: v[mask] if torch.is_tensor(v) and v.shape[0] == mask.shape[0] else v for k, v in d.items()}
        chunks.append(d)
        rows_n += int(d["move_idx"].shape[0])
    if not chunks:
        raise SystemExit(f"no rows for split={split} in {out_dir}")
    keys = chunks[0].keys()
    merged = {k: torch.cat([c[k] for c in chunks], dim=0) if torch.is_tensor(chunks[0][k]) else chunks[0][k] for k in keys}
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, dest)
    return rows_n


def compare_ft(args) -> None:
    """Matched fine-tunes: existing mix vs new SF19, same ckpt / steps / val."""
    from experiments.exp201_recurrent_64 import trial_config
    from scripts.autoresearch_8gb.pipeline import make_val_membership
    from scripts.autoresearch_8gb.train_trial import train_trial

    out = Path(args.out_dir)
    new_train = out / "train_soft.pt"
    new_all = out / "all_soft.pt"
    n_all = pack_train_cache(out, new_all, split=None)
    n_tr = pack_train_cache(out, new_train, split=0)
    all_data = torch.load(new_all, map_location="cpu", weights_only=False)
    man = make_val_membership(all_data, n_hold=0, seed=201, source="sf19")
    if man.get("method") != "saved_split_v1":
        raise SystemExit(f"loader did not honor saved split: {man.get('method')}")
    (out / "val_manifest_soft.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    eval_mask = all_data["split"] != 0
    eval_part = {k: v[eval_mask] if torch.is_tensor(v) and v.shape[0] == eval_mask.shape[0] else v for k, v in all_data.items()}
    src = Path(args.baseline_cache)
    base = torch.load(src, map_location="cpu", weights_only=False)
    take = min(int(n_tr), int(base["move_idx"].shape[0]))
    rng = np.random.RandomState(19)
    idx = rng.choice(int(base["move_idx"].shape[0]), size=take, replace=False)
    idx.sort()
    base_s = {k: v[idx] if torch.is_tensor(v) and v.shape[0] == base["move_idx"].shape[0] else v for k, v in base.items()}
    if "split" not in base_s:
        base_s["split"] = torch.zeros(take, dtype=torch.int8)
    else:
        base_s["split"] = torch.zeros(take, dtype=torch.int8)
    # same frozen eval rows appended so both runs use saved_split_v1
    for k, v in eval_part.items():
        if k in base_s and torch.is_tensor(base_s[k]) and torch.is_tensor(v) and base_s[k].dim() > 0 and base_s[k].shape[0] == take:
            if v.shape[0] and base_s[k].shape[1:] == v.shape[1:]:
                base_s[k] = torch.cat([base_s[k], v], dim=0)
        elif k not in base_s and torch.is_tensor(v):
            base_s[k] = v
    base_path = out / "baseline_soft.pt"
    torch.save(base_s, base_path)
    del base, base_s, all_data
    trial = trial_config()
    trial["train"]["elo_every_steps"] = 0
    trial["train"]["save_every_steps"] = 0
    steps = int(args.ft_steps)
    ckpt = Path(args.ckpt)
    reports = {}
    for name, cache in (("baseline", base_path), ("sf19", new_train)):
        dest = out / f"ft_{name}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "val_manifest_soft.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        rec = train_trial(
            trial, dest,
            soft_cache=new_all if name == "sf19" else cache,
            max_steps=steps,
            max_minutes=float(args.ft_minutes),
            resume_ckpt=ckpt,
        )
        reports[name] = rec
        log(f"ft {name} status={rec.get('status')} steps={rec.get('steps')} {rec}")
    (out / "ft_compare.json").write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    log(f"packed train={n_tr} all={n_all} baseline={take} manifest={man['method']}")
