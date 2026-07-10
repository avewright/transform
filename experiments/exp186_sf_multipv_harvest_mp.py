"""exp186 MP: multiprocessing Stockfish MultiPV harvest (full strength, depth 2-8)."""
from __future__ import annotations
import argparse, json, math, os, random, shutil, signal, sqlite3, sys, time
from datetime import datetime, timezone
from pathlib import Path
from multiprocessing import Process, Queue, Event, cpu_count
import chess, chess.engine
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
LABEL_TAU = 120.0
DEPTH_WEIGHTS = {2: 2, 3: 3, 4: 4, 5: 3, 6: 2, 7: 1, 8: 1}
HF_SOURCES = ["avewright/chess-positions-sf-labeled", "avewright/chess-positions-lichess-sf"]

def resolve_sf():
    c = os.environ.get("STOCKFISH_PATH")
    cands = [Path(c)] if c else []
    w = shutil.which("stockfish")
    if w: cands.append(Path(w))
    cands += [ROOT/"stockfish/stockfish/stockfish-ubuntu-x86-64-avx2", Path("/usr/games/stockfish")]
    for p in cands:
        if p and p.exists(): return p
    raise FileNotFoundError(cands)

SF = resolve_sf()

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def fen_key(fen: str) -> str:
    parts = fen.split()
    return " ".join(parts[:4]) if len(parts) >= 4 else fen

def phase_name(board):
    pieces = sum(1 for p in board.piece_map().values() if p.piece_type != chess.KING)
    return "opening" if pieces >= 20 else ("middlegame" if pieces >= 10 else "endgame")

def score_to_cp(score_obj, pov):
    s = score_obj.pov(pov)
    if s.is_mate():
        mate = s.mate() or 0
        sign = 1 if mate > 0 else -1
        return sign * (100000 - min(abs(mate), 1000)), "mate"
    cp = s.score(mate_score=100000)
    return int(cp if cp is not None else 0), "cp"

def sample_depth(rng, dmin, dmax):
    ds = list(range(dmin, dmax+1))
    ws = [DEPTH_WEIGHTS.get(d, 1) for d in ds]
    return rng.choices(ds, weights=ws, k=1)[0]

def analyze(engine, board, depth, multipv, tau):
    n = board.legal_moves.count()
    if n == 0: return None
    try:
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=min(multipv, n))
    except Exception:
        return None
    if not isinstance(infos, list): infos = [infos]
    best = {}
    for info in infos:
        pv = info.get("pv") or []
        score = info.get("score")
        if not pv or score is None: continue
        cp, et = score_to_cp(score, board.turn)
        uci = pv[0].uci()
        if uci not in best or cp > best[uci][0]:
            best[uci] = (cp, et, [m.uci() for m in pv[:8]])
    if not best: return None
    items = sorted(best.items(), key=lambda x: -x[1][0])
    cps = [v[0] for _, v in items]
    probs = F.softmax(torch.tensor(cps, dtype=torch.float32)/tau, dim=0).tolist()
    soft = []
    for i,((uci,(cp,et,pv)), pr) in enumerate(zip(items, probs)):
        soft.append({"uci":uci,"prob":float(pr),"cp":int(cp),"eval_type":et,"rank":i+1,"pv":pv})
    return {
        "fen": board.fen(),
        "best_move": soft[0]["uci"],
        "best_cp": soft[0]["cp"],
        "soft_targets": soft,
        "label_depth": depth,
        "label_multipv": len(soft),
        "label_tau": tau,
        "label_mode": "multipv_topk",
        "num_legal": n,
        "phase": phase_name(board),
        "ply": board.fullmove_number*2 - (0 if board.turn else 1),
        "source": "exp186_sf_multipv",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "value_target": 2 if soft[0]["cp"]>100 else (0 if soft[0]["cp"]<-100 else 1),
        "cp_gap_top1_top2": int(soft[0]["cp"]-soft[1]["cp"]) if len(soft)>1 else 0,
        "teacher_entropy": float(-(sum(p*math.log(max(p,1e-12)) for p in probs))),
    }

def worker(wid, task_q, result_q, stop_ev, dmin, dmax, multipv, tau, hash_mb):
    rng = random.Random(10000+wid)
    eng = chess.engine.SimpleEngine.popen_uci(str(SF))
    eng.configure({"Threads": 1, "Hash": hash_mb})
    try:
        while not stop_ev.is_set():
            try:
                item = task_q.get(timeout=1.0)
            except Exception:
                continue
            if item is None:
                break
            fen, meta = item
            try:
                board = chess.Board(fen)
            except Exception:
                result_q.put(("bad", None)); continue
            if board.is_game_over(claim_draw=True):
                result_q.put(("skip", None)); continue
            depth = sample_depth(rng, dmin, dmax)
            rec = analyze(eng, board, depth, multipv, tau)
            if rec is None:
                result_q.put(("fail", None)); continue
            rec["hf_repo"] = meta.get("hf_repo")
            result_q.put(("ok", rec))
    finally:
        try: eng.quit()
        except Exception: pass

def build_cache(dataset_dir, out_path, max_rows=None):
    os.environ.setdefault("MOVE_VOCAB_VERSION","compact")
    from move_vocab import UCI_TO_IDX
    boards,turns,castles,eps,moves,cps,mates,soft_idx,soft_pr=[],[],[],[],[],[],[],[],[]
    skipped=0
    for shard in sorted(Path(dataset_dir).glob("positions_*.jsonl")):
        with open(shard) as f:
            for line in f:
                if max_rows is not None and len(boards)>=max_rows: break
                try: row=json.loads(line)
                except: skipped+=1; continue
                fen,best,soft=row.get("fen"),row.get("best_move"),row.get("soft_targets") or []
                if not fen or not best or best not in UCI_TO_IDX or not soft:
                    skipped+=1; continue
                try:
                    board=chess.Board(fen)
                except: skipped+=1; continue
                if chess.Move.from_uci(best) not in board.legal_moves:
                    skipped+=1; continue
                arr=[0]*64
                for sq,piece in board.piece_map().items():
                    arr[sq]=piece.piece_type if piece.color==chess.WHITE else piece.piece_type+6
                ba=torch.tensor([arr],dtype=torch.int8)
                turn=torch.tensor([0 if board.turn else 1],dtype=torch.int8)
                castling=torch.tensor([0],dtype=torch.int8)
                if board.has_kingside_castling_rights(chess.WHITE): castling[0]|=1
                if board.has_queenside_castling_rights(chess.WHITE): castling[0]|=2
                if board.has_kingside_castling_rights(chess.BLACK): castling[0]|=4
                if board.has_queenside_castling_rights(chess.BLACK): castling[0]|=8
                ep=torch.tensor([board.ep_square if board.ep_square is not None else 0],dtype=torch.int8)
                idx,pr=[],[]
                for it in soft[:8]:
                    u=it.get("uci")
                    if u and u in UCI_TO_IDX:
                        idx.append(UCI_TO_IDX[u]); pr.append(float(it.get("prob",0)))
                if not idx: skipped+=1; continue
                s=sum(pr) or 1.0; pr=[p/s for p in pr]
                while len(idx)<8: idx.append(-1); pr.append(0.0)
                boards.append(ba); turns.append(turn); castles.append(castling); eps.append(ep)
                moves.append(torch.tensor([UCI_TO_IDX[best]],dtype=torch.long))
                cps.append(torch.tensor([int(row.get("best_cp",0) or 0)],dtype=torch.int32))
                mates.append(torch.tensor([0],dtype=torch.int32))
                soft_idx.append(torch.tensor(idx,dtype=torch.long))
                soft_pr.append(torch.tensor(pr,dtype=torch.float32))
        if max_rows is not None and len(boards)>=max_rows: break
    if not boards: return 0
    data={"board_array":torch.cat(boards),"turn":torch.cat(turns),"castling":torch.cat(castles),
          "ep_square":torch.cat(eps),"move_idx":torch.cat(moves),"cp":torch.cat(cps),"mate":torch.cat(mates),
          "soft_indices":torch.stack(soft_idx),"soft_probs":torch.stack(soft_pr)}
    tmp=Path(str(out_path)+".tmp"); torch.save(data,tmp); os.replace(tmp,out_path)
    log(f"soft_cache {data['board_array'].shape[0]:,} → {out_path} (skip {skipped})")
    return int(data['board_array'].shape[0])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--target", type=int, default=2_000_000)
    ap.add_argument("--depth-min", type=int, default=2)
    ap.add_argument("--depth-max", type=int, default=8)
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--tau", type=float, default=LABEL_TAU)
    ap.add_argument("--hash-mb", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--cache-every", type=int, default=50000)
    ap.add_argument("--output-dir", type=str, default=str(ROOT/"outputs/exp186_sf_multipv"))
    ap.add_argument("--build-cache-only", action="store_true")
    args=ap.parse_args()
    out=Path(args.output_dir); dsdir=out/"dataset"; dsdir.mkdir(parents=True, exist_ok=True)
    if args.build_cache_only:
        build_cache(dsdir, out/"soft_cache.pt"); return
    if not args.go:
        print("Pass --go"); return
    if args.smoke:
        args.target=300; args.workers=min(8,args.workers); args.cache_every=150; args.shard_size=100

    log(f"exp186-MP SF={SF} workers={args.workers} depth=[{args.depth_min},{args.depth_max}] target={args.target:,}")
    stop_ev=Event()
    def _stop(*_):
        stop_ev.set(); log("STOP")
    signal.signal(signal.SIGINT,_stop); signal.signal(signal.SIGTERM,_stop)

    # seen set (boot on main; producer thread opens its own connection)
    dbp=out/"seen_positions.sqlite"
    _boot=sqlite3.connect(str(dbp)); _boot.execute("CREATE TABLE IF NOT EXISTS seen(key TEXT PRIMARY KEY, ts REAL)")
    seen={r[0] for r in _boot.execute("SELECT key FROM seen")}
    _boot.close()
    log(f"seen loaded {len(seen):,}")

    written=0
    for sp in sorted(dsdir.glob("positions_*.jsonl")):
        with open(sp) as f:
            written += sum(1 for _ in f)
    log(f"resume written={written:,}")

    task_q=Queue(maxsize=args.workers*64)
    result_q=Queue(maxsize=args.workers*64)
    procs=[]
    for wid in range(args.workers):
        p=Process(target=worker, args=(wid, task_q, result_q, stop_ev, args.depth_min, args.depth_max, args.multipv, args.tau, args.hash_mb), daemon=True)
        p.start(); procs.append(p)

    # producer process inline in main via thread-like loop with prefetch
    from datasets import load_dataset
    import threading
    def producer():
        rng=random.Random(42)
        epoch=0
        pending=[]
        conn=sqlite3.connect(str(dbp), check_same_thread=False)
        while not stop_ev.is_set():
            for repo in HF_SOURCES:
                if stop_ev.is_set(): break
                log(f"feed {repo} epoch={epoch}")
                try:
                    if "lichess-sf" in repo:
                        ds=load_dataset(repo, split="train", streaming=True).shuffle(seed=42+epoch, buffer_size=20000)
                        it=iter(ds)
                        while not stop_ev.is_set():
                            try: row=next(it)
                            except StopIteration: break
                            fen=row.get("fen")
                            if not fen: continue
                            k=fen_key(fen)
                            if k in seen: continue
                            seen.add(k); pending.append(k)
                            while not stop_ev.is_set():
                                try:
                                    task_q.put((fen,{"hf_repo":repo}), timeout=0.5); break
                                except Exception: continue
                            if len(pending)>=2000:
                                conn.executemany("INSERT OR IGNORE INTO seen(key,ts) VALUES (?,?)", [(k,time.time()) for k in pending])
                                conn.commit(); pending.clear()
                    else:
                        ds=load_dataset(repo, split="train"); n=len(ds); log(f"  mat {n:,}")
                        order=list(range(n)); rng.shuffle(order)
                        for i in order:
                            if stop_ev.is_set(): break
                            row=ds[i]; fen=row.get("fen")
                            if not fen: continue
                            k=fen_key(fen)
                            if k in seen: continue
                            seen.add(k); pending.append(k)
                            while not stop_ev.is_set():
                                try:
                                    task_q.put((fen,{"hf_repo":repo}), timeout=0.5); break
                                except Exception: continue
                            if len(pending)>=2000:
                                conn.executemany("INSERT OR IGNORE INTO seen(key,ts) VALUES (?,?)", [(k,time.time()) for k in pending])
                                conn.commit(); pending.clear()
                        del ds
                except Exception as e:
                    log(f"feed err {repo}: {e}")
            epoch += 1
        if pending:
            conn.executemany("INSERT OR IGNORE INTO seen(key,ts) VALUES (?,?)", [(k,time.time()) for k in pending])
            conn.commit()
        try: conn.close()
        except Exception: pass
        for _ in procs:
            try: task_q.put(None, timeout=1)
            except Exception: pass
    prod=threading.Thread(target=producer, daemon=True); prod.start()

    # writer
    existing=sorted(dsdir.glob("positions_*.jsonl"))
    if existing:
        last=existing[-1]; shard_idx=int(last.stem.split("_")[-1])
        with open(last) as f: shard_count=sum(1 for _ in f)
        if shard_count>=args.shard_size:
            shard_idx+=1; shard_count=0; shard_path=dsdir/f"positions_{shard_idx:06d}.jsonl"
        else:
            shard_path=last
    else:
        shard_idx=1; shard_count=0; shard_path=dsdir/f"positions_{shard_idx:06d}.jsonl"
    shard_f=open(shard_path,"a",encoding="utf-8")
    t0=time.time(); start_w=written; last_st=t0; last_cache=written
    ok=fail=skip=bad=0; depth_hist={}
    try:
        while written < args.target and not stop_ev.is_set():
            try:
                kind, rec = result_q.get(timeout=1.0)
            except Exception:
                if not prod.is_alive() and task_q.empty(): break
                continue
            if kind!="ok":
                if kind=="fail": fail+=1
                elif kind=="skip": skip+=1
                else: bad+=1
                continue
            shard_f.write(json.dumps(rec,separators=(",",":"))+"\n")
            written+=1; shard_count+=1; ok+=1
            depth_hist[rec["label_depth"]]=depth_hist.get(rec["label_depth"],0)+1
            if shard_count>=args.shard_size:
                shard_f.flush(); os.fsync(shard_f.fileno()); shard_f.close()
                shard_idx+=1; shard_count=0
                shard_path=dsdir/f"positions_{shard_idx:06d}.jsonl"
                shard_f=open(shard_path,"a",encoding="utf-8")
            now=time.time()
            if now-last_st>=15:
                rate=(written-start_w)/max(now-t0,1e-6)
                eta=(args.target-written)/max(rate,1e-6)
                log(f"labeled={written:,}/{args.target:,} | {rate:.1f}/s | eta={eta/3600:.1f}h | depths={dict(sorted(depth_hist.items()))}")
                Path(out/"status.json").write_text(json.dumps({"written":written,"target":args.target,"rate_pos_s":rate,"depth_hist":depth_hist,"updated_at":datetime.now(timezone.utc).isoformat()},indent=2))
                last_st=now
            if written-last_cache>=args.cache_every:
                shard_f.flush(); build_cache(dsdir, out/"soft_cache.pt"); last_cache=written
    finally:
        stop_ev.set()
        try: shard_f.flush(); shard_f.close()
        except: pass
        for p in procs: p.join(timeout=3)
    build_cache(dsdir, out/"soft_cache.pt")
    log(f"Done written={written:,}")

if __name__=="__main__":
    main()
