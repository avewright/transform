#!/usr/bin/env python3
"""exp280: train the full MoE (router + all five 99M experts) on general Lichess.

Data: avewright/chess-soft-multipv-lichess (soft MultiPV). Switch-style
forward: incumbent encode → router → chosen expert. Incumbent also gets
the policy loss every step (generalist). Router gets a phase-prior aux.

Does not write incumbent / puzzle / endgame / opening / middlegame HF repos.

  MOVE_VOCAB_VERSION=compact python experiments/exp280_moe_full.py --go
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_moe import (  # noqa: E402
    EXPERTS,
    EXPERT_NAMES,
    N_EXPERTS,
    FrozenExpertMoE,
    build_router,
    load_expert_weights,
    phase_label_to_expert,
    phase_prior,
)
from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG, build_squares64, count_parameters
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "exp280_moe_full"
EXPERT_DIR = ROOT / "outputs" / "hf_models" / "experts"
ROUTER_CKPT = ROOT / "outputs" / "exp279_moe_router" / "latest.pt"
SOFT_REPO = "avewright/chess-soft-multipv-lichess"
SOURCE_PACK = ROOT / "outputs" / "exp279_moe_router" / "source_pack.pt"
MIN_DEPTH = 12
MIX_KEYS = (
    "board_array",
    "turn",
    "castling",
    "ep_square",
    "move_idx",
    "cp",
    "mate",
    "soft_indices",
    "soft_probs",
    "source_id",
    "source_expert",
)
SOURCE_NAMES = ("opening", "middlegame", "endgame", "general", "puzzles")
# dataset name → expert id
SOURCE_EXPERT = {
    "opening": 3,
    "middlegame": 4,
    "endgame": 2,
    "general": 0,
    "puzzles": 1,
}


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit("Expected compact vocab 1968. Export MOVE_VOCAB_VERSION=compact.")


def load_expert(name: str, device: torch.device) -> torch.nn.Module:
    ckpt = EXPERT_DIR / name / "latest.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing expert {ckpt}")
    model = build_squares64(DEFAULT_100M_SQUARES64_CONFIG)
    load_expert_weights(model, str(ckpt))
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    return model


def load_router(path: Path, device: torch.device) -> torch.nn.Module:
    router = build_router()
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        router.load_state_dict(blob["model_state_dict"])
        print(f"router from {path} steps={blob.get('steps')}", flush=True)
    router.to(device)
    for p in router.parameters():
        p.requires_grad_(True)
    return router


def parquet_to_soft(path: Path) -> dict:
    import pyarrow.parquet as pq
    from build_hf_elo_mix import CORE, META, load_parquet_rows

    have = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in list(CORE) + list(META) if c in have]
    data = load_parquet_rows(path, cols)
    if "label_depth" in data:
        keep = data["label_depth"] >= MIN_DEPTH
        if int((~keep).sum()) > 0:
            data = {k: v[keep] for k, v in data.items()}
    return data


def concat_soft(chunks: list[dict]) -> dict:
    keys = [k for k in chunks[0] if all(k in c and torch.is_tensor(c[k]) for c in chunks)]
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}


def local_parquet_files() -> list[Path]:
    snap = ROOT / ".hf_cache" / "hub" / "datasets--avewright--chess-soft-multipv-lichess"
    found = sorted(snap.rglob("data-*.parquet"))
    return found


def list_remote_parquets(token: str | None) -> list[str]:
    from huggingface_hub import list_repo_files

    return [f for f in list_repo_files(SOFT_REPO, repo_type="dataset", token=token) if f.endswith(".parquet")]


def download_parquet(name: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(SOFT_REPO, name, repo_type="dataset", token=token))


def _only(data: dict, keys: tuple[str, ...] = MIX_KEYS) -> dict:
    n = int(data["board_array"].shape[0])
    out = {}
    for k in keys:
        if k not in data:
            continue
        v = data[k]
        if torch.is_tensor(v) and int(v.shape[0]) == n:
            out[k] = v
    return out


def tag_source(data: dict, name: str) -> dict:
    from exp279_moe_router import attach_onehot

    data = attach_onehot({k: data[k] for k in data if torch.is_tensor(data[k])})
    n = int(data["board_array"].shape[0])
    data["source_id"] = torch.full((n,), SOURCE_NAMES.index(name), dtype=torch.long)
    data["source_expert"] = torch.full((n,), SOURCE_EXPERT[name], dtype=torch.long)
    return _only(data)


def subsample_rows(data: dict, n_take: int, seed: int) -> dict:
    n = int(data["board_array"].shape[0])
    take = min(int(n_take), n)
    if take <= 0:
        return {k: v[:0] for k, v in data.items()}
    if take == n:
        return data
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:take]
    return {k: v[idx] for k, v in data.items()}


def load_source_pack_parts() -> dict[str, dict]:
    if not SOURCE_PACK.exists():
        return {}
    raw = torch.load(SOURCE_PACK, map_location="cpu", weights_only=False)
    parts: dict[str, dict] = {}
    for i, name in enumerate(SOURCE_NAMES):
        mask = raw["source_id"] == i
        if not bool(mask.any()):
            continue
        parts[name] = _only({k: raw[k][mask] for k in raw if torch.is_tensor(raw[k])})
        print(f"  pack {name} n={int(mask.sum()):,}", flush=True)
    return parts


def sample_inbox_or_pack(name: str, n_take: int, seed: int) -> dict | None:
    from exp279_moe_router import sample_specialist_rows

    rng = torch.Generator().manual_seed(seed)
    got = sample_specialist_rows(name, n_take, rng, token=None, seed=seed)
    if got is None or int(got["move_idx"].shape[0]) == 0:
        return None
    return tag_source(got, name)


def build_local_mix(*, per_source: int, val_n: int, seed: int) -> tuple[dict, dict]:
    """Concat opening / mid / end / puzzle / general from local caches."""
    print("building local 5-source mix", flush=True)
    packed = load_source_pack_parts()
    chunks_tr: list[dict] = []
    chunks_va: list[dict] = []
    for name in SOURCE_NAMES:
        if name == "general":
            files = local_parquet_files()
            if not files:
                print("  skip general (no local parquet)", flush=True)
                continue
            gen = tag_source(concat_soft([parquet_to_soft(p) for p in files]), "general")
            gen = subsample_rows(gen, per_source + val_n, seed + 17)
            tr, va = holdout_split(gen, val_n, seed)
            print(f"  general local train={int(tr['board_array'].shape[0]):,} val={int(va['board_array'].shape[0]):,}", flush=True)
            chunks_tr.append(tr)
            chunks_va.append(va)
            continue
        data = packed.get(name)
        if data is None or int(data["board_array"].shape[0]) < 1024:
            print(f"  sample {name} from inbox/hf", flush=True)
            data = sample_inbox_or_pack(name, per_source + val_n, seed + SOURCE_NAMES.index(name))
        if data is None:
            print(f"  skip {name}", flush=True)
            continue
        data = subsample_rows(data, per_source + val_n, seed + SOURCE_NAMES.index(name))
        hold = packed.get(name) and "is_holdout" in packed[name]
        if hold and "is_holdout" in data:
            va = {k: data[k][data["is_holdout"]] for k in data}
            tr = {k: data[k][~data["is_holdout"]] for k in data}
            if int(va["board_array"].shape[0]) > val_n:
                va = subsample_rows(va, val_n, seed)
            if int(tr["board_array"].shape[0]) > per_source:
                tr = subsample_rows(tr, per_source, seed + 1)
        else:
            tr, va = holdout_split(data, val_n, seed + SOURCE_NAMES.index(name))
        print(f"  {name} train={int(tr['board_array'].shape[0]):,} val={int(va['board_array'].shape[0]):,}", flush=True)
        chunks_tr.append(_only(tr))
        chunks_va.append(_only(va))
    if not chunks_tr:
        raise SystemExit("local mix empty")
    train = concat_soft(chunks_tr)
    val = concat_soft(chunks_va)
    print(
        f"mix train={int(train['board_array'].shape[0]):,} val={int(val['board_array'].shape[0]):,}",
        flush=True,
    )
    return train, val


def source_buckets(source_id: torch.Tensor) -> list[torch.Tensor]:
    idx = torch.arange(int(source_id.numel()))
    buckets = [idx[source_id == i] for i in range(5)]
    return [b for b in buckets if int(b.numel()) > 0]


def stratified_pick(buckets: list[torch.Tensor], bs: int) -> torch.Tensor:
    parts = []
    left = int(bs)
    for i, bkt in enumerate(buckets):
        take = left // (len(buckets) - i)
        parts.append(bkt[torch.randint(0, int(bkt.numel()), (take,))])
        left -= take
    return torch.cat(parts)


def holdout_split(data: dict, val_n: int, seed: int) -> tuple[dict, dict]:
    n = int(data["board_array"].shape[0])
    val_n = min(int(val_n), max(256, n // 10))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:val_n], perm[val_n:]
    return {k: v[train_idx] for k, v in data.items()}, {k: v[val_idx] for k, v in data.items()}


def build_moe(device: torch.device, router_ckpt: Path) -> FrozenExpertMoE:
    experts = [load_expert(name, device) for name, _ in EXPERTS]
    router = load_router(router_ckpt, device)
    moe = FrozenExpertMoE(router, experts, freeze=False)
    print(
        f"full MoE params={count_parameters(moe):,} experts={EXPERT_NAMES} device={device}",
        flush=True,
    )
    return moe


def _save(moe: FrozenExpertMoE, path: Path, step: int, extra: dict | None = None) -> None:
    blob = {
        "arch": "chess_moe_full",
        "vocab_version": "compact",
        "model_state_dict": moe.router.state_dict(),
        "expert_state_dicts": [e.state_dict() for e in moe.experts],
        "experts": EXPERTS,
        "steps": step,
    }
    if extra:
        blob.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def eval_soft(moe: FrozenExpertMoE, data: dict, device: torch.device, n: int = 2048) -> dict:
    from autoresearch_8gb.pipeline import prepare_soft_batch, soft_policy_loss

    moe.eval()
    idx = torch.randperm(int(data["board_array"].shape[0]))[: min(n, int(data["board_array"].shape[0]))]
    hist = [0] * N_EXPERTS
    loss_sum = hard_sum = 0.0
    for start in range(0, int(idx.numel()), 64):
        take = idx[start : start + 64]
        bi, move_idx, _wdl, soft_i, soft_p = prepare_soft_batch(data, take, device, hflip_p=0.0)
        out = moe(bi)
        loss_sum += float(soft_policy_loss(out.policy_logits, soft_i, soft_p)) * int(take.numel())
        hard_sum += float(F.cross_entropy(out.policy_logits, move_idx)) * int(take.numel())
        for e in range(N_EXPERTS):
            hist[e] += int((out.expert_id == e).sum())
    moe.train()
    w = max(int(idx.numel()), 1)
    return {
        "soft_ce": loss_sum / w,
        "hard_ce": hard_sum / w,
        "route_hist": [h / w for h in hist],
    }


def train(args: argparse.Namespace) -> None:
    from autoresearch_8gb.pipeline import prepare_soft_batch, soft_policy_loss
    from autoresearch_8gb.train_trial import build_polar_normuon_optimizer
    from upload_exp201_hf import load_hf_token

    _assert_compact()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "train.log"
    token = load_hf_token()

    train_data, val_data = build_local_mix(
        per_source=int(args.per_source),
        val_n=int(args.val_n),
        seed=int(args.seed),
    )
    mix_path = OUT / "mix.pt"
    torch.save({"train_n": int(train_data["board_array"].shape[0]), "val_n": int(val_data["board_array"].shape[0])}, mix_path)

    moe = build_moe(device, Path(args.router_ckpt) if args.router_ckpt else ROUTER_CKPT)
    opt, muon_n, adam_n = build_polar_normuon_optimizer(
        moe,
        muon_lr=float(args.muon_lr),
        adam_lr=float(args.adam_lr),
        weight_decay=0.01,
        compile_polar=False,
    )
    print(f"polar muon_n={muon_n:,} adam_n={adam_n:,} muon_lr={args.muon_lr} adam_lr={args.adam_lr}", flush=True)

    steps = int(args.max_steps)
    bs = int(args.microbatch)
    router_w = float(args.router_w)
    inc_w = float(args.incumbent_w)
    value_w = float(args.value_weight)
    best = float("inf")
    n_train = int(train_data["board_array"].shape[0])
    buckets = source_buckets(train_data["source_id"]) if "source_id" in train_data else [torch.arange(n_train)]
    print(f"train buckets={[int(b.numel()) for b in buckets]}", flush=True)
    t_step = time.time()

    remote_names: list[str] = []
    remote_i = 0
    if args.stream_remote:
        have = {p.name for p in local_parquet_files()}
        remote_names = [n for n in list_remote_parquets(token) if Path(n).name not in have]
        print(f"remote shards left={len(remote_names)}", flush=True)

    for step in range(1, steps + 1):
        pick = stratified_pick(buckets, bs)
        bi, move_idx, wdl, soft_i, soft_p = prepare_soft_batch(train_data, pick, device, hflip_p=0.5)
        out = moe(bi)
        routed = soft_policy_loss(out.policy_logits, soft_i, soft_p)
        hard = F.cross_entropy(out.policy_logits, move_idx)
        inc_pol = soft_policy_loss(out.stem_policy_logits, soft_i, soft_p)
        if "source_expert" in train_data:
            gold = train_data["source_expert"][pick].to(device)
        elif "phase" in train_data:
            gold = phase_label_to_expert(train_data["phase"][pick].to(device))
        else:
            gold = phase_prior(train_data["board_array"][pick].ne(0).sum(1).to(device))
        route_loss = F.cross_entropy(out.gate_logits, gold)
        v_loss = F.cross_entropy(out.value_logits, wdl)
        loss = routed + inc_w * inc_pol + router_w * route_loss + value_w * v_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(moe.parameters(), 1.0)
        opt.step()

        if step % 20 == 0 or step == 1:
            now = datetime.now().strftime("%H:%M:%S")
            dt = max(time.time() - t_step, 1e-6)
            pos_s = (bs if step == 1 else 20 * bs) / dt
            t_step = time.time()
            hist = " ".join(f"r{EXPERT_NAMES[i][:3]}={float((out.expert_id == i).float().mean()):.2f}" for i in range(N_EXPERTS))
            vram = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            line = (
                f"[{now}] step {step}/{steps} | loss={float(loss):.4f} "
                f"soft_ce={float(routed):.4f} hard_ce={float(hard):.4f} "
                f"inc_ce={float(inc_pol):.4f} route_ce={float(route_loss):.4f} "
                f"| {hist} | {pos_s:.1f} pos/s | vram={vram:.2f}GB"
            )
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        if step % int(args.val_every) == 0 or step == 1:
            ev = eval_soft(moe, val_data, device, n=int(args.val_eval_n))
            now = datetime.now().strftime("%H:%M:%S")
            share = " ".join(f"{EXPERT_NAMES[i][:3]}={ev['route_hist'][i]:.2f}" for i in range(N_EXPERTS))
            vline = (
                f"[{now}] val/lichess hard_ce={ev['hard_ce']:.4f} "
                f"soft_ce={ev['soft_ce']:.4f} soft_temp_ce={ev['soft_ce']:.4f} "
                f"| {share}"
            )
            print(vline, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(vline + "\n")
            if ev["soft_ce"] < best:
                best = ev["soft_ce"]
                _save(moe, OUT / "best.pt", step, {"val_soft_ce": best})
                print(f"  best val_soft_ce={best:.4f} step={step}", flush=True)

        if step % int(args.save_every) == 0:
            _save(moe, OUT / "latest.pt", step)

        if args.stream_remote and remote_names and step % int(args.absorb_every) == 0:
            name = remote_names[remote_i % len(remote_names)]
            remote_i += 1
            try:
                path = download_parquet(name, token)
                extra = parquet_to_soft(path)
                train_data = concat_soft([train_data, extra])
                n_train = int(train_data["board_array"].shape[0])
                print(f"absorb {path.name} +{int(extra['board_array'].shape[0]):,} train={n_train:,}", flush=True)
            except Exception as e:
                print(f"skip absorb {name}: {e}", flush=True)

    _save(moe, OUT / "latest.pt", steps)
    print(f"wrote {OUT / 'latest.pt'} best_val_soft_ce={best:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--router-ckpt", default=str(ROUTER_CKPT))
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--microbatch", type=int, default=96)
    ap.add_argument("--val-n", type=int, default=8192)
    ap.add_argument("--val-eval-n", type=int, default=2048)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--absorb-every", type=int, default=400)
    ap.add_argument("--per-source", type=int, default=400_000)
    ap.add_argument("--stream-remote", action="store_true", default=False)
    ap.add_argument("--no-stream-remote", action="store_false", dest="stream_remote")
    ap.add_argument("--muon-lr", type=float, default=5e-4)
    ap.add_argument("--adam-lr", type=float, default=1e-5)
    ap.add_argument("--router-w", type=float, default=0.15)
    ap.add_argument("--incumbent-w", type=float, default=0.4)
    ap.add_argument("--value-weight", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=280)
    args = ap.parse_args()
    if args.go:
        train(args)
    else:
        print("pass --go", flush=True)


if __name__ == "__main__":
    main()
