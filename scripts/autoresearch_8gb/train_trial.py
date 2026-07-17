"""Train one autoresearch trial on ~25M chess transformer (8GB-friendly)."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[2]

ADAM_NAME_HINTS = (
    "embed", "policy_head", "value_head", "cls_token", "cls_pos",
    "pos_embed", "norm", "bn", "shaw_", "rel_bias",
)


def _log(path: Path | None, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path is not None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def soft_policy_loss(logits, soft_indices, soft_probs):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    valid = (soft_indices >= 0) & (soft_probs > 0)
    safe = soft_indices.clamp(min=0).long()
    gathered = log_probs.gather(1, safe) * valid.float()
    return -(soft_probs.float() * gathered).sum(dim=-1).mean()


def soft_temp_policy_loss(logits, soft_indices, soft_probs, temperature: float = 4.0):
    """Chessformer soft-policy aux: visit dist raised to 1/T (paper T=4, c_softpol=8)."""
    p = soft_probs.float().clamp_min(1e-8)
    p_t = p.pow(1.0 / max(temperature, 1e-6))
    p_t = p_t / p_t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return soft_policy_loss(logits, soft_indices, p_t)


def _ema_update(ema_state: dict[str, torch.Tensor], model: nn.Module, n: int) -> int:
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if not torch.is_floating_point(v):
                ema_state[k] = v.detach().clone()
                continue
            if k not in ema_state:
                ema_state[k] = v.detach().clone()
            else:
                ema_state[k].mul_(n / (n + 1)).add_(v.detach(), alpha=1.0 / (n + 1))
    return n + 1


def _split_muon_adam_params(model):
    muon_params, adam_params = [], []
    muon_n = adam_n = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if any(h in name for h in ADAM_NAME_HINTS) or param.ndim < 2:
            adam_params.append(param)
            adam_n += n
        else:
            muon_params.append(param)
            muon_n += n
    return muon_params, adam_params, muon_n, adam_n


def build_normuon_optimizer(model, muon_lr, adam_lr, weight_decay):
    from normuon import SingleDeviceNorMuonWithAuxAdam

    muon_params, adam_params, muon_n, adam_n = _split_muon_adam_params(model)
    opt = SingleDeviceNorMuonWithAuxAdam([
        dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=weight_decay,
             momentum=0.95, beta2=0.95),
        dict(params=adam_params, use_muon=False, lr=adam_lr, betas=(0.9, 0.95),
             weight_decay=weight_decay),
    ])
    return opt, muon_n, adam_n


def build_polar_normuon_optimizer(model, muon_lr, adam_lr, weight_decay):
    from polar_normuon import SingleDeviceNorMuonPolarWithAuxAdam

    muon_params, adam_params, muon_n, adam_n = _split_muon_adam_params(model)
    opt = SingleDeviceNorMuonPolarWithAuxAdam(
        [
            dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=weight_decay,
                 momentum=0.95, beta2=0.95),
            dict(params=adam_params, use_muon=False, lr=adam_lr, betas=(0.9, 0.95),
                 weight_decay=weight_decay),
        ],
        cautious_wd=True,
        compile_polar=False,  # safer on Windows / 8GB
    )
    return opt, muon_n, adam_n


def prepare_soft_batch(data, indices, device, hflip_p=0.0, rng=None):
    from data_loader import (
        board_array_to_fused, compute_wdl, ep_square_to_file,
        hflip_board_array, hflip_ep_square, hflip_move_idx,
    )

    ba = data["board_array"][indices].clone()
    turn = data["turn"][indices].clone()
    castling = data["castling"][indices].clone()
    ep = data["ep_square"][indices].clone()
    move_idx = data["move_idx"][indices].clone()
    cp = data["cp"][indices]
    mate = data["mate"][indices]
    soft_i = data["soft_indices"][indices].clone()
    soft_p = data["soft_probs"][indices].clone()

    if hflip_p > 0:
        flip_mask = torch.rand(ba.shape[0], generator=rng) < hflip_p
        if flip_mask.any():
            ba[flip_mask] = hflip_board_array(ba[flip_mask])
            move_idx[flip_mask] = hflip_move_idx(move_idx[flip_mask]).to(move_idx.dtype)
            castling[flip_mask] = 0
            ep[flip_mask] = hflip_ep_square(ep[flip_mask]).to(ep.dtype)
            si = soft_i[flip_mask]
            valid = si >= 0
            if valid.any():
                si2 = si.clone()
                si2[valid] = hflip_move_idx(si[valid]).to(si.dtype)
                soft_i[flip_mask] = si2

    nb = device.type == "cuda"
    board_input = {
        "fused_ids": board_array_to_fused(ba).to(device, non_blocking=nb),
        "turn": turn.long().to(device, non_blocking=nb),
        "castling": castling.long().to(device, non_blocking=nb),
        "ep_file": ep_square_to_file(ep).long().to(device, non_blocking=nb),
    }
    wdl = compute_wdl(cp, mate).to(device, non_blocking=nb)
    return (
        board_input,
        move_idx.long().to(device, non_blocking=nb),
        wdl,
        soft_i.to(device, non_blocking=nb),
        soft_p.to(device, non_blocking=nb),
    )


def resolve_trial_config(raw: dict, space: dict) -> dict:
    """Expand inherits / overrides into a concrete trial dict."""
    if "inherits" not in raw:
        return {
            "id": raw["id"],
            "desc": raw.get("desc", ""),
            "model": dict(raw["model"]),
            "train": dict(raw["train"]),
        }
    base = None
    for t in space["trials"]:
        if t["id"] == raw["inherits"]:
            base = resolve_trial_config(t, space) if "inherits" in t else t
            break
    if base is None:
        raise KeyError(f"inherits unknown trial {raw['inherits']}")
    model = dict(base["model"])
    train = dict(base["train"])
    model.update(raw.get("model_overrides") or {})
    train.update(raw.get("train_overrides") or {})
    return {"id": raw["id"], "desc": raw.get("desc", ""), "model": model, "train": train}


def find_cache(candidates: list[str]) -> Path | None:
    for c in candidates:
        p = ROOT / c if not Path(c).is_absolute() else Path(c)
        if p.exists():
            return p
    return None


def train_trial(
    trial: dict[str, Any],
    out_dir: Path,
    *,
    soft_cache: Path | None,
    deep_cache: Path | None = None,
    max_steps: int = 3000,
    max_minutes: float = 45.0,
    smoke: bool = False,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train one trial. Returns metrics dict including ckpt_path / pos_s / status."""
    os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from chess_transformer_factory import ChessTransformerConfig, build_model, count_parameters
    from data_loader import stream_hf_batches
    from move_vocab import VOCAB_SIZE

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    cfg_path = out_dir / "config.json"

    train = dict(trial["train"])
    model_kw = dict(trial["model"])
    if "grad_checkpoint" in train:
        model_kw["gradient_checkpointing"] = bool(train["grad_checkpoint"])

    if smoke:
        max_steps = min(max_steps, 15)
        max_minutes = min(max_minutes, 3.0)
        train["batch_size"] = min(int(train.get("batch_size", 32)), 8)
        train["accum_steps"] = 1
        train["warmup"] = 2
        train["elo_every_steps"] = 0
        train["torch_compile"] = False

    model_cfg = ChessTransformerConfig(**{
        k: v for k, v in model_kw.items()
        if k in ChessTransformerConfig.__dataclass_fields__
    })
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"model": asdict(model_cfg), "train": train, "trial": trial}, f, indent=2)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    _log(log_path, f"trial={trial['id']} device={device} vocab={VOCAB_SIZE}")
    _log(
        log_path,
        f"train bs={train.get('batch_size')} accum={train.get('accum_steps')} "
        f"grad_ckpt={model_kw.get('gradient_checkpointing')} "
        f"compile={train.get('torch_compile')} elo_every={train.get('elo_every_steps', 0)}",
    )
    _log(log_path, f"model={model_cfg}")

    try:
        model = build_model(model_cfg).to(device)
    except torch.cuda.OutOfMemoryError as e:
        return {"status": "oom", "error": str(e), "pos_s": 0.0, "ckpt_path": None}

    n_params = count_parameters(model)
    _log(log_path, f"params={n_params/1e6:.2f}M")

    opt_name = train.get("optimizer", "normuon")
    try:
        if opt_name == "adamw":
            optimizer = AdamW(
                model.parameters(),
                lr=float(train.get("adam_lr", 3e-4)),
                weight_decay=float(train.get("weight_decay", 0.01)),
            )
            _log(log_path, "optimizer=AdamW")
        elif opt_name == "polar_normuon":
            optimizer, muon_n, adam_n = build_polar_normuon_optimizer(
                model,
                float(train.get("muon_lr", 0.02)),
                float(train.get("adam_lr", 3e-4)),
                float(train.get("weight_decay", 0.01)),
            )
            _log(log_path, f"optimizer=PolarNorMuon ({muon_n/1e6:.1f}M) + AdamW aux ({adam_n/1e6:.1f}M)")
        else:
            optimizer, muon_n, adam_n = build_normuon_optimizer(
                model,
                float(train.get("muon_lr", 0.02)),
                float(train.get("adam_lr", 3e-4)),
                float(train.get("weight_decay", 0.01)),
            )
            _log(log_path, f"optimizer=NorMuon ({muon_n/1e6:.1f}M) + AdamW aux ({adam_n/1e6:.1f}M)")
    except ImportError as e:
        optimizer = AdamW(
            model.parameters(),
            lr=float(train.get("adam_lr", 3e-4)),
            weight_decay=float(train.get("weight_decay", 0.01)),
        )
        _log(log_path, f"optimizer fallback AdamW ({e})")

    compiled = False
    if train.get("torch_compile") and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
            compiled = True
            _log(log_path, "torch.compile enabled (mode=default); will fall back if inductor fails")
        except Exception as e:
            _log(log_path, f"torch.compile skipped: {e}")

    soft_data = None
    deep_data = None
    train_soft_n = train_deep_n = 0
    if soft_cache and soft_cache.exists():
        soft_data = torch.load(soft_cache, map_location="cpu", weights_only=False)
        n = soft_data["board_array"].shape[0]
        hold = min(2000, max(512, n // 40))
        train_soft_n = max(1, n - hold)
        _log(log_path, f"soft train={train_soft_n:,}")
    elif not smoke:
        return {
            "status": "failed",
            "error": f"soft cache required: {soft_cache}",
            "pos_s": 0.0,
            "ckpt_path": None,
            "n_params": n_params,
        }

    if deep_cache and deep_cache.exists():
        deep_data = torch.load(deep_cache, map_location="cpu", weights_only=False)
        n = deep_data["board_array"].shape[0]
        hold = min(1000, max(256, n // 40))
        train_deep_n = max(1, n - hold)
        _log(log_path, f"deep soft train={train_deep_n:,}")

    hard_iter = None
    if float(train.get("soft_frac", 1.0)) < 1.0 and not smoke:
        hard_iter = iter(stream_hf_batches(
            batch_size=int(train["batch_size"]), device=device, seed=42,
            shuffle_buffer=2048, min_depth=int(train.get("min_depth", 12)),
        ))

    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    warmup = int(train.get("warmup", 200))
    min_lr_frac = float(train.get("min_lr_frac", 0.05))
    bs = int(train["batch_size"])
    accum = int(train.get("accum_steps", 1))
    # Fit largest batch that works without grad-checkpoint on 8GB.
    # Also detects torch.compile/inductor failures (common on Windows without MSVC).
    if device.type == "cuda" and not smoke:
        while bs >= 32:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                probe = {
                    "fused_ids": torch.randint(0, 13, (bs, 64), device=device),
                    "turn": torch.zeros(bs, dtype=torch.long, device=device),
                    "castling": torch.zeros(bs, dtype=torch.long, device=device),
                    "ep_file": torch.zeros(bs, dtype=torch.long, device=device),
                }
                model.train()
                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(probe)
                    loss = out["policy_logits"].float().mean() + out["value_logits"].float().mean()
                loss.backward()
                optimizer.zero_grad(set_to_none=True)
                peak = torch.cuda.max_memory_allocated() / 1e9
                vram_cap = float(train.get("max_vram_gb", 6.8))
                if peak > vram_cap:
                    new_bs = max(32, (bs * 3) // 4)
                    _log(
                        log_path,
                        f"batch probe high vram bs={bs} peak={peak:.2f}GB > {vram_cap}; retry bs={new_bs}",
                    )
                    torch.cuda.empty_cache()
                    if new_bs >= bs:
                        break
                    bs = new_bs
                    continue
                _log(log_path, f"batch probe ok bs={bs} peak_vram={peak:.2f}GB compile={compiled}")
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                new_bs = max(32, (bs * 3) // 4)
                _log(log_path, f"batch probe OOM at bs={bs}; retry bs={new_bs}")
                if new_bs >= bs:
                    bs = 32
                    break
                bs = new_bs
            except Exception as e:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                msg = str(e)
                if compiled and hasattr(model, "_orig_mod"):
                    model = model._orig_mod
                    compiled = False
                    _log(log_path, f"torch.compile disabled after runtime error: {msg[:200]}")
                    continue
                raise
        train["batch_size"] = bs
    soft_frac = float(train.get("soft_frac", 0.85))
    soft_alpha = float(train.get("soft_alpha", 0.45))
    deep_mix = float(train.get("deep_mix_frac", 0.35))
    hflip_p = float(train.get("hflip_p", 0.5))
    value_weight = float(train.get("value_weight", 0.1))
    grad_clip = float(train.get("grad_clip", 1.0))
    # Chessformer (Monroe & Chalmers, arXiv:2409.12272) soft-policy + SWA
    soft_temp = float(train.get("soft_temp", 0.0))  # 0 = off; paper uses 4
    soft_temp_weight = float(train.get("soft_temp_weight", 0.5))
    use_swa = bool(train.get("use_swa", False))
    swa_start_frac = float(train.get("swa_start_frac", 0.75))
    label_smoothing = float(train.get("label_smoothing", 0.0))
    elo_every = int(train.get("elo_every_steps", 0) or 0)
    elo_history_path = out_dir / "elo_gauntlet.jsonl"

    def set_lrs(step: int) -> None:
        if step < warmup:
            scale = (step + 1) / max(warmup, 1)
        else:
            progress = (step - warmup) / max(max_steps - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            scale = min_lr_frac + (1.0 - min_lr_frac) * cosine
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base * scale

    model.train()
    optimizer.zero_grad(set_to_none=True)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(hash(trial["id"]) % (2**31 - 1))
    t0 = time.time()
    # Active-time budget: ignore laptop sleep gaps (>120s between steps).
    active_budget_s = max_minutes * 60.0
    active_elapsed = 0.0
    last_tick = t0
    step = 0
    positions = 0
    peak_vram = 0.0
    swa_state: dict[str, torch.Tensor] | None = None
    swa_n = 0
    swa_start_step = int(max_steps * swa_start_frac) if use_swa else max_steps + 1
    if soft_temp > 0:
        _log(log_path, f"chessformer soft_temp={soft_temp} weight={soft_temp_weight}")
    if use_swa:
        _log(log_path, f"chessformer SWA from step {swa_start_step}/{max_steps}")

    stop_path = out_dir / "STOP"
    interrupted = False

    def _save_ckpt(status: str) -> dict[str, Any]:
        wall = max(time.time() - t0, 1e-6)
        elapsed = max(active_elapsed, 1e-6)
        pos_s = positions / elapsed
        _log(log_path, f"timing wall={wall:.0f}s active={elapsed:.0f}s steps={step} status={status}")
        ckpt_path = out_dir / "latest.pt"
        to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        state = to_save.state_dict()
        if use_swa and swa_state is not None and swa_n > 0:
            state = swa_state
            _log(log_path, f"saving SWA weights (n={swa_n})")
        torch.save({
            "model_state_dict": state,
            "config": asdict(model_cfg),
            "vocab": "compact",
            "vocab_version": "compact",
            "trial_id": trial["id"],
            "steps": step,
            "pos_s": pos_s,
            "n_params": n_params,
            "swa_n": swa_n if use_swa else 0,
            "status": status,
        }, ckpt_path)
        with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(model_cfg), f, indent=2)
        _log(log_path, f"done steps={step} pos_s={pos_s:.1f} params={n_params/1e6:.2f}M status={status}")
        return {
            "status": status,
            "ckpt_path": str(ckpt_path),
            "model_config_path": str(out_dir / "model_config.json"),
            "pos_s": pos_s,
            "peak_vram_gb": peak_vram,
            "steps": step,
            "n_params": n_params,
            "elapsed_s": elapsed,
        }

    try:
        while step < max_steps and active_elapsed < active_budget_s:
            if stop_path.exists():
                interrupted = True
                _log(log_path, f"STOP file seen at step {step}; saving and exiting")
                break
            now = time.time()
            gap = now - last_tick
            if gap < 120.0:
                active_elapsed += gap
            last_tick = now
            use_soft = soft_data is not None and (
                smoke or torch.rand(1, generator=rng).item() < soft_frac
            )
            for _ in range(accum):
                if use_soft and soft_data is not None:
                    use_deep = (
                        deep_data is not None and train_deep_n > 0
                        and torch.rand(1, generator=rng).item() < deep_mix
                    )
                    src = deep_data if use_deep else soft_data
                    n_src = train_deep_n if use_deep else train_soft_n
                    idx = torch.randint(0, n_src, (bs,), generator=rng)
                    bi, hard, wdl, si, sp = prepare_soft_batch(
                        src, idx, device, hflip_p=hflip_p, rng=rng,
                    )
                    with autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        out = model(bi)
                        hard_ce = F.cross_entropy(
                            out["policy_logits"], hard, label_smoothing=label_smoothing,
                        )
                        soft_ce = soft_policy_loss(out["policy_logits"], si, sp)
                        p_loss = (1.0 - soft_alpha) * hard_ce + soft_alpha * soft_ce
                        if soft_temp > 0:
                            p_loss = p_loss + soft_temp_weight * soft_temp_policy_loss(
                                out["policy_logits"], si, sp, temperature=soft_temp,
                            )
                        v_loss = F.cross_entropy(out["value_logits"], wdl)
                        loss = (p_loss + value_weight * v_loss) / accum
                    loss.backward()
                elif smoke:
                    bi = {
                        "fused_ids": torch.randint(0, 13, (bs, 64), device=device),
                        "turn": torch.zeros(bs, dtype=torch.long, device=device),
                        "castling": torch.zeros(bs, dtype=torch.long, device=device),
                        "ep_file": torch.zeros(bs, dtype=torch.long, device=device),
                    }
                    hard = torch.randint(0, VOCAB_SIZE, (bs,), device=device)
                    wdl = torch.randint(0, 3, (bs,), device=device)
                    with autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        out = model(bi)
                        loss = (
                            F.cross_entropy(
                                out["policy_logits"], hard, label_smoothing=label_smoothing,
                            )
                            + value_weight * F.cross_entropy(out["value_logits"], wdl)
                        ) / accum
                    loss.backward()
                else:
                    try:
                        bi, move_t, wdl_t = next(hard_iter)
                    except StopIteration:
                        hard_iter = iter(stream_hf_batches(
                            batch_size=bs, device=device, seed=43 + step,
                            shuffle_buffer=2048, min_depth=int(train.get("min_depth", 12)),
                        ))
                        bi, move_t, wdl_t = next(hard_iter)
                    with autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        out = model(bi)
                        loss = (
                            F.cross_entropy(
                                out["policy_logits"], move_t, label_smoothing=label_smoothing,
                            )
                            + value_weight * F.cross_entropy(out["value_logits"], wdl_t)
                        ) / accum
                    loss.backward()
                positions += bs

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            set_lrs(step + 1)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if use_swa and step >= swa_start_step:
                to_avg = model._orig_mod if hasattr(model, "_orig_mod") else model
                if swa_state is None:
                    swa_state = {
                        k: v.detach().clone() for k, v in to_avg.state_dict().items()
                    }
                    swa_n = 1
                else:
                    swa_n = _ema_update(swa_state, to_avg, swa_n)
            if device.type == "cuda":
                peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)
            if step % 25 == 0 or step == 1:
                elapsed = max(time.time() - t0, 1e-6)
                _log(log_path, f"step {step}/{max_steps} | {positions/elapsed:.0f} pos/s | vram={peak_vram:.2f}GB")
            if elo_every > 0 and step % elo_every == 0:
                mid = _save_ckpt("mid_elo")
                _log(log_path, f"elo gauntlet at step {step} -> {mid['ckpt_path']}")
                try:
                    from autoresearch_8gb.elo_trial import run_elo_trial
                    elo_result = run_elo_trial(
                        mid["ckpt_path"],
                        f"ar8gb_{trial['id']}_s{step}",
                        model_config=mid.get("model_config_path"),
                        smoke=False,
                    )
                    row = {
                        "step": step,
                        "elo": elo_result.get("elo"),
                        "estimate": elo_result.get("estimate"),
                        "rc": elo_result.get("rc"),
                        "json_path": elo_result.get("json_path"),
                        "at": datetime.now().isoformat(timespec="seconds"),
                    }
                    with open(elo_history_path, "a", encoding="utf-8") as ef:
                        ef.write(json.dumps(row) + "\n")
                    _log(
                        log_path,
                        f"elo@{step} estimate={elo_result.get('elo')} rc={elo_result.get('rc')}",
                    )
                except Exception as e:
                    _log(log_path, f"elo@{step} failed: {e}")
                model.train()
                last_tick = time.time()  # don't charge gauntlet wall time to train budget
    except KeyboardInterrupt:
        interrupted = True
        _log(log_path, f"KeyboardInterrupt at step {step}; saving and exiting")
    except torch.cuda.OutOfMemoryError as e:
        return {
            "status": "oom",
            "error": str(e),
            "pos_s": positions / max(time.time() - t0, 1e-6),
            "ckpt_path": None,
            "n_params": n_params,
            "steps": step,
        }

    if stop_path.exists():
        try:
            stop_path.unlink()
        except OSError:
            pass
    return _save_ckpt("interrupted" if interrupted else "trained")
