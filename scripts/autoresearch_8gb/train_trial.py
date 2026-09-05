"""Train one autoresearch trial on ~25M chess transformer (8GB-friendly)."""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
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
import sys
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ADAM_NAME_HINTS = (
    "embed", "policy_head", "value_head", "cls_token", "cls_pos",
    "pos_embed", "norm", "bn", "shaw_", "rel_bias", "film",
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device):
    """BF16 on CUDA; disable AMP on MPS (view/stride autograd bugs)."""
    if device.type == "cuda":
        return autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _log(path: Path | None, msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path is not None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


from autoresearch_8gb.pipeline import (  # noqa: E402
    apply_membership,
    attach_static_targets,
    audit_soft_targets,
    cheap_eval_losses,
    classify_checkpoint,
    exposure_report,
    legal_policy_diagnostics,
    load_model_state,
    lr_scale,
    make_val_membership,
    muon_update_scale_note,
    prepare_soft_batch,
    restore_rng_state,
    save_training_checkpoint,
    soft_policy_loss,
    soft_temp_policy_loss,
    write_shard_manifest,
)


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


def build_polar_normuon_optimizer(model, muon_lr, adam_lr, weight_decay, compile_polar=True):
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
        compile_polar=compile_polar,
    )
    return opt, muon_n, adam_n


def _is_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    msg = str(err).lower()
    return "out of memory" in msg or "oom" in msg


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
    resume_ckpt: Path | None = None,
) -> dict[str, Any]:
    """Train one trial. Returns metrics dict including ckpt_path / pos_s / status."""
    os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)

    from data_loader import stream_hf_batches
    from move_vocab import VOCAB_SIZE

    device = device or pick_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    cfg_path = out_dir / "config.json"

    train = dict(trial["train"])
    model_kw = dict(trial["model"])
    if "grad_checkpoint" in train:
        model_kw["gradient_checkpointing"] = bool(train["grad_checkpoint"])

    arch = str(trial.get("arch") or train.get("arch") or "factory")
    average_recurrent_grads = None
    if arch == "squares64":
        from chess_squares64 import (
            Squares64RecurrentConfig,
            average_recurrent_grads as _avg_rec_grads,
            build_squares64,
            count_parameters,
        )
        average_recurrent_grads = _avg_rec_grads
        cfg_cls = Squares64RecurrentConfig
        model_builder = build_squares64
    else:
        from chess_transformer_factory import ChessTransformerConfig, build_model, count_parameters
        cfg_cls = ChessTransformerConfig
        model_builder = build_model

    if smoke:
        max_steps = min(max_steps, 15)
        max_minutes = min(max_minutes, 3.0)
        train["batch_size"] = min(int(train.get("batch_size", 32)), 8)
        train["accum_steps"] = 1
        train["warmup"] = 2
        train["elo_every_steps"] = 0
        train["torch_compile"] = False

    # MPS / Apple unified memory (e.g. M5 Pro 24GB): fat micro-batches are fine;
    # only clamp extreme CUDA-oriented sizes. torch.compile still flaky on MPS.
    if device.type == "mps" and not smoke:
        bs = int(train.get("batch_size", 96))
        mps_cap = int(train.get("mps_batch_cap", 192))
        if bs > mps_cap:
            scale = math.ceil(bs / mps_cap)
            train["batch_size"] = mps_cap
            train["accum_steps"] = int(train.get("accum_steps", 1)) * scale
        if train.get("torch_compile"):
            train["torch_compile"] = False

    model_cfg = cfg_cls(**{
        k: v for k, v in model_kw.items()
        if k in cfg_cls.__dataclass_fields__
    })
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"arch": arch, "model": asdict(model_cfg), "train": train, "trial": trial}, f, indent=2)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    _log(log_path, f"trial={trial['id']} arch={arch} device={device} vocab={VOCAB_SIZE}")
    _log(
        log_path,
        f"train bs={train.get('batch_size')} accum={train.get('accum_steps')} "
        f"grad_ckpt={model_kw.get('gradient_checkpointing')} "
        f"compile={train.get('torch_compile')} elo_every={train.get('elo_every_steps', 0)} "
        f"save_every={train.get('save_every_steps', 0)}",
    )
    _log(log_path, f"model={model_cfg}")

    try:
        model = model_builder(model_cfg).to(device)
    except Exception as e:
        if _is_oom(e):
            return {"status": "oom", "error": str(e), "pos_s": 0.0, "ckpt_path": None}
        raise

    n_params = count_parameters(model)
    _log(log_path, f"params={n_params/1e6:.2f}M")

    start_step = 0
    resume_kind = "fresh"
    resume_payload: dict[str, Any] | None = None
    if resume_ckpt is not None:
        resume_ckpt = Path(resume_ckpt)
        if not resume_ckpt.exists():
            raise FileNotFoundError(f"resume ckpt missing: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        if ckpt.get("eval_only"):
            raise SystemExit(
                f"{resume_ckpt} is an evaluation/SWA snapshot, not a training resume file. "
                "Use latest.pt or known_good.pt (live weights)."
            )
        model.load_state_dict(load_model_state(ckpt), strict=True)
        start_step = int(ckpt.get("steps", ckpt.get("global_step", 0)) or 0)
        resume_kind = classify_checkpoint(ckpt)
        resume_payload = ckpt if isinstance(ckpt, dict) else None
        if resume_kind == "full":
            _log(log_path, f"FULL RESUME {resume_ckpt} steps={start_step}")
        else:
            _log(
                log_path,
                f"WEIGHTS-ONLY WARM START {resume_ckpt} steps={start_step} "
                f"(no optimizer/RNG in checkpoint; optimizer is re-initialized; "
                f"LR follows this process's train config)",
            )

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
            compile_polar = bool(train.get("compile_polar", device.type == "cuda"))
            optimizer, muon_n, adam_n = build_polar_normuon_optimizer(
                model,
                float(train.get("muon_lr", 0.02)),
                float(train.get("adam_lr", 3e-4)),
                float(train.get("weight_decay", 0.01)),
                compile_polar=compile_polar,
            )
            _log(
                log_path,
                f"optimizer=PolarNorMuon ({muon_n/1e6:.1f}M) + AdamW aux ({adam_n/1e6:.1f}M) "
                f"compile_polar={compile_polar}",
            )
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

    if resume_kind == "full" and resume_payload and resume_payload.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            _log(log_path, "optimizer state restored")
        except Exception as e:
            resume_kind = "weights_only"
            _log(log_path, f"optimizer state load failed ({e}); continuing as weights-only warm start")

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
    train_soft_idx = train_deep_idx = None
    val_soft_idx = val_deep_idx = None
    train_soft_n = train_deep_n = 0
    val_manifests: dict[str, Any] = {}
    if soft_cache and soft_cache.exists():
        soft_data = torch.load(soft_cache, map_location="cpu", weights_only=False)
        attach_static_targets(soft_data)
        n = int(soft_data["board_array"].shape[0])
        hold = min(2000, max(64 if smoke else 512, n // 40), max(0, n // 5))
        man_path = out_dir / "val_manifest_soft.json"
        if man_path.exists() and not smoke:
            val_manifests["soft"] = json.loads(man_path.read_text(encoding="utf-8"))
        else:
            val_manifests["soft"] = make_val_membership(soft_data, n_hold=hold, seed=201, source="soft")
            man_path.write_text(json.dumps(val_manifests["soft"], indent=2), encoding="utf-8")
        train_soft_idx, val_soft_idx = apply_membership(soft_data, val_manifests["soft"])
        train_soft_n = int(train_soft_idx.numel())
        audit = audit_soft_targets(soft_data)
        _log(
            log_path,
            f"soft train={train_soft_n:,} val={int(val_soft_idx.numel()):,} "
            f"blocked={val_manifests['soft']['n_blocked']} "
            f"targets empty={audit['empty_rows']} unnorm={audit['unnormalized_rows']}",
        )
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
        attach_static_targets(deep_data)
        n = int(deep_data["board_array"].shape[0])
        hold = min(1000, max(64 if smoke else 256, n // 40))
        man_path = out_dir / "val_manifest_deep.json"
        if man_path.exists() and not smoke:
            val_manifests["deep"] = json.loads(man_path.read_text(encoding="utf-8"))
        else:
            val_manifests["deep"] = make_val_membership(deep_data, n_hold=hold, seed=202, source="deep")
            man_path.write_text(json.dumps(val_manifests["deep"], indent=2), encoding="utf-8")
        train_deep_idx, val_deep_idx = apply_membership(deep_data, val_manifests["deep"])
        train_deep_n = int(train_deep_idx.numel())
        _log(log_path, f"deep train={train_deep_n:,} val={int(val_deep_idx.numel()):,}")

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
    # Fit batch to VRAM. Shrinks on OOM / over-cap; grows when fill_vram is set.
    # Also detects torch.compile/inductor failures (common on Windows without MSVC).
    if device.type == "cuda" and not smoke:
        min_bs = int(train.get("min_batch_size", 32))
        max_bs = int(train.get("max_batch_size", 2048))
        vram_cap = float(train.get("max_vram_gb", 6.8))
        fill_vram = bool(train.get("fill_vram", False))
        grew = False
        while bs >= min_bs:
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
                if peak > vram_cap:
                    new_bs = max(min_bs, (bs * 3) // 4)
                    _log(
                        log_path,
                        f"batch probe high vram bs={bs} peak={peak:.2f}GB > {vram_cap}; retry bs={new_bs}",
                    )
                    torch.cuda.empty_cache()
                    if new_bs >= bs:
                        break
                    bs = new_bs
                    if grew:
                        _log(log_path, f"batch probe settle bs={bs} after fill overshoot")
                        break
                    continue
                if fill_vram and peak < vram_cap * 0.78 and bs < max_bs:
                    scale = (vram_cap * 0.90) / max(peak, 0.1)
                    new_bs = min(max_bs, max(bs + 16, int(bs * scale)))
                    new_bs = max(min_bs, (new_bs // 8) * 8)
                    if new_bs > bs:
                        _log(
                            log_path,
                            f"batch probe fill_vram bs={bs} peak={peak:.2f}GB < {vram_cap}; grow bs={new_bs}",
                        )
                        torch.cuda.empty_cache()
                        bs = new_bs
                        grew = True
                        continue
                _log(log_path, f"batch probe ok bs={bs} peak_vram={peak:.2f}GB compile={compiled}")
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                new_bs = max(min_bs, (bs * 3) // 4)
                _log(log_path, f"batch probe OOM at bs={bs}; retry bs={new_bs}")
                if new_bs >= bs:
                    bs = min_bs
                    break
                bs = new_bs
                if grew:
                    _log(log_path, f"batch probe settle bs={bs} after fill OOM")
                    break
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
    save_every = int(train.get("save_every_steps", 0) or 0)
    keep_step_every = int(train.get("keep_step_every", 0) or 0)
    keep_last = int(train.get("keep_last_ckpts", 4))
    elo_history_path = out_dir / "elo_gauntlet.jsonl"
    val_every = int(train.get("val_every_steps", 500) or 0)
    legal_every = int(train.get("legal_eval_every", 0) or 0)
    train["max_steps"] = max_steps
    train["batch_size"] = bs

    def set_lrs(step: int) -> None:
        scale = lr_scale(step, warmup=warmup, max_steps=max_steps, min_lr_frac=min_lr_frac)
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base * scale

    model.train()
    optimizer.zero_grad(set_to_none=True)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(hash(trial["id"]) % (2**31 - 1))
    if resume_kind == "full" and resume_payload and resume_payload.get("rng"):
        restore_rng_state(resume_payload["rng"], sampler_rng=rng)
        _log(log_path, "RNG state restored")
    t0 = time.time()
    # Active-time budget: ignore laptop sleep gaps (>120s between steps).
    active_budget_s = max_minutes * 60.0
    active_elapsed = 0.0
    last_tick = t0
    step = start_step
    positions = int((resume_payload or {}).get("positions") or 0)
    peak_vram = 0.0
    window_loss_t: torch.Tensor | None = None
    window_n = 0
    clip_hits = 0
    shallow_seen = 0
    deep_seen = 0
    swa_state: dict[str, torch.Tensor] | None = None
    swa_n = 0
    swa_start_step = int(max_steps * swa_start_frac) if use_swa else max_steps + 1
    if soft_temp > 0:
        _log(log_path, f"chessformer soft_temp={soft_temp} weight={soft_temp_weight}")
    if use_swa:
        _log(log_path, f"chessformer SWA from step {swa_start_step}/{max_steps} (eval_swa.pt only)")
    _log(log_path, muon_update_scale_note())
    write_shard_manifest(
        out_dir / "dataset_manifest.json",
        live=Path(soft_cache) if soft_cache else out_dir,
        shards=[],
        val_manifests=val_manifests,
        notes="Live process tensors; READY shards are NOT auto-merged.",
    )
    exp = exposure_report(
        shallow_n=train_soft_n, deep_n=train_deep_n, deep_mix_frac=deep_mix,
        shallow_seen=0, deep_seen=0,
    )
    _log(
        log_path,
        f"exposure deep/shallow odds={exp['deep_vs_shallow_odds']:.1f}x "
        f"(mix={deep_mix} deep_n={train_deep_n} shallow_n={train_soft_n})",
    )

    stop_path = out_dir / "STOP"
    interrupted = False

    def _save_ckpt(status: str, *, also_step: bool = False) -> dict[str, Any]:
        wall = max(time.time() - t0, 1e-6)
        elapsed = max(active_elapsed, 1e-6)
        pos_s = positions / elapsed if positions else 0.0
        ckpt_path = out_dir / "latest.pt"
        known = out_dir / "known_good.pt"
        eval_path = out_dir / "eval_swa.pt"
        save_training_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            step=step,
            positions=positions,
            model_cfg=asdict(model_cfg),
            train_cfg=train,
            trial_id=trial["id"],
            arch=arch,
            n_params=n_params,
            sampler_rng=rng,
            manifest=val_manifests,
            swa_state=swa_state,
            swa_n=swa_n,
            resume_kind="full",
            status=status,
            extra={"pos_s": pos_s},
            eval_path=eval_path if swa_n > 0 else None,
            known_good_path=known if also_step or status in ("trained", "interrupted") else None,
            include_cuda_rng=device.type == "cuda",
        )
        with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(model_cfg), f, indent=2)
        if also_step and step > 0:
            step_path = out_dir / f"step_{step:06d}.pt"
            shutil.copy2(ckpt_path, step_path)
            kept = sorted(out_dir.glob("step_*.pt"))
            for old in kept[:-max(keep_last, 1)]:
                try:
                    old.unlink()
                except OSError:
                    pass
            _log(log_path, f"ckpt {step_path.name} + latest.pt (live) steps={step} status={status} kind=full")
        elif status in ("trained", "interrupted"):
            _log(log_path, f"timing wall={wall:.0f}s active={elapsed:.0f}s steps={step} status={status}")
            _log(log_path, f"done steps={step} pos_s={pos_s:.1f} params={n_params/1e6:.2f}M status={status} kind=full")
        else:
            _log(log_path, f"ckpt latest.pt (live weights) steps={step} status={status}")
            if swa_n > 0:
                _log(log_path, f"eval_swa.pt n={swa_n}")
        return {
            "status": status,
            "ckpt_path": str(ckpt_path),
            "model_config_path": str(out_dir / "model_config.json"),
            "pos_s": pos_s,
            "peak_vram_gb": peak_vram,
            "steps": step,
            "n_params": n_params,
            "elapsed_s": elapsed,
            "resume_kind": "full",
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
                    pool = train_deep_idx if use_deep else train_soft_idx
                    local = torch.randint(0, int(pool.numel()), (bs,), generator=rng)
                    idx = pool[local]
                    if use_deep:
                        deep_seen += bs
                    else:
                        shallow_seen += bs
                    bi, hard, wdl, si, sp = prepare_soft_batch(
                        src, idx, device, hflip_p=hflip_p, rng=rng,
                    )
                    with amp_context(device):
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
                    # Search-value-head aux: learn the backed-up best-child scalar
                    # against the game-result scalar (Stockfish-style retrogression).
                    if "searched_value" in out:
                        # wdl is a (B, 3) probability distribution.
                        scalar_t = wdl[:, 0] - wdl[:, 2]
                        sv_loss = F.mse_loss(out["searched_value"], scalar_t)
                        loss = loss + 0.1 * sv_loss
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
                    with amp_context(device):
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
                    with amp_context(device):
                        out = model(bi)
                        loss = (
                            F.cross_entropy(
                                out["policy_logits"], move_t, label_smoothing=label_smoothing,
                            )
                            + value_weight * F.cross_entropy(out["value_logits"], wdl_t)
                        ) / accum
                    loss.backward()
                positions += bs
                det = loss.detach()
                window_loss_t = det if window_loss_t is None else window_loss_t + det
                window_n += 1

            if average_recurrent_grads is not None:
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                average_recurrent_grads(raw)
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            try:
                clip_hits += int(float(total_norm) > grad_clip)
            except TypeError:
                clip_hits += 0
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
                avg_loss = float((window_loss_t or torch.zeros(1)).float().item()) * accum / max(window_n, 1)
                finite = math.isfinite(avg_loss)
                lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
                _log(
                    log_path,
                    f"step {step}/{max_steps} | loss={avg_loss:.4f} | "
                    f"lr={','.join(lrs)} | clip={clip_hits}/25 | "
                    f"mix s/d={shallow_seen}/{deep_seen} | "
                    f"{positions/max(elapsed, 1e-6):.0f} pos/s | vram={peak_vram:.2f}GB"
                    + ("" if finite else " NON-FINITE"),
                )
                window_loss_t = None
                window_n = 0
                clip_hits = 0
                if not finite:
                    interrupted = True
                    _log(log_path, f"stopping: non-finite loss at step {step}")
                    break
            if val_every > 0 and step % val_every == 0 and not smoke:
                raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
                if val_soft_idx is not None and val_soft_idx.numel():
                    take = val_soft_idx[torch.arange(min(256, int(val_soft_idx.numel())))]
                    metrics = cheap_eval_losses(raw_m, soft_data, take, device, soft_temp=soft_temp or 4.0)
                    _log(log_path, "val/soft " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                if val_deep_idx is not None and val_deep_idx.numel() and deep_data is not None:
                    take = val_deep_idx[torch.arange(min(256, int(val_deep_idx.numel())))]
                    metrics = cheap_eval_losses(raw_m, deep_data, take, device, soft_temp=soft_temp or 4.0)
                    _log(log_path, "val/deep " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                if legal_every > 0 and step % legal_every == 0 and val_soft_idx is not None:
                    take = val_soft_idx[torch.arange(min(48, int(val_soft_idx.numel())))]
                    diag = legal_policy_diagnostics(raw_m, soft_data, take, device)
                    _log(log_path, "val/legal " + " ".join(f"{k}={v:.4f}" for k, v in diag.items()))
                model.train()
                last_tick = time.time()
            if save_every > 0 and step % save_every == 0:
                _save_ckpt(
                    "mid",
                    also_step=keep_step_every > 0 and step % keep_step_every == 0,
                )
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
    except Exception as e:
        if not _is_oom(e):
            raise
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
