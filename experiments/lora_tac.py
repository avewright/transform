#!/usr/bin/env python3
"""LoRA fine-tune the 437M meta-attention transformer on the 60k puzzle cache.

Efficient-M5 design:
  * Wrap only the attention projections (q_c,k_c,v_c,q_p,k_p,out_proj) with LoRA
    (rank=16): trains a tiny fraction of 437M -> fast steps / low memory on MPS.
  * Freeze all base weights + everything else; only LoRA adapters train.
  * Validates on the SF tactical test set (avg cp cost of picked vs SF best).

If LoRA closes the tactical gap much faster than all-params, it's the M5 path.

Usage:
  python experiments/lora_tac.py --steps 2000 --lr 5e-4
"""
from __future__ import annotations
import os, sys, argparse, json, time, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
import chess, torch
import torch.nn.functional as F
from torch.optim import AdamW
from chess_transformer_factory import build_model, ChessTransformerConfig
from move_vocab import legal_move_mask, IDX_TO_UCI
from chess_features import batch_boards_to_fused_token_ids

TEST = ROOT / "data" / "sf_test_500.jsonl"


class LoRALinear(torch.nn.Module):
    """LoRA adapter wrapping an existing nn.Linear (base frozen)."""
    def __init__(self, original, rank=16, alpha=32.0, dropout=0.0):
        super().__init__()
        self.original = original
        for p in original.parameters():
            p.requires_grad_(False)
        in_f = original.in_features; out_f = original.out_features
        self.scaling = alpha / rank
        self.lora_A = torch.nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(out_f, rank))
        self.lora_dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def forward(self, x):
        base = self.original(x)
        lo = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lo

    def extra_repr(self):
        return f"(rank={self.lora_A.shape[0]}, in={self.lora_A.shape[1]}, out={self.lora_B.shape[0]})"


LORA_TARGETS = ("q_c", "k_c", "v_c", "q_p", "k_p", "out_proj")


def apply_lora(model, rank=16, alpha=32.0, dropout=0.0):
    count = 0
    for name, module in model.named_modules():
        if not name.startswith("layers."):
            continue
        parts = name.split(".")
        if len(parts) < 3 or parts[-1] != "attn":
            continue
        # parts: ['layers', '0', 'attn'] -> wrap attn submodules
        attn = module
        for sub in LORA_TARGETS:
            orig = getattr(attn, sub, None)
            if orig is None or not isinstance(orig, torch.nn.Linear):
                continue
            setattr(attn, sub, LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout))
            count += 1
    return count


def freeze_all(model):
    p_total = 0
    for p in model.parameters():
        p.requires_grad_(False); p_total += 1
    return p_total


def load_puzzle(path, max_rows):
    c = torch.load(path, map_location="cpu", weights_only=False)
    def ef(ep):
        r = torch.zeros_like(ep); m = ep >= 0; r[m] = (ep[m] % 8) + 1; return r
    return {
        "ba": c["board_array"][:max_rows],
        "turn": c["turn"][:max_rows],
        "castling": c["castling"][:max_rows],
        "ep": ef(c["ep_square"][:max_rows]),
        "move_idx": c["move_idx"][:max_rows],
    }


def probe(model, rows, dev):
    agree = cost_sum = ncost = within50 = 0
    for d in rows:
        b = chess.Board(d["fen"])
        inp = batch_boards_to_fused_token_ids([b], dev)
        with torch.no_grad():
            r = model(inp)
        lg = r["policy_logits"][0].float()
        mask = legal_move_mask(b).to(dev); lg[~mask] = float("-inf")
        picked = IDX_TO_UCI[lg.argmax().item()]
        cp_by = {m["uci"]: m["cp"] for m in d.get("move_values", [])}
        if d["best_uci"] not in cp_by or picked not in cp_by:
            continue
        cost = cp_by[d["best_uci"]] - cp_by[picked]
        agree += (picked == d["best_uci"]); cost_sum += cost; ncost += 1
        within50 += (cost <= 50)
    return agree/max(1,ncost), cost_sum/max(1,ncost), within50/max(1,ncost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--ckpt", default="outputs/hf_437m/best_model.pt")
    ap.add_argument("--cache", default="outputs/exp193_tactical/soft_cache.pt")
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--out", default="outputs/lora_tac")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("mps")
    random.seed(args.seed)
    d = load_puzzle(args.cache, args.max_train)
    n = d["ba"].shape[0]
    print(f"puzzle cache rows: {n} | LoRA rank={args.rank} on {LORA_TARGETS}")

    # build + load (cpu then mps)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config"); cfg = ChessTransformerConfig(**cfg) if not isinstance(cfg, ChessTransformerConfig) else cfg
    model = build_model(cfg)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck.get("model_state_dict", ck).items()})

    freeze_all(model)
    n_lora = apply_lora(model, rank=args.rank, alpha=args.alpha)
    model = model.to(dev); model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"wrapped {n_lora} projections, trainable params: {trainable/1e6:.2f}M")

    test_rows = [json.loads(l) for l in open(TEST) if l.strip()][:500]
    def report(tag):
        model.eval()
        a, c, w = probe(model, test_rows, dev)
        print(f"[{tag}] test agree={a:.0%} avgcost={c:+.0f}cp within50={w:.0%}", flush=True)
        model.train()
    report("before")

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    for s in range(args.steps):
        idx = torch.randint(0, n, (args.batch,))
        inp = {
            "fused_ids": d["ba"][idx].to(dev).long(),
            "turn": d["turn"][idx].to(dev).long(),
            "castling": d["castling"][idx].to(dev).long(),
            "ep_file": d["ep"][idx].to(dev).long(),
        }
        targets = d["move_idx"][idx].to(dev).long()
        opt.zero_grad()
        out = model(inp)
        loss = F.cross_entropy(out["policy_logits"], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if (s + 1) % 500 == 0:
            print(f"step {s+1}: loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    report("after")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # save only lora adapters
    st = {k: v.clone() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
    torch.save({"lora_state": st, "config": cfg}, f"{args.out}/lora.pt")
    print("saved LoRA adapters:", args.out)


if __name__ == "__main__":
    main()
