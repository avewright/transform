"""Smoke test: ChessQwenModel with board tower + Unsloth/LoRA Qwen backbone.

Usage:
  python experiments/exp180_qwen_peft.py
  python experiments/exp180_qwen_peft.py --load-qwen --backbone-mode unsloth
  python experiments/exp180_qwen_peft.py --load-qwen --backbone-mode lora
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_qwen_factory import (
    ChessQwenConfig,
    ChessQwenModel,
    count_parameters,
    get_optimizer_param_groups,
)
from chess_transformer_factory import ChessTransformerEncoderLayer


def _dummy_backbone(hidden_size: int = 1024, n_layers: int = 2):
    layer = ChessTransformerEncoderLayer(hidden_size, 8, hidden_size * 4, use_swiglu=True)

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([layer for _ in range(n_layers)])

        def forward(self, inputs_embeds=None, use_cache=False, **kwargs):
            x = inputs_embeds
            for block in self.layers:
                x = block(x)
            return type("Out", (), {"last_hidden_state": x})()

    return DummyBackbone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-qwen", action="store_true", help="Load real Qwen3-0.6B weights")
    parser.add_argument(
        "--backbone-mode",
        default="lora",
        choices=["frozen", "lora", "peft", "full", "last_n", "unsloth"],
    )
    args = parser.parse_args()

    config = ChessQwenConfig(
        backbone_mode=args.backbone_mode,
        tower_layers=4,
        lora_rank=16,
        load_in_4bit=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.load_qwen:
        if args.backbone_mode == "unsloth" and device.type != "cuda":
            raise SystemExit("Unsloth requires CUDA — use --backbone-mode lora on CPU")
        from chess_qwen_factory import build_chess_qwen
        model = build_chess_qwen(config, device=device)
    else:
        from chess_qwen_factory import configure_backbone
        backbone = _dummy_backbone().to(device)
        configure_backbone(backbone, config)
        model = ChessQwenModel(backbone, hidden_size=1024, config=config).to(device)

    stats = count_parameters(model)
    groups = get_optimizer_param_groups(model)
    print(f"Mode: {config.backbone_mode}")
    print(f"Params: total={stats['total']:,} trainable={stats['trainable']:,} frozen={stats['frozen']:,}")
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['name']}: {n:,} params @ lr={g['lr']}")

    batch = 2
    token_ids = {
        "fused_ids": torch.randint(0, 13, (batch, 64), device=device),
        "turn": torch.zeros(batch, dtype=torch.long, device=device),
        "castling": torch.zeros(batch, dtype=torch.long, device=device),
        "ep_file": torch.zeros(batch, dtype=torch.long, device=device),
    }
    targets = torch.randint(0, 1968, (batch,), device=device)
    win_pct = torch.rand(batch, device=device)

    out = model(token_ids, move_targets=targets, value_targets=win_pct)
    print(
        f"Forward OK: policy={tuple(out['policy_logits'].shape)} "
        f"value={tuple(out['value_logits'].shape)} loss={out['loss'].item():.4f}"
    )


if __name__ == "__main__":
    main()
