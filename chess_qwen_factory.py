"""Chess + Qwen fusion: trainable board tower in parallel with PEFT/full backbone.

Architecture:
  1. SmallFusedEncoder — 128d piece/square/context lookups (cheap)
  2. BoardTower — bidirectional chess transformer (big, always trained)
  3. Linear proj → Qwen hidden_size (1024 for Qwen3-0.6B)
  4. Readout token appended; Qwen backbone via inputs_embeds (causal mixer)
  5. Policy head: Linear → move vocab softmax
  6. Value head: optional 128-bin HL-Gauss distributional head

Backbone adaptation modes (backbone_mode):
  - unsloth: Unsloth FastLanguageModel + LoRA (default, recommended)
  - lora / peft: manual LoRA without Unsloth
  - full: all Qwen weights trainable
  - last_n: unfreeze top N transformer layers only
  - frozen: ablation baseline only
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from chess_features import (
    NUM_CASTLING_STATES,
    NUM_EP_STATES,
    NUM_FUSED_TOKENS,
    batch_boards_to_fused_token_ids,
)
from chess_model import ChessRelativeBias
from chess_transformer_factory import ChessTransformerEncoderLayer
from move_vocab import VOCAB_SIZE, index_to_move, legal_move_mask

BackboneMode = Literal["frozen", "lora", "peft", "full", "last_n", "unsloth"]

DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class ChessQwenConfig:
    qwen_name_or_path: str = "Qwen/Qwen3-0.6B"
    torch_dtype: str = "bfloat16"

    piece_embed_dim: int = 128
    tower_dim: int = 768
    tower_layers: int = 6
    tower_heads: int = 12
    tower_ffn_ratio: int = 4
    tower_dropout: float = 0.05
    use_rel_bias: bool = True

    backbone_mode: BackboneMode = "unsloth"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    unfreeze_last_n_layers: int = 4

    # Unsloth loading (backbone_mode="unsloth")
    max_seq_length: int = 128
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    gradient_checkpointing: bool = True
    use_rslora: bool = False

    n_value_bins: int = 128
    value_hidden: int = 256
    policy_hidden: int = 0

    lr_tower: float = 3e-4
    lr_heads: float = 3e-4
    lr_lora: float = 1e-4
    lr_backbone: float = 5e-6

    @classmethod
    def from_json(cls, path: str | Path) -> ChessQwenConfig:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model_data = data.get("model", data)
        if "lora_targets" in model_data and isinstance(model_data["lora_targets"], list):
            model_data = {**model_data, "lora_targets": tuple(model_data["lora_targets"])}
        return cls(**model_data)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CHESS_QWEN_CONFIG = ChessQwenConfig()


class SmallFusedEncoder(nn.Module):
    """128d fused piece-color + square + context token embeddings."""

    NUM_CONTEXT = 3

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        if embed_dim > 128:
            raise ValueError(f"piece_embed_dim must be <= 128, got {embed_dim}")
        self.embed_dim = embed_dim
        self.piece_color_embed = nn.Embedding(NUM_FUSED_TOKENS, embed_dim)
        self.square_embed = nn.Embedding(64, embed_dim)
        self.turn_embed = nn.Embedding(2, embed_dim)
        self.castling_embed = nn.Embedding(NUM_CASTLING_STATES, embed_dim)
        self.ep_embed = nn.Embedding(NUM_EP_STATES, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        fused_ids = token_ids["fused_ids"]
        sq_emb = self.piece_color_embed(fused_ids)
        sq_idx = torch.arange(64, device=fused_ids.device)
        sq_emb = sq_emb + self.square_embed(sq_idx)

        turn_tok = self.turn_embed(token_ids["turn"]).unsqueeze(1)
        castle_tok = self.castling_embed(token_ids["castling"]).unsqueeze(1)
        ep_tok = self.ep_embed(token_ids["ep_file"]).unsqueeze(1)

        return self.norm(torch.cat([turn_tok, castle_tok, ep_tok, sq_emb], dim=1))

    def prepare_input(self, board: chess.Board, device: torch.device):
        return batch_boards_to_fused_token_ids([board], device)

    def prepare_batch(self, boards: list[chess.Board], device: torch.device):
        return batch_boards_to_fused_token_ids(boards, device)


class BoardTower(nn.Module):
    """Bidirectional chess transformer — spatial mixing before causal Qwen."""

    N_CTX = 3

    def __init__(self, config: ChessQwenConfig):
        super().__init__()
        self.config = config
        self.embed = SmallFusedEncoder(config.piece_embed_dim)
        self.in_proj = nn.Linear(config.piece_embed_dim, config.tower_dim)
        ffn_dim = config.tower_dim * config.tower_ffn_ratio

        self.rel_bias = (
            ChessRelativeBias(config.tower_heads, n_ctx=self.N_CTX)
            if config.use_rel_bias else None
        )
        self.layers = nn.ModuleList([
            ChessTransformerEncoderLayer(
                d_model=config.tower_dim,
                nhead=config.tower_heads,
                ffn_dim=ffn_dim,
                dropout=config.tower_dropout,
                use_swiglu=True,
            )
            for _ in range(config.tower_layers)
        ])
        self.norm = nn.LayerNorm(config.tower_dim)

    @property
    def embed_dim(self) -> int:
        return self.config.tower_dim

    def _attn_bias(self, batch_size: int, seq_len: int, device: torch.device):
        if self.rel_bias is None:
            return None
        bias = self.rel_bias()
        nhead = self.config.tower_heads
        return bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(
            batch_size * nhead, seq_len, seq_len,
        ).to(device)

    def forward(self, token_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.in_proj(self.embed(token_ids))
        batch_size, seq_len, _ = x.shape
        attn_bias = self._attn_bias(batch_size, seq_len, x.device)
        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias)
        return self.norm(x)

    def prepare_input(self, board: chess.Board, device: torch.device):
        return self.embed.prepare_input(board, device)

    def prepare_batch(self, boards: list[chess.Board], device: torch.device):
        return self.embed.prepare_batch(boards, device)


# ── LoRA (PEFT-compatible fallback when peft package absent) ──

class LoRALinear(nn.Module):
    def __init__(
        self,
        original: nn.Linear,
        rank: int = 32,
        alpha: float = 64.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, original.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        delta = self.lora_dropout(x).float() @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + delta.to(base.dtype)


def apply_custom_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    targets: tuple[str, ...],
) -> int:
    added = 0
    modules_dict = dict(module.named_modules())
    for name, child in list(module.named_modules()):
        for target in targets:
            if target in name and isinstance(child, nn.Linear):
                parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
                parent = modules_dict[parent_name] if parent_name else module
                wrapper = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr, wrapper)
                added += wrapper.lora_A.numel() + wrapper.lora_B.numel()
                break
    return added


def apply_peft_lora(backbone: nn.Module, config: ChessQwenConfig) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "Install peft for backbone_mode='peft': pip install peft"
        ) from exc

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(backbone, peft_config)


def configure_backbone(backbone: nn.Module, config: ChessQwenConfig) -> dict[str, int]:
    """Apply backbone adaptation. Returns stats dict."""
    stats = {"mode": config.backbone_mode, "lora_params": 0}

    for param in backbone.parameters():
        param.requires_grad = False

    mode = config.backbone_mode
    if mode == "frozen":
        return stats

    if mode in ("lora", "peft"):
        if mode == "peft":
            wrapped = apply_peft_lora(backbone, config)
            # Caller replaces backbone reference when using peft wrapper
            stats["lora_params"] = sum(
                p.numel() for p in wrapped.parameters() if p.requires_grad
            )
            stats["peft_wrapped"] = True
            return stats

        stats["lora_params"] = apply_custom_lora(
            backbone,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            targets=config.lora_targets,
        )
        return stats

    if mode == "full":
        for param in backbone.parameters():
            param.requires_grad = True
        stats["trainable_backbone"] = sum(
            p.numel() for p in backbone.parameters() if p.requires_grad
        )
        return stats

    if mode == "last_n":
        layers = _get_transformer_layers(backbone)
        n = min(config.unfreeze_last_n_layers, len(layers))
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        stats["unfrozen_layers"] = n
        stats["trainable_backbone"] = sum(
            p.numel() for p in backbone.parameters() if p.requires_grad
        )
        return stats

    raise ValueError(f"Unknown backbone_mode: {mode}")


def _get_transformer_layers(backbone: nn.Module) -> nn.ModuleList:
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "h"):
        return backbone.h
    raise AttributeError("Cannot find transformer layers on backbone")


def _resolve_dtype(config: ChessQwenConfig) -> torch.dtype:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(config.torch_dtype, torch.bfloat16)


def _normalize_qwen_path(name: str) -> str:
    """Map Unsloth HF ids to base Qwen weights for transformers loading."""
    if name.startswith("unsloth/"):
        name = "Qwen/" + name.split("/", 1)[1]
    return name.replace("-unsloth-bnb-4bit", "").replace("-bnb-4bit", "")


def load_bnb_peft_qwen(
    config: ChessQwenConfig,
    device: str | torch.device = "cuda",
) -> tuple[nn.Module, int, object | None]:
    """4-bit Qwen + PEFT LoRA via transformers/bitsandbytes (Windows-friendly fallback)."""
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit Qwen requires CUDA.")

    try:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("Install: pip install bitsandbytes peft") from exc

    model_path = _normalize_qwen_path(config.qwen_name_or_path)
    compute_dtype = _resolve_dtype(config)
    quant = None
    if config.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant,
        dtype=compute_dtype if quant is None else None,
        device_map={"": str(device)},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if config.backbone_mode != "frozen" and not config.full_finetuning:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_targets),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, lora_config)

    hidden_size = model.config.hidden_size
    return model, hidden_size, None


def load_unsloth_qwen(
    config: ChessQwenConfig,
    device: str | torch.device = "cuda",
) -> tuple[nn.Module, int, object | None]:
    """Load Qwen3 via Unsloth with optional 4-bit quant + LoRA adapters."""
    if not torch.cuda.is_available():
        raise RuntimeError("Unsloth requires CUDA. Use backbone_mode='lora' for CPU tests.")

    # Unsloth pulls vllm on import; skip when vllm C extension is missing (common on Windows).
    import importlib.util
    if importlib.util.find_spec("vllm._C") is None:
        print("  Unsloth needs vllm GPU build; using bitsandbytes+PEFT instead")
        return load_bnb_peft_qwen(config, device=device)

    try:
        from unsloth import FastLanguageModel
    except (ImportError, NotImplementedError, ModuleNotFoundError) as exc:
        print(f"  Unsloth unavailable ({exc}); falling back to bitsandbytes+PEFT")
        return load_bnb_peft_qwen(config, device=device)

    torch_dtype = _resolve_dtype(config)
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.qwen_name_or_path,
            max_seq_length=config.max_seq_length,
            dtype=torch_dtype,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            full_finetuning=config.full_finetuning,
        )
    except Exception as exc:
        print(f"  Unsloth load failed ({exc}); falling back to bitsandbytes+PEFT")
        return load_bnb_peft_qwen(config, device=device)

    if config.backbone_mode == "unsloth" and not config.full_finetuning:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            target_modules=list(config.lora_targets),
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth" if config.gradient_checkpointing else False,
            use_rslora=config.use_rslora,
        )
    elif config.backbone_mode == "frozen":
        for param in model.parameters():
            param.requires_grad = False

    model = model.to(device)
    hidden_size = model.config.hidden_size
    return model, hidden_size, tokenizer


def _forward_backbone(backbone: nn.Module, embeds: torch.Tensor) -> torch.Tensor:
    """Run Qwen backbone (Unsloth CausalLM or bare Qwen3Model) on board embeds."""
    outputs = backbone(
        inputs_embeds=embeds,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None and getattr(outputs, "hidden_states", None):
        hidden = outputs.hidden_states[-1]
    if hidden is None:
        raise RuntimeError("Backbone forward did not return hidden states")
    return hidden.float()


def load_qwen_backbone(
    config: ChessQwenConfig,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, int, object | None]:
    """Load Qwen backbone. Returns (backbone, hidden_size, tokenizer_or_none)."""
    if config.backbone_mode == "unsloth":
        return load_unsloth_qwen(config, device=device)

    if config.backbone_mode == "peft" and config.load_in_4bit and torch.cuda.is_available():
        return load_bnb_peft_qwen(config, device=device)

    torch_dtype = _resolve_dtype(config)
    full = AutoModelForCausalLM.from_pretrained(
        config.qwen_name_or_path,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    if hasattr(full, "model"):
        backbone = full.model
        hidden_size = full.config.hidden_size
    else:
        backbone = full
        hidden_size = backbone.config.hidden_size

    del full
    backbone = backbone.to(device)

    if config.backbone_mode == "peft":
        backbone = apply_peft_lora(backbone, config)
    else:
        configure_backbone(backbone, config)

    return backbone, hidden_size, None


class ChessQwenModel(nn.Module):
    """Board tower + trainable Qwen backbone + policy/value heads."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        config: ChessQwenConfig,
        tokenizer=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.encoder = BoardTower(config)
        self.input_proj = nn.Linear(config.tower_dim, hidden_size)
        self.readout_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        if config.policy_hidden > 0:
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_size, config.policy_hidden),
                nn.GELU(),
                nn.Linear(config.policy_hidden, VOCAB_SIZE),
            )
        else:
            self.policy_head = nn.Linear(hidden_size, VOCAB_SIZE)

        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, config.value_hidden),
            nn.GELU(),
            nn.Linear(config.value_hidden, config.n_value_bins),
        )

    def encode_board(self, token_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        tower_out = self.encoder(token_ids)
        embeds = self.input_proj(tower_out)
        readout = self.readout_token.expand(embeds.shape[0], -1, -1)
        return torch.cat([embeds, readout], dim=1)

    def forward(
        self,
        board_input: dict[str, torch.Tensor],
        move_targets: torch.Tensor | None = None,
        value_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embeds = self.encode_board(board_input)
        embeds = embeds.to(self._backbone_embed_dtype())

        hidden = _forward_backbone(self.backbone, embeds)
        readout_hidden = hidden[:, -1, :]

        policy_logits = self.policy_head(readout_hidden)
        value_logits = self.value_head(readout_hidden)

        result = {"policy_logits": policy_logits, "value_logits": value_logits}
        device = board_input["fused_ids"].device
        total_loss = torch.tensor(0.0, device=device)

        if move_targets is not None:
            policy_loss = F.cross_entropy(policy_logits, move_targets)
            result["policy_loss"] = policy_loss
            total_loss = total_loss + policy_loss

        if value_targets is not None:
            if value_targets.dtype in (torch.long, torch.int, torch.int64):
                value_loss = F.cross_entropy(value_logits, value_targets)
            else:
                value_loss = hl_gauss_loss(value_logits, value_targets, self.config.n_value_bins)
            result["value_loss"] = value_loss
            total_loss = total_loss + 0.3 * value_loss

        result["loss"] = total_loss
        return result

    @torch.no_grad()
    def predict_move(self, board: chess.Board) -> tuple[chess.Move, torch.Tensor]:
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        logits = self.forward(board_input)["policy_logits"][0]
        logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return index_to_move(probs.argmax().item()), probs

    def _backbone_embed_dtype(self) -> torch.dtype:
        if hasattr(self.backbone, "get_input_embeddings"):
            emb = self.backbone.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.dtype
        return next(self.backbone.parameters()).dtype

    def save_checkpoint(self, path: str | Path) -> None:
        """Save chess tower/heads plus Unsloth/LoRA adapter weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.to_dict(),
                "tower_state": {
                    k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")
                },
            },
            path / "chess_modules.pt",
        )
        if self.config.backbone_mode == "unsloth" and self.tokenizer is not None:
            self.backbone.save_pretrained(str(path / "unsloth_lora"))
            self.tokenizer.save_pretrained(str(path / "unsloth_lora"))
        elif hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(str(path / "backbone"))


def hl_gauss_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 128,
    sigma: float | None = None,
) -> torch.Tensor:
    """HL-Gauss loss for win-percent targets in [0, 1]."""
    if sigma is None:
        sigma = 1.0 / n_bins * 2.5
    bin_centers = (torch.arange(n_bins, device=logits.device, dtype=logits.dtype) + 0.5) / n_bins
    diff = bin_centers.unsqueeze(0) - targets.unsqueeze(1)
    target_dist = F.softmax(-0.5 * (diff / sigma) ** 2, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, target_dist, reduction="batchmean")


def _load_backbone_with_adapter(
    config: ChessQwenConfig,
    adapter_dir: Path,
    device: str | torch.device,
) -> tuple[nn.Module, int, object | None]:
    """Load 4-bit Qwen + saved LoRA adapter weights."""
    try:
        from peft import PeftModel
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("Install peft + bitsandbytes to load checkpoints") from exc

    model_path = _normalize_qwen_path(config.qwen_name_or_path)
    compute_dtype = _resolve_dtype(config)
    quant = None
    if config.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit checkpoint loading requires CUDA.")
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant,
        dtype=compute_dtype if quant is None else None,
        device_map={"": str(device)},
        trust_remote_code=True,
    )
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_disable"):
        base.gradient_checkpointing_disable()
    backbone = PeftModel.from_pretrained(base, str(adapter_dir))
    if hasattr(backbone, "gradient_checkpointing_disable"):
        backbone.gradient_checkpointing_disable()
    return backbone, backbone.config.hidden_size, None


def _disable_backbone_gradient_checkpointing(backbone: nn.Module) -> None:
    for mod in (
        backbone,
        getattr(backbone, "base_model", None),
        getattr(getattr(backbone, "base_model", None), "model", None),
    ):
        if mod is None:
            continue
        if hasattr(mod, "gradient_checkpointing_disable"):
            mod.gradient_checkpointing_disable()
        if hasattr(mod, "config"):
            mod.config.use_cache = False


def load_chess_qwen_checkpoint(
    checkpoint_dir: str | Path,
    device: str | torch.device | None = None,
    for_training: bool = False,
) -> ChessQwenModel:
    """Load ChessQwen tower/heads + LoRA backbone from an exp181-style checkpoint dir."""
    checkpoint_dir = Path(checkpoint_dir)
    modules_path = checkpoint_dir / "chess_modules.pt"
    if not modules_path.exists():
        raise FileNotFoundError(f"Missing {modules_path}")

    ckpt = torch.load(modules_path, map_location="cpu", weights_only=False)
    config = ChessQwenConfig(**ckpt["config"])
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    adapter_dir = None
    if (checkpoint_dir / "unsloth_lora").exists():
        adapter_dir = checkpoint_dir / "unsloth_lora"
    elif (checkpoint_dir / "backbone").exists():
        adapter_dir = checkpoint_dir / "backbone"

    if adapter_dir is not None:
        backbone, hidden_size, tokenizer = _load_backbone_with_adapter(config, adapter_dir, dev)
    else:
        backbone, hidden_size, tokenizer = load_qwen_backbone(config, device=dev)

    model = ChessQwenModel(backbone, hidden_size, config, tokenizer=tokenizer).to(dev)
    missing, unexpected = model.load_state_dict(ckpt["tower_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
    non_backbone_missing = [k for k in missing if not k.startswith("backbone.")]
    if non_backbone_missing:
        raise RuntimeError(f"Missing tower/head keys: {non_backbone_missing[:5]}")
    if for_training:
        _disable_backbone_gradient_checkpointing(model.backbone)
        model.train()
    else:
        prepare_for_inference(model)
        model.eval()
    return model


def value_logits_to_white_win(value_logits: torch.Tensor) -> float:
    """Expected White win probability from HL-Gauss value bins."""
    logits = value_logits.float()
    n_bins = logits.shape[-1]
    centers = (torch.arange(n_bins, device=logits.device, dtype=logits.dtype) + 0.5) / n_bins
    probs = F.softmax(logits, dim=-1)
    return (probs * centers).sum().item()


def build_chess_qwen(
    config: ChessQwenConfig | dict | str | Path | None = None,
    device: str | torch.device | None = None,
) -> ChessQwenModel:
    if config is None:
        resolved = DEFAULT_CHESS_QWEN_CONFIG
    elif isinstance(config, ChessQwenConfig):
        resolved = config
    elif isinstance(config, (str, Path)):
        resolved = ChessQwenConfig.from_json(config)
    else:
        resolved = ChessQwenConfig(**config)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone, hidden_size, tokenizer = load_qwen_backbone(resolved, device=dev)
    model = ChessQwenModel(backbone, hidden_size, resolved, tokenizer=tokenizer).to(dev)
    if resolved.backbone_mode == "unsloth":
        prepare_for_training(model)
    return model


def prepare_for_training(model: ChessQwenModel) -> None:
    """Enable Unsloth training kernels on the Qwen backbone."""
    if model.config.backbone_mode != "unsloth":
        return
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(model.backbone)
    except ImportError:
        pass


def prepare_for_inference(model: ChessQwenModel) -> None:
    """Switch Unsloth backbone to inference mode."""
    if model.config.backbone_mode != "unsloth":
        return
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model.backbone)
    except ImportError:
        pass


def get_optimizer_param_groups(
    model: ChessQwenModel,
    config: ChessQwenConfig | None = None,
) -> list[dict]:
    """Three-way LR: tower+proj | heads | backbone (LoRA or full)."""
    cfg = config or model.config
    tower_params, head_params, backbone_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        elif "policy_head" in name or "value_head" in name:
            head_params.append(param)
        else:
            tower_params.append(param)

    groups = [
        {"params": tower_params, "lr": cfg.lr_tower, "name": "tower"},
        {"params": head_params, "lr": cfg.lr_heads, "name": "heads"},
    ]
    if backbone_params:
        lr = cfg.lr_lora if cfg.backbone_mode in ("lora", "peft", "unsloth", "last_n") else cfg.lr_backbone
        groups.append({"params": backbone_params, "lr": lr, "name": "backbone"})
    return groups


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
