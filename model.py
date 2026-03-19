"""Model loading, wrapping, and saving utilities."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attnres import AttnResWrapper
from config import Config


def load_base_model(cfg: Config) -> tuple:
    """Load the pretrained model and tokenizer.

    Returns:
        (model, tokenizer) tuple
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.model.torch_dtype, torch.bfloat16)

    print(f"Loading model: {cfg.model.name_or_path} (dtype={cfg.model.torch_dtype})")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        dtype=torch_dtype,
        device_map=cfg.device,
        trust_remote_code=True,
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params / 1e6:.1f}M parameters")

    return model, tokenizer


def wrap_with_attnres(model, cfg: Config) -> AttnResWrapper:
    """Wrap a model with Block AttnRes."""
    if not cfg.attnres.enabled:
        return model

    print(f"Wrapping model with Block AttnRes (block_size={cfg.attnres.block_size})")
    wrapper = AttnResWrapper(
        model,
        block_size=cfg.attnres.block_size,
        head_dim=cfg.attnres.head_dim,
    )
    print(f"AttnRes: {wrapper.num_blocks} blocks of {wrapper.block_size} layers")
    return wrapper


def save_randopt_results(
    top_k_results,
    output_dir: str | Path,
    cfg: Config,
):
    """Save the top K perturbation noise vectors and config."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model": cfg.model.__dict__,
            "attnres": cfg.attnres.__dict__,
            "randopt": {k: v for k, v in cfg.randopt.__dict__.items()
                       if not k.startswith("_")},
            "data": cfg.data.__dict__,
        }, f, indent=2, default=str)

    # Save each perturbation's noise vectors
    for i, result in enumerate(top_k_results):
        noise_path = output_dir / f"perturbation_{i:04d}.pt"
        torch.save({
            "index": result.index,
            "accuracy": result.accuracy,
            "noise_vectors": {
                name: tensor.cpu()
                for name, tensor in result.noise_vectors.items()
            },
        }, noise_path)

    print(f"Saved {len(top_k_results)} perturbations to {output_dir}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "num_perturbations_sampled": cfg.randopt.n_perturbations,
            "top_k": cfg.randopt.top_k,
            "noise_std": cfg.randopt.noise_std,
            "results": [
                {"rank": i, "original_index": r.index, "accuracy": r.accuracy}
                for i, r in enumerate(top_k_results)
            ],
        }, f, indent=2)


def load_randopt_results(output_dir: str | Path, device: torch.device | str = "cpu"):
    """Load saved perturbation results for ensemble inference."""
    output_dir = Path(output_dir)

    from randopt import PerturbationResult

    results = []
    for path in sorted(output_dir.glob("perturbation_*.pt")):
        data = torch.load(path, map_location=device, weights_only=True)
        results.append(PerturbationResult(
            index=data["index"],
            accuracy=data["accuracy"],
            noise_vectors=data["noise_vectors"],
        ))

    print(f"Loaded {len(results)} perturbations from {output_dir}")
    return results
