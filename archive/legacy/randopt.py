"""RandOpt — Random Optimization post-training.

Implements the core algorithm from:
  "Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights"
  (Gan & Isola, MIT CSAIL, 2026)
  https://arxiv.org/abs/2603.12228

Algorithm:
  1. Start with pretrained weight vector θ₀
  2. Sample N perturbations: θ_i = θ₀ + ε_i, where ε_i ~ N(0, σ²I)
  3. Evaluate each θ_i on the task (chess move prediction)
  4. Select top K perturbations by accuracy
  5. Ensemble predictions via majority vote at inference

Key insight from the paper: for large pretrained models, the density of
task-improving solutions in a Gaussian neighborhood around θ₀ scales with
model size. Random sampling becomes effective post-training.
"""

import copy
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from config import RandOptConfig


@dataclass
class PerturbationResult:
    """Result of evaluating a single weight perturbation."""
    index: int
    accuracy: float
    noise_vectors: dict[str, torch.Tensor]  # param_name → noise tensor


def get_perturbable_params(
    model: nn.Module,
    perturb_patterns: list[str],
    skip_patterns: list[str],
) -> dict[str, nn.Parameter]:
    """Identify which parameters to perturb based on name patterns."""
    params = {}
    for name, param in model.named_parameters():
        # Check skip patterns first
        if any(re.match(pat, name) for pat in skip_patterns):
            continue
        # Check perturb patterns
        if any(re.match(pat, name) for pat in perturb_patterns):
            params[name] = param
    return params


def apply_perturbation(
    model: nn.Module,
    noise_vectors: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> None:
    """Apply noise vectors to model parameters in-place."""
    param_dict = dict(model.named_parameters())
    for name, noise in noise_vectors.items():
        if name in param_dict:
            param_dict[name].data.add_(noise * scale)


def remove_perturbation(
    model: nn.Module,
    noise_vectors: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> None:
    """Remove previously applied noise vectors (reverse the perturbation)."""
    apply_perturbation(model, noise_vectors, scale=-scale)


# ---------------------------------------------------------------------------
# Seed-based perturbation (memory-efficient, from official RandOpt)
# ---------------------------------------------------------------------------

def apply_seed_perturbation(
    model: nn.Module,
    seed: int,
    sigma: float,
    perturbable_names: set[str],
    negate: bool = False,
) -> None:
    """Apply deterministic perturbation using only a seed + sigma.

    Instead of storing full noise tensors, regenerate them from the seed.
    Matches the official RandOpt implementation pattern.
    """
    sign = -1.0 if negate else 1.0
    for name, p in model.named_parameters():
        if name not in perturbable_names:
            continue
        gen = torch.Generator(device=p.device)
        gen.manual_seed(seed)
        noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
        p.data.add_(sign * sigma * noise)
        del noise


def remove_seed_perturbation(
    model: nn.Module,
    seed: int,
    sigma: float,
    perturbable_names: set[str],
) -> None:
    """Undo a seed-based perturbation (subtract what was added)."""
    apply_seed_perturbation(model, seed, sigma, perturbable_names, negate=True)


def sample_noise(
    param_shapes: dict[str, torch.Size],
    std: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample Gaussian noise vectors for each parameter."""
    return {
        name: torch.randn(shape, device=device, dtype=dtype, generator=generator) * std
        for name, shape in param_shapes.items()
    }


@torch.no_grad()
def evaluate_perturbation(
    model: nn.Module,
    eval_data: list[dict],
    batch_size: int = 64,
) -> float:
    """Evaluate model accuracy on chess move prediction.

    Args:
        model: The model with perturbation already applied.
        eval_data: List of dicts with input_ids, labels, target_uci.
        batch_size: Batch size for evaluation.

    Returns:
        Accuracy (fraction of correctly predicted best moves).
    """
    model.eval()
    correct = 0
    total = 0

    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i : i + batch_size]
        input_ids = torch.stack([ex["input_ids"] for ex in batch])
        attention_mask = torch.stack([ex["attention_mask"] for ex in batch])
        targets = [ex["target_uci"] for ex in batch]

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        if hasattr(model, "forward_with_attnres"):
            outputs = model.forward_with_attnres(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs["logits"]
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # For each example, get the predicted tokens after the prompt
        for j, ex in enumerate(batch):
            # Find where the prompt ends (where labels != -100)
            labels = ex["labels"]
            prompt_end = (labels != -100).nonzero(as_tuple=True)[0]
            if len(prompt_end) == 0:
                continue
            start_pos = prompt_end[0].item() - 1  # predict from last prompt token

            # Get predicted token ids
            pred_ids = logits[j, start_pos:start_pos + 6].argmax(dim=-1)
            # Decode
            # We need the tokenizer — pass it through or use a simpler check
            total += 1
            # Store prediction for later decoding
            ex["_pred_ids"] = pred_ids

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_perturbation_simple(
    model: nn.Module,
    eval_data: list[dict],
    tokenizer,
    batch_size: int = 64,
) -> float:
    """Evaluate model accuracy with tokenizer-based decoding.

    Checks if the model's generated move matches the target UCI move.
    """
    model.eval()
    correct = 0
    total = 0

    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i : i + batch_size]
        input_ids = torch.stack([ex["input_ids"] for ex in batch])
        attention_mask = torch.stack([ex["attention_mask"] for ex in batch])

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass — get logits
        if hasattr(model, "forward_with_attnres"):
            outputs = model.forward_with_attnres(
                input_ids=input_ids, attention_mask=attention_mask,
            )
            logits = outputs["logits"]
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        for j, ex in enumerate(batch):
            labels = ex["labels"]
            target_uci = ex["target_uci"]

            # Find where target starts (first non -100 label)
            target_positions = (labels != -100).nonzero(as_tuple=True)[0]
            if len(target_positions) == 0:
                continue

            start = target_positions[0].item()
            end = min(start + 5, logits.shape[1])  # UCI moves are 4-5 chars

            # Greedy decode from model's logits
            pred_ids = logits[j, start - 1 : end - 1].argmax(dim=-1)
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            # Check if predicted move matches target
            # Normalize: strip spaces, lowercase
            pred_move = pred_text.replace(" ", "").lower()[:5]
            target_move = target_uci.lower()

            if pred_move == target_move:
                correct += 1
            total += 1

    return correct / max(total, 1)


@torch.no_grad()
def randopt(
    model: nn.Module,
    eval_data: list[dict],
    tokenizer,
    cfg: RandOptConfig,
    device: torch.device | None = None,
) -> list[PerturbationResult]:
    """Run RandOpt: sample N perturbations, evaluate each, return top K.

    This is the core algorithm from the Neural Thickets paper.

    Args:
        model: Pretrained model (will NOT be permanently modified).
        eval_data: Chess positions for evaluation.
        tokenizer: Model tokenizer for decoding predictions.
        cfg: RandOpt configuration.
        device: Device to use.

    Returns:
        Top K PerturbationResults sorted by accuracy (descending).
    """
    if device is None:
        device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Identify parameters to perturb
    perturbable = get_perturbable_params(
        model, cfg.perturb_patterns, cfg.skip_patterns,
    )
    param_shapes = {name: p.shape for name, p in perturbable.items()}
    num_params = sum(p.numel() for p in perturbable.values())

    print(f"RandOpt: {len(perturbable)} parameter tensors, {num_params:,} parameters to perturb")
    print(f"RandOpt: sampling {cfg.n_perturbations} perturbations, selecting top {cfg.top_k}")

    # Subsample eval data if needed
    eval_subset = eval_data[: cfg.eval_positions]

    # Setup RNG
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    results: list[PerturbationResult] = []

    # Evaluate baseline (no perturbation)
    baseline_acc = evaluate_perturbation_simple(
        model, eval_subset, tokenizer, cfg.eval_batch_size,
    )
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # Sample and evaluate perturbations
    for i in tqdm(range(cfg.n_perturbations), desc="RandOpt sampling"):
        # Sample noise
        noise = sample_noise(param_shapes, cfg.noise_std, device, dtype, generator)

        # Apply perturbation
        apply_perturbation(model, noise)

        # Evaluate
        acc = evaluate_perturbation_simple(
            model, eval_subset, tokenizer, cfg.eval_batch_size,
        )

        # Remove perturbation (restore original weights)
        remove_perturbation(model, noise)

        results.append(PerturbationResult(
            index=i,
            accuracy=acc,
            noise_vectors=noise,
        ))

        if (i + 1) % 100 == 0:
            top_so_far = sorted(results, key=lambda r: r.accuracy, reverse=True)[:cfg.top_k]
            best = top_so_far[0].accuracy
            mean_topk = sum(r.accuracy for r in top_so_far) / len(top_so_far)
            print(f"  [{i+1}/{cfg.n_perturbations}] best={best:.4f} mean_top{cfg.top_k}={mean_topk:.4f}")

    # Select top K
    results.sort(key=lambda r: r.accuracy, reverse=True)
    top_k = results[: cfg.top_k]

    print(f"\nRandOpt complete. Top {cfg.top_k} accuracies:")
    for r in top_k:
        print(f"  perturbation {r.index}: acc={r.accuracy:.4f}")

    return top_k


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def ensemble_predict(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    top_k_results: list[PerturbationResult],
    tokenizer,
    max_new_tokens: int = 6,
) -> list[str]:
    """Majority-vote ensemble over top K perturbations.

    For each perturbation, apply it, generate a move, remove it.
    Then majority-vote over the K predictions.

    Args:
        model: Base pretrained model.
        input_ids: Batched input token IDs.
        attention_mask: Batched attention mask.
        top_k_results: Top K perturbation results from randopt().
        tokenizer: Tokenizer for decoding.
        max_new_tokens: Max tokens to generate per perturbation.

    Returns:
        List of predicted UCI moves (one per batch element).
    """
    batch_size = input_ids.shape[0]
    all_predictions: list[list[str]] = [[] for _ in range(batch_size)]

    for result in top_k_results:
        # Apply perturbation
        apply_perturbation(model, result.noise_vectors)

        # Generate
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy for consistency
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Remove perturbation
        remove_perturbation(model, result.noise_vectors)

        # Decode new tokens only
        new_ids = gen_ids[:, input_ids.shape[1]:]
        for j in range(batch_size):
            text = tokenizer.decode(new_ids[j], skip_special_tokens=True).strip()
            # Extract UCI move (4-5 chars like "e2e4" or "e7e8q")
            move = text.replace(" ", "").lower()[:5]
            all_predictions[j].append(move)

    # Majority vote
    final_moves = []
    for preds in all_predictions:
        from collections import Counter
        vote = Counter(preds).most_common(1)[0][0]
        final_moves.append(vote)

    return final_moves
