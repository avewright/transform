"""Quick smoke test: Qwen3-0.6B chess move generation."""
import sys
import time

import chess
import torch

sys.path.insert(0, ".")

from constrained import build_token_text_map, make_legal_move_processor
from data import fen_to_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded: {n_params/1e6:.1f}M params")

# Test raw generation (no constraint) with different encodings
board = chess.Board()
fen = board.fen()

for enc in ["fen", "grid_compact"]:
    prompt = fen_to_prompt(fen, encoding=enc)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    n_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        raw = tokenizer.decode(gen[0, n_tokens:], skip_special_tokens=True)
    print(f"\n[{enc}] ({n_tokens} tokens) Raw output: {repr(raw[:80])}")

# Test constrained decoding
print("\nBuilding token map...")
t0 = time.time()
token_texts = build_token_text_map(tokenizer)
print(f"Token map built: {len(token_texts)} tokens in {time.time()-t0:.1f}s")

for enc in ["fen", "grid_compact"]:
    prompt = fen_to_prompt(fen, encoding=enc)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    processor = make_legal_move_processor(board, tokenizer, prompt_length, token_texts=token_texts)

    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=6, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            logits_processor=processor,
        )
    elapsed = time.time() - t0
    new_text = tokenizer.decode(gen[0, prompt_length:], skip_special_tokens=True).strip()
    uci = new_text.replace(" ", "").lower()[:5].rstrip()
    print(f"[{enc}] Constrained move: {uci} (raw: {repr(new_text)}) in {elapsed:.2f}s")

# Sample 5 moves with temperature
print("\nSampled moves (grid_compact, temp=0.8):")
prompt = fen_to_prompt(fen, encoding="grid_compact")
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
inputs = {k: v.to("cuda") for k, v in inputs.items()}
prompt_length = inputs["input_ids"].shape[1]

for i in range(5):
    proc = make_legal_move_processor(board, tokenizer, prompt_length, token_texts=token_texts)
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=6, do_sample=True, temperature=0.8, top_k=50,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            logits_processor=proc,
        )
    new_text = tokenizer.decode(gen[0, prompt_length:], skip_special_tokens=True).strip()
    uci = new_text.replace(" ", "").lower()[:5].rstrip()
    print(f"  Sample {i+1}: {uci}")
