"""Quick NNUE benchmark: verify distilled model works and measure speed."""
import torch, chess, time
from nnue_model import NNUEModel, batch_boards_to_halfka_sparse
from chess_features import batch_boards_to_planes

# Load NNUE
student = NNUEModel(accumulator_size=512, hidden1=32, hidden2=32, policy_channels=32)
ckpt = torch.load("outputs/exp126_nnue_distill/best_nnue.pt", weights_only=False)
student.load_state_dict(ckpt["model_state_dict"])
student.eval()
device = torch.device("cuda")
student = student.to(device)

# Test on starting position
board = chess.Board()
halfka = batch_boards_to_halfka_sparse([board], device)
planes = batch_boards_to_planes([board]).to(device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        out = student(halfka, planes)

# Single inference benchmark
t0 = time.time()
n = 1000
for _ in range(n):
    with torch.no_grad():
        out = student(halfka, planes)
elapsed = time.time() - t0
print(f"Single eval: {n/elapsed:.0f} evals/s ({elapsed/n*1000:.2f} ms/eval)")

# Batch-8 benchmark
boards = [chess.Board() for _ in range(8)]
halfka_b = batch_boards_to_halfka_sparse(boards, device)
planes_b = batch_boards_to_planes(boards).to(device)

for _ in range(10):
    with torch.no_grad():
        out = student(halfka_b, planes_b)

t0 = time.time()
n = 500
for _ in range(n):
    with torch.no_grad():
        out = student(halfka_b, planes_b)
elapsed = time.time() - t0
print(f"Batch-8 eval: {n*8/elapsed:.0f} evals/s ({elapsed/n*1000:.2f} ms/batch)")

# Print outputs
with torch.no_grad():
    out = student(halfka, planes)
val = out["value_logits"][0]
pol = out["policy_logits"][0]
import torch.nn.functional as F
wdl = F.softmax(val, dim=-1)
print(f"\nStarting position WDL: W={wdl[0]:.3f} D={wdl[1]:.3f} L={wdl[2]:.3f}")

from move_vocab import legal_move_mask, IDX_TO_UCI
mask = legal_move_mask(board).to(device)
pol[~mask] = float("-inf")
probs = F.softmax(pol, dim=-1)
topk = torch.topk(probs, 5)
print("Top 5 moves:")
for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
    print(f"  {IDX_TO_UCI[idx]}: {p*100:.1f}%")
