"""Quick data quality check for training datasets."""
import json
import glob
import statistics
import sys

dataset_glob = sys.argv[1] if len(sys.argv) > 1 else "outputs/exp095_endgame_harvest/dataset/positions_*.jsonl"
files = sorted(glob.glob(dataset_glob))
if not files:
    print(f"No files matching: {dataset_glob}")
    sys.exit(1)

n = 0
plies = []
cps = []
n_tgt = []
gaps = []

for f in files[:10]:
    for line in open(f):
        d = json.loads(line)
        n += 1
        plies.append(int(d.get("ply", 0)))
        cps.append(float(d.get("best_cp", 0)))
        n_tgt.append(len(d.get("soft_targets", [])))
        gaps.append(float(d.get("cp_gap_top1_top2", 0)))

print(f"Sampled {n} from {len(files)} files")
print(f"Ply: mean={statistics.mean(plies):.1f} med={statistics.median(plies):.0f} min={min(plies)} max={max(plies)}")

opening = sum(1 for p in plies if p < 20)
middle = sum(1 for p in plies if 20 <= p < 60)
endgame = sum(1 for p in plies if p >= 60)
print(f"  opening(<20): {opening} ({100*opening/n:.1f}%)")
print(f"  middlegame(20-60): {middle} ({100*middle/n:.1f}%)")
print(f"  endgame(>=60): {endgame} ({100*endgame/n:.1f}%)")

print(f"CP: mean={statistics.mean(cps):.0f} med={statistics.median(cps):.0f}")
print(f"CP gap: mean={statistics.mean(gaps):.0f} med={statistics.median(gaps):.0f}")
print(f"Targets: mean={statistics.mean(n_tgt):.1f} med={statistics.median(n_tgt):.0f}")

# Format check
d0 = json.loads(open(files[0]).readline())
print(f"\nFormat check:")
for k in ["fen", "best_move", "best_cp", "cp_gap_top1_top2", "ply"]:
    v = d0.get(k, "MISSING")
    print(f"  {k}: {v}")
st = d0.get("soft_targets", [])
print(f"  soft_targets: type={type(st).__name__}, len={len(st)}")
if st:
    print(f"  first entry: {st[0]}")
