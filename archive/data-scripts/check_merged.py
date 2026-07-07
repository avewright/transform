"""Quick check merged dataset has all sources."""
import json
import glob

files = sorted(glob.glob("outputs/exp100_diverse_data/dataset/positions_*.jsonl"))
print(f"{len(files)} files total")

# Sample from different sections
for idx in [0, 50, 114, 115, 116, 130, 131, 145, 149]:
    if idx < len(files):
        d = json.loads(open(files[idx]).readline())
        ply = d.get("ply", "N/A")
        cp = d.get("best_cp", "?")
        src = d.get("source", d.get("relabel_source", "unknown"))
        n_tgt = len(d.get("soft_targets", []))
        print(f"  File {idx:3d}: ply={ply:>5} cp={str(cp):>6} targets={n_tgt:>3} source={src}")

# Full distribution
opening = 0
middle = 0
endgame = 0
total = 0
for f in files:
    for line in open(f):
        d = json.loads(line)
        total += 1
        ply = d.get("ply", 0) or 0
        if ply == 0:
            endgame += 1  # synthetic endgames have no ply
        elif ply < 20:
            opening += 1
        elif ply < 60:
            middle += 1
        else:
            endgame += 1

print(f"\nTotal: {total}")
print(f"  opening(<20): {opening} ({100*opening/total:.1f}%)")
print(f"  middlegame(20-60): {middle} ({100*middle/total:.1f}%)")
print(f"  endgame(>=60 or synthetic): {endgame} ({100*endgame/total:.1f}%)")
