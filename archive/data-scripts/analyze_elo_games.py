"""Analyze game patterns from ELO eval logs."""
import json
import sys

log_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/elo_eval_exp093_d8_ema.log"

with open(log_path) as f:
    games = []
    for line in f:
        line = line.strip()
        if line.startswith("game "):
            g = json.loads(line[5:])
            games.append(g)

for elo in sorted(set(g["sf_elo"] for g in games)):
    elo_games = [g for g in games if g["sf_elo"] == elo]
    losses = [g for g in elo_games if g["score"] == 0.0]
    wins = [g for g in elo_games if g["score"] == 1.0]
    draws = [g for g in elo_games if g["score"] == 0.5]
    
    n = len(elo_games)
    score = sum(g["score"] for g in elo_games) / n
    avg_ply = sum(g["plies"] for g in elo_games) / n
    
    print(f"\n=== SF {elo} === Score: {score:.1%}  W/D/L: {len(wins)}/{len(draws)}/{len(losses)}  Avg plies: {avg_ply:.0f}")
    
    if losses:
        loss_avg_ply = sum(g["plies"] for g in losses) / len(losses)
        print(f"  Loss avg plies: {loss_avg_ply:.0f}")
        # Short losses (< 40 plies) = opening blunders
        short = [g for g in losses if g["plies"] < 40]
        mid = [g for g in losses if 40 <= g["plies"] < 80]
        long_l = [g for g in losses if g["plies"] >= 80]
        print(f"  Short (<40ply): {len(short)}  Mid (40-80): {len(mid)}  Long (80+): {len(long_l)}")
        
        for g in sorted(losses, key=lambda x: x["plies"]):
            print(f"    {g['color']:5s} {g['opening']:20s} plies={g['plies']:3d} {g['termination']}")
    
    if wins:
        win_avg_ply = sum(g["plies"] for g in wins) / len(wins)
        print(f"  Win avg plies: {win_avg_ply:.0f}")
        
    # Termination stats
    terms = {}
    for g in elo_games:
        t = g["termination"]
        terms[t] = terms.get(t, 0) + 1
    print(f"  Terminations: {terms}")
