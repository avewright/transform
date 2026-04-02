"""Temporary script to stage, commit, and push experiment files."""
import subprocess, os, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PAT = os.environ.get("GH_PAT") or open(".env").read().strip().split("=", 1)[1]

def run(cmd):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip())
    return r.returncode

# Show status
run("git status --short")

# Stage experiment files and README
files = [
    "experiments/exp013_action_value.py",
    "experiments/exp014_mcts.py",
    "experiments/exp015_lora.py",
    "experiments/exp016_rich_features.py",
    "experiments/exp017_scaling_law.py",
    "README.md",
]
for f in files:
    if os.path.exists(f):
        run(f"git add {f}")
        print(f"  Staged: {f}")
    else:
        print(f"  MISSING: {f}")

# Show what's staged
run("git diff --cached --stat")

# Commit
msg = "Add exp013-017 experiment designs and update research roadmap\n\nNew experiments:\n- exp013: Action-value Q(s,a) training (all legal moves labeled)\n- exp014: MCTS search at inference\n- exp015: LoRA fine-tuning of backbone\n- exp016: Rich board features (attack maps, pawn structure)\n- exp017: Data scaling law measurement\n\nUpdated README with Phase 3 research roadmap and future improvements."
rc = run(f'git commit -m "{msg}"')
if rc != 0:
    print("Commit failed or nothing to commit")

# Set remote with PAT and push
remote_url = f"https://avewright:{PAT}@github.com/avewright/transform.git"
run(f"git remote set-url origin {remote_url}")
rc = run("git push origin main")

# Reset remote to not store PAT
run("git remote set-url origin https://github.com/avewright/transform.git")

if rc == 0:
    print("\n=== SUCCESS: Pushed to origin/main ===")
else:
    print("\n=== PUSH FAILED ===")
    # Try current branch name
    run("git branch --show-current")
    sys.exit(1)
