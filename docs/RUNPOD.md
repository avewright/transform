# RunPod — max-Elo GPU workflow

Balance is limited (~$8–10). Prefer cheap community GPUs when available; **terminate** the pod when idle.

## One-time: SSH key in RunPod account

Local key used by this machine:

```bash
cat ~/.ssh/runpod.pub
# ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE8xf81opZlRj2DmHoPqTm2k78q+m2AFPUFcA/vB9acH runpod
```

1. Open [RunPod → Settings → SSH Public Keys](https://www.console.runpod.io/user/settings)
2. Paste the **full** line from `~/.ssh/runpod.pub` (must start with `ssh-ed25519`, not `SHA256:`)
3. Save. Each key on its own line if you have several.

Without this, TCP SSH often rejects the key even if `PUBLIC_KEY` is set on the pod.

## Connect (after a pod is up)

SSH config host (updated when we spawn):

```text
Host runpod-transform
```

In Cursor / VS Code: **Remote-SSH: Connect to Host…** → `runpod-transform`.

Or terminal:

```bash
ssh runpod-transform
```

## On the pod (efficient path)

```bash
cd /workspace
git clone https://github.com/avewright/transform.git   # or pull if present
cd transform
bash scripts/runpod_setup.sh

# Upload from Mac (separate terminal) — only what you need:
# scp -P <PORT> outputs/hf_437m_ft3h_hub/best_model.pt root@<IP>:/workspace/transform/outputs/champion/champion.pt
# scp -P <PORT> outputs/hf_soft_mix/soft_cache.pt outputs/hf_soft_mix/deep_soft.pt root@<IP>:/workspace/transform/outputs/hf_soft_mix/

export MOVE_VOCAB_VERSION=compact
export PYTHONUNBUFFERED=1

# Short soft-FT + pure-policy Elo + promote (crown by Elo, not soft_loss)
python -m harness.loop \
  --name rp_ft3j \
  --init outputs/champion/champion.pt \
  --soft-cache outputs/hf_soft_mix/soft_cache.pt \
  --deep-soft-cache outputs/hf_soft_mix/deep_soft.pt \
  --steps 3000 \
  --batch-size 48 \
  --soft-frac 0.85 --soft-alpha 0.38 --deep-mix-frac 0.42
```

Download `best.pt` / `outputs/champion/` **before** terminate.

## Tear down (save money)

```bash
# from Mac, with RUNPOD_API in .env
set -a && source .env && set +a
curl -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" \
  -H "Authorization: Bearer $RUNPOD_API"
```

Or stop/terminate from the [Pods console](https://console.runpod.io/pods).

Stopped pods still cost for disk; **terminate** when done.

## Why SSH failed before

We offered `~/.ssh/runpod` over public TCP; the server rejected it (`authorized_keys` never got the key). Setting env `PUBLIC_KEY` alone is not enough — account SSH keys + image start script that writes `authorized_keys` are required. New pods use a start command that does that injection.

## Active pod (updated by agent)

| Field | Value |
|-------|-------|
| Host alias | `runpod-transform` |
| Pod ID | `aywi9oap3c03zd` |
| GPU | RTX 3090 @ **$0.22/hr** |
| Connect | Remote-SSH → `runpod-transform` or `ssh runpod-transform` |

Terminate when done: `curl -X DELETE https://rest.runpod.io/v1/pods/aywi9oap3c03zd -H "Authorization: Bearer $RUNPOD_API"`
