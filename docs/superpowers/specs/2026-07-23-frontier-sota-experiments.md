# Frontier SOTA Chess Model — Experiment Card

**Date:** 2026-07-23  
**North star:** Maximize **verified policy Elo** (then search Elo), not soft top-1.  
**Lab now:** M5 Pro 24GB unified → promote winners to A40/A100.  
**Current floor:** ~25M autoresearch recipes all ~1320 Elo under short budgets (noise floor). Real frontier work needs **more soft data + longer train + Elo-gated promotion**.

Promotion rule (unchanged): beat champion by **>100 Elo** or (within noise **and** ≥20% faster pos/s). Soft metrics are diagnostic only.

---

## Stack we are building toward

```text
Data: MultiPV soft (shallow+deep) + puzzle/Syzygy + phase-balanced
  → Trunk: meta content×position (± Shaw / piece-square dual)
  → Heads: spatial prior + latent neural search + distributional value
  → Train: NorMuon → SWA → expert-iter (MCTS teacher) → optional PPO/KL
  → Infer: legal-mask π_refined (0 sims) → Gumbel-MCTS when needed
```

Frontier ≠ one clever arch. It is **data scale × calibrated value × amortized search × Elo loop**.

---

## Phase 0 — Unblock Elo signal (do first)

Short budgets + 26k soft rows pin everything at the 1320 bracket floor. Fix the measuring stick.

| ID | Experiment | Hypothesis | Primary metric | Budget |
|----|------------|------------|----------------|--------|
| **F0.1** | Soft cache → **≥500k–2M** MultiPV (SF18, depth 4–10, MultiPV=8) | More dense teacher signal raises Elo ceiling before arch matters | Elo vs fixed gauntlet | CPU harvest overnight |
| **F0.2** | Deep pack: phase-balanced SF depth≥14 + Syzygy perfect EG | Deep mix fixes middlegame/EG collapse | Phase-sliced Elo + EG score | CPU + merge |
| **F0.3** | Longer Elo protocol: 32–48 games/level, bracket 1200–2000 | Reduce ±100 noise so champions are real | Elo CI width | Eval only |
| **F0.4** | Train budget ladder: 2h / 8k steps / same data for top-3 recipes | Discriminate recipes that tie at 45m | Elo delta vs baseline | 3× GPU runs |

**Exit:** ≥1 recipe clearly > baseline Elo (or clear Pareto speed win) under F0.3 protocol.

---

## Phase 1 — Architecture (small → promote)

Keep ~20–40M for lab; promote only Elo winners to 200M/400M.

| ID | Experiment | Hypothesis | Fixed | Variable | Metric |
|----|------------|------------|-------|----------|--------|
| **F1.1** | `meta_shaw_elo` vs `baseline_deep_small` | Content×pos + Shaw > handcrafted rel_bias | data, steps, opt | attn factorization | Elo, pos/s |
| **F1.2** | `piece_square_dual` vs meta-only | Updating both streams helps tactics | same | dual QKV | Elo |
| **F1.3** | Depth↔width iso-param (`arch_deep_thin` 36L/192 vs 28L/256 vs 16L/384) | Depth wins for chess relational reasoning | params±10% | L/d | Elo |
| **F1.4** | QK-Norm + zero-init residual outs | Stability → better late Elo under NorMuon | recipe | init/norm | Elo, train crash rate |
| **F1.5** | GAB alone vs GAB+rel_bias vs neither | Learned geometry bias replaces handcrafted | same | gab/rel_bias | Elo |
| **F1.6** | Distributional value (64–128 bin HL-Gauss) vs 3-class WDL | Finer value unlocks search later | trunk | value head | Elo@0sims + Elo@32sims |
| **F1.7** | **Latent neural search head** (`exp195` / `meta_latent_search`) | Internal refine > one-shot spatial @ 0 external sims | trunk+data | topK, refine steps | Elo@0sims, latency |

**Exit:** Lab champion with (a) best Elo and (b) a speed champion within noise. Scale-up card filled.

---

## Phase 2 — Data & supervision (highest Elo ROI historically)

| ID | Experiment | Hypothesis | Notes |
|----|------------|------------|-------|
| **F2.1** | Soft α / T sweep: α∈{0.35,0.55,0.75}, T∈{1,2,4} | Chessformer soft-temp aux raises Elo without soft-loss overfitting | Elo-gate every 1k steps |
| **F2.2** | Hard ballast mix: soft_frac∈{0.7,0.85,1.0} + min_depth≥15 | Some hard SF CE prevents MultiPV mush | Needs HF or local hard shards |
| **F2.3** | Puzzle-heavy + Syzygy 30–50% deep_mix | Tactics+perfect EG lift policy Elo | Use `avewright/chess-soft-syzygy` when authed |
| **F2.4** | Blunder/hard-negative pack | Oversample positions model loses vs SF | Filter from self-play or SF diffs |
| **F2.5** | Elo-weighted sampling of teacher rows | Weight by teacher margin / human Elo (searchless-chess finding) | Don’t filter; reweight |
| **F2.6** | STM / Black-side oversampling | Fixes known Black weakness in progress notes | Balance STM in batches |

**Exit:** Soft+deep mix that beats Phase-1 champ by >100 Elo at matched steps.

---

## Phase 3 — Training systems (performance / efficiency / speed)

| ID | Experiment | Hypothesis |
|----|------------|------------|
| **F3.1** | NorMuon vs Polar-NorMuon vs AdamW @ matched wall-clock | Geometry-aware opt wins Elo/hour |
| **F3.2** | Hot Muon LR schedule (warmup → cosine floor 5%) | Higher LR early, don’t melt RL later |
| **F3.3** | SWA last 20–30% of steps | Averaged weights win Elo over last.pt |
| **F3.4** | Effective batch sweep (MPS: 64–192 microbatch) | Larger batch → better Elo/hour on 24GB unified |
| **F3.5** | Grad checkpoint vs fat batch tradeoff | On A40: no ckpt + huge batch; on 8GB: ckpt |
| **F3.6** | Compile / TF32 / bf16 (CUDA) vs MPS fp32 | Throughput Pareto without Elo loss |
| **F3.7** | Mid-train Elo probe every 500–1k steps | Early-kill weak recipes; Elo-gated ckpt select |

**Exit:** Documented train recipe: opt + LR + batch + SWA + compile flags for lab and A40.

---

## Phase 4 — Neural search (replace/augment MCTS)

Goal: great Elo **without** classical alpha-beta; MCTS only as teacher if needed.

| ID | Experiment | Hypothesis | Deploy |
|----|------------|------------|--------|
| **F4.1** | Latent search topK∈{8,16,32}, steps∈{2,3,5} | More refine compute → Elo until latency wall | π_refined @ 0 sims |
| **F4.2** | Legal-aware topK (already in `legal_mask`) ablate | Illegal topK poisons refine | Elo |
| **F4.3** | Backup-value aux vs WDL scalar / root Q | Differentiable minimax backup improves value | Elo@sims |
| **F4.4** | Distill MCTS visits into latent head (`exp183` teacher) | Amortize tree search into weights | Drop MCTS at deploy |
| **F4.5** | Recurrent think steps (shared refine block × T) | Adaptive compute; T↑ at test time | Elo vs latency curve |
| **F4.6** | MuZero-lite dynamics: g(h,a)→h' + r | Latent multi-ply without external tree | Hard; after F4.1–4.4 |

**Exit:** 0-sim Elo ≥ prior MCTS-32 Elo of the old spatial head (amortized search win).

---

## Phase 5 — RL / expert iteration (after supervised floor)

Do **not** RL from scratch. Anchor with KL to supervised prior.

| ID | Experiment | Hypothesis |
|----|------------|------------|
| **F5.1** | Expert-iter vs SF (`exp183 --mode sf`) | MCTS/SF visit targets lift Elo past supervised plateau |
| **F5.2** | Self-play expert-iter + historical league (20 ckpts) | Population diversity prevents style collapse |
| **F5.3** | SF-shaped reward: ΔWDL / Δcp per move + terminal z | Dense reward > sparse outcome REINFORCE |
| **F5.4** | PPO/KL trust region vs frozen prior | Prevents blunder-mode collapse |
| **F5.5** | RL LR = 0.05–0.1× pretrain; AdamW trunk, freeze early layers optional | NorMuon-hot destroys good policies in RL |
| **F5.6** | Gumbel-MCTS at iter gen (not PUCT AB) | Better low-sim improvement signal |

**Exit:** RL champ beats supervised champ on same Elo protocol; no soft-loss crowning.

---

## Phase 6 — Scale-up (frontier compute)

| ID | Experiment | When |
|----|------------|------|
| **F6.1** | Replay lab champion @ **200M** compact soft | After Phase 2 mix locked |
| **F6.2** | Replay @ **400M meta** (`exp191` recipe) | After F6.1 Elo↑ |
| **F6.3** | 700M deep-narrow A100 pretrain → expert-iter | Only if 400M saturates |
| **F6.4** | Multi-GPU soft harvest + continuous FT | Production loop |

Use `docs/superpowers/specs/2026-07-15-elo-autoresearch-scaleup.md` as the promote checklist.

---

## Phase 7 — Inference frontier (free Elo)

| ID | Experiment | Notes |
|----|------------|-------|
| **F7.1** | Policy temp / value blend / dynamic value weight (`exp117`) | No train |
| **F7.2** | Syzygy probe ≤5–6 men | Perfect EG |
| **F7.3** | Book + tablebase + π_refined | Deploy stack |
| **F7.4** | Optional Gumbel-MCTS 16–64 sims only if Elo↑ / latency OK | Tournament mode |
| **F7.5** | Batched leaf eval + CUDA graphs (A40+) | Search throughput |

---

## Suggested near-term schedule (M5 → cloud)

```text
Week 1 (M5):  F0.1–F0.4 + finish elo_wave (F1.*, F3.*)
Week 2 (M5):  F2 mixes + F4.1–F4.3 latent search Elo
Week 3 (A40): F6.1 200M replay + F5.1 expert-iter SF
Week 4 (A40): F6.2 400M meta + F5.2–F5.4 RL + F7 deploy
```

Harness entrypoints already in repo:
- Autoresearch: `scripts/run_autoresearch_elo_wave.sh`
- Latent search: `experiments/exp195_meta_latent_search.py`
- Expert-iter: `experiments/exp183_selfplay.py`
- Elo: `elo_eval_latest.py`
- Soft harvest: `scripts/harvest_local_multipv.py` / `experiments/exp186_sf_multipv_harvest_mp.py`

---

## Anti-goals (known failure modes)

1. Crowning on soft top-1 / soft_loss (1850 wall).  
2. Pure REINFORCE from scratch on laptop.  
3. Deep alpha-beta with a bad value head.  
4. Growing to 400M before data mix + Elo protocol are solid.  
5. Stacking every arch trick at once without iso-param ablations.

---

## One-line north star

**Dense MultiPV data → meta trunk → latent neural search + strong value → Elo-gated expert-iter → scale the winner.**
