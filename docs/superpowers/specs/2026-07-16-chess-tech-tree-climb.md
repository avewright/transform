# Chess Tech Tree Climb (8GB → A40 → best model)

**Date:** 2026-07-16  
**North star:** Max **policy Elo** (then scale champion recipes)

## Tree (unlocked → next)

```text
L0  Square tokens + STM + SpatialPolicyHead          ✅ shipped
L1  Rel_bias / Shaw / Meta factored attn             ✅ wave1
L2  Soft multipv distill + NorMuon                   ✅ lab default
L3  Modded-nanogpt: QK-Norm, zero-init, Polar        ✅ wave2 queued
L4  Chessformer CF-240M: soft T=4, SWA, value weight ✅ wave3 queued
L5  Chessformer-2026 GAB (board-conditioned bias)    ✅ wave4 (this)
L6  Fusion stacks (GAB⊗QK⊗SWA⊗softT / meta⊗…)       ✅ wave4
L7  History planes (7-ply) + deeper soft harvest     🔜 data
L8  Value@1 / light search Elo transfer              🔜 eval
L9  Scale champion → 200M / 400M on A40              🔜 promote
L10 Self-play RL / MCTS closed loop                  🔮 later
```

## Wave map

| Wave | Focus | Trial ids |
|------|-------|-----------|
| 1 | Arch ablations | `baseline_deep_small`, `meta_shaw_*`, width/depth, GELU, fused |
| 2 | NanoGPT speedrun | `qk_norm`, `zero_init_out`, `polar_normuon`, … |
| 3 | CF-240M losses | `cf_soft_temp*`, `cf_swa`, `cf_shaw_recipe`, `cf_value_heavy` |
| 4 | GAB + fusions + train knobs | `gab*`, `stack_ultimate`, `meta_shaw_soft_swa`, `muon_hot`, … |
| 5 | Data / Elo protocol | deeper soft, value@1 probe (not in space yet) |

## Promotion rule (unchanged)

Elo-only (±100 noise). Soft holdout is diagnostic. Never promote on soft_loss.

## Scale-up card (when A40 free)

1. Take Pareto Elo champion from 8GB journal  
2. Re-train at `DEFAULT_400M_META_CONFIG` / CF-scale with same recipe flags  
3. Re-Elo with same SF protocol; optionally MCTS probe  

## Papers

- Monroe & Chalmers CF-240M — [arXiv:2409.12272](https://arxiv.org/abs/2409.12272)  
- Chessformer GAB — [arXiv:2605.19091](https://arxiv.org/html/2605.19091v1)  
- Ruoss GC-270M — [arXiv:2402.04494](https://arxiv.org/abs/2402.04494)  
- Keller Jordan modded-nanogpt — local `modded-nanogpt/`
