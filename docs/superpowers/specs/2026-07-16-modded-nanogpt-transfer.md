# Modded-NanoGPT → Chess Autoresearch Transfer Notes

Source: local clone `modded-nanogpt/` (KellerJordan NanoGPT speedrun).

## Landed in codebase (wave 2)

| Idea | Flag / trial id | Where |
|------|-----------------|-------|
| QK-RMSNorm | `use_qk_norm` / `qk_norm` | `chess_transformer_factory.py` |
| Zero-init out/FFN down | `zero_init_out_proj` / `zero_init_out` | same |
| Combo | `qk_norm_zero_init` | search_space |
| Meta + QK-Norm | `meta_qk_norm` | search_space |
| Polar Express NorMuon + cautious WD | `optimizer=polar_normuon` | `polar_normuon.py` + train_trial |

## Deferred (8GB / chess topology)

- FP8 head / FA3 / sliding window — H100-oriented; skip on 4060
- RoPE — board topology prefers Shaw / relative chess bias
- Multi-token prediction — policy is already next-move; optional later
- Batch-size schedule — easy follow-up once wave 2 Elo ranks settle

## How to run wave 2

After wave 1 finishes (or on free GPU):

```bash
bash scripts/run_autoresearch_wave2.sh
# Windows:
$env:MOVE_VOCAB_VERSION='compact'
python experiments/exp194_autoresearch_8gb.py --go --soft-cache outputs/autoresearch_8gb/soft_cache.pt --train-minutes 180 --max-steps 5000 --only qk_norm zero_init_out qk_norm_zero_init meta_qk_norm polar_normuon
```
