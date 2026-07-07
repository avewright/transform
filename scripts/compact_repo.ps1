# One-shot repo compaction. Safe: moves to archive/, does not delete.
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path "$Root\chess_model.py")) { $Root = Split-Path $PSScriptRoot -Parent }
Set-Location $Root

$dirs = @(
    "archive/scratch",
    "archive/legacy",
    "archive/experiments/early",
    "archive/experiments/tools",
    "archive/data-scripts",
    "archive/docs",
    "docs"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

function Move-IfExists($from, $toDir) {
    if (Test-Path $from) {
        $dest = Join-Path $toDir (Split-Path $from -Leaf)
        if (-not (Test-Path $dest)) { Move-Item -LiteralPath $from -Destination $toDir -Force }
    }
}

# Scratch / one-off debug at repo root
$scratch = @(
    "_check_datasets.py", "_debug_train.py", "_git_push.py", "_prefetch_data.py",
    "_profile2.py", "_profile204m.py", "_profileA.py", "_profile_speed.py",
    "_test_loader.py", "_test_train.py", "_test_train2.py",
    "auto_push.py", "bench3.py", "bench_speed.py", "bench_speed2.py",
    "build_err.txt", "build_log.txt", "bug.md", "codex_ideas.md",
    "monitor_and_generate.py", "monitor_exp076.py", "monitor_exp083.py", "monitor_training.py",
    "test_dataloader.py", "test_nnue_speed.py", "throughput_sweep.py",
    "critical_moves_v1_backup.py", "generate_and_upload.py"
)
foreach ($f in $scratch) { Move-IfExists $f "archive/scratch" }

# Legacy Qwen / text / randopt path (chess-native transformer is active)
$legacy = @("model.py", "constrained.py", "attnres.py", "randopt.py", "nnue_model.py")
foreach ($f in $legacy) { Move-IfExists $f "archive/legacy" }

# Data pipeline one-offs → archive
$dataScripts = @(
    "build_hf_dataset_v2.py", "build_hf_dataset_v3.py", "analyze_elo_games.py", "analyze_lichess.py",
    "cache_lichess_data.py", "check_data_quality.py", "check_merged.py", "check_status.py",
    "generate_data_cpu.py", "generate_massive.py", "generate_sf_games.py", "hf_data.py",
    "label_positions.py", "lichess_data.py", "merge_datasets.py", "prepare_hf_dataset.py",
    "process_lichess_parquets.py", "relabel_deep.py", "relabel_depth8.py",
    "upload_dataset_hf.py", "upload_model_hf.py", "train_action_value.py"
)
foreach ($f in $dataScripts) { Move-IfExists $f "archive/data-scripts" }

# Extra architecture docs
$docs = @(
    "ARCHITECTURE_V1.md", "CURRENT_ARCHITECTURE.md", "CRITICAL_MOVES_EXPLAINED.md",
    "EXPORT_SUMMARY_2026-03-31.json", "EXPORT_SUMMARY_2026-03-31.md",
    "HF_DATASET_EXP085_README.md", "HUGGINGFACE_DATASETS.md", "MODEL_ARCHITECTURE_EXP083.md",
    "ROADMAP_3000_ELO.md", "SEARCH_WITH_ATTENTION.md"
)
foreach ($f in $docs) { Move-IfExists $f "archive/docs" }

# Shell one-offs
$shell = @(
    "exp110_pipeline.sh", "run_2hr_pipeline.sh", "run_exp074_tmux.sh",
    "run_prepare_hf_dataset_tmux.sh", "run_process_lichess_parquets_tmux.sh",
    "watchdog_exp075.sh", "watchdog_exp076.sh", "watchdog_exp101_102.ps1"
)
foreach ($f in $shell) { Move-IfExists $f "archive/scratch" }
Move-IfExists "watchdog_exp083.ps1" "archive/scratch"

# Experimental alt transformer
Move-IfExists "rope_transformer.py" "archive/legacy"

# Early experiments (keep exp052+ as current line)
Get-ChildItem "experiments" -Filter "exp0*.py" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^exp0([0-4][0-9]|050|051)_' } |
    ForEach-Object { Move-IfExists $_.FullName "archive/experiments/early" }

# Underscore experiment utilities
Get-ChildItem "experiments" -Filter "_*.py" -ErrorAction SilentlyContinue |
    ForEach-Object { Move-IfExists $_.FullName "archive/experiments/tools" }

Write-Host "Done. Core kept at root: chess_model, chess_features, move_vocab, config, data*, train, evaluate, selfplay, chess_transformer_factory, play*, uci_engine, search/*, experiments/exp052+"
