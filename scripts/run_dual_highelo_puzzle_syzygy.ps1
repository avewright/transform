# Train piece-square dual attention on high-Elo puzzles + HF Syzygy.
# Native python stderr (warnings) must not abort the pipeline.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:MOVE_VOCAB_VERSION = "compact"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:STOCKFISH_PATH) {
  $sf = "stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"
  if (Test-Path $sf) { $env:STOCKFISH_PATH = (Resolve-Path $sf).Path }
}

$puzzle = "outputs/exp193_puzzle_highelo/soft_cache.pt"
$syz = "outputs/syzygy_hf/soft_cache.pt"
$mix = "outputs/autoresearch_8gb/highelo_puzzle_syzygy_mix.pt"
$out = "outputs/autoresearch_8gb"
$mins = if ($env:TRAIN_MINUTES) { $env:TRAIN_MINUTES } else { "240" }
$steps = if ($env:MAX_STEPS) { $env:MAX_STEPS } else { "8000" }
$minPuzzle = if ($env:MIN_PUZZLE_ROWS) { [int]$env:MIN_PUZZLE_ROWS } else { 200000 }

function Get-SoftRows([string]$path) {
  if (-not (Test-Path $path)) { return 0 }
  $code = @"
import torch
d = torch.load(r'$path', map_location='cpu', weights_only=False)
print(int(d['board_array'].shape[0]))
"@
  $n = & python -c $code 2>$null
  if (-not $n) { return 0 }
  return [int]$n
}

Write-Host "waiting for high-Elo puzzle cache >= $minPuzzle rows: $puzzle"
while ($true) {
  $n = Get-SoftRows $puzzle
  if ($n -ge $minPuzzle) {
    Write-Host "puzzle cache ready: $n rows"
    break
  }
  Write-Host "  puzzle rows=$n (need $minPuzzle); sleep 30s"
  Start-Sleep -Seconds 30
}
if (-not (Test-Path $syz)) { throw "missing $syz - download avewright/chess-soft-syzygy first" }

Write-Host "merging puzzle + syzygy -> $mix"
& python -u scripts/autoresearch_8gb/merge_soft_caches.py $puzzle $syz --out $mix
if ($LASTEXITCODE -ne 0) { throw "merge failed exit=$LASTEXITCODE" }
$mixN = Get-SoftRows $mix
Write-Host "mix $mixN"

Write-Host "=== train dual_highelo_puzzle_syzygy steps=$steps minutes=$mins ==="
$log = Join-Path $out "dual_highelo_train.log"
& python -u experiments/exp194_autoresearch_8gb.py --go --force `
  --soft-cache $mix `
  --train-minutes $mins `
  --max-steps $steps `
  --min-steps-done 0 `
  --only dual_highelo_puzzle_syzygy `
  --output-dir $out *>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "train failed exit=$LASTEXITCODE" }
Write-Host "=== done ==="
