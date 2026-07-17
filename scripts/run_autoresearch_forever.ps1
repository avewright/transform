# Continuous Elo autoresearch: after fixed waves, mutate champion forever.
# Safe to run while wave1 is active — waits for GPU/journal quiescence.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$OUT = if ($env:OUT) { $env:OUT } else { "outputs/autoresearch_8gb" }
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
$env:MOVE_VOCAB_VERSION = "compact"
$env:PYTHONUNBUFFERED = "1"
if (-not $env:STOCKFISH_PATH) {
  $sf = "stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"
  if (Test-Path $sf) { $env:STOCKFISH_PATH = (Resolve-Path $sf).Path }
}
$mins = if ($env:TRAIN_MINUTES) { $env:TRAIN_MINUTES } else { "180" }
$steps = if ($env:MAX_STEPS) { $env:MAX_STEPS } else { "5000" }
$maxGen = if ($env:MAX_GENERATIONS) { [int]$env:MAX_GENERATIONS } else { 50 }
$log = Join-Path $OUT "forever.log"

function Write-Log($msg) {
  $line = "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] $msg"
  Write-Host $line
  Add-Content -Path $log -Value $line
}

if ($env:SOFT_CACHE) {
  $SOFT = $env:SOFT_CACHE
} elseif (Test-Path "$OUT\soft_cache_200k.pt") {
  $SOFT = "$OUT\soft_cache_200k.pt"
} else {
  $SOFT = "$OUT\soft_cache.pt"
}
Write-Log "soft_cache=$SOFT"

function Get-DoneIds {
  $done = @{}
  $path = Join-Path $OUT "trials.jsonl"
  if (-not (Test-Path $path)) { return $done }
  Get-Content $path | ForEach-Object {
    try {
      $j = $_ | ConvertFrom-Json
      if ($j.id -and $j.status -eq "done" -and $null -ne $j.elo_estimate) {
        $done[$j.id] = $true
      }
    } catch {}
  }
  return $done
}

function Test-Exp194Running {
  # Avoid slow WMI: look for recent train.log activity + nvidia util is optional
  $lock = Join-Path $OUT "forever.lock"
  if (Test-Path $lock) {
    $age = (Get-Date) - (Get-Item $lock).LastWriteTime
    if ($age.TotalMinutes -lt 3) { return $true }
  }
  return $false
}

function Invoke-Trials($name, $only) {
  Write-Log "=== $name start ==="
  $onlyArgs = @($only | Where-Object { $_ })
  if ($onlyArgs.Count -eq 0) {
    Write-Log "${name}: empty only list, skip"
    return
  }
  $lock = Join-Path $OUT "forever.lock"
  Set-Content -Path $lock -Value (Get-Date -Format o)
  try {
    & python -u experiments/exp194_autoresearch_8gb.py --go `
      --soft-cache $SOFT `
      --train-minutes $mins `
      --max-steps $steps `
      --min-steps-done 4000 `
      --only @onlyArgs `
      --output-dir $OUT `
      2>&1 | ForEach-Object {
        $_ | Tee-Object -FilePath (Join-Path $OUT "forever_$name.log") -Append
        Set-Content -Path $lock -Value (Get-Date -Format o)
      }
  } finally {
    Remove-Item $lock -Force -ErrorAction SilentlyContinue
  }
  Write-Log "=== $name end ==="
}

# Wait for wave1 + existing chain to finish fixed waves if chain is mid-flight
# meta_shaw omitted from gate (too slow on 8GB); piece_square_dual covers the idea.
$wave1 = @(
  "baseline_deep_small", "no_relbias_vanilla",
  "wider_shallower", "gelu_no_swiglu", "fused_encoder"
)
Write-Log "forever: waiting for wave1 Elo-complete..."
while ($true) {
  $done = Get-DoneIds
  $n = (@($wave1 | Where-Object { $done.ContainsKey($_) })).Count
  Write-Log "wave1 elo-done $n/$($wave1.Count)"
  if ($n -ge $wave1.Count) { break }
  Start-Sleep -Seconds 180
}

# Fixed waves (skip if already done)
$w2 = @("qk_norm","zero_init_out","qk_norm_zero_init","meta_qk_norm","polar_normuon")
$w3 = @("cf_soft_temp","cf_soft_temp_heavy","cf_swa","cf_shaw_recipe","cf_value_heavy")
# piece_square_dual + puzzle/Syzygy data mixes (priority after wave1)
$w4 = @("piece_square_dual","puzzle_syzygy_mix","puzzle_syzygy_heavy","dual_puzzle_syzygy","gab","gab_no_relbias","gab_qk_norm","stack_ultimate","meta_shaw_soft_swa","muon_hot","label_smooth","dropout_zero","warmup_long")

function Pending($ids) {
  $done = Get-DoneIds
  return @($ids | Where-Object { -not $done.ContainsKey($_) })
}

$p = Pending $w2; if ($p.Count) { Invoke-Trials "wave2" $p }
$p = Pending $w3; if ($p.Count) { Invoke-Trials "wave3" $p }
$p = Pending $w4; if ($p.Count) { Invoke-Trials "wave4" $p }

function Get-MaxDoneElo {
  $m = -1.0
  $path = Join-Path $OUT "trials.jsonl"
  if (-not (Test-Path $path)) { return $m }
  Get-Content $path | ForEach-Object {
    try {
      $j = $_ | ConvertFrom-Json
      if ($j.status -eq "done" -and $null -ne $j.elo_estimate) {
        $e = [double]$j.elo_estimate
        if ($e -gt $m) { $m = $e }
      }
    } catch {}
  }
  return $m
}

# Continuous mutation generations
for ($g = 0; $g -lt $maxGen; $g++) {
  # If everyone is stuck on the SF LimitStrength floor (~1320), train longer.
  $maxElo = Get-MaxDoneElo
  if ($maxElo -ge 0 -and $maxElo -le 1350 -and $g -ge 1) {
    $steps = ([int]$steps + 3000).ToString()
    $mins = ([double]$mins + 90).ToString()
    Write-Log "Elo floor detected (max=$maxElo) -> escalate budget steps=$steps minutes=$mins"
  }

  Write-Log "=== mutate generation $g ==="
  $json = & python -u scripts/autoresearch_8gb/mutate.py --n 6 --generation $g --out-root $OUT
  Write-Log "mutate: $json"
  $obj = $json | ConvertFrom-Json
  $added = @($obj.added)
  if ($added.Count -eq 0) {
    Write-Log "no new mutations; bumping generation tag and retrying wider n"
    $json = & python -u scripts/autoresearch_8gb/mutate.py --n 10 --generation (100 + $g) --out-root $OUT
    $obj = $json | ConvertFrom-Json
    $added = @($obj.added)
  }
  if ($added.Count -eq 0) {
    Write-Log "still empty; sleeping 10m"
    Start-Sleep -Seconds 600
    continue
  }
  Invoke-Trials "mut_g$g" $added

  # Snapshot champion
  if (Test-Path (Join-Path $OUT "champion.json")) {
    Copy-Item -Force (Join-Path $OUT "champion.json") (Join-Path $OUT "champion_gen$g.json")
    Write-Log "champion snapshot: $((Get-Content (Join-Path $OUT 'champion.json') -Raw).Substring(0,[Math]::Min(200,(Get-Item (Join-Path $OUT 'champion.json')).Length)))"
  }
}

Write-Log "=== forever complete (max generations) ==="
