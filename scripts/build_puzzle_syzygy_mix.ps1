# Wait for puzzle + Syzygy soft caches, merge into autoresearch deep mix.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$puzzle = "outputs/exp193_puzzle_soft/soft_cache.pt"
$syz = "outputs/syzygy_hf/soft_cache.pt"
$out = "outputs/autoresearch_8gb/puzzle_syzygy_mix.pt"
$env:MOVE_VOCAB_VERSION = "compact"
$env:PYTHONUNBUFFERED = "1"

Write-Host "waiting for $puzzle and $syz ..."
while (-not ((Test-Path $puzzle) -and (Test-Path $syz))) {
  Start-Sleep -Seconds 60
  $p = if (Test-Path $puzzle) { "yes" } else { "no" }
  $s = if (Test-Path $syz) { "yes" } else { "no" }
  Write-Host "$(Get-Date -Format HH:mm:ss) puzzle=$p syzygy=$s"
}
python -u scripts/autoresearch_8gb/merge_soft_caches.py $puzzle $syz --out $out
Write-Host "merged -> $out"
# smoke shapes
python -c "import torch; d=torch.load(r'$out',map_location='cpu',weights_only=False); print('mix rows', d['board_array'].shape[0])"
