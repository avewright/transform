# Deprecated entrypoint: delegates to forever runner (waves 2–4 + mutate loop).
# Kept so older launchers still work.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_autoresearch_forever.ps1
