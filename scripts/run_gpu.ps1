# Use Python 3.12 — it has CUDA PyTorch. Default `python` (3.14) is CPU-only.
$Py = "C:\Program Files\Python312\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "py -3.12"
}

& $Py -c @"
import torch
assert torch.cuda.is_available(), (
    'CUDA not available. Reinstall GPU torch:\n'
    '  py -3.12 -m pip install torch==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129'
)
print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
"@

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Set-Location (Split-Path $PSScriptRoot -Parent)
& $Py @args
