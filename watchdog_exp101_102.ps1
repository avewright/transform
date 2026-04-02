# Watchdog: monitor exp101, auto-launch exp102 when done
# Usage: powershell -File watchdog_exp101_102.ps1

$exp101Log = "C:\temp\chess_training\exp101\exp101.log"
$checkInterval = 120  # seconds

Write-Host "[watchdog] Monitoring exp101 at $exp101Log"
Write-Host "[watchdog] Will launch exp102 when exp101 finishes"

while ($true) {
    $procs = Get-Process python -ErrorAction SilentlyContinue
    $exp101Running = $false
    foreach ($p in $procs) {
        try {
            $cmd = (Get-WmiObject Win32_Process -Filter "ProcessId=$($p.Id)").CommandLine
            if ($cmd -match "exp101") { $exp101Running = $true; break }
        } catch {}
    }

    if (-not $exp101Running) {
        Write-Host "[watchdog] exp101 finished at $(Get-Date -Format 'HH:mm:ss')"
        
        # Show final log lines
        if (Test-Path $exp101Log) {
            Write-Host "[watchdog] exp101 final output:"
            Get-Content $exp101Log | Select-Object -Last 10
        }

        # Check for checkpoints
        $ckpts = Get-ChildItem "C:\temp\chess_training\exp101\*.pt" -ErrorAction SilentlyContinue
        Write-Host "[watchdog] exp101 checkpoints: $($ckpts.Count)"
        
        # Launch exp102
        Write-Host "[watchdog] Launching exp102..."
        Set-Location "C:\Users\AWright\OneDrive - Kahua, Inc\Projects\transform"
        python experiments/exp102_auxiliary_losses.py --max-files 1 --batch-size 16 --accum-steps 32 --lr 3e-5 --value-weight 0.25 --aux-weight 0.10
        Write-Host "[watchdog] exp102 finished at $(Get-Date -Format 'HH:mm:ss')"
        break
    }

    # Show latest log line
    if (Test-Path $exp101Log) {
        $last = Get-Content $exp101Log | Select-Object -Last 1
        Write-Host "[watchdog $(Get-Date -Format 'HH:mm:ss')] exp101 running. $last"
    }
    Start-Sleep $checkInterval
}
