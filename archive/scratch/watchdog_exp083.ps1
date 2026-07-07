$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\AWright\OneDrive - Kahua, Inc\Projects\transform"
$ExpDir = Join-Path $RepoRoot "outputs\exp083_sf_opening_stream"
$WatchdogLog = Join-Path $ExpDir "watchdog.log"
$StdoutLog = Join-Path $ExpDir "launcher_stdout.log"
$StderrLog = Join-Path $ExpDir "launcher_stderr.log"
$TrainScript = "experiments/exp083_sf_opening_stream.py"
$PythonExe = "python"
$CheckIntervalSec = 60
$RestartDelaySec = 20
$MaxRestarts = 100000
$RestartCount = 0

New-Item -ItemType Directory -Force -Path $ExpDir | Out-Null

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    Write-Output $line
    Add-Content -Path $WatchdogLog -Value $line
}

function Get-TrainingProcesses {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "^python(\.exe)?$" -and
            $_.CommandLine -like "*$TrainScript*"
        }
}

function Launch-Training {
    Write-Log "Launching exp083 trainer"
    $proc = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $TrainScript `
        -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -WindowStyle Hidden `
        -PassThru
    Start-Sleep -Seconds 5
    Write-Log "Started PID=$($proc.Id)"
}

function Get-StatusSummary {
    $statusPath = Join-Path $ExpDir "status.json"
    if (-not (Test-Path $statusPath)) {
        return "status=missing"
    }
    try {
        $status = Get-Content $statusPath -Raw | ConvertFrom-Json
        return "train_steps=$($status.train_steps) games=$($status.generator.games_generated) positions=$($status.generator.positions_generated) buffer=$($status.buffer_size) queue=$($status.queue_size)"
    } catch {
        return "status=parse_error"
    }
}

Write-Log "=========================================="
Write-Log "WATCHDOG STARTED for exp083_sf_opening_stream"
Write-Log "Repo=$RepoRoot"
Write-Log "Script=$TrainScript"
Write-Log "=========================================="

if (-not (Get-TrainingProcesses)) {
    Launch-Training
} else {
    $existing = Get-TrainingProcesses | Select-Object -ExpandProperty ProcessId
    Write-Log "Existing trainer found: PID=$($existing -join ',')"
}

while ($true) {
    Start-Sleep -Seconds $CheckIntervalSec
    $procs = Get-TrainingProcesses
    $summary = Get-StatusSummary

    if ($procs) {
        $pids = $procs | Select-Object -ExpandProperty ProcessId
        Write-Log "OK PID=$($pids -join ',') $summary"
        continue
    }

    $RestartCount += 1
    Write-Log "DEAD trainer missing; restart_count=$RestartCount $summary"

    if ($RestartCount -ge $MaxRestarts) {
        Write-Log "FATAL max restarts reached"
        exit 1
    }

    Start-Sleep -Seconds $RestartDelaySec
    Launch-Training
}
