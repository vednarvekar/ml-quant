# Daily paper-trading run (no real orders). Scheduled via Windows Task Scheduler, Mon-Fri 19:30.
# Logs to data/paper/logs/YYYY-MM-DD.log
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$logDir = Join-Path $root "data\paper\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ((Get-Date -Format "yyyy-MM-dd") + ".log")
$env:PYTHONIOENCODING = "utf-8"
"=== start $(Get-Date -Format s) ===" | Out-File $log -Append -Encoding utf8
# Use the real interpreter + explicit paths (the venv launcher breaks under Task Scheduler)
$py = "$root\.venv-sched\Scripts\python.exe"   # project-local Python (visible to Task Scheduler)
$env:PYTHONPATH = "$root\src"
try {
    & $py -W ignore -m mlq.paper.run --update 2>&1 | Out-File $log -Append -Encoding utf8
} catch {
    "SCRIPT ERROR: $_" | Out-File $log -Append -Encoding utf8
}
"=== end $(Get-Date -Format s) exit=$LASTEXITCODE ===" | Out-File $log -Append -Encoding utf8
