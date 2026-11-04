<#
.SYNOPSIS
  Unattended overnight run of the accuracy improvements (retrieval re-ranking/routing, verifier v2, Qwen2.5-3B).

.DESCRIPTION
  Steps (each is logged and the run continues if one fails; earlier result files are never overwritten):
    1. pip install bitsandbytes + accelerate (4-bit loading of the 3B model)
    2. download the re-ranker (BAAI/bge-reranker-base) and entailment model (cross-encoder/nli-deberta-v3-small)
    3. retrieval v2 evaluation              -> data\evidence\evaluation_evidence_v2.json
    4. verifier v2 on the existing drafts   -> data\evidence\generator_reverify.json
    5. download Qwen2.5-3B-Instruct (~6 GB)
    6. generator comparison v2 (all models, new retriever + verifier v2) -> data\evidence\generator_comparison_v2.json
    7. backend tests
  Keeps the laptop awake while it runs. Log: logs\overnight_<time>.log. Expected 2-3 hours.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\overnight.ps1
#>
[CmdletBinding()]
param([switch]$Skip3B)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir ("overnight_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$CacheDir = Join-Path $Root ".cache"
$TmpDir = Join-Path $CacheDir "tmp"
New-Item -ItemType Directory -Force -Path $CacheDir, $TmpDir | Out-Null
$env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"
$env:HF_HOME = Join-Path $CacheDir "huggingface"
$env:TORCH_HOME = Join-Path $CacheDir "torch"
$env:TEMP = $TmpDir
$env:TMP = $TmpDir

# keep the computer awake (not the display) while this script runs; reverts automatically when it exits
try {
    Add-Type -Namespace PV -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);' -ErrorAction Stop
    [void][PV.Power]::SetThreadExecutionState([uint32]2147483649)
    $awake = "on"
} catch { $awake = "could not be set - set Windows sleep to 'Never' for tonight" }

$results = [ordered]@{}
function Log($msg) { Add-Content -Path $Log -Value $msg -Encoding UTF8 }
function Invoke-Py([string[]]$PyArgs) {
    & $VenvPy @PyArgs 2>&1 | ForEach-Object { $line = "$_"; Log $line; $line } | Out-Host
    return $LASTEXITCODE
}
function Step([string]$name, [scriptblock]$body) {
    Write-Host ""
    Write-Host ("==> {0}   ({1})" -f $name, (Get-Date -Format "HH:mm")) -ForegroundColor Cyan
    Log "==> $name  $(Get-Date -Format s)"
    $t = Get-Date
    try { $code = & $body } catch { $code = 99; Log "EXCEPTION: $_"; Write-Host "EXCEPTION: $_" -ForegroundColor Red }
    $code = @($code)[-1]
    $mins = [math]::Round(((Get-Date) - $t).TotalMinutes, 1)
    $status = if ($code -eq 0) { "OK" } else { "FAILED (exit $code)" }
    $results[$name] = "$status, $mins min"
    $color = if ($code -eq 0) { "Green" } else { "Red" }
    Write-Host "    $status in $mins min" -ForegroundColor $color
    Log "    $status in $mins min"
}

$start = Get-Date
Write-Host "Overnight run - log: $Log"
Write-Host "Keep-awake: $awake. Keep the laptop plugged in. You can go to sleep now."
if (-not (Test-Path $VenvPy)) { Write-Host "No .venv - run run.cmd and scripts\build_evidence.ps1 first." -ForegroundColor Red; Read-Host "Press Enter"; exit 1 }

Step "1 install bitsandbytes + accelerate" { Invoke-Py @("-m", "pip", "install", "--quiet", "bitsandbytes", "accelerate", "huggingface_hub") }
Step "2 download re-ranker + entailment model" { Invoke-Py @((Join-Path $Root "scripts\download_generators.py"), "--only-improvements") }
Push-Location $Backend
try {
    Step "3 retrieval v2 evaluation" { Invoke-Py @("-m", "pvsys.pipeline", "evidence-eval-v2") }
    Step "4 verifier v2 on existing drafts" { Invoke-Py @("-m", "pvsys.pipeline", "evidence-reverify") }
} finally { Pop-Location }
if (-not $Skip3B) {
    Step "5 download Qwen2.5-3B-Instruct" { Invoke-Py @((Join-Path $Root "scripts\download_generators.py"), "--with-3b") }
}
Push-Location $Backend
try {
    Step "6 generator comparison v2" { Invoke-Py @("-m", "pvsys.pipeline", "evidence-compare", "--out", "generator_comparison_v2.json") }
    Step "7 backend tests" { Invoke-Py @("-m", "pytest", "-q", "-p", "no:cacheprovider") }
} finally { Pop-Location }

$total = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host ""
Write-Host "Summary ($total min):" -ForegroundColor Cyan
Log "SUMMARY ($total min)"
foreach ($k in $results.Keys) { Write-Host ("  {0,-45} {1}" -f $k, $results[$k]); Log ("  {0,-45} {1}" -f $k, $results[$k]) }
Write-Host ""
Write-Host "Log: $Log"
Write-Host "Tell Claude 'done' - the results are in data\evidence\*_v2.json and generator_reverify.json." -ForegroundColor Green
Log "DONE"
Read-Host "Press Enter to close"
