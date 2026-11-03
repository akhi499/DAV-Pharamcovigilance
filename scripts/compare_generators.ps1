<#
.SYNOPSIS
  Extractive vs generative answering across local language models (for the research paper).

.DESCRIPTION
  1. Downloads Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct (about 6.5 GB, once) into models\ with provenance.
     -With3B also downloads Qwen2.5-3B-Instruct (~6 GB, Qwen research licence; too large for a 4 GB GPU, so it
     runs on the CPU and takes much longer).
  2. Runs every models\generator* model (0.5B, 1.5B, 1.7B[, 3B]) on the same 89 test drugs x 2 tasks with
     identical retrieved passages, on the NVIDIA GPU if available.
  Results: data\evidence\generator_comparison.json. Log: logs\compare_generators_<time>.log.
  Needs scripts\build_evidence.ps1 to have been run before (evidence index + CUDA PyTorch).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\compare_generators.ps1
  powershell -ExecutionPolicy Bypass -File scripts\compare_generators.ps1 -With3B
  powershell -ExecutionPolicy Bypass -File scripts\compare_generators.ps1 -Limit 5      # 2-minute smoke test
#>
[CmdletBinding()]
param(
    [switch]$With3B,
    [switch]$SkipDownload,
    [int]$Limit = 0
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir ("compare_generators_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
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

function Write-Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan; Add-Content -Path $Log -Value "==> $msg" -Encoding UTF8 }
function Fail($msg) {
    Write-Host ""; Write-Host "FAILED: $msg" -ForegroundColor Red; Add-Content -Path $Log -Value "FAILED: $msg" -Encoding UTF8
    Write-Host "Log: $Log"; Read-Host "Press Enter to close"; exit 1
}
function Invoke-Py([string[]]$PyArgs) {
    & $VenvPy @PyArgs 2>&1 | ForEach-Object { $line = "$_"; Add-Content -Path $Log -Value $line -Encoding UTF8; $line } | Out-Host
    return $LASTEXITCODE
}

$start = Get-Date
Write-Host "Log file: $Log"
if (-not (Test-Path $VenvPy)) { Fail "The Python environment (.venv) does not exist. Run run.cmd and scripts\build_evidence.ps1 first." }
if (-not (Test-Path (Join-Path $Root "data\evidence\chunks.parquet"))) { Fail "The evidence index is missing. Run scripts\build_evidence.ps1 first." }

$code = Invoke-Py @("-c", "import sys, torch; ok = torch.cuda.is_available(); print('Using GPU:', torch.cuda.get_device_name(0) if ok else 'none - models will run on the CPU (slow)'); print('GPU memory GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if ok else '-')")
if ($code -ne 0) { Fail "PyTorch is not working in .venv - run scripts\build_evidence.ps1 first." }

if (-not $SkipDownload) {
    Write-Step "1/2  Downloading generator models (about 6.5 GB once; kept in models\, cache in .cache\)"
    $null = Invoke-Py @("-m", "pip", "install", "--quiet", "huggingface_hub")
    $dlArgs = @((Join-Path $Root "scripts\download_generators.py"))
    if ($With3B) { $dlArgs += "--with-3b" }
    $code = Invoke-Py $dlArgs
    if ($code -ne 0) { Fail "Model download failed (exit $code). Check the internet connection and re-run; finished models are kept." }
}

Write-Step "2/2  Comparing extractive and generative answers (89 test drugs x 2 tasks per model)"
Push-Location $Backend
try {
    $cmpArgs = @("-m", "pvsys.pipeline", "evidence-compare")
    if ($Limit -gt 0) { $cmpArgs += @("--limit", "$Limit") }
    $code = Invoke-Py $cmpArgs
    if ($code -ne 0) { Fail "Comparison failed (exit $code)." }
} finally { Pop-Location }

$mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host ""
Write-Host "Done in $mins minutes. Results: data\evidence\generator_comparison.json" -ForegroundColor Green
Write-Host "Log: $Log"
Add-Content -Path $Log -Value "DONE in $mins minutes" -Encoding UTF8
Read-Host "Press Enter to close"
