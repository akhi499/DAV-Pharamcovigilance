<#
.SYNOPSIS
  One-command launcher for the FAERS Drug Safety Workbench (FastAPI backend + React dashboard).

.DESCRIPTION
  First run: creates .venv, installs Python and Node dependencies, builds the
  processed data if it is missing, then starts the backend (port 8000) and the
  dashboard (port 5173), opens the browser and waits. Press Enter (or Ctrl+C)
  to stop both processes cleanly. Works in Windows PowerShell 5.1 and PowerShell 7.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run.ps1
  powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Stop
  powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Rebuild     # re-run the whole pipeline
  powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -NoNlp       # skip installing torch/transformers
  powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Test        # run all checks before starting
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$Rebuild,
    [switch]$NoNlp,
    [switch]$NoBrowser,
    [switch]$Test,
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173
)

$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 turns any stderr output of a native program into a terminating error when
# $ErrorActionPreference is "Stop" and stderr is redirected. Probing commands (python -c "import torch",
# taskkill) legitimately write to stderr, so they run through this helper with errors relaxed.
function Invoke-Quiet([scriptblock]$Block) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Block 2>&1 | Out-String } finally { $ErrorActionPreference = $old }
}
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$RunDir = Join-Path $Root ".run"
$LogDir = Join-Path $Root "logs"
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path $RunDir, $LogDir | Out-Null
# keep every download / cache (pip, npm, PyTorch, Hugging Face) on the project's drive
$CacheDir = Join-Path $Root ".cache"
New-Item -ItemType Directory -Force -Path (Join-Path $CacheDir "tmp") | Out-Null
$env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"
$env:npm_config_cache = Join-Path $CacheDir "npm"
$env:HF_HOME = Join-Path $CacheDir "huggingface"
$env:TORCH_HOME = Join-Path $CacheDir "torch"
$env:TEMP = Join-Path $CacheDir "tmp"
$env:TMP = $env:TEMP

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

function Stop-Tree([int]$ProcessId) {
    # taskkill /T stops the process and every child (npm -> node -> esbuild, python -> uvicorn workers)
    if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        Invoke-Quiet { & taskkill.exe /PID $ProcessId /T /F } | Out-Null
    }
}

function Stop-All {
    foreach ($name in "backend", "frontend") {
        $pidFile = Join-Path $RunDir "$name.pid"
        if (Test-Path $pidFile) {
            $p = [int](Get-Content $pidFile -Raw)
            Stop-Tree $p
            Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
            Write-Host "    stopped $name (PID $p)"
        }
    }
}

if ($Stop) { Write-Step "Stopping services"; Stop-All; exit 0 }

function Find-Python {
    foreach ($cand in @(@("py", "-3"), @("python"), @("python3"))) {
        $exe = $cand[0]
        if (Get-Command $exe -ErrorAction SilentlyContinue) {
            $args2 = @()
            if ($cand.Count -gt 1) { $args2 = $cand[1..($cand.Count - 1)] }
            $ver = (Invoke-Quiet { & $exe @args2 -c "import sys; print('PYVER=%d.%d' % sys.version_info[:2])" }) -split "`r?`n" |
                Where-Object { $_ -like "PYVER=*" } | Select-Object -First 1
            if ($ver) { $ver = $ver.Substring(6) }
            if ($LASTEXITCODE -eq 0 -and $ver) {
                $parts = $ver.Trim().Split(".")
                if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10) { return @($exe) + $args2 }
            }
        }
    }
    throw "Python 3.10+ was not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH') and run this script again."
}

function Test-Port([int]$Port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $c.Connect("127.0.0.1", $Port); return $true } catch { return $false } finally { $c.Close() }
}

function Wait-Http([string]$Url, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { return $true }
        } catch { Start-Sleep -Milliseconds 700 }
    }
    return $false
}

# ---------------------------------------------------------------- setup ----
Stop-All  # clean up anything left from a previous run

if (-not (Test-Path $VenvPy)) {
    $py = @(Find-Python)
    Write-Step "Creating virtual environment (.venv) with $($py -join ' ')"
    $pyArgs = @()
    if ($py.Count -gt 1) { $pyArgs = $py[1..($py.Count - 1)] }
    & $py[0] @pyArgs -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}

$stamp = Join-Path $RunDir "deps.stamp"
$reqHash = (Get-FileHash (Join-Path $Backend "requirements.txt")).Hash
if (-not (Test-Path $stamp) -or ((Get-Content $stamp -Raw).Trim() -ne $reqHash)) {
    Write-Step "Installing Python dependencies"
    & $VenvPy -m pip install --upgrade pip --quiet
    & $VenvPy -m pip install -r (Join-Path $Backend "requirements-dev.txt") --quiet
    if ($LASTEXITCODE -ne 0) { throw "pip install failed - see the messages above." }
    Set-Content -Path $stamp -Value $reqHash
}

if (-not $NoNlp) {
    Invoke-Quiet { & $VenvPy -c "import torch, transformers" } | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Step "Installing the optional language-model stack (CPU-only torch, ~200 MB)"
        & $VenvPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
        & $VenvPy -m pip install -r (Join-Path $Backend "requirements-nlp.txt") --quiet
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Language-model packages could not be installed. The dashboard still works:"
            Write-Warn "precomputed categories are shown and search falls back to spelling similarity."
        }
    }
    $weights = Join-Path $Root "models\SapBERT-from-PubMedBERT-fulltext\model.safetensors"
    if (-not (Test-Path $weights)) {
        Write-Step "Downloading SapBERT weights (pinned revision, ~440 MB)"
        & $VenvPy (Join-Path $Root "scripts\download_model.py")
        if ($LASTEXITCODE -ne 0) { Write-Warn "Model download failed; continuing in lexical-fallback mode." }
    }
}

if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "Node.js 18+ was not found. Install the LTS version from https://nodejs.org/ and run this script again."
}
if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
    Write-Step "Installing dashboard dependencies (npm ci)"
    Push-Location $Frontend
    try {
        & npm.cmd ci --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { & npm.cmd install --no-audit --no-fund }
        if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
    } finally { Pop-Location }
}

# ----------------------------------------------------------------- data ----
$signals = Join-Path $Root "data\processed\signals.parquet"
if ($Rebuild -or -not (Test-Path $signals)) {
    Push-Location $Backend
    try {
        $zip = Join-Path $Root "data\raw\faers_ascii_2025q1.zip"
        if (-not (Test-Path $zip)) {
            Write-Step "Downloading FAERS 2025 Q1 from fis.fda.gov (~67 MB)"
            & $VenvPy -m pvsys.pipeline download
            if ($LASTEXITCODE -ne 0) { throw "Download failed. Place faers_ascii_2025q1.zip in data\raw and retry." }
        }
        Write-Step "Running the pipeline (ingest, clean, quality, signals, NLP) - a few minutes"
        & $VenvPy -m pvsys.pipeline all
        if ($LASTEXITCODE -ne 0) { throw "Pipeline failed - see the messages above." }
    } finally { Pop-Location }
}

$kb = Join-Path $Root "data\nlp\kb_chunks.parquet"
$preds = Join-Path $Root "data\nlp\predictions.parquet"
if ((Test-Path $preds) -and -not (Test-Path $kb)) {
    Write-Step "Building the question-answering knowledge base (FDA FAERS documents + CTCAE definitions)"
    Push-Location $Backend
    try {
        & $VenvPy -m pvsys.pipeline rag-index
        if ($LASTEXITCODE -ne 0) { Write-Warn "Knowledge base build failed; the 'Ask the data' page will report it. Other pages are unaffected." }
    } finally { Pop-Location }
}

$evidenceIdx = Join-Path $Root "data\evidence\chunks.parquet"
if (-not (Test-Path $evidenceIdx)) {
    Write-Warn "Clinical sources (FDA labels, safety communications, PubMed) are not indexed yet; 'Ask the data' will answer"
    Write-Warn "from FAERS only. Run scripts\build_evidence.ps1 once to add them."
}

if ($Test) {
    Write-Step "Running test suites"
    Push-Location $Backend; & $VenvPy -m pytest -q; & $VenvPy -m ruff check .; Pop-Location
    Push-Location $Frontend; & npm.cmd test; & npm.cmd run typecheck; & npm.cmd run build; Pop-Location
}

# ---------------------------------------------------------------- start ----
foreach ($port in $ApiPort, $WebPort) {
    if (Test-Port $port) { throw "Port $port is already in use. Close the program using it or pass -ApiPort/-WebPort." }
}

Write-Step "Starting backend on http://127.0.0.1:$ApiPort"
$api = Start-Process -FilePath $VenvPy -ArgumentList "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "$ApiPort" `
    -WorkingDirectory $Backend -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $LogDir "backend.out.log") -RedirectStandardError (Join-Path $LogDir "backend.err.log")
Set-Content -Path (Join-Path $RunDir "backend.pid") -Value $api.Id

Write-Step "Starting dashboard on http://localhost:$WebPort"
$env:PV_API_URL = "http://127.0.0.1:$ApiPort"
$web = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev", "--", "--port", "$WebPort", "--strictPort" `
    -WorkingDirectory $Frontend -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $LogDir "frontend.out.log") -RedirectStandardError (Join-Path $LogDir "frontend.err.log")
Set-Content -Path (Join-Path $RunDir "frontend.pid") -Value $web.Id

try {
    if (-not (Wait-Http "http://127.0.0.1:$ApiPort/api/health" 90)) {
        throw "Backend did not start. See logs\backend.err.log"
    }
    if (-not (Wait-Http "http://localhost:$WebPort/" 90)) {
        throw "Dashboard did not start. See logs\frontend.err.log"
    }
    Write-Host ""
    Write-Host "Dashboard: http://localhost:$WebPort" -ForegroundColor Green
    Write-Host "API docs : http://127.0.0.1:$ApiPort/docs" -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process "http://localhost:$WebPort" }
    Write-Host ""
    Read-Host "Press Enter to stop both services"
} finally {
    Write-Step "Stopping services"
    Stop-All
}
