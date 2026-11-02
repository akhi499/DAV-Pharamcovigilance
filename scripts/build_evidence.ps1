<#
.SYNOPSIS
  Builds the external clinical-evidence index (FDA labels, FDA Drug Safety Communications, PubMed reviews)
  and evaluates retrieval and generated answers. Run once; re-run any time to refresh.

.DESCRIPTION
  Steps (each logged to logs\build_evidence_<time>.log):
    1. fetch  - download labels (openFDA + DailyMed), FDA Drug Safety Communications and PubMed abstracts for the
                most-reported FAERS drugs. Cached in data\evidence\raw, so re-runs only fetch what is missing.  (~5-15 min)
    2. index  - chunk the documents and embed them with models\bge-small-en-v1.5 on the CPU.                  (~10-30 min)
    3. eval   - retrieval evaluation, insufficient-evidence threshold, generated-answer evaluation with
                models\generator (Qwen2.5-0.5B-Instruct) and signal-label concordance.                        (~10-30 min)
  Needs internet for step 1 only. Run scripts\run.ps1 (or run.cmd) at least once first so .venv exists.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\build_evidence.ps1
  powershell -ExecutionPolicy Bypass -File scripts\build_evidence.ps1 -Drugs 50 -Questions 20     # quicker, smaller
  powershell -ExecutionPolicy Bypass -File scripts\build_evidence.ps1 -SkipFetch                   # re-index + re-evaluate only
  powershell -ExecutionPolicy Bypass -File scripts\build_evidence.ps1 -Cpu                         # do not use the NVIDIA GPU

  GPU: if an NVIDIA GPU is present, the CUDA build of PyTorch (~2.5 GB) is installed into .venv once and the
  models run on the GPU. All downloads and caches (pip, PyTorch, Hugging Face, temp files) stay inside the
  project folder (.cache\), i.e. on the same drive as the project.
#>
[CmdletBinding()]
param(
    [int]$Drugs = 150,
    [int]$Questions = 40,
    [switch]$SkipFetch,
    [switch]$SkipEval,
    [switch]$Refresh,
    [switch]$Cpu
)

# Native programs write progress to stderr; with "Stop", Windows PowerShell 5.1 would abort on it.
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Log = Join-Path $LogDir ("build_evidence_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$env:PYTHONIOENCODING = "utf-8"
# keep every download / cache on the project's drive
$CacheDir = Join-Path $Root ".cache"
$TmpDir = Join-Path $CacheDir "tmp"
New-Item -ItemType Directory -Force -Path $CacheDir, $TmpDir | Out-Null
$env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"
$env:HF_HOME = Join-Path $CacheDir "huggingface"
$env:TORCH_HOME = Join-Path $CacheDir "torch"
$env:XDG_CACHE_HOME = $CacheDir
$env:TEMP = $TmpDir
$env:TMP = $TmpDir
$env:PYTHONUNBUFFERED = "1"

function Write-Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan; Add-Content -Path $Log -Value "==> $msg" -Encoding UTF8 }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow; Add-Content -Path $Log -Value "WARNING: $msg" -Encoding UTF8 }
function Fail($msg) {
    Write-Host ""; Write-Host "FAILED: $msg" -ForegroundColor Red; Add-Content -Path $Log -Value "FAILED: $msg" -Encoding UTF8
    Write-Host "Log: $Log"; Read-Host "Press Enter to close"; exit 1
}

# Runs the venv's python with stderr merged into stdout; each line is converted to plain text (so Windows
# PowerShell 5.1 shows log lines instead of error records), printed live and appended to the log.
function Invoke-Py([string[]]$PyArgs) {
    & $VenvPy @PyArgs 2>&1 | ForEach-Object { $line = "$_"; Add-Content -Path $Log -Value $line -Encoding UTF8; $line } | Out-Host
    return $LASTEXITCODE
}

$start = Get-Date
Write-Host "Log file: $Log"
if (-not (Test-Path $VenvPy)) { Fail "The Python environment (.venv) does not exist yet. Run run.cmd once, then run this script again." }

Write-Step "Checking Python packages"
Push-Location $Backend
try {
    $code = Invoke-Py @("-c", "import torch, transformers, httpx, pypdf; print('packages OK - torch', torch.__version__, '- transformers', transformers.__version__)")
    if ($code -ne 0) {
        Write-Step "Installing missing packages (CPU-only PyTorch, transformers, httpx, pypdf)"
        $null = Invoke-Py @("-m", "pip", "install", "--quiet", "-r", "requirements.txt")
        $null = Invoke-Py @("-m", "pip", "install", "--quiet", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
        $code = Invoke-Py @("-m", "pip", "install", "--quiet", "-r", "requirements-nlp.txt")
        if ($code -ne 0) { Fail "Package installation failed - see the messages above." }
    }

    if ($Cpu) {
        $env:PV_DEVICE = "cpu"
        Write-Host "    -Cpu given: models will run on the CPU"
    } elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Step "NVIDIA GPU found - checking for a CUDA build of PyTorch"
        $null = Invoke-Py @("-c", "import torch; print('torch', torch.__version__, '- CUDA available:', torch.cuda.is_available())")
        $code = Invoke-Py @("-c", "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)")
        if ($code -ne 0) {
            Write-Step "Installing the CUDA 12.6 build of PyTorch into .venv (~2.5 GB download, once; cached in .cache\pip)"
            $null = Invoke-Py @("-m", "pip", "uninstall", "-y", "torch")
            $code = Invoke-Py @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu126")
            if ($code -ne 0) {
                Write-Warn "CUDA PyTorch could not be installed - reinstalling the CPU build and continuing on the CPU."
                $null = Invoke-Py @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
            }
        }
        $code = Invoke-Py @("-c", "import sys, torch; ok = torch.cuda.is_available(); print('Using GPU:', torch.cuda.get_device_name(0) if ok else 'none (CPU)'); sys.exit(0 if ok else 1)")
        if ($code -eq 0) { $env:PV_DEVICE = "cuda" }
        else { Write-Warn "PyTorch cannot use the GPU (often an old NVIDIA driver: update it from nvidia.com). Continuing on the CPU." }
    } else {
        Write-Host "    no NVIDIA GPU detected - models run on the CPU"
    }

    foreach ($m in @(@("bge-small-en-v1.5", "embedding model (vector search)"), @("generator", "generative model (Qwen2.5-0.5B-Instruct)"))) {
        $w = Join-Path $Root ("models\{0}\model.safetensors" -f $m[0])
        if (Test-Path $w) { Write-Host ("    found {0}: models\{1}" -f $m[1], $m[0]) }
        else { Write-Warn ("{0} not found in models\{1} - continuing without it (see README 'Local models')." -f $m[1], $m[0]) }
    }

    if (-not $SkipFetch) {
        Write-Step "1/3  Downloading FDA labels, FDA safety communications and PubMed reviews for the top $Drugs FAERS drugs"
        $fetchArgs = @("-m", "pvsys.pipeline", "evidence-fetch", "--n", "$Drugs")
        if ($Refresh) { $fetchArgs += "--refresh" }
        $code = Invoke-Py $fetchArgs
        if ($code -ne 0) { Fail "Download step failed (exit $code). Check your internet connection and re-run; finished downloads are kept." }
    }

    Write-Step "2/3  Chunking and embedding the documents (CPU)"
    $code = Invoke-Py @("-m", "pvsys.pipeline", "evidence-index")
    if ($code -ne 0) { Fail "Indexing failed (exit $code)." }

    if (-not $SkipEval) {
        Write-Step "3/3  Evaluating retrieval and generated answers ($Questions generated answers)"
        $code = Invoke-Py @("-m", "pvsys.pipeline", "evidence-eval", "--n", "$Questions")
        if ($code -ne 0) { Fail "Evaluation failed (exit $code)." }
    }
} finally { Pop-Location }

$mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host ""
Write-Host "Done in $mins minutes. Results are in data\evidence\ (evaluation_evidence.json, index_manifest.json)." -ForegroundColor Green
Write-Host "Log: $Log"
Add-Content -Path $Log -Value "DONE in $mins minutes" -Encoding UTF8
Read-Host "Press Enter to close"
