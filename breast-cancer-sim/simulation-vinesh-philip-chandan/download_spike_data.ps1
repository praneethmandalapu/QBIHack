# Download Option B spike data for TCGA-AR-A1AX (gitignored — share this script, not data/).
#
# From breast-cancer-sim/ in PowerShell:
#   .\simulation-vinesh-philip-chandan\download_spike_data.ps1
#   .\simulation-vinesh-philip-chandan\download_spike_data.ps1 -ExportRaw
#
# Baseline only (~50 MB DICOM). Adds follow-up with -IncludeFollowup.
#
# If script execution is blocked, run once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$IncludeFollowup,
    [switch]$ExportRaw,
    [switch]$ExportRawOnly
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating venv at breast-cancer-sim\.venv ..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv .venv
    }
    else {
        throw "Python not found. Install Python 3.10+ from https://www.python.org/downloads/"
    }
}

& $VenvPython -m pip install -q -r requirements.txt

$pyArgs = @()
if ($IncludeFollowup) { $pyArgs += "--include-followup" }
if ($ExportRaw) { $pyArgs += "--export-raw" }
if ($ExportRawOnly) { $pyArgs += "--export-raw-only" }

& $VenvPython "simulation-vinesh-philip-chandan\download_spike_data.py" @pyArgs
exit $LASTEXITCODE
