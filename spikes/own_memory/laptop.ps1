# Engramscope own-memory test on SmolLM2-1.7B, Windows laptop runner (instructions: LAPTOP.md, rules: PREREG-1.7B.md).
#
#   powershell -ExecutionPolicy Bypass -File spikes\own_memory\laptop.ps1 check    # about 15 minutes, run first
#   powershell -ExecutionPolicy Bypass -File spikes\own_memory\laptop.ps1          # everything else (hours)
#
# Rerun the same command to resume after a stop: finished steps are skipped and training resumes from its checkpoint.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

# keep Windows awake while this window is running (the display may still turn off)
Add-Type -Namespace Engramscope -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Engramscope.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (a Python package manager, about 30 MB)..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

$smi = (& nvidia-smi 2>$null | Out-String)
$m = [regex]::Match($smi, "CUDA Version:\s*([0-9]+)\.([0-9]+)")
if (-not $m.Success) { throw "nvidia-smi did not report a CUDA version. Is the NVIDIA driver installed?" }
$cuda = [int]$m.Groups[1].Value * 100 + [int]$m.Groups[2].Value
if ($cuda -ge 1302) { $torch = "torch==2.15.0+cu132"; $index = "https://download.pytorch.org/whl/cu132" }
elseif ($cuda -ge 1300) { $torch = "torch==2.14.1+cu130"; $index = "https://download.pytorch.org/whl/cu130" }
elseif ($cuda -ge 1206) { $torch = "torch==2.14.1+cu126"; $index = "https://download.pytorch.org/whl/cu126" }
else { throw "The NVIDIA driver supports CUDA $($m.Groups[1].Value).$($m.Groups[2].Value). Update the driver (NVIDIA app or nvidia.com), then rerun." }
Write-Host "Driver supports CUDA $($m.Groups[1].Value).$($m.Groups[2].Value): using $torch"

if (-not $env:ENGRAMSCOPE_OWNMEM) { $env:ENGRAMSCOPE_OWNMEM = "$env:USERPROFILE\engramscope-ownmem" }
New-Item -ItemType Directory -Force -Path "$env:ENGRAMSCOPE_OWNMEM\data" | Out-Null
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Write-Host "Working folder: $env:ENGRAMSCOPE_OWNMEM (log: $env:ENGRAMSCOPE_OWNMEM\scale\log.txt)"

uv run --no-project --python 3.12 --with $torch --with transformers==5.24.0 --with peft==0.22.0 --with pyarrow `
    --with tokenizers==0.23.3 --with sympy --with numpy --with requests --with psutil `
    --index-strategy unsafe-best-match --extra-index-url $index python -u scale.py @args
if ($LASTEXITCODE -ne 0) { Write-Host "scale.py stopped with an error (exit code $LASTEXITCODE). Paste the last lines above into the chat." }
