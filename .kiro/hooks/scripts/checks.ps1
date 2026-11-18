# Engramscope post-task checks for Windows (Windows PowerShell 5.1 or PowerShell 7+).
# Python: ruff and pytest. Web: eslint and vitest. Each part skips until Task 2 creates it.
# Mirrors the sh hook in .kiro/hooks/checks-after-task.json.

$ErrorActionPreference = 'Stop'

# This script lives in <repo>/.kiro/hooks/scripts/, so the repo root is three levels up.
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $repoRoot

function Invoke-Step([string]$exe, [string[]]$arguments, [string]$where) {
    Push-Location $where
    try {
        & $exe @arguments
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($code -ne 0) {
        Write-Output ("engramscope checks: '{0} {1}' failed with exit code {2}" -f $exe, ($arguments -join ' '), $code)
        exit $code
    }
}

$ran = $false

if (Test-Path (Join-Path $repoRoot 'pyproject.toml')) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Output 'engramscope checks: uv is not on PATH; install it from https://docs.astral.sh/uv/'
        exit 1
    }
    Invoke-Step 'uv' @('run', 'ruff', 'check', '.') $repoRoot
    Invoke-Step 'uv' @('run', 'ruff', 'format', '--check', '.') $repoRoot
    Invoke-Step 'uv' @('run', 'pytest', '-q') $repoRoot
    $ran = $true
}

$web = Join-Path $repoRoot 'web'
if (Test-Path (Join-Path $web 'package.json')) {
    Invoke-Step 'npm' @('run', 'lint') $web
    Invoke-Step 'npx' @('vitest', '--run') $web
    $ran = $true
}

if ($ran) {
    Write-Output 'engramscope checks: all passed'
} else {
    Write-Output 'engramscope checks: nothing to check yet (before Task 2), skipping'
}
exit 0
