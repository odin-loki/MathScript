# Run the MathScript test suite on Windows
# Usage: .\scripts\run_tests.ps1 [-Filter <pattern>] [-Verbose]
param(
    [string]$Filter = "",
    [switch]$Verbose
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BuildDir = Join-Path $Root "build-msvc"

if (-not (Test-Path $BuildDir)) {
    Write-Host "Build directory not found. Run .\build.ps1 first." -ForegroundColor Red
    exit 1
}

$ctestArgs = @("--test-dir", $BuildDir, "-C", "Release", "--output-on-failure")
if ($Filter) { $ctestArgs += @("-R", $Filter) }
if ($Verbose) { $ctestArgs += @("-VV") }

ctest @ctestArgs
$code = $LASTEXITCODE
if ($code -eq 0) { Write-Host "All tests passed." -ForegroundColor Green }
else { Write-Host "Some tests failed (exit $code)." -ForegroundColor Red }
exit $code
