# Pre-tag gate for MathScript v1.0.0 (read-only checklist; does not modify version).
#
# Usage:
#   .\scripts\tag_1.0.0_checklist.ps1
#
# After fuzz-24h.yml completes with zero crashes, see docs/RELEASE.md for tag steps.

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$CMakeLists = Join-Path $Root "CMakeLists.txt"
$VersionLine = (Select-String -Path $CMakeLists -Pattern '^project\(MathScript VERSION ').Line

Write-Host "=== MathScript v1.0.0 tag checklist ==="
Write-Host "Current: $VersionLine"
Write-Host ""
Write-Host "Required before tagging (see docs/RELEASE.md):"
Write-Host "  [ ] CI green on main (.github/workflows/ci.yml)"
Write-Host "  [ ] 106/106 CTest suites passing (1180 test cases)"
Write-Host "  [ ] Coverage >= 90% (coverage-linux)"
Write-Host "  [ ] Valgrind clean (valgrind-linux)"
Write-Host "  [ ] fuzz-24h.yml: 86400 s x 7 targets, zero crashes"
Write-Host "  [ ] plugin-linux + jit-linux green"
Write-Host "  [ ] benchmark-linux within tolerance"
Write-Host "  [ ] unsafe_delta clean"
Write-Host "  [ ] CHANGELOG + docs current"
Write-Host ""
Write-Host "Fuzz marathon: bash scripts/fuzz_24h_dispatch.sh (requires gh CLI on Linux/WSL)"
Write-Host "Local gate:    bash scripts/pre_release.sh (Linux/WSL) or build + ctest on Windows"
Write-Host ""

$BuildDir = Join-Path $Root "build-msvc"
if (Test-Path $BuildDir) {
    Write-Host "Running ctest in $BuildDir ..."
    ctest --test-dir $BuildDir -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) {
        throw "CTest failed — fix before tagging."
    }
    Write-Host "CTest passed."
} else {
    Write-Host "NOTE: build-msvc not found — run .\build.ps1 -Test first."
}

if ($VersionLine -match "1\.0\.0") {
    Write-Host "NOTE: CMakeLists.txt already references 1.0.0 — verify fuzz marathon completed."
} else {
    Write-Host "NOTE: Version not yet 1.0.0 in CMakeLists.txt (expected until fuzz marathon passes)."
}
