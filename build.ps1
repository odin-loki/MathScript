# Build MathScript on native Windows (MSVC + Ninja)
# Usage: .\build.ps1 [-Test] [-Configure] [-Clean]
#   -Configure  Run CMake configure only (no build)
#   -Clean      Remove build-msvc before configure+build

param(
    [switch]$Test,
    [switch]$Configure,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $Root "build-msvc"

if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item -Recurse -Force $BuildDir
    Write-Host "Removed $BuildDir"
}

# Locate VsDevCmd.bat — search Program Files first (VS 2026+ installs there)
$VsDevCmdCandidates = @(
    "${env:ProgramFiles}\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat",
    "${env:ProgramFiles}\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
)

$VsDevCmd = $VsDevCmdCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $VsDevCmd) {
    throw "Visual Studio Build Tools not found. Install VS 2022/2026 with C++ workload."
}

# Resolve MSVC toolset root via vswhere so we can pin cl.exe / link.exe / lib paths.
# VsDevCmd.bat from some VS2026 installs omits the MSVC lib path from LIB; we add it
# explicitly to avoid LNK1104 on msvcrtd.lib during CMake compiler detection.
$vsInstall = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -property installationPath 2>$null
if ($vsInstall) {
    $msvcRoot = Get-ChildItem "$vsInstall\VC\Tools\MSVC" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $msvcRoot) { $msvcRoot = "" }  # fall back: rely on PATH

$CmakeConfigureArgs = "-S `"$Root`" -B `"$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
    "-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF"

# Build the batch content that runs inside a single cmd session so VsDevCmd
# environment variables (PATH, INCLUDE, LIB) are visible to cmake.
function New-BuildBatch([string]$extraCmds) {
    $libFix = if ($msvcRoot) { "set `"LIB=$msvcRoot\lib\x64;%LIB%`"" +
                               "`r`nset `"INCLUDE=$msvcRoot\include;%INCLUDE%`"" } else { "rem msvcRoot not found" }
    return @"
@echo off
call "$VsDevCmd" -arch=amd64
if errorlevel 1 exit /b 1
$libFix
cmake $CmakeConfigureArgs
if errorlevel 1 exit /b 1
$extraCmds
"@
}

if ($Configure) {
    $batchFile = [System.IO.Path]::GetTempFileName() + ".bat"
    (New-BuildBatch "") | Out-File -FilePath $batchFile -Encoding ASCII
    cmd /c $batchFile
    Remove-Item $batchFile -Force -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE" }
    Write-Host "Configure complete. Build directory: $BuildDir"
    exit 0
}

$buildSteps = "cmake --build `"$BuildDir`" --config Release"
$batchFile = [System.IO.Path]::GetTempFileName() + ".bat"
(New-BuildBatch $buildSteps) | Out-File -FilePath $batchFile -Encoding ASCII
cmd /c $batchFile
Remove-Item $batchFile -Force -ErrorAction SilentlyContinue
if ($LASTEXITCODE -ne 0) { throw "MathScript build failed with exit code $LASTEXITCODE" }

Write-Host "Build complete. Binaries in $BuildDir\bin"

if ($Test) {
    ctest --test-dir $BuildDir -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "CTest failed with exit code $LASTEXITCODE" }
    Write-Host "All tests passed."
}
