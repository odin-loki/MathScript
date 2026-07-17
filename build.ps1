# Build MathScript on native Windows (MSVC + Ninja)
# Usage: .\build.ps1 [-Test] [-Benchmark] [-Configure] [-Clean]
#   -Configure  Run CMake configure only (no build)
#   -Clean      Remove build directory before configure+build
#   -Benchmark  Configure build-msvc-bench with MS_BUILD_BENCHMARKS=ON

param(
    [switch]$Test,
    [switch]$Benchmark,
    [switch]$Configure,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = if ($Benchmark) {
    Join-Path $Root "build-msvc-bench"
} else {
    Join-Path $Root "build-msvc"
}

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

# Resolve MSVC toolset root — prefer the same install as VsDevCmd, then vswhere.
$msvcRoot = ""
if ($VsDevCmd -match 'Visual Studio\\([^\\]+)\\([^\\]+)\\Common7') {
    $vsEdition = $Matches[2]
    $vsBase = Split-Path (Split-Path (Split-Path $VsDevCmd -Parent) -Parent) -Parent
    $msvcCandidates = Get-ChildItem "$vsBase\VC\Tools\MSVC" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($c in $msvcCandidates) {
        if (Test-Path (Join-Path $c.FullName "bin\Hostx64\x64\cl.exe")) {
            $msvcRoot = $c.FullName
            break
        }
    }
}
if (-not $msvcRoot) {
    $vsInstall = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
        -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath 2>$null
    if ($vsInstall) {
        $msvcCandidates = Get-ChildItem "$vsInstall\VC\Tools\MSVC" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        foreach ($c in $msvcCandidates) {
            if (Test-Path (Join-Path $c.FullName "bin\Hostx64\x64\cl.exe")) {
                $msvcRoot = $c.FullName
                break
            }
        }
    }
}
if (-not $msvcRoot) { $msvcRoot = "" }  # fall back: rely on PATH

$ClExe = if ($msvcRoot) { Join-Path $msvcRoot "bin\Hostx64\x64\cl.exe" } else { "cl" }
$LinkExe = if ($msvcRoot) { Join-Path $msvcRoot "bin\Hostx64\x64\link.exe" } else { "link" }
$LibExe = if ($msvcRoot) { Join-Path $msvcRoot "bin\Hostx64\x64\lib.exe" } else { "lib" }

$CmakeConfigureArgs = "-S `"$Root`" -B `"$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
    "-DCMAKE_C_COMPILER=`"$ClExe`" -DCMAKE_CXX_COMPILER=`"$ClExe`" " +
    "-DCMAKE_LINKER=`"$LinkExe`" -DCMAKE_AR=`"$LibExe`" " +
    "-DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF"
if ($Benchmark) {
    $CmakeConfigureArgs += " -DMS_BUILD_BENCHMARKS=ON"
}

# Build the batch content that runs inside a single cmd session so VsDevCmd
# environment variables (PATH, INCLUDE, LIB) are visible to cmake.
function New-BuildBatch([string]$extraCmds) {
    $libFix = if ($msvcRoot) {
        "set `"PATH=$msvcRoot\bin\Hostx64\x64;%PATH%`"" +
        "`r`nset `"LIB=$msvcRoot\lib\x64;%LIB%`"" +
        "`r`nset `"INCLUDE=$msvcRoot\include;%INCLUDE%`""
    } else { "rem msvcRoot not found" }
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

if ($Benchmark) {
    $BenchDir = Join-Path $BuildDir "tests\performance"
    foreach ($bench in @("bench_matmul", "bench_fft")) {
        $exe = Join-Path $BenchDir "$bench.exe"
        if (-not (Test-Path $exe)) {
            throw "Missing benchmark executable: $exe"
        }
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $exe --benchmark_min_time=0.01s 2>&1 | Out-Null
        $ErrorActionPreference = $prevEap
        if ($LASTEXITCODE -ne 0) { throw "Benchmark smoke failed: $bench (exit $LASTEXITCODE)" }
    }
    Write-Host "Benchmark smoke OK (bench_matmul, bench_fft)"
}

if ($Test) {
    ctest --test-dir $BuildDir -C Release --output-on-failure
    if ($LASTEXITCODE -ne 0) { throw "CTest failed with exit code $LASTEXITCODE" }
    Write-Host "All tests passed."
}
