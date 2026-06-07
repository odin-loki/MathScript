# Build MathScript on native Windows (MSVC + Ninja)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $Root "build-msvc"

$VsDevCmdCandidates = @(
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
)

$VsDevCmd = $VsDevCmdCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $VsDevCmd) {
    throw "Visual Studio Build Tools not found. Install VS 2022/2026 Build Tools with C++ workload."
}

$BuildCmd = @(
    "`"$VsDevCmd`" -arch=amd64",
    "cmake -S `"$Root`" -B `"$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF",
    "cmake --build `"$BuildDir`" --config Release"
) -join " && "

cmd /c $BuildCmd
if ($LASTEXITCODE -ne 0) {
    throw "MathScript build failed with exit code $LASTEXITCODE"
}

Write-Host "Build complete. Binaries in $BuildDir\bin"
