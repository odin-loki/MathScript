# Build MathScript on native Windows (MSVC + Ninja)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $Root "build-msvc"
$VsDevCmd = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"

if (-not (Test-Path $VsDevCmd)) {
    throw "Visual Studio Build Tools not found. Install VS 2022/2026 Build Tools with C++ workload."
}

cmd /c "`"$VsDevCmd`" -arch=amd64 && cmake -S `"$Root`" -B `"$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF && cmake --build `"$BuildDir`" --config Release"

Write-Host "Build complete. Binaries in $BuildDir\bin"
