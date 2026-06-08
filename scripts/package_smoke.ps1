# Verify MathScript install tree after cmake --install (Windows).
param(
    [string]$BuildDir = "build-msvc",
    [string]$Prefix = "install-smoke"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

if (Test-Path -LiteralPath $BuildDir -PathType Container) {
    $BuildPath = (Resolve-Path -LiteralPath $BuildDir).Path
} elseif (Test-Path -LiteralPath (Join-Path $Root $BuildDir) -PathType Container) {
    $BuildPath = (Resolve-Path -LiteralPath (Join-Path $Root $BuildDir)).Path
} else {
    Write-Error "Build directory not found: $BuildDir"
}

if ([System.IO.Path]::IsPathRooted($Prefix)) {
    $PrefixPath = $Prefix
} else {
    $PrefixPath = Join-Path $Root $Prefix
}

& cmake --install $BuildPath --prefix $PrefixPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($bin in @("mathscript-repl", "mathscriptc", "mathscript-server")) {
    $path = Join-Path $PrefixPath "bin\$bin.exe"
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Error "Missing: $path"
    }
}

$libPath = Join-Path $PrefixPath "lib\ms_core.lib"
if (-not (Test-Path -LiteralPath $libPath)) {
    Write-Error "Missing: $libPath"
}

$versionPath = Join-Path $PrefixPath "include\ms\version.hpp"
if (-not (Test-Path -LiteralPath $versionPath)) {
    Write-Error "Missing: $versionPath"
}

Write-Host "Install smoke OK: $PrefixPath"
