# Regenerate MSVC Release benchmark baseline medians on Windows.
# Usage: .\scripts\bench_write_msvc_baseline.ps1 [-BuildDir build-msvc-bench]
#
# Requires a Release benchmark build: .\build.ps1 -Benchmark
# Runs all add_ms_bench targets from tests/performance/CMakeLists.txt with 0.1s min time
# and 3 repetitions, then writes median_time_ns into tests/performance/baselines/msvc-release.json.

param(
    [string]$BuildDir = "build-msvc-bench"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BuildPath = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $Root $BuildDir }
$BenchDir = Join-Path $BuildPath "tests\performance"
$BaselinePath = Join-Path $Root "tests\performance\baselines\msvc-release.json"

$MinTime = "0.1s"
$Repetitions = 3

$CmakeListsPath = Join-Path $Root "tests\performance\CMakeLists.txt"
if (-not (Test-Path $CmakeListsPath)) {
    throw "CMakeLists not found: $CmakeListsPath"
}

$BenchTargets = Get-Content $CmakeListsPath | ForEach-Object {
    if ($_ -match '^\s*add_ms_bench\s*\(\s*(\w+)') {
        $matches[1]
    }
}

if ($BenchTargets.Count -eq 0) {
    throw "No add_ms_bench targets found in $CmakeListsPath"
}

Write-Host "Benchmark targets ($($BenchTargets.Count)): $($BenchTargets -join ', ')"

if (-not (Test-Path $BaselinePath)) {
    throw "Baseline file not found: $BaselinePath"
}

foreach ($bench in $BenchTargets) {
    $exe = Join-Path $BenchDir "$bench.exe"
    if (-not (Test-Path $exe)) {
        throw "Missing benchmark executable: $exe (run .\build.ps1 -Benchmark first)"
    }
}

$MergeScript = @'
import json, sys
merged_path, bench_path = sys.argv[1], sys.argv[2]
with open(merged_path, encoding="utf-8-sig") as f:
    merged = json.load(f)
with open(bench_path, encoding="utf-8-sig") as f:
    data = json.load(f)
merged["benchmarks"].extend(data.get("benchmarks", []))
with open(merged_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(merged, f)
'@

$WriteScript = @'
import json, re, statistics, sys

results_path, baseline_path, min_time, repetitions = sys.argv[1:5]
repetitions = int(repetitions)

with open(results_path, encoding="utf-8-sig") as f:
    data = json.load(f)

def canonical_name(name):
    return re.sub(r"/min_time:[^/]+$", "", name)

def to_ns(entry):
    unit = entry.get("time_unit", "s")
    real = float(entry["real_time"])
    if unit == "s":
        return real * 1e9
    if unit == "ms":
        return real * 1e6
    if unit == "us":
        return real * 1e3
    if unit == "ns":
        return real
    raise SystemExit(f"Unsupported time_unit: {unit}")

by_name = {}
for entry in data.get("benchmarks", []):
    name = entry.get("name") or entry.get("run_name")
    if not name:
        continue
    by_name.setdefault(canonical_name(name), []).append(to_ns(entry))

medians = {
    name: int(statistics.median(samples))
    for name, samples in by_name.items()
}

with open(baseline_path, encoding="utf-8") as f:
    baseline = json.load(f)

baseline["_comment"] = (
    "MSVC Release regression baseline. Wave 218-225 profiling targets; all add_ms_bench targets "
    "medians captured via scripts/bench_write_msvc_baseline.ps1 (0.1s min_time, 3 repetitions). "
    "Null median_time_ns entries are schema placeholders until regenerated; regression skips null entries."
)
baseline["_generate"] = ".\\scripts\\bench_write_msvc_baseline.ps1"
baseline["profile"] = {
    "os": "windows",
    "compiler": "msvc",
    "build_type": "Release",
    "benchmark_min_time": min_time,
    "benchmark_repetitions": repetitions,
}

benchmarks = baseline.setdefault("benchmarks", {})

filled = 0
for name, spec in benchmarks.items():
    if name in medians:
        spec["median_time_ns"] = medians[name]
        filled += 1

added = 0
for name, median in medians.items():
    if name not in benchmarks:
        benchmarks[name] = {"median_time_ns": median}
        added += 1

with open(baseline_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(baseline, f, indent=2)
    f.write("\n")

print(f"Wrote baseline: {baseline_path}")
print(
    f"Filled {filled} median(s), added {added} new benchmark(s) "
    f"from {len(medians)} captured benchmark name(s)"
)
for name in sorted(benchmarks):
    spec = benchmarks[name]
    median = spec.get("median_time_ns")
    if median is not None:
        print(f"  {name}: {median} ns")
'@

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("ms-bench-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempDir | Out-Null

try {
    $mergedPath = Join-Path $TempDir "bench_results.json"
    [System.IO.File]::WriteAllText($mergedPath, '{"benchmarks":[]}', (New-Object System.Text.UTF8Encoding $false))

    foreach ($bench in $BenchTargets) {
        $exe = Join-Path $BenchDir "$bench.exe"
        $benchJson = Join-Path $TempDir "$bench.json"
        Write-Host "Running $bench ..."
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $exe `
            --benchmark_min_time=$MinTime `
            --benchmark_repetitions=$Repetitions `
            --benchmark_report_aggregates_only=false `
            --benchmark_out=$benchJson `
            --benchmark_out_format=json `
            2>&1 | Out-Host
        $ErrorActionPreference = $prevEap
        if ($LASTEXITCODE -ne 0) {
            throw "Benchmark failed: $bench (exit $LASTEXITCODE)"
        }

        $MergeScript | python - $mergedPath $benchJson
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to merge JSON for $bench"
        }
    }

    $WriteScript | python - $mergedPath $BaselinePath $MinTime $Repetitions
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to write baseline JSON"
    }
}
finally {
    if (Test-Path $TempDir) {
        Remove-Item -Recurse -Force $TempDir
    }
}
