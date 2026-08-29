# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Wave-by-wave implementation history (thousands of entries) is in [`docs/WAVES.md`](docs/WAVES.md). Current architecture, API, and release criteria are in [`docs/`](docs/).

## [Unreleased]

CMake project version **1.0.0**. The git tag `v1.0.0` is not cut; it still follows the gates in [`docs/RELEASE.md`](docs/RELEASE.md) (CI green, coverage ≥ 90%, Valgrind clean, fuzz-24h, packaging, benches, compliance, JIT). After local prove-out, packaging is next.

- Local prove-out (Windows MSVC Release, CUDA off): **816/816** CTest suites passed (~36 s at `-j 32`); 28 Google Benchmark targets passed with `--benchmark_min_time=0.001s`. Windows ZIP smoke: `scripts/package_smoke.ps1` → `mathscript-1.0.0-win64.zip`.
- MSVC `/W4` compile warnings in library and test TUs were cleared (unused locals, `[[nodiscard]]`, `size_t`→`int`, unused statics).
- Tests are grouped by mathematical domain (`tests/unit/linalg`, `tests/integration/fft`, …). Duplicate remigration wave pipelines were collapsed.
- REPL matrix calls use a name-keyed handler registry (`src/interp/matrix_calls/<domain>/`); CMake generates `matrix_call_register_all.cpp`.
- 28 Google Benchmark targets in the same tree as the library (`MS_BUILD_BENCHMARKS=ON`).
- Local Windows build directory: `build-msvc` only.
