# MathScript HPC Computer Algebra System

[![CI](https://github.com/odin-loki/MathScript/actions/workflows/ci.yml/badge.svg)](https://github.com/odin-loki/MathScript/actions/workflows/ci.yml)

A high-performance Computer Algebra System built in C++23 with CPU math libraries, runtime dispatch, and a console REPL.

## Project Status: Phase 10 (Hardening) — In Progress

Phase 9 (own BLAS/LAPACK SVD pipeline) is complete. Phase 10 adds CI, coverage reporting, Valgrind memcheck, fuzz testing, and install/packaging. **All CI jobs on `main` are green** (Windows + Linux build/test, coverage, fuzz smoke, Valgrind).

### Completed Components

- **Build system:** Native Windows MSVC + Ninja; Linux GCC 13 / Clang; GoogleTest via FetchContent
- **Core types:** Matrix, Tensor, Sparse, Scalar, Sym
- **Linear algebra:** matmul, LU, QR, Cholesky, solve, eig, SVD, expm, trace, det, norm, and helpers
- **Math modules:** fft, stats, prob, optim, signal, special (CPU implementations)
- **Runtime:** Topology detection, thread pool, dispatch layer, own BLAS/LAPACK CPU kernels
- **Executables:** `mathscriptc`, `mathscript-repl`, `mathscript-server`
- **Unit tests:** 34 CTest suites — all passing (CUDA disabled)
- **CI baseline:** ~60% line coverage (58% enforced in CI); path to 85% for 1.0.0

### Build Instructions (Native Windows)

#### Prerequisites
- Visual Studio 2022/2026 Build Tools with C++ workload
- CMake 3.28+
- Ninja (included with Strawberry Perl or VS)

#### Build
```powershell
.\build.ps1
```

Or manually:
```powershell
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=amd64 && cmake -S . -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF && cmake --build build-msvc --config Release"
```

#### Run tests
```powershell
ctest --test-dir build-msvc --output-on-failure
```

#### Run REPL
```powershell
.\build-msvc\bin\mathscript-repl.exe
```

### Linux (headless CI-style build)

Ubuntu 24.04 with GCC 13 (matches CI):

```bash
cmake -S . -B build-linux -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build-linux
ctest --test-dir build-linux --output-on-failure
```

For libFuzzer targets, use Clang with the project’s Linux Clang flags (see `cmake/options.cmake`).

### Coverage (Linux, GCC/Clang)

```bash
cmake -S . -B build-cov -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF \
  -DMS_ENABLE_COVERAGE=ON
cmake --build build-cov
ctest --test-dir build-cov --output-on-failure
MS_COVERAGE_MIN=58 bash scripts/coverage_report.sh build-cov
```

Set `MS_COVERAGE_MIN=85` to enforce the Phase 10 shipping target once the baseline reaches it.

### Valgrind (Linux)

```bash
cmake -S . -B build-valgrind -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF
cmake --build build-valgrind
bash scripts/valgrind_tests.sh build-valgrind
```

### Install and packages

```bash
cmake --install build-linux --prefix /opt/mathscript
cd build-linux && cpack -G TGZ && cpack -G ZIP
```

### CI

GitHub Actions (`.github/workflows/ci.yml`) on every push/PR to `main`:

- Windows MSVC build + full test suite + install smoke
- Linux GCC 13 build + full test suite + install/package smoke + unsafe surface audit
- Coverage report (~60% line; 58% minimum) + artifact upload
- libFuzzer smoke on `fuzz_special_fns`
- Valgrind memcheck (excludes long `test_fuzz_stress`)

Weekly extended fuzz runs: `.github/workflows/nightly.yml` (1 hour libFuzzer session).

### Unsafe surface audit

```bash
bash scripts/unsafe_report.sh
```

Review entries and the approved baseline live in `UNSAFE_REVIEW.md`.

## Phase Progress

| Phase | Name | Status |
|-------|------|--------|
| 0 | Foundation | Complete |
| 1 | Core Types + REPL | Complete |
| 2 | Math Library CPU | Complete |
| 3 | GPU / CUDA | Stub + optional backend |
| 4–8 | IDE, distributed, frameworks, special functions | Substantial progress |
| 9 | Own BLAS/LAPACK core | Complete |
| 10 | Hardening (CI, coverage, packaging) | In Progress |

## Documentation

See `mathscript-master-plan.md` for the complete 10-phase delivery plan.
