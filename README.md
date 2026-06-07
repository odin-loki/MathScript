# MathScript HPC Computer Algebra System

A high-performance Computer Algebra System built in C++23 with CPU math libraries, runtime dispatch, and a console REPL.

## Project Status: Phase 10 (Hardening) — In Progress

Phase 9 (own BLAS/LAPACK SVD pipeline) is complete. Phase 10 adds CI, coverage reporting, fuzz smoke tests, and install/packaging.

### Completed Components

- **Build system:** Native Windows MSVC + Ninja; Linux Clang toolchains; GoogleTest via FetchContent
- **Core types:** Matrix, Tensor, Sparse, Scalar, Sym
- **Linear algebra:** matmul, LU, QR, Cholesky, solve, eig, SVD, expm, trace, det, norm, and helpers
- **Math modules:** fft, stats, prob, optim, signal, special (CPU implementations)
- **Runtime:** Topology detection, thread pool, dispatch layer, own BLAS/LAPACK CPU kernels
- **Executables:** `mathscriptc`, `mathscript-repl`, `mathscript-server`
- **Unit tests:** 34 CTest suites — all passing (CUDA disabled)

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

```bash
cmake -S . -B build-linux -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF \
  -DMS_USE_LIBCXX=ON \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++" \
  -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++abi"
cmake --build build-linux
ctest --test-dir build-linux --output-on-failure
```

On older Linux distros without complete C++23 in libstdc++, install `libc++-dev` and `libc++abi-dev` and enable `MS_USE_LIBCXX=ON` as above.

### Coverage (Linux, Clang/GCC)

```bash
cmake -S . -B build-cov -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF \
  -DMS_ENABLE_COVERAGE=ON
cmake --build build-cov
ctest --test-dir build-cov --output-on-failure
bash scripts/coverage_report.sh build-cov
```

Set `MS_COVERAGE_MIN=85` to enforce the Phase 10 line-coverage target once the baseline reaches it.

### Install and packages

```bash
cmake --install build-linux --prefix /opt/mathscript
cmake --build build-linux --target package   # TGZ/ZIP; DEB/RPM on Linux; NSIS on Windows
```

### CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR to `main`:

- Windows MSVC build + full test suite + install smoke
- Linux Clang build + full test suite + install/package smoke + unsafe surface audit
- Coverage report (artifact upload; minimum threshold currently 50%)
- libFuzzer smoke test on `fuzz_special_fns`

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
