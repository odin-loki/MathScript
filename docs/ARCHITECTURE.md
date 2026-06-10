# MathScript Architecture

MathScript is a C++23 computer algebra and HPC math library with a console REPL, optional CUDA/MPI backends, and an in-tree BLAS/LAPACK CPU kernel layer. The repository is organized as a CMake monorepo: static libraries under `src/`, public headers under `include/ms/`, and GoogleTest suites under `tests/`.

## Repository Layout

```
MathsScript/
├── cmake/           # options.cmake, platform.cmake, coverage, install rules
├── include/ms/      # Public API headers (installed with the library)
├── src/             # Implementation libraries and executables
├── tests/           # unit/ GoogleTest suites; fuzz/ libFuzzer targets
├── scripts/         # coverage, Valgrind, unsafe-surface audit helpers
├── docs/            # Architecture and API reference (this tree)
└── .github/workflows/  # CI and nightly fuzz workflows
```

## Source Modules (`src/`)

Each subdirectory builds a static library (`ms_<name>`) linked into the umbrella `mathscript` INTERFACE target.

| Module | Library | Role |
|--------|---------|------|
| `core` | `ms_core` | Matrix, Tensor, Sparse, Sym, Scalar, RNG, error formatting |
| `runtime` | `ms_runtime` | Topology detection, thread pool, dispatch, load balancer |
| `linalg` | `ms_linalg` | matmul, LU/QR/Cholesky, solve, eig, SVD, expm, norms, iterative solvers |
| `fft` | `ms_fft` | 1D/2D FFT, DCT/DST, rfft helpers |
| `stats` / `prob` | `ms_stats`, `ms_prob` | Descriptive statistics, regression, common distributions |
| `optim` | `ms_optim` | Gradient descent, Newton/Broyden root finders, golden-section search |
| `signal` | `ms_signal` | Filters, convolution, window functions |
| `special` | `ms_special` | Bessel, gamma, elliptic, hypergeometric, and extended special functions |
| `simd` | `ms_simd` | ISA detection and vectorized elementwise kernels (AVX2/AVX-512) |
| `cuda` | `ms_cuda` | Optional GPU buffers, BLAS, FFT, LU/solve stubs (off when `MS_ENABLE_CUDA=OFF`) |
| `interp` | `ms_interp` | REPL interpreter, session state, plotting hooks |
| `ode` / `pde` | `ms_ode`, `ms_pde` | Euler/RK4 ODE integrators; 1D heat equation |
| `poly` / `symbolic` / `domain` | `ms_poly`, `ms_symbolic`, `ms_domain` | Polynomial ops, symbolic AST, combinatorics/graph helpers |
| `distributed` | `ms_distributed` | MPI context, distributed matrices, gather/scatter linear algebra |
| `frameworks` | `ms_frameworks` | Axiom, CellAI, Cypha, GRIA, Izaac research frameworks |
| `exe` | — | `mathscriptc`, `mathscript-repl`, `mathscript-server` binaries |

CPU BLAS/LAPACK kernels live in `src/linalg/` and are exposed via `include/ms/cpu/blas.hpp` and `include/ms/cpu/lapack.hpp`. They are the default path when external BLAS is not required.

## Public Headers (`include/ms/`)

Headers mirror the module layout above. `include/ms/ms.hpp` is the umbrella include that pulls in core types, linalg, memory, error, fft, stats, prob, optim, signal, special, simd, and cuda interfaces. Finer-grained includes are preferred for compile-time cost (see `docs/API.md`).

## Build System

- **CMake** 3.28+, **C++23** (`CMAKE_CXX_STANDARD 23`)
- **Generators:** Ninja (Windows MSVC, Linux GCC/Clang)
- **Entry:** root `CMakeLists.txt` includes `cmake/options.cmake` and adds `src/` then `tests/` when enabled

### Key CMake Options (`cmake/options.cmake`)

| Flag | Default | Purpose |
|------|---------|---------|
| `MS_BUILD_TESTS` | ON | Fetch GoogleTest and register CTest suites |
| `MS_ENABLE_CUDA` | OFF | CUDA GPU backend (`ms_cuda`) |
| `MS_ENABLE_MPI` | OFF | Distributed MPI support |
| `MS_ENABLE_AVX512` | ON | AVX-512 compile-time gate (`MS_ENABLE_AVX512` macro) |
| `MS_ENABLE_COVERAGE` | OFF | gcov/lcov instrumentation (Linux) |
| `MS_BUILD_FUZZ` | OFF | libFuzzer executables under `tests/fuzz/` |
| `MS_BUILD_GUI` | OFF | Qt6 IDE (optional) |
| `MS_BUILD_JIT` | OFF | LLVM ORC JIT backend for scalar REPL assignments (Linux/Clang + LLVM 18) |
| `MS_BUILD_PLUGIN` | OFF | Clang plugin compliance enforcement |
| `MS_BUILD_BENCHMARKS` | OFF | Google Benchmark targets (`bench_matmul`, `bench_fft`) |
| `MS_ENABLE_ASAN` | OFF | AddressSanitizer |
| `MS_VERBOSE_DISPATCH` | OFF | Runtime dispatch logging |

CI builds typically pass `-DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF` for deterministic, portable runs.

### Windows Quick Build

```powershell
.\build.ps1
# or: cmake -S . -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF
ctest --test-dir build-msvc --output-on-failure
```

Binaries land in `build-msvc/bin/` (or `build-linux/bin/` on Linux).

## Test Layout

- **`tests/unit/`** — **57** GoogleTest executables registered via `add_ms_test()` in `tests/CMakeLists.txt`; each binary is one CTest case.
- **`tests/fuzz/`** — seven libFuzzer targets (built when `MS_BUILD_FUZZ=ON`); `test_fuzz_stress` is always registered as a long-running smoke target.
- Tests link the `mathscript` INTERFACE library and `GTest::gtest_main`.

Coverage and Valgrind scripts (`scripts/coverage_report.sh`, `scripts/valgrind_tests.sh`) expect a Debug Linux build with tests enabled.

## CI Pipeline (`.github/workflows/ci.yml`)

Triggered on push/PR to `main`:

1. **build-test-windows** — MSVC x64, Release, full `ctest`, install smoke (`mathscript-repl.exe`)
2. **build-test-linux** — GCC 13, Release, full `ctest`, install + CPack TGZ/ZIP smoke, unsafe surface audit
3. **coverage-linux** — Debug + `MS_ENABLE_COVERAGE`, 90% minimum line coverage enforced
4. **fuzz-linux** — libFuzzer smoke on all 7 targets (Clang)
5. **valgrind-linux** — Debug memcheck over the suite (excludes `test_fuzz_stress`)
6. **plugin-linux** — Clang + LLVM 18, `MS_BUILD_PLUGIN=ON`; builds `ms_plugin`, runs `test_plugin_smoke`, and compliance rule tests (`compliance_*`). **Twenty** compile-fail rules enforced (all `DiagnosticID`s in `diagnostics.hpp` except partial **`UnsafeAudit`** via `MS_UNSAFE` + `scripts/unsafe_report.sh`). Each rule has `fail.cpp` + `ok.cpp`; `[[ms::unsafe]]` escape where applicable. When `MS_UNSAFE` is defined, also runs `compliance_unsafe_annotation`.
7. **jit-linux** — Clang + LLVM 18, `-DMS_BUILD_JIT=ON`; links ORC LLJIT, runs `test_jit_backend` and `test_plot_console`. LLVM backend JIT-compiles scalar assignments; matrix calls (`matmul`, `solve`, `transpose`, `chol`, multi-target `lu`/`qr`/`svd`/`eig_sym`, scalar `det`/…) dispatch to native kernels; other lines delegate to REPL when unsupported.
8. **benchmark-linux** — Google Benchmark smoke (`bench_matmul`, `bench_fft`) when `MS_BUILD_BENCHMARKS=ON`

### Performance benchmarks

Build with `-DMS_BUILD_BENCHMARKS=ON` (see `tests/performance/`). CI runs a fast smoke via `scripts/bench_regression.sh` with `MS_BENCH_REGRESSION=off`.

Regression detection compares JSON output from Google Benchmark against `tests/performance/baselines/linux-gcc13.json`. A run more than 10% slower than the stored median fails (`MS_BENCH_TOLERANCE`, default 10). Skip compare for smoke-only runs: `MS_BENCH_REGRESSION=off`.

```bash
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=OFF -DMS_BUILD_BENCHMARKS=ON
cmake --build build-bench --target bench_matmul bench_fft
bash scripts/bench_regression.sh build-bench                    # compare (default)
MS_BENCH_REGRESSION=off bash scripts/bench_regression.sh build-bench  # smoke only
bash scripts/bench_regression.sh --write-baseline build-bench   # refresh baseline medians
```

Baselines are captured with `--benchmark_out=… --benchmark_out_format=json` (wrapped by `bench_regression.sh`). Calibrate on the target runner profile (Linux GCC 13 Release) before enabling the CI regression gate.

Weekly extended fuzz: `.github/workflows/nightly.yml` (15 min × 7 targets; manual dispatch also runs a 30 min/target marathon).

Manual Phase 10 fuzz campaign: `.github/workflows/fuzz-24h.yml` (`workflow_dispatch` only) — 7 parallel jobs, **86400 s (24 h) per target**, **1500 min** job timeout each. Required before tagging **v1.0.0** (zero crashes). Trigger from GitHub Actions UI or `gh workflow run fuzz-24h.yml`.

Vendor integrity: `scripts/verify_vendor.sh` + `vendor/CHECKSUMS.sha256` (empty until sources are vendored).

## Executables

| Binary | Purpose |
|--------|---------|
| `mathscriptc` | Batch/script driver linked to full `mathscript` |
| `mathscript-repl` | Interactive console or one-shot `-e` / `--eval` (`ms_interp` + core math) |
| `mathscript-gui` | Qt6 IDE (`MS_BUILD_GUI=ON`): REPL + `PlotWidget` / OpenGL `PlotSurfWidget` (PNG export via File menu) |
| `mathscript-server` | Distributed/MPI-oriented server entry point |

Install rules and CPack packaging are defined in `cmake/install.cmake`.
