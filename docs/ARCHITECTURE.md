# MathScript Architecture

For function-level API detail see [`docs/API.md`](API.md); for REPL usage and workflows see [`docs/USER_GUIDE.md`](USER_GUIDE.md). This document is the structural reference: repository layout, modules, build system, tests, and CI.

MathScript is a C++23 computer algebra and HPC math library with a console REPL, optional CUDA/MPI backends, and an in-tree BLAS/LAPACK CPU kernel layer. The repository is organized as a CMake monorepo: static libraries under `src/`, public headers under `include/ms/`, and GoogleTest suites under `tests/`.

## Repository Layout

```
MathsScript/
├── cmake/              # options.cmake, platform.cmake, coverage, install, jit, verify_vendor
├── include/ms/         # Public API headers (installed with the library)
├── src/                # Implementation libraries and executables
├── tests/              # unit/, numerical/, integration/, fuzz/, compliance/, performance/
├── scripts/            # coverage, Valgrind, bench regression, packaging smoke, unsafe audit
├── docs/               # Architecture, API, user guide
├── vendor/             # Vendored third-party sources; CHECKSUMS.sha256
├── exe/                # Reference entry-point sources (binaries built from src/exe/)
├── build.ps1 / build.sh
└── .github/workflows/  # ci.yml, nightly.yml, fuzz-24h.yml
```

## Source Modules (`src/`)

Each library subdirectory builds a static target `ms_<name>` (except `plugin`, `gui`, and `exe`). Libraries listed below are linked into the umbrella `mathscript` INTERFACE target unless noted.

### Numerical core

| Module | Library | Role |
|--------|---------|------|
| `core` | `ms_core` | Matrix, Tensor, Sparse, Sym, Scalar, RNG, checked arithmetic, units |
| `runtime` | `ms_runtime` | Topology detection, thread pool, dispatch, load balancer |
| `linalg` | `ms_linalg` | matmul, LU/QR/Cholesky/LDL, solve, eig, SVD, expm, norms, iterative solvers, matrix functions |
| `fft` | `ms_fft` | 1D/2D FFT, DCT/DST, rfft helpers |
| `ode` | `ms_ode` | Fixed-step and adaptive ODE integrators (Euler, RK4, RK45, stiff solvers) |
| `pde` | `ms_pde` | 1D/2D heat, wave, advection, Poisson PDE solvers |
| `poly` | `ms_poly` | Polynomial arithmetic, roots, resultants, discriminants, Bernstein basis |
| `optim` | `ms_optim` | Gradient descent, Newton/Broyden, golden-section, global search, Levenberg–Marquardt |
| `special` | `ms_special` | Bessel, gamma, elliptic, hypergeometric, Painlevé, Voigt, and extended special functions |
| `symbolic` | `ms_symbolic` | Symbolic expression AST: parse, differentiate, simplify, numeric eval |
| `domain` | `ms_domain` | Basic combinatorics (`factorial`, `nchoosek`, `gcd`) and simple graph edge helpers |
| `simd` | `ms_simd` | ISA detection and vectorized elementwise kernels (AVX2/AVX-512) |

CPU BLAS/LAPACK kernels live in `src/linalg/` and are exposed via `include/ms/cpu/blas.hpp` and `include/ms/cpu/lapack.hpp`. They are the default path when external BLAS is not required.

### Statistics & data

| Module | Library | Role |
|--------|---------|------|
| `stats` | `ms_stats` | Descriptive statistics, regression, hypothesis tests, time-series analysis |
| `prob` | `ms_prob` | Common probability distributions, PDF/CDF/quantile, sampling |
| `ml` | `ms_ml` | Supervised/unsupervised learning, PCA, autodiff, neural nets, LDA/QDA |
| `info` | `ms_info` | Shannon/Rényi/Tsallis entropy, divergences, channel capacity, complexity measures |

### Applied domains

| Module | Library | Role |
|--------|---------|------|
| `finance` | `ms_finance` | Black–Scholes/Greeks, bonds, portfolio metrics, Monte Carlo and exotic options |
| `control` | `ms_control` | Transfer functions, state space, Bode, LQR, Lyapunov/Riccati, Gramians |
| `signal` | `ms_signal` | Filters, convolution, window functions, adaptive LMS |
| `image` | `ms_image` | Float image type, color transforms, filters, edges, morphology, features |
| `graph` | `ms_graph` | Graph ADT, traversal, shortest paths, MST, centrality, max-flow, TSP heuristic |
| `geo` | `ms_geo` | 2D/3D primitives, convex hull, Delaunay, KD-tree, Bézier, Minkowski sum |
| `diffgeo` | `ms_diffgeo` | Metric tensor, Christoffel/Riemann/Ricci, geodesics, surface curvature |
| `topo` | `ms_topo` | Simplicial complexes, persistent homology, Betti numbers, Wasserstein distance |
| `quantum` | `ms_quantum` | Gates, kets/density matrices, QFT, entanglement and fidelity measures |
| `numthy` | `ms_numthy` | Primality, factorization, divisor/arithmetic functions (φ, μ, τ, σ, …) |
| `combo` | `ms_combo` | Factorials, binomials, permutations/combinations, Stirling/Catalan/Bell numbers |
| `bignum` | `ms_bignum` | Arbitrary-precision BigInt and Rational |
| `compress` | `ms_compress` | RLE, Huffman, arithmetic/range coding, LZ/LZW/BWT |
| `cplx` | `ms_cplx` | Residues, contour integrals, Möbius transforms, conformal mapping |
| `tensorops` | `ms_tensorops` | Dense tensors, einsum, CP/Tucker/HOSVD, Khatri–Rao, symmetrize |

### Interpreter & tooling

| Module | Library | Role |
|--------|---------|------|
| `interp` | `ms_interp` | REPL interpreter, session state, plotting hooks, optional ORC JIT backend |
| `cuda` | `ms_cuda` | Optional GPU buffers, BLAS, FFT, LU/solve, NCCL (off when `MS_ENABLE_CUDA=OFF`) |
| `distributed` | `ms_distributed` | MPI context, distributed matrices, gather/scatter linear algebra |
| `plugin` | `ms_plugin` | Optional Clang AST enforcement plugin (`MS_BUILD_PLUGIN=ON`; not linked into `mathscript`) |

### Research frameworks (`frameworks/`)

| Subdir | Namespace | Role |
|--------|-----------|------|
| `axiom` | `ms::axiom` | Genetic programming over `ms::Sym` algorithm representations |
| `cellai` | `ms::cellai` | Hebbian updates and multi-timescale cellular associative memory |
| `cypha` | `ms::cypha` | Mixture-of-experts differentiable information filtering (DIF) |
| `gria` | `ms::gria` | Entropy-based reversible/critical/irreversible compute classification |
| `izaac` | `ms::izaac` | VRF, CSPRNG, and cryptographic/statistical simulation utilities |

All five subdirs compile into `ms_frameworks`.

### Executables & GUI

| Module | Target | Role |
|--------|--------|------|
| `exe` | — | `mathscriptc`, `mathscript-repl`, `mathscript-server` binaries |
| `gui` | `ms_gui` | Optional Qt6 IDE (`MS_BUILD_GUI=ON`): REPL + `PlotWidget` / OpenGL `PlotSurfWidget` |

## Public Headers (`include/ms/`)

Headers mirror the module layout above, plus cross-cutting trees: `cpu/` (BLAS/LAPACK), `memory/` (allocators), `error/` (`Result`, error types), and `unsafe/` (`[[ms::unsafe]]` escape hatch). `include/ms/ms.hpp` is the umbrella include that pulls in all `mathscript`-linked modules. Finer-grained includes are preferred for compile-time cost (see `docs/API.md`).

## Build System

- **CMake** 3.28+, **C++23** (`CMAKE_CXX_STANDARD 23`)
- **Generators:** Ninja (Windows MSVC, Linux GCC/Clang)
- **Entry:** root `CMakeLists.txt` includes `cmake/options.cmake` and adds `src/` then `tests/` when enabled

### Key CMake Options (`cmake/options.cmake`)

| Flag | Default | Purpose |
|------|---------|---------|
| `MS_BUILD_TESTS` | ON | Fetch GoogleTest and register CTest suites |
| `MS_BUILD_BENCHMARKS` | OFF | Google Benchmark targets under `tests/performance/` |
| `MS_BUILD_GUI` | OFF | Qt6 IDE (`mathscript-gui`) |
| `MS_BUILD_FUZZ` | OFF | libFuzzer executables under `tests/fuzz/` |
| `MS_BUILD_PLUGIN` | OFF | Clang AST enforcement plugin (`ms_plugin`) |
| `MS_BUILD_JIT` | OFF | LLVM ORC JIT backend for scalar REPL assignments (Linux/Clang + LLVM 18) |
| `MS_ENABLE_CUDA` | OFF | CUDA GPU backend (`ms_cuda`) |
| `MS_ENABLE_MPI` | OFF | Distributed MPI support |
| `MS_ENABLE_NCCL` | ON | Multi-GPU NCCL (when CUDA enabled) |
| `MS_ENABLE_AVX512` | ON | AVX-512 compile-time gate (`MS_ENABLE_AVX512` macro) |
| `MS_ENABLE_COVERAGE` | OFF | gcov/lcov instrumentation (Linux) |
| `MS_ENABLE_ASAN` | OFF | AddressSanitizer + UBSan (GCC/Clang) |
| `MS_USE_LIBCXX` | OFF | Use libc++ on Linux (when libstdc++ lacks full C++23) |
| `MS_VERBOSE_DISPATCH` | OFF | Runtime dispatch logging |
| `MS_CUDA_ARCHITECTURES` | `70;80;86;89;90` | CUDA SM architectures (CACHE string) |

CI builds typically pass `-DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF` for deterministic, portable runs.

### Windows Quick Build

```powershell
.\build.ps1
# or: cmake -S . -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF
ctest --test-dir build-msvc --output-on-failure
.\build.ps1 -Benchmark   # build-msvc-bench with MS_BUILD_BENCHMARKS=ON (+ matmul/fft smoke)
```

Binaries land in `build-msvc/bin/` (or `build-linux/bin/` on Linux).

## Test Layout

**374 CTest suites** as of Wave 223 (CHANGELOG); each suite is one GoogleTest executable or CLI smoke target. Individual suites contain many `TEST()` cases (several thousand total across the tree). Tests link the `mathscript` INTERFACE library and `GTest::gtest_main` unless noted.

| Directory | Contents |
|-----------|----------|
| **`tests/unit/`** | ~156 GoogleTest executables via `add_ms_test()`: core math, REPL, linalg, special functions, typed LU/solve, memory/error/runtime, Wave 57–60 domain modules, frameworks, and data-driven coverage. Plus `test_mathscriptc_cli`, `test_repl_cli`, `test_server_cli` (CLI smoke), and always-registered `test_fuzz_stress` (long-running smoke, not libFuzzer). |
| **`tests/numerical/`** | 62 NIST/DLMF and reference-value accuracy regressions: Bessel, LU/SVD/solve/chol/eig residuals, FFT impulse/Parseval/roundtrip, erf/gamma/digamma, ODE/PDE reference trajectories, iterative solvers. |
| **`tests/integration/`** | 146 cross-module pipeline tests: REPL→plot→save, session roundtrip, `mathscriptc` multi-line scripts, wave pipelines (57–99+), JIT/REPL parity, scientific and tensor decomposition pipelines. |
| **`tests/compliance/`** | `test_plugin_smoke`; 20 compile-fail/pass rule pairs when `MS_BUILD_PLUGIN=ON` (`compliance_*`). Each rule has `fail.cpp` + `ok.cpp`; `[[ms::unsafe]]` escape where applicable. When `MS_UNSAFE` is defined, also runs `compliance_unsafe_annotation`. |
| **`tests/fuzz/`** | Seven libFuzzer targets (built when `MS_BUILD_FUZZ=ON`): `fuzz_special_fns`, `fuzz_matrix_ops`, `fuzz_repl_input`, `fuzz_sym_parser`, `fuzz_poly_ops`, `fuzz_bignum`, `fuzz_mpi_message`. Corpus seeds under `tests/fuzz/corpus/<target>/`. |
| **`tests/performance/`** | Google Benchmark targets (`bench_matmul`, `bench_fft`, `bench_linalg`, `bench_repl`, `bench_special`, `bench_stats`, `bench_signal_filters`, …) built when `MS_BUILD_BENCHMARKS=ON`. Windows: `.\build.ps1 -Benchmark` uses `build-msvc-bench` and runs matmul/fft smoke. Linux: `scripts/bench_smoke.sh` covers all registered targets. |

CUDA-off is the CI default. Coverage and Valgrind scripts (`scripts/coverage_report.sh`, `scripts/valgrind_tests.sh`) expect a Debug Linux build with tests enabled.

## CI Pipeline (`.github/workflows/ci.yml`)

Triggered on push/PR to `main` (concurrency group cancels in-flight runs on the same ref):

1. **build-test-windows** — MSVC x64, Release, full `ctest`, install smoke (`scripts/package_smoke.ps1`), upload Windows ZIP artifact, optional NSIS/WiX CPack smoke when tools are present
2. **build-test-linux** — GCC 13, Release, full `ctest`, install smoke, CPack TGZ/ZIP/DEB/RPM smoke, unsafe surface audit (`scripts/unsafe_report.sh`), unsafe delta check (`scripts/unsafe_delta.sh`), vendor checksum verify (`scripts/verify_vendor.sh`)
3. **coverage-linux** — Debug + `MS_ENABLE_COVERAGE`, full `ctest`, 90% minimum line coverage enforced (`scripts/coverage_report.sh`, `MS_COVERAGE_MIN=90`)
4. **fuzz-linux** — libFuzzer smoke on all 7 targets (Clang, 4096 runs / 30 s per target); uses `tests/fuzz/corpus/<target>` when present
5. **valgrind-linux** — Debug memcheck over the suite (`scripts/valgrind_tests.sh`; excludes `test_fuzz_stress`)
6. **sanitizer-linux** — Debug + AddressSanitizer/UBSan (GCC 13); full `ctest` excluding `test_fuzz_stress`, `test_cuda_matmul`, and `test_cuda_stub`
7. **plugin-linux** — Clang + LLVM 18, `MS_BUILD_PLUGIN=ON`; builds `ms_plugin`, runs `test_plugin_smoke`, and compliance rule tests (`compliance_*`). **Twenty** compile-fail rules enforced (all `DiagnosticID`s in `diagnostics.hpp` except partial **`UnsafeAudit`** via `MS_UNSAFE` + `scripts/unsafe_report.sh`). Each rule has `fail.cpp` + `ok.cpp`; `[[ms::unsafe]]` escape where applicable. When `MS_UNSAFE` is defined, also runs `compliance_unsafe_annotation`.
8. **jit-linux** — Clang + LLVM 18, `-DMS_BUILD_JIT=ON`; links ORC LLJIT, runs `test_jit_backend` and `test_plot_console`. LLVM backend JIT-compiles scalar assignments; matrix calls dispatch to native kernels; other lines delegate to REPL when unsupported.
9. **benchmark-linux** — Google Benchmark regression (`bench_matmul`, `bench_fft`, `bench_linalg`, `bench_repl`, `bench_special`, `bench_stats`) when `MS_BUILD_BENCHMARKS=ON`

### Performance benchmarks

Build with `-DMS_BUILD_BENCHMARKS=ON` (see `tests/performance/`). On Windows MSVC: `.\build.ps1 -Benchmark` configures `build-msvc-bench` with benchmarks and tests enabled, builds all targets, and runs a short `bench_matmul` / `bench_fft` smoke. CI **`benchmark-linux`** runs `scripts/bench_regression.sh` with `MS_BENCH_REGRESSION=on` and `MS_BENCH_TOLERANCE=10`.

Regression detection compares JSON output from Google Benchmark against `tests/performance/baselines/linux-gcc13.json`. A run more than 10% slower than the stored median fails (`MS_BENCH_TOLERANCE`, default 10). Skip compare for smoke-only runs: `MS_BENCH_REGRESSION=off`.

```bash
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=OFF -DMS_BUILD_BENCHMARKS=ON
cmake --build build-bench --target bench_matmul bench_fft
bash scripts/bench_regression.sh build-bench                    # compare (default)
MS_BENCH_REGRESSION=off bash scripts/bench_regression.sh build-bench  # smoke only
bash scripts/bench_regression.sh --write-baseline build-bench   # refresh baseline medians
```

Baselines are captured with `--benchmark_out=… --benchmark_out_format=json` (wrapped by `bench_regression.sh`). Calibrate on the target runner profile (Linux GCC 13 Release) before enabling the CI regression gate.

### Scheduled and manual fuzz workflows

**`.github/workflows/nightly.yml`** — weekly (Sunday 04:00 UTC) **fuzz-extended**: 15 min/target × 7 libFuzzer targets. Manual **`fuzz-marathon`** dispatch (`workflow_dispatch` only): 60 min/target × 7. All jobs use `tests/fuzz/corpus/<target>` when present.

**`.github/workflows/fuzz-24h.yml`** — manual only (`workflow_dispatch`): 7 parallel matrix jobs, **86400 s (24 h) per target**, **1500 min** job timeout each. Required before tagging **v1.0.0** (zero crashes). Trigger from GitHub Actions UI or `gh workflow run fuzz-24h.yml`.

Vendor integrity: `scripts/verify_vendor.sh` + `vendor/CHECKSUMS.sha256`.

## Executables

| Binary | Purpose |
|--------|---------|
| `mathscriptc` | Batch/script driver linked to full `mathscript` |
| `mathscript-repl` | Interactive console or one-shot `-e` / `--eval`, `--load`, `--jit` (`ms_interp` + core math) |
| `mathscript-gui` | Qt6 IDE (`MS_BUILD_GUI=ON`): REPL + `PlotWidget` / OpenGL `PlotSurfWidget` (PNG export via File menu) |
| `mathscript-server` | Distributed/MPI-oriented server entry point |

Install rules and CPack packaging are defined in `cmake/install.cmake`.
