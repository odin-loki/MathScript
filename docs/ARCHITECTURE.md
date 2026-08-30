# MathScript Architecture

Function-level API: [`docs/API.md`](API.md). REPL usage: [`docs/USER_GUIDE.md`](USER_GUIDE.md). This document is the structural reference.

MathScript is a C++23 numerical and computer-algebra library with a console REPL, optional CUDA/MPI backends, and an in-tree BLAS/LAPACK layer. The repository is a CMake monorepo: static libraries under `src/`, public headers under `include/ms/`, GoogleTest under `tests/`.

## Layout

```
MathsScript/
├── cmake/              # options, platform, coverage, install, JIT, vendor verify
├── include/ms/        # Public API (installed)
├── src/                # Libraries and executables
├── tests/              # unit, numerical, integration, fuzz, compliance, performance
├── scripts/            # coverage, ASan helpers, bench, packaging, unsafe audit
├── docs/               # This file, API, user guide, waves
├── vendor/             # Third-party sources; CHECKSUMS.sha256
├── build.ps1 / build.sh
└── .github/workflows/  # ci.yml, nightly.yml, fuzz-24h.yml
```

Local Windows builds use a single directory: **`build-msvc`**. Linux CI uses `build` (or a job-specific name in the workflow). Do not keep extra `build-*` trees on a workstation.

## Source modules (`src/`)

Each library subdirectory builds `ms_<name>` except `plugin`, `gui`, and `exe`. Libraries below link into the umbrella `mathscript` INTERFACE target unless noted.

### Numerical core

| Module | Library | Role |
|--------|---------|------|
| `core` | `ms_core` | Matrix, Tensor, Sparse, Sym, Scalar, RNG, checked arithmetic, units |
| `runtime` | `ms_runtime` | Topology, thread pool, dispatch, load balancer, CPU BLAS/LAPACK kernels |
| `linalg` | `ms_linalg` | matmul, LU/QR/Cholesky/LDL, solve, eig, SVD, expm, iterative solvers |
| `fft` | `ms_fft` | 1D/2D FFT, DCT/DST, rfft helpers |
| `ode` | `ms_ode` | Fixed-step and adaptive ODE integrators |
| `pde` | `ms_pde` | Heat, wave, advection, Poisson, Helmholtz |
| `fem` | `ms_fem` | 1D/2D/3D P1 Poisson assembly and solve |
| `cfd` | `ms_cfd` | Finite-volume upwind advection (1D/2D/3D) |
| `poly` | `ms_poly` | Polynomial arithmetic, roots, factorisation, interpolation |
| `optim` | `ms_optim` | Local, global, and least-squares optimisation |
| `special` | `ms_special` | Bessel, gamma, elliptic, hypergeometric, and related functions |
| `symbolic` | `ms_symbolic` | Expression AST: parse, differentiate, simplify, transforms |
| `domain` | `ms_domain` | Small combinatorics and graph helpers |
| `simd` | `ms_simd` | ISA detection and xsimd vectorised kernels |

CPU BLAS/LAPACK kernels are in `src/runtime/cpu/` and `src/cpu/`, exposed via `include/ms/cpu/blas.hpp` and `include/ms/cpu/lapack.hpp`.

### Statistics and data

| Module | Library | Role |
|--------|---------|------|
| `stats` | `ms_stats` | Descriptive stats, tests, regression, time series |
| `prob` | `ms_prob` | PDF/CDF/quantile for common distributions |
| `ml` | `ms_ml` | Supervised/unsupervised learning, PCA, autodiff |
| `info` | `ms_info` | Entropy, divergences, channel capacity |

### Applied domains

| Module | Library | Role |
|--------|---------|------|
| `finance` | `ms_finance` | Options, bonds, portfolio, Monte Carlo |
| `control` | `ms_control` | Transfer functions, state space, LQR, Gramians |
| `signal` | `ms_signal` | Filters, convolution, spectral estimates |
| `image` | `ms_image` | Filters, morphology, features, geometry ops |
| `graph` | `ms_graph` | Traversal, shortest paths, MST, centrality, matching |
| `geo` | `ms_geo` | Hulls, Delaunay, KD-tree, curves |
| `diffgeo` | `ms_diffgeo` | Metrics, curvature, geodesics |
| `topo` | `ms_topo` | Simplicial complexes, persistent homology |
| `quantum` | `ms_quantum` | Gates, density matrices, entanglement |
| `numthy` | `ms_numthy` | Primality, factorisation, modular arithmetic |
| `combo` | `ms_combo` | Combinatorics, ranking, special integer sequences |
| `bignum` | `ms_bignum` | Arbitrary-precision integers and rationals |
| `compress` | `ms_compress` | RLE, Huffman, LZ, BWT, wavelet, ANS |
| `cplx` | `ms_cplx` | Residues, contours, conformal maps |
| `tensorops` | `ms_tensorops` | Einsum, CP/Tucker/HOSVD/TT, NMF |
| `crypto` | `ms_crypto` | SHA, HMAC, HKDF, PBKDF2, AES, ChaCha, X25519, Ed25519 |

### Interpreter and tooling

| Module | Library | Role |
|--------|---------|------|
| `interp` | `ms_interp` | REPL, session state, plot hooks, optional ORC JIT |
| `cuda` | `ms_cuda` | Optional GPU buffers, BLAS, FFT, solvers, `.cu` elementwise kernels with xsimd host fallback, NCCL stubs |
| `distributed` | `ms_distributed` | MPI context, block/gather `dist_*` solvers |
| `plugin` | `ms_plugin` | Optional Clang AST plugin (`MS_BUILD_PLUGIN=ON`) |

### REPL exposure

The REPL exposes user-facing mathematical operations. It does not bind kernel, allocator, or interpreter-internal APIs. BLAS/LAPACK kernels are reached through `linalg`. Lifecycle MPI calls (`init` / `finalize` / `barrier`) are not expressions.

Matrix assignments `name = callee(args)` dispatch through `ms::interp::dispatch_matrix_call` (`src/interp/matrix_call.{hpp,cpp}`): an unordered_map from callee name to a function pointer. Handlers are one `.cpp` per callee under `src/interp/matrix_calls/<domain>/`. CMake generates `matrix_call_register_all.cpp` so the static library keeps every handler. This avoids MSVC C1061 (nested if limit) and C1060 (huge translation units). Helpers live in `repl_engine_internal.cpp`.

### Research frameworks (`frameworks/`)

All five compile into `ms_frameworks`:

| Subdir | Namespace | Role |
|--------|-----------|------|
| `axiom` | `ms::axiom` | Genetic programming over `ms::Sym` |
| `cellai` | `ms::cellai` | Hebbian / multi-timescale associative memory |
| `cypha` | `ms::cypha` | Mixture-of-experts DIF with NIG uncertainty |
| `gria` | `ms::gria` | Entropy-based compute classification |
| `izaac` | `ms::izaac` | VRF, CSPRNG, DP/MPC/backtest utilities |

### Executables

| Binary | Role |
|--------|--------|
| `mathscriptc` | Batch script driver |
| `mathscript-repl` | Interactive REPL or `-e` / `--eval` |
| `mathscript-server` | MPI-oriented server entry |
| `mathscript-gui` | Qt6 IDE (`MS_BUILD_GUI=ON`) |

## Public headers

Headers mirror `src/` plus `cpu/`, `memory/`, `error/` (`Result<T>`), and `unsafe/`. `include/ms/ms.hpp` is the umbrella include. Prefer finer includes for compile-time cost.

## Build system

- CMake 3.28+, C++23
- Generator: Ninja
- Windows: `.\build.ps1` → `build-msvc`
- Linux: `cmake -S . -B build …`

### CMake options (`cmake/options.cmake`)

| Flag | Default | Purpose |
|------|---------|---------|
| `MS_BUILD_TESTS` | ON | GoogleTest + CTest |
| `MS_BUILD_INTEGRATION` | ON | Per-file integration executables |
| `MS_LINK_TESTS_SHARED` | OFF | PIC `libms_bundle.so` so Debug coverage/ASan can link 800 tests without copying the static library |
| `MS_BUILD_BENCHMARKS` | OFF | 28 Google Benchmark targets |
| `MS_BUILD_GUI` | OFF | Qt6 IDE |
| `MS_BUILD_FUZZ` | OFF | libFuzzer targets |
| `MS_BUILD_PLUGIN` | OFF | Clang enforcement plugin |
| `MS_BUILD_JIT` | OFF | LLVM ORC JIT for scalar REPL |
| `MS_ENABLE_CUDA` | OFF | GPU backend |
| `MS_ENABLE_MPI` | OFF | Distributed MPI |
| `MS_ENABLE_NCCL` | ON | NCCL when CUDA is on (stubs if not implemented) |
| `MS_ENABLE_AVX512` | ON | AVX-512 kernels |
| `MS_ENABLE_COVERAGE` | OFF | gcov (Linux) |
| `MS_ENABLE_ASAN` | OFF | ASan + UBSan |
| `MS_USE_LIBCXX` | OFF | libc++ on Linux |

CI typically uses `-DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF`.

## Tests

Tests are grouped by **mathematical domain** (`linalg`, `fft`, `special`, …), not by the wave number they were written in. `ctest -R int_fft` runs FFT pipelines; `ctest -R test_fft` runs FFT unit tests.

| Directory | Contents |
|-----------|----------|
| `tests/unit/<domain>/` | Library unit tests for that module |
| `tests/unit/repl/` | REPL session tests plus `test_repl_commands_<domain>.cpp` |
| `tests/numerical/<domain>/` | NIST/DLMF and residual/accuracy regressions |
| `tests/integration/<domain>/` | REPL/cross-module pipelines for that domain |
| `tests/compliance/` | Clang plugin compile-fail/pass pairs |
| `tests/fuzz/` | Seven libFuzzer targets + corpora; `test_fuzz_stress` always built |
| `tests/performance/<domain>/` | 28 `bench_*` executables when `MS_BUILD_BENCHMARKS=ON` |

CTest catalogue: **816** suites (Windows MSVC, CUDA off). Suite count is the number of test executables after configure. Wave-numbered pipelines were collapsed to one file per unique command set.

REPL matrix handlers live in `src/interp/matrix_calls/<domain>/`.

## CI (`.github/workflows/ci.yml`)

On push/PR to `main`:

1. **build-test-windows** — MSVC Release, full CTest, ZIP smoke
2. **build-test-linux** — GCC 13, no-exceptions syntax gate, CTest, CPack, unsafe audit
3. **coverage-linux** — 80% line coverage minimum (90% is the v1.0.0 tag goal)
4. **fuzz-linux** — 7 libFuzzer smokes
5. **sanitizer-linux** — ASan/UBSan (full CTest via shared test bundle; leak detection off)
6. **plugin-linux** — twenty compile-fail rules
7. **jit-linux** — ORC JIT smoke
8. **benchmark-linux** — 28 benches, 10% regression vs baseline

Nightly and `fuzz-24h.yml` cover longer fuzz campaigns. Tag criteria: [`RELEASE.md`](RELEASE.md).
