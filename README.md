# MathScript

[![CI](https://github.com/odin-loki/MathScript/actions/workflows/ci.yml/badge.svg)](https://github.com/odin-loki/MathScript/actions/workflows/ci.yml)

MathScript is a C++23 computer-algebra and numerical library with a console REPL. It implements dense and sparse linear algebra, special functions, statistics, ODE/PDE/FEM/CFD, optimisation, signal and image processing, number theory, graphs, geometry, topology, quantum primitives, control, finance, compression, and a small symbolic CAS — all against one set of conventions (`Result<T>` instead of exceptions, no external BLAS/LAPACK).

Executables: `mathscriptc`, `mathscript-repl`, `mathscript-server`. Optional Qt GUI, CUDA, MPI, and LLVM ORC JIT are CMake-gated.

## Documentation

| Document | Contents |
|----------|----------|
| [User guide](docs/USER_GUIDE.md) | Install, first REPL session, syntax, plotting, JIT |
| [Architecture](docs/ARCHITECTURE.md) | Layout, modules, CMake options, tests, CI |
| [API index](docs/API.md) | Public headers under `include/ms/` and REPL bindings |
| [Contributing](docs/CONTRIBUTING.md) | Build, test, coverage, fuzz, plugin, packaging |
| [Performance](docs/PERFORMANCE.md) | Benchmarks, baselines, known complexity trade-offs |
| [Release](docs/RELEASE.md) | 1.0.0 tag criteria and remaining gates |
| [1.0 scope](docs/RELEASE_DECISIONS.md) | Stubs and post-1.0 deferrals (decided, not remaining) |
| [Unsafe surface](UNSAFE_REVIEW.md) | Approved `MS_UNSAFE` sites |
| [Wave history](docs/WAVES.md) | How the library was built, wave by wave |

## What it is

The surface is a numerical library: linear algebra, FFT, statistics, solvers, optimisation, signal/image processing, number theory, graphs, geometry, topology, quantum primitives, control, and finance. Implementations share `std::vector` coefficient polynomials, `Result<T>` error handling, and defensive checks on malformed input.

Two properties of the tree:

- **In-tree BLAS/LAPACK.** `linalg` owns LU/QR/SVD/eig/Cholesky and LAPACK-style kernels (`dorgbr`, `dlartg`, `dbdsqr`, …). There is no Eigen or OpenBLAS dependency for the default path.
- **Restricted C++ subset.** A Clang plugin can enforce twenty compile-time rules (no raw `new`, no `throw`, no C-style casts, …). Production code returns `Result<T>`.

## Modules

Static libraries under `src/` (35 libraries, plus `exe` / `gui` / `plugin`):

| Group | Modules |
|-------|---------|
| Core and dispatch | `core`, `runtime`, `linalg`, `simd`, `cuda`, `distributed` |
| Numerical methods | `fft`, `ode`, `pde`, `fem`, `cfd`, `poly`, `optim`, `special`, `signal` |
| Statistics and ML | `stats`, `prob`, `ml`, `info` |
| Applied | `finance`, `control`, `graph`, `geo`, `diffgeo`, `topo`, `quantum`, `cplx`, `tensorops`, `numthy`, `combo`, `bignum`, `compress`, `image`, `crypto` |
| Symbolic and REPL | `symbolic`, `interp` |
| Frameworks | `frameworks` (`axiom`, `cellai`, `cypha`, `gria`, `izaac`) |
| Helpers | `domain` |

CPU BLAS/LAPACK kernels live in `linalg` and are declared in `include/ms/cpu/blas.hpp` and `include/ms/cpu/lapack.hpp`.

## Status

- **Version:** CMake project version **1.0.0**. The `v1.0.0` git tag is not cut; remaining gates are in [`docs/RELEASE.md`](docs/RELEASE.md).
- **Tests:** **816** CTest suites, 100% passed on Windows MSVC Release (CUDA off, ~36 s at `-j 32`). Tests live under `tests/{unit,numerical,integration,performance}/<domain>/`. CI enforces **90%** line coverage. REPL matrix calls dispatch through a name-keyed handler registry (`src/interp/matrix_calls/<domain>/`).
- **Benchmarks:** 28 Google Benchmark targets passed locally with `--benchmark_min_time=0.001s`. CI regression uses 10% tolerance vs `linux-gcc13.json`.
- **CI:** Windows MSVC and Linux GCC 13; coverage; libFuzzer smoke (7 targets); AddressSanitizer + UBSan; 28-bench regression (10% tolerance); Clang plugin; vendor checksums; optional JIT and plugin jobs.

## Build

Windows (single tree `build-msvc`):

```powershell
.\build.ps1
ctest --test-dir build-msvc --output-on-failure
.\build-msvc\bin\mathscript-repl.exe
```

Linux:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build
ctest --test-dir build --output-on-failure
./build/bin/mathscript-repl
```

Optional: `.\build.ps1 -Benchmark` (same `build-msvc` tree, `MS_BUILD_BENCHMARKS=ON`). Full options: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).
