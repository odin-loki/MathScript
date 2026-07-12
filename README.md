# MathScript

[![CI](https://github.com/odin-loki/MathScript/actions/workflows/ci.yml/badge.svg)](https://github.com/odin-loki/MathScript/actions/workflows/ci.yml)

MathScript is a C++23 computer-algebra and HPC math system: 35 static libraries under `src/` (dense/sparse linear algebra through number theory and computational geometry), an in-tree BLAS/LAPACK CPU kernel layer with no external linear-algebra dependency, a console REPL (`mathscript-repl`) with optional LLVM ORC JIT for scalar expressions, and a Clang plugin that enforces a restricted C++ subset (`Result<T>` instead of exceptions, no raw `new`, twenty compile-time compliance rules). Executables: `mathscriptc`, `mathscript-repl`, `mathscript-server`; optional Qt GUI and CUDA/MPI backends behind CMake flags.

## Documentation

- [User guide](docs/USER_GUIDE.md) — new to the REPL? Start here for install, first session, and common commands.
- [Architecture](docs/ARCHITECTURE.md) — repository layout, module table, CMake options, and CI job reference for contributors.
- [API index](docs/API.md) — public headers under `include/ms/`, grouped by module with function-level notes.
- [Contributing](docs/CONTRIBUTING.md) — full build matrix: coverage, libFuzzer, Valgrind, Clang plugin, JIT, benchmarks, packaging.
- [Release checklist](docs/RELEASE.md) — Phase 10 exit criteria and the 1.0.0 tag procedure.
- [Changelog](CHANGELOG.md) — authoritative wave-by-wave history (Waves 56–206 and onward); the README does not duplicate it.
- [Master plan](mathscript-master-plan.md) — original ten-phase design spec and delivery roadmap.
- [Unsafe surface review](UNSAFE_REVIEW.md) — approved `MS_UNSAFE` escape hatches and the scripted audit baseline.

## What this actually is

Most of the surface area is a fairly conventional (if unusually broad) numerical library: linear algebra, FFT, statistics, ODE/PDE solvers, optimization, signal/image processing, number theory, graph theory, computational/differential geometry, topology, quantum computing primitives, control theory, and financial math, all implemented from scratch against the same conventions (`std::vector`-of-coefficients polynomials, `Result<T>` instead of exceptions, defensive early-returns on malformed input, doc comments that cite the reference formula rather than just restating the signature).

Two things make it a bit more interesting than "yet another math library":

- **It doesn't borrow BLAS/LAPACK.** The `linalg` module implements its own LU/QR/SVD/eig/Cholesky kernels, LAPACK-style (`dorgbr`, `dlartg`, `dbdsqr`, ...). That's more work than linking Eigen, but it means every numerical bug is *this repository's* bug to find, not a black box — see [Bugs that were worth finding](#bugs-that-were-worth-finding) below.
- **It's built almost entirely by waves of parallel AI subagents**, each working in an isolated git worktree on one self-contained module addition, tested and merged independently. See `mathscript-master-plan.md` and [`CHANGELOG.md`](CHANGELOG.md) for the full history.

## Architecture, briefly

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full repository layout, module table, and CMake option reference. The short version: static libraries per module under `src/`, mirrored public headers under `include/ms/`, GoogleTest suites under `tests/{unit,numerical,integration,fuzz}/`, everything wired through a single CMake monorepo build.

Verified module directories under `src/` (35 libraries, excluding `exe`/`gui`/`plugin`):

| Group | Modules |
|-------|---------|
| **Core & dispatch** | `core`, `runtime`, `linalg`, `simd`, `cuda`, `distributed` |
| **Numerical methods** | `fft`, `ode`, `pde`, `poly`, `optim`, `special`, `signal` |
| **Statistics & ML** | `stats`, `prob`, `ml`, `info` |
| **Applied domains** | `finance`, `control`, `graph`, `geo`, `diffgeo`, `topo`, `quantum`, `cplx`, `tensorops`, `numthy`, `combo`, `bignum`, `compress`, `image` |
| **Symbolic & REPL** | `symbolic`, `interp` |
| **Research frameworks** | `frameworks` (`axiom`, `cellai`, `cypha`, `gria`, `izaac`) |
| **Legacy helpers** | `domain` |

CPU BLAS/LAPACK kernels live inside `linalg` and are exposed via `include/ms/cpu/blas.hpp` and `include/ms/cpu/lapack.hpp`.

## Bugs that were worth finding

A few things surfaced during development that are worth calling out on their own merits, independent of which wave found them:

- **A sign error in `dlartg` and a wrong transpose flag in `dorgbr`** silently corrupted right singular vectors and `U`/`V` orthogonality in `ms::linalg::svd` for dozens of waves before anyone noticed, for essentially any `m×n` matrix with `n≥3` — masked because reconstruction (`U·Σ·Vᵗ`) still looked self-consistent even with the wrong orthogonal factors. Found while investigating an unrelated rank-deficient-tensor-decomposition bug, fixed by computing the Givens rotation directly via `hypot`/`cs=f/r`/`sn=g/r` instead of a sign-fragile `copysign` formula.
- **`ms::izaac::CSPRNG` was badly biased** — only 8 of 32 internal state bytes were mutated per call. It was caught by a Monte Carlo π estimator returning `4.0` and `3.0` instead of something near `3.14159`, which is about as visible as a broken RNG can get. Replaced with a proper xoshiro256** generator with SplitMix64 seed expansion.
- **A mirror-index bug in the shared low-pass FFT filter helper** made every "low-pass" filter in `ms::signal` (`lowpass`, `butterworth`, `highpass`, `bandpass`) zero out nearly all non-DC frequency content regardless of the requested cutoff — functionally a no-op beyond the mean. A pre-existing test had been passing by silently relying on the bug.

None of these were caught by code review; all three were caught by a test asserting a *property* of the correct answer (orthogonality, a known constant, a known filter response) rather than checking that the code ran without crashing.

## Status

- **Version:** 1.0.0 — Phase 10 (hardening) complete; post-1.0 work continues via the wave process above (latest: Wave 207 in [`CHANGELOG.md`](CHANGELOG.md))
- **Tests:** **368** CTest suites, 100% passing (CUDA disabled in CI); ~91% line coverage (**90%** minimum enforced in CI)
- **CI:** Windows MSVC + Linux GCC 13 build/test; coverage; libFuzzer smoke (7 targets); Valgrind memcheck; benchmark regression gate (`bench_matmul`, `bench_fft`, 10% tolerance); Clang plugin compliance (20 enforced rules); vendor checksum verification; optional `jit-linux` and `plugin-linux` jobs

## Build (quickstart)

Enough to get a first build and REPL session. For coverage, fuzz, Valgrind, plugin, JIT, and packaging options, see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

### Windows

```powershell
.\build.ps1
ctest --test-dir build-msvc --output-on-failure
.\build-msvc\bin\mathscript-repl.exe
```

One-shot eval: `.\build-msvc\bin\mathscript-repl.exe -e "surf([1, 2; 3, 4])"`. Optional JIT: `--jit -e "x = 1 + 2"` when built with `-DMS_BUILD_JIT=ON`.

### Linux (CI-style)

```bash
cmake -S . -B build-linux -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build-linux
ctest --test-dir build-linux --output-on-failure
./build-linux/bin/mathscript-repl
```

## Phase progress

| Phase | Name | Status |
|-------|------|--------|
| 0 | Foundation | Complete |
| 1 | Core types + REPL | Complete |
| 2 | Math library (CPU) | Complete |
| 3 | GPU / CUDA | Stub + optional backend |
| 4–8 | IDE, distributed, frameworks, special functions | Substantial progress |
| 9 | Own BLAS/LAPACK core | Complete |
| 10 | Hardening (CI, coverage, packaging) | Complete — **v1.0.0** |

Per-wave additions (REPL bindings, new modules, bug fixes) are recorded only in [`CHANGELOG.md`](CHANGELOG.md).
