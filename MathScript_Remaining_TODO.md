# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-18 (Wave 233 complete — **381 CTest suites** on `main` @ Wave 233; next: Wave 234+ GUI/REPL + 237+ API gaps)

---

## Performance profiling iteration ✅ FULLY COMPLETE (Waves 218–230)

Twelve waves optimized and **certified** all identified hot paths across every `src/` module (35 libraries), closed benchmark infra, and established baseline refresh workflows. Wave 228 was a certification pass; Waves 229–230 closed infra and baseline path only (no code profiling).

| Wave | Focus |
|------|--------|
| 218–224 | (see CHANGELOG.md) |
| **225** | tensorops einsum, imfilter/sharpen/LoG, Floyd-Warshall, CZT, hilbert buffers, smoke fix |
| **226** | topo/geo/ode-pde/ml benches, smoke-safe args, MSVC baseline populated, REPL micro-opts |
| **227** | quantum/bignum/compress benches, finance/symbolic/frameworks opts, Linux+MSVC baseline sync, `docs/PERFORMANCE.md` |
| **228** | Certification audit (`numthy`, `cplx`, `optim`, `diffgeo`, `domain`, `runtime/cpu`); `bench_finance`; Linux schema + smoke guard |
| **229** | Benchmark infra closure — Linux CI builds all **28** benches; `bench_cmake_targets.sh`; docs sync |
| **230** | Baseline refresh path — `bench-baseline-linux.yml` (`workflow_dispatch`); docs final sign-off |

**PROFILING ITERATION FULLY COMPLETE — no further profiling, benchmark-infra, or baseline-path waves.**

| Item | Status |
|------|--------|
| Linux CI bench build | ✅ `benchmark-linux` builds all 28 targets |
| Linux baseline refresh | ✅ `bench-baseline-linux.yml` — `workflow_dispatch`; artifact download → maintainer commit |
| Windows baseline refresh | ✅ `bench_write_msvc_baseline.ps1` (local) |

---

## Wave 231 ✅ COMPLETE (crypto AES/ChaCha + FEM 2D + CFD 2D)

| Branch | Shipped |
|--------|---------|
| `wave231/crypto-chacha` | ChaCha20 stream cipher |
| `wave231/crypto-*` | AES-128/256 ECB + AES-128 CBC encrypt/decrypt |
| `wave231/fem-2d` | 2D P1 triangular Poisson assembly + solve |
| `wave231/cfd-2d` | 2D structured FVM upwind advection |

**376 CTest suites** (+2 registrations: `integration_crypto_wave231`, `test_cfd_2d`; ~27 new cases in existing/new binaries). Profiling iteration unchanged — **FULLY COMPLETE (Waves 218–230).**

| Branch | Shipped |
|--------|---------|
| `wave231b/bench-crypto` | AES/ChaCha benchmarks in `bench_crypto` |
| `wave231b/bench-fem` | 2D P1 assemble/solve benchmarks in `bench_fem` |
| `wave231b/bench-cfd` | 2D FVM advection benchmarks in `bench_ode_pde` |

**Wave 231 benchmark coverage complete** (`bench_crypto`, `bench_fem`, `bench_ode_pde`; **28-bench smoke unchanged**). Profiling iteration still **FULLY COMPLETE (Waves 218–230)** — no further profiling waves.

---

## Product profiling & adjustment ✅ CLOSED

The profiling and product-adjustment program is **closed**. Waves **218–230** certified all hot paths, benchmark infra, and baseline refresh; Wave **231** (+ **231b**) added feature work and benchmark coverage only. **No further profiling, benchmark-infra, or product-adjustment waves.** Wave **232+** is feature work only — see `mathscript-master-plan.md` and `CHANGELOG.md`.

---

## Wave 232 ✅ COMPLETE (symbolic transforms + CUDA + MPI REPL + plugin audit + GUI history)

| Branch | Shipped |
|--------|---------|
| `wave232/sym-laplace` | `sym_laplace` / `sym_ilaplace` table-driven MVP |
| `wave232/sym-fourier` | `sym_fourier` / `sym_ifourier`, `sym_ztransform` / `sym_iztransform` MVP |
| `wave232/sym-repl` | REPL bindings for expand/collect/series/limit/solve/substitute + transforms |
| `wave232/cuda-lu` | `cuda::lu()` via cuSOLVER |
| `wave232/cuda-stream` | `StreamPool::acquire()` + real `device_stats` via `cudaMemGetInfo` |
| `wave232/mpi-repl` | Stub-safe `mpi_rank`, `mpi_size`, `mpi_allreduce_sum`, `dist_solve` |
| `wave232/plugin-audit` | `UnsafeRegistry` + `${CMAKE_BINARY_DIR}/ms-unsafe-audit.json` |
| `wave232/gui-repl` | GUI Up-arrow REPL history; Wave 231 crypto/fem/cfd REPL wrappers |

**379 CTest suites** (+3 registrations: `test_symbolic_transforms`, `integration_repl_wave232_pipeline`, `integration_distributed_repl_pipeline`; ~24 transform unit tests + pipeline coverage). Profiling iteration unchanged — **FULLY COMPLETE (Waves 218–230).**

---

## Wave 233 ✅ COMPLETE (GUI polish + optim/control/quantum REPL + sym_dsolve + CUDA REPL)

| Branch | Shipped |
|--------|---------|
| `wave233/gui-polish` | Syntax highlighting, layout persistence, variable inspector, red errors, Stop/cancel, richer GPU status |
| `wave233/repl-optim` | `bfgs`, `lbfgs`, `nelder_mead`, `adam`, `golden_section`, `levenberg_marquardt` |
| `wave233/repl-control` | `control_poles`, `control_zeros`, `control_step_info`, `control_nyquist` |
| `wave233/repl-quantum` | `quantum_purity`, `quantum_schmidt_rank`, `quantum_uncertainty`, `quantum_grover_optimal_iterations` |
| `wave233/sym-dsolve` | `sym_dsolve` separable first-order ODE MVP |
| `wave233/cuda-repl` | `cuda_lu`, `cuda_add` REPL bindings |

**381 CTest suites** (+2 registrations: `test_symbolic_dsolve`, `integration_repl_wave233_pipeline`; 7 dsolve unit tests + optim/control/quantum pipeline coverage). Profiling iteration unchanged — **FULLY COMPLETE (Waves 218–230).**

---

## Feature work (Wave 234+) — IN PROGRESS

Wave 233 closed GUI polish (syntax highlight, layout persistence, variable inspector, error styling, Stop/cancel, GPU status) and a major REPL binding tranche (optim, control, quantum, `sym_dsolve`, CUDA). **Next up: Wave 234–236 remaining GUI/REPL bindings**; Wave **237+** for remaining API gaps.

| Wave | Focus |
|------|--------|
| **234–236** | Remaining GUI polish + REPL bindings |
| **237+** | Remaining API gaps (curve25519, AES-GCM, 3D FEM/CFD, scalable distributed LA, etc.) |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
