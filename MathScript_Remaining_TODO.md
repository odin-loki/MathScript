# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-18 (Wave 232 kickoff — **376 CTest suites** on `main` @ Wave 231; feature branches in flight)

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

## Feature work (Wave 232+) — IN PROGRESS

### Wave 232 (kickoff — parallel branches)

| Branch | Status | Deliverable |
|--------|--------|-------------|
| `wave232/sym-laplace` | in progress | `sym_laplace` / `sym_ilaplace` table-driven MVP |
| `wave232/sym-fourier` | in progress | `sym_fourier` / `sym_ifourier`, `sym_ztransform` / `sym_iztransform` MVP |
| `wave232/sym-repl` | in progress | REPL bindings for existing + new symbolic commands |
| `wave232/cuda-lu` | in progress | `cuda::lu()` via cuSOLVER |
| `wave232/cuda-stream` | in progress | `StreamPool::acquire()` + real device memory stats |
| `wave232/mpi-repl` | in progress | Stub-safe MPI/distributed REPL commands |
| `wave232/plugin-audit` | in progress | Unsafe-site registry + JSON audit report |
| `wave232/gui-repl` | in progress | GUI command history + Wave 231 crypto/fem/cfd REPL wrappers |
| `wave232/docs` | in progress | CHANGELOG, TODO, `docs/API.md` kickoff |

Wave 231 closed AES/ChaCha, 2D FEM, and 2D CFD. Wave 232 extends symbolic transforms, CUDA stubs, MPI REPL exposure, plugin audit, and GUI/REPL polish in parallel isolated worktrees; merges land independently.

| Wave | Focus |
|------|--------|
| **232–236** | GUI polish + REPL bindings (continued) |
| **237+** | Remaining API gaps (curve25519, AES-GCM, 3D FEM/CFD, scalable distributed LA, etc.) |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
