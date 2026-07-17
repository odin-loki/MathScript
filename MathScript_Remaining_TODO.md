# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-18 (reconciled against `main` @ Wave 230 — **374 CTest suites**, **28-bench smoke verified on MSVC and Linux CI**)

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

## Next: Feature work (Wave 231+)

| Wave | Focus |
|------|--------|
| **231** | crypto AES/ChaCha + fem 2D + cfd 2D |
| **231–232** | sym transforms + CUDA/MPI/plugin |
| **232–236** | GUI polish + REPL bindings |
| **237+** | Remaining API gaps |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
