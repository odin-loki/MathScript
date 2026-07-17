# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-18 (reconciled against `main` @ Wave 229 — **374 CTest suites**, **28-bench smoke verified on MSVC and Linux CI**)

---

## Performance profiling + infra ✅ COMPLETE (Waves 218–229)

Eleven profiling waves optimized and **certified** all identified hot paths across every `src/` module (35 libraries). Wave 228 was a certification pass; Wave 229 closed benchmark **infra** only (no code profiling).

| Wave | Focus |
|------|--------|
| 218–224 | (see CHANGELOG.md) |
| **225** | tensorops einsum, imfilter/sharpen/LoG, Floyd-Warshall, CZT, hilbert buffers, smoke fix |
| **226** | topo/geo/ode-pde/ml benches, smoke-safe args, MSVC baseline populated, REPL micro-opts |
| **227** | quantum/bignum/compress benches, finance/symbolic/frameworks opts, Linux+MSVC baseline sync, `docs/PERFORMANCE.md` |
| **228** | Certification audit (`numthy`, `cplx`, `optim`, `diffgeo`, `domain`, `runtime/cpu`); `bench_finance`; Linux schema + smoke guard |
| **229** | Benchmark infra closure — Linux CI builds all **28** benches; `bench_cmake_targets.sh`; docs sync |

**PROFILING + INFRA DONE — no further profiling or benchmark-infra waves.**

| Item | Status |
|------|--------|
| Linux CI bench build | ✅ Fixed — `benchmark-linux` builds all 28 targets |
| Baseline median refresh | Manual or scheduled — `bench_regression.sh --write-baseline` (Linux) / `bench_write_msvc_baseline.ps1` (Windows); null entries still skipped in compare |

---

## Next: Feature work (Wave 230+)

| Wave | Focus |
|------|--------|
| **230** | crypto AES/ChaCha + fem 2D + cfd 2D |
| **230–231** | sym transforms + CUDA/MPI/plugin |
| **231–235** | GUI polish + REPL bindings |
| **236+** | Remaining API gaps |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
