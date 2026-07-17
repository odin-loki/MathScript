# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-18 (reconciled against `main` @ Wave 228 — **374 CTest suites**, **27-bench smoke verified on MSVC**; `bench_finance` on parallel branch)

---

## Performance profiling ✅ COMPLETE (Waves 218–228)

Eleven profiling waves optimized and **certified** all identified hot paths across every `src/` module (35 libraries). Wave 228 was a certification pass — not a new optimization sweep.

| Wave | Focus |
|------|--------|
| 218–224 | (see CHANGELOG.md) |
| **225** | tensorops einsum, imfilter/sharpen/LoG, Floyd-Warshall, CZT, hilbert buffers, smoke fix |
| **226** | topo/geo/ode-pde/ml benches, smoke-safe args, MSVC baseline populated, REPL micro-opts |
| **227** | quantum/bignum/compress benches, finance/symbolic/frameworks opts, Linux+MSVC baseline sync, `docs/PERFORMANCE.md` |
| **228** | Certification audit (`numthy`, `cplx`, `optim`, `diffgeo`, `domain`, `runtime/cpu`); `bench_finance`; Linux schema + smoke guard |

**PROFILING ITERATION DONE — do not start Wave 229 profiling.**

Remaining perf infra only: Linux CI `bench_regression.sh --write-baseline` to populate null medians.

---

## Next: Feature work (Wave 229+)

| Wave | Focus |
|------|--------|
| **229** | crypto AES/ChaCha + fem 2D + cfd 2D |
| **229–230** | sym transforms + CUDA/MPI/plugin |
| **231–235** | GUI polish + REPL bindings |
| **236+** | Remaining API gaps |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
