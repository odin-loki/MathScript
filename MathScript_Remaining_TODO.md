# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-18 (reconciled against `main` @ Wave 226 — **374 CTest suites**, **25-bench smoke verified on MSVC**)

---

## Performance profiling ✅ COMPLETE (Waves 218–226)

Nine profiling waves optimized all identified hot paths across signal, fft, simd, stats, linalg, graph, image, tensorops, topo, geo, ode/pde/cfd, ml/prob/control, interp, crypto, fem, and benchmark infra.

| Wave | Focus |
|------|--------|
| 218–224 | (see CHANGELOG.md) |
| **225** | tensorops einsum, imfilter/sharpen/LoG, Floyd-Warshall, CZT, hilbert buffers, smoke fix |
| **226** | topo/geo/ode-pde/ml benches, smoke-safe args, MSVC baseline populated, REPL micro-opts |

**No further profiling waves planned.** Remaining: Linux CI `bench_regression.sh --write-baseline` only.

---

## Next: Feature work (Wave 227+)

| Wave | Focus |
|------|--------|
| **227** | crypto AES/ChaCha + fem 2D + cfd 2D |
| **227–228** | sym transforms + CUDA/MPI/plugin |
| **229–233** | GUI polish + REPL bindings |
| **234+** | Remaining API gaps |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
