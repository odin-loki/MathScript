# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-18 (reconciled against `main` @ Wave 227 — **374 CTest suites**, **25-bench smoke verified on MSVC**)

---

## Performance profiling ✅ COMPLETE (Waves 218–227)

Ten profiling waves optimized all identified hot paths across signal, fft, simd, stats, linalg, graph, image, tensorops, topo, geo, ode/pde/cfd, ml/prob/control, interp, crypto, fem, and benchmark infra.

| Wave | Focus |
|------|--------|
| 218–224 | (see CHANGELOG.md) |
| **225** | tensorops einsum, imfilter/sharpen/LoG, Floyd-Warshall, CZT, hilbert buffers, smoke fix |
| **226** | topo/geo/ode-pde/ml benches, smoke-safe args, MSVC baseline populated, REPL micro-opts |
| **227** | documentation closure; intentional O(n²) audit; `docs/PERFORMANCE.md`; final verify |

**No further profiling waves planned.** Remaining: Linux CI `bench_regression.sh --write-baseline` only.

---

## Next: Feature work (Wave 228+)

| Wave | Focus |
|------|--------|
| **228** | crypto AES/ChaCha + fem 2D + cfd 2D |
| **228–229** | sym transforms + CUDA/MPI/plugin |
| **230–234** | GUI polish + REPL bindings |
| **235+** | Remaining API gaps |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
