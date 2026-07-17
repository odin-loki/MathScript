# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-17 (reconciled against `main` @ Wave 225 — **374 CTest suites**, **23-bench smoke OK**, profiling **COMPLETE**)

---

## Performance profiling ✅ COMPLETE (Waves 218–225)

Eight profiling waves optimized all identified hot paths across signal, fft, simd, stats, linalg, graph, image, tensorops, interp, crypto, fem, and benchmark infra.

| Wave | Focus |
|------|--------|
| 218–224 | (see CHANGELOG.md) |
| **225** | tensorops einsum, imfilter/sharpen/LoG, Floyd-Warshall, CZT, hilbert buffers, bootstrap parity, baseline schema |

**No further profiling waves planned.** Remaining: Linux CI `bench_regression.sh --write-baseline` only.

---

## Next: Feature work (Wave 226+)

| Wave | Focus |
|------|--------|
| **226** | crypto AES/ChaCha + fem 2D + cfd 2D |
| **227–228** | sym transforms + CUDA/MPI/plugin |
| **229–233** | GUI polish + REPL bindings |
| **234+** | Remaining API gaps |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
