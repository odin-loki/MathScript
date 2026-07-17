# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-17 (reconciled against `main` @ Wave 223 — **374 CTest suites**, MSVC green, `.\build.ps1 -Benchmark` smoke OK — 7 benches)

This document tracks open items from `mathscript-master-plan.md` and the original audit below. Work proceeds in **waves** of up to 8 parallel Composer 2.5 subagents (isolated git worktrees).

---

## Status vs original audit (2026-07-17)

| Original claim | Current reality |
|---|---|
| Wave 141 / 342 tests | **Wave 223 / 374 tests** — full MSVC suite passing |
| `ms::sym` still 13 functions | **Expanded:** sub/div/neg/tan/sqrt, expand, collect, integrate, substitute, sym_parse, sym_limit, sym_series, sym_solve_linear |
| UNSAFE `approved_sites: 6` | **40** sites documented in `UNSAFE_REVIEW.md` |
| Build broken on Linux `-fno-exceptions` | **Fixed (Wave 220)** — matrix/numa fallbacks |
| `ms::crypto` / `ms::fem` / `ms::cfd` absent | **MVP present (Wave 220)** — sha256/512/hmac; 1D Poisson FEM; 1D upwind advection |
| GUI incomplete | **Partial (Wave 220)** — REPL on `QThread`; syntax highlight / multi-plot still open |
| `vendor/` empty | **Fixed (Wave 220)** — GoogleTest v1.14.0 vendored |

---

## Performance profiling ✅ (Waves 218–223 — done)

Six profiling waves completed. Hot paths optimized across signal, fft, simd, stats, linalg, graph, image, interp, crypto, fem, and bench infra.

- [x] FFT convolve, moving_average, iterative FFT, twiddle cache (218)
- [x] welch/spectrogram, rfft, poly batch, median_filter, FMA dot, percentile (219)
- [x] xcorr, sosfilt in-place, savgol, hybrid conv2, SHA, FEM Thomas, norm_l2 (221)
- [x] coherence/spectrogram buffers, periodogram, BFS, SLIC, REPL eval, sum_squares (222)
- [x] explicit fft2 dims, 2D FFT conv2, watershed, gaussian blur, welch/filtfilt/resample, baseline schema (223)
- [ ] **Linux CI:** run `bench_regression.sh --write-baseline` to fill null medians (infra only)

---

## P0 — Blocking ✅ (Wave 220 — done)

- [x] **Remove `throw` from `matrix.hpp` and `numa_allocator.hpp`**
- [ ] **Verify Linux CI green** after exception fix
- [x] **Vendor GoogleTest** into `vendor/`
- [x] **Regenerate `UNSAFE_REVIEW.md`** baseline

---

## P1 — Missing §2 modules (Waves 224+)

- [x] **`ms::crypto` MVP start** — sha256/sha512, hmac (Wave 220–221)
- [ ] **`ms::crypto` finish** — aes256_cbc, chacha20
- [x] **`ms::fem` 1D MVP** — mesh1d, P1, assemble, Dirichlet, Thomas solve (Wave 220–221)
- [ ] **`ms::fem` 2D** — lagrange P1 on quads/tris
- [x] **`ms::cfd` 1D MVP** — upwind FVM advection (Wave 220)
- [ ] **`ms::cfd` 2D** — coupled heat or 2D advection

---

## P2 — Symbolic CAS depth (Waves 225–226)

- [x] `sym_limit`, `sym_series`, `sym_solve_linear` (Wave 220)
- [ ] `sym_dsolve` (separable ODEs)
- [ ] `sym_laplace_transform`, `sym_fourier_transform`

---

## P3 — Infrastructure (Waves 227–228)

- [ ] **CUDA** — real `cuda::lu` or document+test stub boundary
- [ ] **MPI** — non-stub `MpiContext` when `mpiexec -n > 1`
- [ ] **`MS_BUILD_PLUGIN=ON` in CI** — Clang profile plugin smoke

---

## P4 — GUI (Waves 229–231)

- [x] Phase 1: REPL/script eval on worker `QThread` (Wave 220)
- [ ] Phase 2: Syntax highlighting in editor
- [ ] Phase 3: Multi-plot manager, heatmap thumbnails, live hardware bar, `QSettings` persistence

---

## P5 — Master-plan API gaps (Waves 232+)

- [ ] REPL bindings for post-Wave-187 functions
- [ ] `ms::signal`: ellip/bessel/remez, EMD/VMD
- [ ] `ms::geo`: `poly_diff`, robust non-convex booleans
- [ ] 24h fuzz marathon, packaging, full ORC JIT v2

---

## Wave schedule (Composer 2.5 subagents)

| Wave | Focus | Status |
|------|--------|--------|
| **218–223** | Performance profiling passes I–V | ✅ |
| **224** | crypto AES/ChaCha + fem 2D + cfd 2D | planned |
| **225–226** | sym transforms + CUDA/MPI/plugin | planned |
| **227–231** | GUI polish + REPL bindings | planned |
| **232+** | Remaining API gaps | ongoing |

---

## Suggested immediate order

1. Wave 224 — extend crypto/fem/cfd to 2D
2. Waves 225–226 — sym transforms + infrastructure
3. Waves 227–231 — GUI + REPL bindings
4. Linux CI baseline regeneration (when benchmark-linux runs)

---

*Original audit sections preserved below for history.*

<details>
<summary>Original audit (pre-Wave 219 snapshot)</summary>

Sixteen of seventeen §2 math modules existed with real implementations. Still absent: `ms::crypto`, `ms::fem`, `ms::cfd`. Build blocked by two `throw` sites under `-fno-exceptions`. GUI plan deferred to separate document.

</details>
