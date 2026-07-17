# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-17 (reconciled against `main` @ Wave 221 — **374 CTest suites**, MSVC green, `.\build.ps1 -Benchmark` smoke OK)

This document tracks open items from `mathscript-master-plan.md` and the original audit below. Work proceeds in **waves** of up to 8 parallel Composer 2.5 subagents (isolated git worktrees).

---

## Status vs original audit (2026-07-17)

| Original claim | Current reality |
|---|---|
| Wave 141 / 342 tests | **Wave 221 / 374 tests** — full MSVC suite passing |
| `ms::sym` still 13 functions | **Expanded:** sub/div/neg/tan/sqrt, expand, collect, integrate, substitute, sym_parse, sym_limit, sym_series, sym_solve_linear |
| UNSAFE `approved_sites: 6` | **40** sites documented in `UNSAFE_REVIEW.md` |
| Build broken on Linux `-fno-exceptions` | **Fixed (Wave 220)** — matrix/numa fallbacks |
| `ms::crypto` / `ms::fem` / `ms::cfd` absent | **MVP present (Wave 220)** — sha256/512/hmac; 1D Poisson FEM; 1D upwind advection |
| GUI incomplete | **Partial (Wave 220)** — REPL on `QThread`; syntax highlight / multi-plot still open |
| `vendor/` empty | **Fixed (Wave 220)** — GoogleTest v1.14.0 vendored |

---

## P0 — Blocking ✅ (Wave 220 — done)

- [x] **Remove `throw` from `matrix.hpp` and `numa_allocator.hpp`**
- [ ] **Verify Linux CI green** after exception fix
- [x] **Vendor GoogleTest** into `vendor/`
- [x] **Regenerate `UNSAFE_REVIEW.md`** baseline

---

## P1 — Missing §2 modules (partial — Waves 220–223)

- [x] **`ms::crypto` MVP start** — sha256/sha512, hmac (Wave 220–221 perf)
- [ ] **`ms::crypto` finish** — aes256_cbc, chacha20
- [x] **`ms::fem` 1D MVP** — mesh1d, P1, assemble, Dirichlet, Thomas solve (Wave 220–221)
- [ ] **`ms::fem` 2D** — lagrange P1 on quads/tris
- [x] **`ms::cfd` 1D MVP** — upwind FVM advection (Wave 220)
- [ ] **`ms::cfd` 2D** — coupled heat or 2D advection

---

## P2 — Symbolic CAS depth (Waves 224–225)

- [x] `sym_limit`, `sym_series`, `sym_solve_linear` (Wave 220)
- [ ] `sym_dsolve` (separable ODEs)
- [ ] `sym_laplace_transform`, `sym_fourier_transform`
- [ ] Boolean/set algebra, codegen (long-term)

---

## P3 — Infrastructure stubs (Waves 226–227)

- [ ] **CUDA** — real `cuda::lu` or document+test stub boundary
- [ ] **MPI** — non-stub `MpiContext` when `mpiexec -n > 1`
- [ ] **`MS_BUILD_PLUGIN=ON` in CI** — Clang profile plugin smoke

---

## P4 — GUI (Waves 228–230)

- [x] Phase 1: REPL/script eval on worker `QThread` (Wave 220)
- [ ] Phase 2: Syntax highlighting in editor
- [ ] Phase 3: Multi-plot manager, heatmap thumbnails, live hardware bar, `QSettings` persistence

---

## P5 — Master-plan API gaps (Waves 231+)

- [ ] REPL bindings for post-Wave-187 functions
- [ ] `ms::signal`: ellip/bessel/remez, periodogram, EMD/VMD
- [ ] Performance: coherence/spectrogram FFT reuse, REPL eval, graph BFS, image filters (Wave 222+)
- [ ] `ms::geo`: `poly_diff`, robust non-convex booleans
- [ ] 24h fuzz marathon, packaging, full ORC JIT v2

---

## P6 — Non-code

- [ ] **Trademark:** USPTO clearance for "MathScript"

---

## Wave schedule (Composer 2.5 subagents)

| Wave | Focus | Status |
|------|--------|--------|
| **220** | P0: exceptions, vendor gtest, crypto/fem/cfd, sym CAS, GUI thread | ✅ |
| **221** | Performance III: xcorr, sosfilt, savgol, conv2, SHA, FEM, norm_l2 | ✅ |
| **222** | Performance IV: coherence, spectrogram, graph BFS, image filters, REPL, bench smoke | in progress |
| **223** | crypto AES/ChaCha + fem 2D + cfd 2D | planned |
| **224–225** | sym transforms + CUDA/MPI/plugin | planned |
| **226–230** | GUI polish + REPL bindings | planned |
| **231+** | Remaining API gaps | ongoing |

---

## Suggested immediate order

1. Wave 222 — continue profiling hot paths (signal coherence, image/graph, REPL, bench infra)
2. Waves 223 — extend crypto/fem/cfd to 2D
3. Waves 224–225 — sym transforms + infrastructure
4. Waves 226–230 — GUI + REPL bindings
5. Waves 231+ — long-tail API + packaging + fuzz

---

*Original audit sections preserved below for history.*

<details>
<summary>Original audit (pre-Wave 219 snapshot)</summary>

Sixteen of seventeen §2 math modules existed with real implementations. Still absent: `ms::crypto`, `ms::fem`, `ms::cfd`. Build blocked by two `throw` sites under `-fno-exceptions`. GUI plan deferred to separate document.

</details>
