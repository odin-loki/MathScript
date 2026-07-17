# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-17 (reconciled against `main` @ Wave 219 — **370 CTest suites**, MSVC green, `.\build.ps1 -Benchmark` smoke OK)

This document tracks open items from `mathscript-master-plan.md` and the original audit below. Work proceeds in **waves** of up to 8 parallel Composer 2.5 subagents (isolated git worktrees).

---

## Status vs original audit (2026-07-17)

| Original claim | Current reality |
|---|---|
| Wave 141 / 342 tests | **Wave 219 / 370 tests** — full MSVC suite passing |
| `ms::sym` still 13 functions | **Expanded:** sub/div/neg/tan/sqrt, expand, collect, integrate, substitute, sym_parse |
| UNSAFE `approved_sites: 6` | **40** sites documented in `UNSAFE_REVIEW.md` |
| Build broken on Linux `-fno-exceptions` | **Still open** — two `throw` sites remain (see Wave 220) |
| `ms::crypto` / `ms::fem` / `ms::cfd` absent | **Still absent** — master plan routes crypto via Izaac; FEM/CFD descoped but listed here for completeness |
| GUI incomplete | **Still incomplete** — REPL runs on UI thread |
| `vendor/` empty | **Still empty** — FetchContent still used for GTest/benchmark |

---

## P0 — Blocking (Wave 220)

- [ ] **Remove `throw` from `matrix.hpp:46` and `numa_allocator.hpp:58`** — required for Linux/GCC `-fno-exceptions` builds
- [ ] **Verify Linux CI green** after exception fix
- [ ] **Vendor GoogleTest** into `vendor/` (offline-build policy §13.8)
- [ ] **Regenerate `UNSAFE_REVIEW.md`** baseline via `scripts/unsafe_report.sh`

---

## P1 — Missing §2 modules (Waves 221–223)

- [ ] **`ms::crypto` MVP** — sha256/sha512, hmac, aes256_cbc, chacha20 (wrap or complement Izaac)
- [ ] **`ms::fem` MVP** — 1D/2D Poisson: mesh1d, lagrange P1, assemble_stiffness, apply_dirichlet, solve_fem
- [ ] **`ms::cfd` MVP** — 1D advection FVM or coupled heat; document limitations

---

## P2 — Symbolic CAS depth (Waves 224–225)

- [ ] `sym_limit`, `sym_series` (Taylor), `sym_solve` (linear), `sym_dsolve` (separable ODEs)
- [ ] `sym_laplace_transform`, `sym_fourier_transform` (table lookup + convolution rule)
- [ ] Boolean/set algebra, codegen (long-term)

---

## P3 — Infrastructure stubs (Waves 226–227)

- [ ] **CUDA** — real `cuda::lu` or document+test stub boundary; device buffer not `"cpu-stub"`
- [ ] **MPI** — non-stub `MpiContext` when `mpiexec -n > 1`; gate multi-rank tests
- [ ] **`MS_BUILD_PLUGIN=ON` in CI** — Clang profile plugin smoke

---

## P4 — GUI (Waves 228–230)

- [ ] Phase 1: REPL/script eval on worker `QThread` (non-blocking Run)
- [ ] Phase 2: Syntax highlighting in editor
- [ ] Phase 3: Multi-plot manager, heatmap thumbnails, live hardware bar, `QSettings` persistence

---

## P5 — Master-plan API gaps (Waves 231+)

- [ ] REPL bindings for post-Wave-187 functions (ml/image/tensorops/stateful)
- [ ] `ms::signal`: ellip/bessel/remez, periodogram, EMD/VMD
- [ ] `ms::geo`: `poly_diff`, robust non-convex booleans
- [ ] `ms::graph`: planarity, Blossom matching
- [ ] `ms::finance`: SABR put, remaining vol models
- [ ] `ms::bignum`: APFloat/APComplex (large rewrite)
- [ ] 24h fuzz marathon (`fuzz-24h.yml` manual)
- [ ] Windows installer / Linux packages
- [ ] Full ORC JIT v2 matrix LLVM IR

---

## P6 — Non-code

- [ ] **Trademark:** USPTO clearance for "MathScript" vs NI LabVIEW MathScript — decide alternate public name before procurement/GitHub polish

---

## Wave schedule (Composer 2.5 subagents)

| Wave | Focus | Agents |
|------|--------|--------|
| **220** | P0: exceptions, vendor gtest, unsafe audit, crypto sha MVP start | 8 |
| **221** | crypto finish + fem 1D Poisson | 8 |
| **222** | cfd 1D + sym limit/series/solve | 8 |
| **223** | sym transforms + CUDA buffer/LU | 8 |
| **224** | MPI real + plugin CI | 8 |
| **225** | GUI thread + syntax highlight | 8 |
| **226–230** | GUI polish + REPL binding batches | 8×5 |
| **231+** | Remaining API gaps until master-plan §2 closed | ongoing |

---

## Suggested immediate order

1. Wave 220 (this session) — unblock Linux + supply chain + crypto/fem seeds  
2. Waves 221–223 — complete missing modules  
3. Waves 224–225 — sym CAS parity with other modules  
4. Waves 226–227 — CUDA/MPI/plugin  
5. Waves 228–230 — GUI  
6. Waves 231+ — REPL + long-tail API + packaging + fuzz marathon  

---

*Original audit sections preserved below for history.*

<details>
<summary>Original audit (pre-Wave 219 snapshot)</summary>

Sixteen of seventeen §2 math modules existed with real implementations. Still absent: `ms::crypto`, `ms::fem`, `ms::cfd`. Build blocked by two `throw` sites under `-fno-exceptions`. GUI plan deferred to separate document.

</details>
