# Performance Guide

MathScript completed a twelve-wave profiling iteration (Waves **218–230**) covering hot paths across all `src/` modules, benchmark infra, and baseline refresh. See [`CHANGELOG.md`](../CHANGELOG.md) for per-wave optimizations.

## Modules covered (Waves 218–228)

| Domain | Modules |
|--------|---------|
| Numerical core | `signal`, `fft`, `simd`, `stats`, `linalg`, `poly`, `numthy`, `cplx`, `optim` |
| Spatial / graph | `graph`, `image`, `tensorops`, `topo`, `geo`, `diffgeo`, `domain` |
| Solvers | `ode`, `pde`, `cfd`, `fem` |
| ML / crypto | `ml`, `prob`, `control`, `crypto`, `interp`, `quantum`, `bignum`, `compress` |
| Frameworks | `frameworks`, `distributed`, `finance`, `info`, `combo`, `symbolic` |
| CPU runtime | `runtime/cpu`, `cpu/blas` |
| Infra | `bench_*` targets, `build.ps1 -Benchmark`, `scripts/bench_cmake_targets.sh`, baseline JSON |

## Benchmark smoke policy

- **28** Google Benchmark executables in `tests/performance/CMakeLists.txt`.
- **Windows:** `.\build.ps1 -Benchmark` builds `build-msvc-bench`, runs every target with `--benchmark_min_time=0.001s` (~15 s total; per-bench timing logged).
- **Linux:** `scripts/bench_smoke.sh` (same targets, 0.001 s min time).
- **Regression gate:** `scripts/bench_regression.sh` compares against stored medians; **10% tolerance** (`MS_BENCH_TOLERANCE`). Set `MS_BENCH_REGRESSION=off` to smoke-only.

## Baselines

| File | Status |
|------|--------|
| `tests/performance/baselines/msvc-release.json` | All medians populated; regenerate via `.\scripts\bench_write_msvc_baseline.ps1`. |
| `tests/performance/baselines/linux-gcc13.json` | Schema placeholders (null medians) for Wave 218+ targets until refreshed; regenerate via **`.github/workflows/bench-baseline-linux.yml`** (`workflow_dispatch`) or locally: `bash scripts/bench_regression.sh --write-baseline build-bench`. Regression skips null entries. |

## Intentional remaining complexity (not perf debt)

These paths are deliberately sub-optimal for correctness, API simplicity, or small-input sizes:

| Location | Complexity | Rationale |
|----------|------------|-----------|
| `image::dft_magnitude` | O(RC·RC) naive DFT | Reference/visualization helper; use `ms::fft` for production sizes. |
| `topo::bottleneck_distance` | O(n²) greedy matching | Exact Hungarian is O(n³); typical persistence diagrams n < 500. |
| `geo::convex_hull_3d` | O(n³) face enumeration | Small point sets; research/education path. |
| `geo::minkowski_sum_convex` | O(n·m) brute fallback | Used when either polygon has fewer than 3 vertices. |

## Profiling certification (Wave 228)

Wave **228** is a **certification pass**, not a new optimization sweep. Eight parallel subagents re-audited every remaining `src/` module not fully signed off in Waves 218–227, plus benchmark infra gaps (finance bench coverage, Linux baseline schema, smoke timing guard).

**Scope:** all **35** library directories under `src/` (excluding `exe`/`gui`/`plugin` runtime stubs) are now audited across Waves **218–228**. Waves 218–227 applied hot-path opts; Wave 228 confirms closure on gaps.

### Wave 228 audit targets

| Area | Modules / files | Outcome |
|------|-----------------|---------|
| Number theory | `numthy` | `isqrt_u64` reuse, `factors.reserve(32)` |
| Complex / optimization | `cplx`, `optim` | Hoisted constants, reused grad/line-search buffers |
| Differential geometry / legacy | `diffgeo`, `domain` | Buffer reuse in geodesic/parallel transport; **domain certified no-change** |
| CPU kernels | `runtime/cpu`, `cpu/blas` | DGEMM micro-kernel opts (~31% matmul/512 on MSVC rank-1 path) |
| Finance / info / combo | `finance`, `info`, `combo` | **`bench_finance`** added (`BM_BlackScholes`, `BM_MCEuropean`, `BM_Entropy`) |
| Benchmark infra | `linux-gcc13.json`, smoke guard | Full schema (434 keys); simd/poly args capped; per-bench smoke timing in `build.ps1` |

### Module certified no-change (Wave 228)

- **`domain`** — helpers already O(n) or minimal; no opts warranted.

**PROFILING ITERATION DONE (Waves 218–228).** No further code-profiling waves.

## Profiling infra closure (Wave 229)

Wave **229** closed remaining **benchmark infrastructure** gaps — not a code-profiling sweep.

| Fix | Detail |
|-----|--------|
| **Linux CI build** | `benchmark-linux` now runs `cmake --build build-bench` (all **28** `add_ms_bench` targets); stale 20-target subset removed. Job timeout raised to **30 min**. |
| **Target discovery** | `scripts/bench_cmake_targets.sh` shares CMakeLists parsing with `bench_smoke.sh` / `build.ps1 -Benchmark`. |
| **Regression gate** | `bench_regression.sh` still compares against `linux-gcc13.json`; null medians skipped. |

**PROFILING + INFRA DONE (Wave 229).** Linux median refresh was the remaining baseline-path gap — closed in Wave 230 below.

## Baseline refresh workflow (Wave 230)

Wave **230** completes the **baseline path** for the profiling iteration — not a code-profiling sweep.

| Path | Refresh |
|------|---------|
| **Linux** | **`.github/workflows/bench-baseline-linux.yml`** — `workflow_dispatch` only: ubuntu-24.04, gcc-13, Release, all **28** `add_ms_bench` targets; runs `bash scripts/bench_regression.sh --write-baseline build-bench`; uploads `linux-gcc13.json` as artifact (**60 min** timeout). Maintainer downloads, reviews, and commits — no auto-commit to the repo. Trigger: GitHub Actions UI or `gh workflow run bench-baseline-linux.yml`. |
| **Windows** | `.\scripts\bench_write_msvc_baseline.ps1` — local refresh; `msvc-release.json` medians populated. |

Baseline refresh is **not** part of the default PR gate (`benchmark-linux` still runs regression compare only).

**PROFILING ITERATION FULLY COMPLETE (Waves 218–230).** Code profiling (218–228), benchmark infra (229), baseline refresh path (230). No further profiling waves.

## Wave 231 benchmark coverage (feature wave — not profiling)

Wave **231** added Google Benchmark cases for new crypto/FEM/CFD 2D hot paths in existing **`bench_crypto`**, **`bench_fem`**, and **`bench_ode_pde`** executables (**28** targets unchanged). Profiling iteration status unchanged — **FULLY COMPLETE (Waves 218–230).**

## Final sign-off

**Product profiling and adjustment is complete.** Twelve waves (218–230) closed code profiling, benchmark infrastructure, and baseline refresh; Wave 231 (+231b) added seven benchmark cases for new crypto/FEM/CFD paths without reopening the iteration. No further profiling or product-adjustment waves are planned — Wave 232+ is feature development only.
