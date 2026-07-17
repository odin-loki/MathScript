# Performance Guide

MathScript completed an eleven-wave profiling iteration (Waves **218–228**) covering hot paths across all `src/` modules. See [`CHANGELOG.md`](../CHANGELOG.md) for per-wave optimizations.

## Modules covered (Waves 218–227)

| Domain | Modules |
|--------|---------|
| Numerical core | `signal`, `fft`, `simd`, `stats`, `linalg`, `poly` |
| Spatial / graph | `graph`, `image`, `tensorops`, `topo`, `geo` |
| Solvers | `ode`, `pde`, `cfd`, `fem` |
| ML / crypto | `ml`, `prob`, `control`, `crypto`, `interp`, `quantum`, `bignum`, `compress` |
| Frameworks | `frameworks`, `distributed`, `finance`, `info`, `combo`, `symbolic` |
| Infra | `bench_*` targets, `build.ps1 -Benchmark`, baseline JSON |

## Benchmark smoke policy

- **27** Google Benchmark executables in `tests/performance/CMakeLists.txt`.
- **Windows:** `.\build.ps1 -Benchmark` builds `build-msvc-bench`, runs every target with `--benchmark_min_time=0.001s`.
- **Linux:** `scripts/bench_smoke.sh` (same targets, 0.001 s min time).
- **Regression gate:** `scripts/bench_regression.sh` compares against stored medians; **10% tolerance** (`MS_BENCH_TOLERANCE`). Set `MS_BENCH_REGRESSION=off` to smoke-only.

## Baselines

| File | Status |
|------|--------|
| `tests/performance/baselines/msvc-release.json` | All medians populated; regenerate via `.\scripts\bench_write_msvc_baseline.ps1`. |
| `tests/performance/baselines/linux-gcc13.json` | Schema placeholders (null medians) for Wave 218+ targets; regenerate on Linux CI: `bash scripts/bench_regression.sh --write-baseline build-bench`. Regression skips null entries. |

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
| Number theory | `numthy` | Certified — no changes |
| Complex / optimization | `cplx`, `optim` | Certified — no changes |
| Differential geometry / legacy | `diffgeo`, `domain` | Certified — no changes |
| CPU kernels | `runtime/cpu`, `cpu/blas` | Certified — no changes |
| Finance / info / combo | `finance`, `info`, `combo` | Wave 227 opts confirmed; **`bench_finance`** added (Wave 228) |
| Benchmark infra | `linux-gcc13.json`, smoke guard | Schema completed; smoke args verified ≤120 s total |

### Modules certified no-change (Wave 228)

These modules were profiled in Wave 228 and required **no code changes** — hot loops already use appropriate reserves, hoisted invariants, or are intentionally small-input paths:

- `numthy` — Pollard rho, GCD, modular exp already reserve and avoid redundant `sqrt`
- `cplx`, `optim` — vector ops and gradient/line-search loops already optimal from Waves 218–226 benches
- `diffgeo`, `domain` — geodesic trajectory reserved in Wave 226; domain helpers are O(n) legacy stubs
- `runtime`, `cpu` — BLAS/dgemm micro-kernels already tuned in Waves 218–220; no safe MSVC micro-opt found

**PROFILING ITERATION DONE.** Do not start Wave 229 profiling.
