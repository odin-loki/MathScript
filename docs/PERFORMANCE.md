# Performance Guide

MathScript completed a ten-wave profiling iteration (Waves **218–227**) covering hot paths across all major modules. See [`CHANGELOG.md`](../CHANGELOG.md) for per-wave optimizations.

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

No further profiling waves are planned.
