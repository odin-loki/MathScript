# Performance

MathScript ships **28** Google Benchmark executables covering the numerical hot paths. There is no separate benchmark build tree: enable them in the same CMake tree as the library.

## How to run

**Windows** (same `build-msvc` as tests):

```powershell
.\build.ps1 -Benchmark
```

That configures `-DMS_BUILD_BENCHMARKS=ON`, builds all `add_ms_bench` targets, and smokes each with `--benchmark_min_time=0.001s`.

**Linux:**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_BUILD_BENCHMARKS=ON \
  -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build
bash scripts/bench_smoke.sh build
```

Regression vs stored medians (10% tolerance, `MS_BENCH_TOLERANCE`):

```bash
bash scripts/bench_regression.sh build
MS_BENCH_REGRESSION=off bash scripts/bench_regression.sh build   # smoke only
bash scripts/bench_regression.sh --write-baseline build          # refresh Linux JSON
```

Windows baseline refresh: `.\scripts\bench_write_msvc_baseline.ps1`.

## Targets

`bench_matmul`, `bench_fft`, `bench_linalg`, `bench_repl`, `bench_special`, `bench_stats`, `bench_rng_dispatch`, `bench_simd`, `bench_signal_linalg`, `bench_signal_filters`, `bench_ode_pde`, `bench_fem`, `bench_special_memory`, `bench_optim_symbolic`, `bench_frameworks`, `bench_tensorops`, `bench_distributed_cellai`, `bench_poly_domain`, `bench_prob`, `bench_optim_ml`, `bench_crypto`, `bench_graph`, `bench_topo`, `bench_image`, `bench_geo`, `bench_quantum`, `bench_compress`, `bench_finance`.

## Baselines

| File | Use |
|------|--------|
| `tests/performance/baselines/msvc-release.json` | Windows medians |
| `tests/performance/baselines/linux-gcc13.json` | Linux CI medians (`benchmark-linux`) |

Null median entries are skipped by the regression script.

## Intentional complexity

These paths are correct-first, not performance debt:

| Location | Complexity | Why |
|----------|------------|-----|
| `image::dft_magnitude` | O(RC·RC) DFT | Visualisation helper; use `ms::fft` for production sizes |
| `topo::bottleneck_distance` | O(n²) greedy matching | Typical persistence diagrams n < 500 |
| `geo::convex_hull_3d` | O(n³) face enumeration | Small point sets |
| `geo::minkowski_sum_convex` | O(n·m) brute | Fallback for polygons with fewer than 3 vertices |
