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
| `tests/performance/baselines/linux-gcc13.json` | Linux CI medians (`benchmark-linux` on ubuntu-24.04, AVX-512 off) |

Null median entries are skipped by the regression script.

## Intentional complexity

These paths are correct-first, not performance debt:

| Location | Complexity | Why |
|----------|------------|-----|
| `image::dft_magnitude` | O(RC·RC) DFT | Visualisation helper; use `ms::fft` for production sizes |
| `topo::bottleneck_distance` | O(n²) greedy matching | Typical persistence diagrams n < 500 |
| `geo::convex_hull_3d` | O(n³) face enumeration | Small point sets |
| `geo::minkowski_sum_convex` | O(n·m) brute | Fallback for polygons with fewer than 3 vertices |

## dgemm kernels

`ms::cpu::blas::dgemm` dispatches, widest first, to whichever kernel the host can
actually run and the problem is large enough to repay:

| Kernel | Micro-kernel | Registers | Source |
|---|---|---|---|
| AVX-512F | 16×8 | 16 accumulators + 2 A + 1 broadcast, of 32 zmm | `src/runtime/cpu/avx512_dgemm.cpp` |
| AVX2/FMA | 8×6 | 12 accumulators + 2 A + 1 broadcast, of 16 ymm | `src/runtime/cpu/avx2_dgemm.cpp` |
| scalar | rank-1 update through `ms::simd::axpy` | — | `src/runtime/cpu/blas_dgemm.cpp` |

Both vector kernels sit on the packing and cache blocking in
`src/runtime/cpu/gemm_blocking.hpp`, which is the Goto/BLIS decomposition: outer
loops over column, depth and row blocks; A and B packed once into contiguous panels;
the micro-kernel at the bottom reading both sequentially.

### Measured

One host, `Intel(R) Xeon(R) Processor @ 2.10GHz`, n = 512, `-O2`, single-threaded,
same inputs through every path:

| Path | GFLOP/s | vs. baseline |
|---|---|---|
| No vector kernel compiled in (`MS_ENABLE_AVX2=OFF`, `MS_ENABLE_AVX512=OFF`) | 5.03 | — |
| AVX2/FMA, packed and blocked | 36.06 | 7.2× |
| AVX-512, gather and no blocking (before this change) | 12.16 | 2.4× |
| AVX-512, packed and blocked | 66.62 | 13.2× |

The first row is the configuration CI actually built until the AVX2 kernel existed:
`-DMS_ENABLE_AVX512=OFF` and no AVX2 path meant matrix multiply fell to a rank-1
update loop.

The two AVX-512 rows agree to `6.0e-14` on the same inputs, so the 5.5× between
them is a comparison between two implementations of one function rather than
between two functions.

**One machine, one size, one thread.** These are not a benchmark suite and should
not be quoted as a general figure. What they establish is direction and rough
magnitude, which is what was missing: the packing and blocking work was justified
by an argument about cache and gather cost, and an argument is not a measurement.

Three things are worth knowing before reading a number from this.

**`available()` is a runtime question.** A binary built with AVX-512 kernels still
has to run on hosts whose OS never enabled the ZMM register state. Both kernels
answer through `ms::simd::detect_isa()`, which reads `CPUID.1:ECX.OSXSAVE` and then
`XGETBV(0)` rather than trusting the CPUID feature bit alone. Dispatching on the
feature bit is a SIGILL on the kernel's first instruction, decided by the deployment
environment rather than by anything the caller passed in.

**Small problems are declined, on purpose.** Packing allocates two aligned panels,
and below roughly 64×64×64 that costs more than the blocking saves, so the blocked
kernels decline and the caller keeps the simpler path. `worthwhile(m, n, k)` is
public so the decision is inspectable rather than a magic number inside a branch.

**The block sizes are defaults, not measurements.** They are chosen so the packed A
block fits a typical L2 and the packed B block fits L3, and they are a `struct`
rather than constants baked into the loop because the right values are a property of
the machine. Treating them as tuned would be inventing a number.

### Reproducibility

The kernels reassociate the sum over k and use fused multiply-add, so a vectorised
run does not reproduce a scalar one bit for bit. That is ordinary floating-point
behaviour, not a defect, but it means the ISA path is part of the description of a
result. `MS_FORCE_ISA` pins the path — `scalar`, `sse2`, `sse41`, `avx`, `avx2`,
`avx512` — and only ever narrows what was detected, so two machines with different
hardware can be made to agree. `ms::runtime::capture()` records which path was
actually taken, alongside the version, worker count and seed. See "Randomness and the
seeding contract" in [`API.md`](API.md).

The tolerance the kernels are held to, and the differential tests that enforce it,
are described in [`PLAN_STATUS.md`](PLAN_STATUS.md) under §9.

### What is not built

No NEON kernel, so Graviton and Apple silicon fall to the scalar path. No `sgemm`.
The kernels are single-threaded. `src/simd/vector_ops.cpp`, `src/linalg`, `src/fft`
and `src/special` are not vectorised beyond what xsimd gives them today.
