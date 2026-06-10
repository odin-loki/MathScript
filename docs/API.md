# MathScript Public API Index

Public headers live under `include/ms/`. Include paths use the `ms/...` prefix (e.g. `#include "ms/core/matrix.hpp"`). For a single umbrella include, use `ms/ms.hpp`.

## Umbrella

| Header | Description |
|--------|-------------|
| `ms/ms.hpp` | Aggregates core, linalg, memory, error, fft, stats, prob, optim, signal, special, simd, and cuda public interfaces |

## Core (`include/ms/core/`)

| Header | Description |
|--------|-------------|
| `core/matrix.hpp` | `Matrix<S, Order, Alloc>` dense matrix type with col/row-major storage |
| `core/tensor.hpp` | Fixed-rank `Tensor<S, N>` multi-dimensional array wrapper |
| `core/sparse.hpp` | COO `Sparse<S>` matrix with `spmv` and dense conversion |
| `core/scalar.hpp` | `Scalar` value with physical units and arithmetic |
| `core/sym.hpp` | Lightweight string-based symbolic `Sym` expression type |
| `core/expr.hpp` | CRTP expression templates for lazy matrix evaluation |
| `core/operations.hpp` | High-level `matmul`, `lu`, `qr`, `solve`, and related `Result<>` wrappers |
| `core/rng.hpp` | Session RNG hooks (`set_session_rng`, `session_uniform`, `session_normal`) |

## Linear Algebra (`include/ms/linalg/`)

| Header | Description |
|--------|-------------|
| `linalg/linalg.hpp` | Eig/SVD/LDL/Schur result types; `rand`, `eig_sym`, `svd`, `expm`, `trace`, `det`, `norm`, decompositions |
| `linalg/matrix_operations.hpp` | `expected`-based matmul, LU, QR, solve, and matrix utilities |
| `linalg/transpose.hpp` | `transpose()` returning row-major aligned matrix |

## CPU BLAS/LAPACK (`include/ms/cpu/`)

| Header | Description |
|--------|-------------|
| `cpu/blas.hpp` | Column-major BLAS: `dgemm`, `dsyrk`, `dtrsm`, `dger` |
| `cpu/lapack.hpp` | LAPACK-style factorizations and solvers: Cholesky, LU, QR, SVD, least squares, symmetric eigen |
| `cpu/blas_kernel.hpp` | AVX-512 micro-kernels for internal BLAS paths |

## Runtime (`include/ms/runtime/`)

| Header | Description |
|--------|-------------|
| `runtime/dispatch.hpp` | `ExecPolicy`, `Backend`, and `DispatchDecision` for CPU/GPU routing |
| `runtime/topology.hpp` | `SystemTopology` CPU/GPU/NUMA discovery |
| `runtime/thread_pool.hpp` | Configurable worker thread pool with futures |
| `runtime/load_balancer.hpp` | `balance()` picks backend, device, and thread count for a workload |

## Memory (`include/ms/memory/`)

| Header | Description |
|--------|-------------|
| `memory/policy.hpp` | `PolicyFlag` alignment/pinning/NUMA/pool allocation flags |
| `memory/aligned_allocator.hpp` | Cross-platform aligned `allocate`/`deallocate` helpers |
| `memory/pinned_allocator.hpp` | Host pinned memory allocator (CPU builds) |
| `memory/arena.hpp` | Per-thread monotonic PMR arena allocator |
| `memory/pool_allocator.hpp` | Slab-based pool for small fixed-size blocks |
| `memory/numa_allocator.hpp` | NUMA topology-aware local allocation |

## Error (`include/ms/error/`)

| Header | Description |
|--------|-------------|
| `error/error_types.hpp` | `DimensionMismatch`, `SingularMatrix`, and other `std::expected` error variants |
| `error/expected.hpp` | Re-exports error types; `Result<T>` alias for `std::expected<T, Error>` |

## Math Modules

| Header | Description |
|--------|-------------|
| `fft/fft.hpp` | `fft`, `ifft`, `rfft`, `dft`, `dct2`/`dst2`, and shift helpers |
| `stats/stats.hpp` | Mean, variance, percentiles, t/z tests, correlation, linear regression |
| `prob/prob.hpp` | PDF/CDF/PPF for normal, exponential, binomial, Poisson, chi-square, t, gamma, uniform |
| `optim/optim.hpp` | `gradient_descent`, `newton_raphson`, `broyden`, `golden_section`, `newton_1d`, `simplex_solver`, `minimize_with_constraints` |
| `signal/signal.hpp` | Butterworth/low/high/band-pass filters, convolution, moving average, window functions |
| `special/special.hpp` | Broad special-function catalog: gamma, Bessel, elliptic, hypergeometric, zeta, Painlevé, etc. |
| `ode/ode.hpp` | `ode_euler` and `ode_rk4` for scalar ODEs `dy/dt = f(t, y)` |
| `pde/pde.hpp` | `pde_heat_1d` explicit heat-equation solver |
| `poly/poly.hpp` | Polynomial eval, add/sub/mul, derivative on coefficient vectors |
| `symbolic/symbolic.hpp` | AST `SymExpr` with simplify, differentiate, and evaluate |
| `domain/domain.hpp` | `factorial`, `nchoosek`, `gcd`, and `Graph` edge counting |

## SIMD (`include/ms/simd/`)

| Header | Description |
|--------|-------------|
| `simd/isa.hpp` | `IsaFeatures` detection and human-readable `isa_summary` |
| `simd/simd.hpp` | Vectorized `add`, `mul`, `dot`, `axpy`, `exp_map`, `gemv` with kernel dispatch |

## CUDA (`include/ms/cuda/`) — optional when `MS_ENABLE_CUDA=ON`

| Header | Description |
|--------|-------------|
| `cuda/buffer.hpp` | `DeviceBuffer`, `PinnedBuffer`, and `StreamPool` GPU memory management |
| `cuda/blas.hpp` | GPU `matmul` for dense `Matrix<double>` |
| `cuda/fft.hpp` | GPU `fft` / `ifft` |
| `cuda/solver.hpp` | GPU `lu` and `solve` |
| `cuda/elementwise.hpp` | In-place `add_inplace` and `fill` on device spans |
| `cuda/sparse.hpp` | COO sparse matrix-vector multiply on GPU |
| `cuda/nvml.hpp` | NVML `DeviceStats` and utilization queries |
| `cuda/nccl.hpp` | NCCL availability and communicator size helpers |

## Profile / Compliance (`include/ms/unsafe/`)

| Header | Description |
|--------|-------------|
| `unsafe/unsafe.hpp` | `MS_UNSAFE(reason)` attribute macro for justified unsafe sites; full enforcement when built with `MS_BUILD_PLUGIN` |

## Interpreter (`include/ms/interp/`)

| Header | Description |
|--------|-------------|
| `interp/repl_engine.hpp` | `Interpreter` REPL session, matrix/scalar state, plot series, save/load |
| `interp/plot_console.hpp` | ASCII plot previews for console REPL (`format_plot_preview`) |
| `interp/jit_backend.hpp` | `JitBackend` abstraction (`Repl` / `OrcJit`); `create_backend()` factory; optional `-DMS_BUILD_JIT=ON` links LLVM ORC LLJIT |

### REPL scalar expressions

Assignments of the form `name = <expr>` support:

- Literals and scalar variables (`x = 3.14`, `y = x`)
- Binary operators with standard precedence: `+`, `-`, `*`, `/` (e.g. `1 + 2 * 3` → 7)
- Parentheses: `(1 + 2) * 3` → 9
- Unary libm calls: `sin`, `cos`, `sqrt`, `exp`, `log`, …
- Two-argument libm calls: `pow(x, 2)`, `min(a, b)`, `max(a, b)`, `atan2(y, x)`

Plot commands: `plot`, `scatter`, `hist`, `imshow`, `spy`, `surf`; `show` redisplays ASCII preview; `saveplot <file>` writes ASCII preview to disk (GUI **Export Plot as PNG** when `MS_BUILD_GUI=ON`). CLI: `mathscript-repl -e`, `--load`, `--jit`. Matrix assignment: `C = matmul(A, B)`, `x = solve(A, b)`, `T = transpose(A)`, `L = chol(A)`. Multi-target: `L, U, P = lu(A)`, `Q, R = qr(A)`, `U, S, V = svd(A)`, `D, V = eig_sym(A)`. Scalar from matrix: `d = det(A)`, etc. Session `save`/`load` persists scalars, matrices, and plot state.

## Distributed (`include/ms/distributed/`) — optional when `MS_ENABLE_MPI=ON`

| Header | Description |
|--------|-------------|
| `distributed/mpi_context.hpp` | `MPIContext` init/finalize, rank/size, `allreduce_sum`, `barrier` |
| `distributed/dist_matrix.hpp` | `DistMatrix` local shards; `scatter`, `gather`, `combine_gather` |
| `distributed/block.hpp` | Block and block-cyclic row partitioning helpers |
| `distributed/linalg.hpp` | Distributed wrappers for `eig_sym`, `svd`, `lu` via gather-on-root |
| `distributed/solve.hpp` | Distributed linear solve `A x = b` |

## Frameworks (`include/ms/frameworks/`)

| Header | Description |
|--------|-------------|
| `frameworks/axiom/axiom.hpp` | Evolutionary `Axiom` algorithm search with `PrimitiveRegistry` |
| `frameworks/cellai/cellai.hpp` | `CellMemory` temporal memory and Hebbian update helpers |
| `frameworks/cypha/cypha.hpp` | `DifModel` mixture-of-experts with NIG uncertainty |
| `frameworks/gria/gria.hpp` | Information-theoretic `alpha`, GF(2^n), cellular automata, LFSR utilities |
| `frameworks/izaac/izaac.hpp` | VRF-based `CSPRNG`, session seeding, and Monte Carlo helpers |
