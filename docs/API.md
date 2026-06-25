# MathScript Public API Index

Public headers live under `include/ms/`. Include paths use the `ms/...` prefix (e.g. `#include "ms/core/matrix.hpp"`). For a single umbrella include, use `ms/ms.hpp`.

## Umbrella

| Header | Description |
|--------|-------------|
| `ms/ms.hpp` | Aggregates core, linalg, memory, error, fft, stats, prob, optim, signal, special, simd, cuda, distributed, frameworks, and Waves 57–60 modules (numthy through bignum) |

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

## Number Theory — Wave 57 (`include/ms/numthy/`)

| Header | Description |
|--------|-------------|
| `numthy/numthy.hpp` | Primality (`isprime`, `primes`, `prime_pi`), factorisation (`factor`, `factor_exp`), divisor functions (`divisors`, `euler_phi`, `mobius`), modular arithmetic (`mod_inv`, `mod_pow`, `crt`), Legendre/Jacobi/Kronecker symbols, discrete log, Tonelli–Shanks, continued fractions, `partition` |

## Combinatorics — Wave 57 (`include/ms/combo/`)

| Header | Description |
|--------|-------------|
| `combo/combo.hpp` | Factorials and derangements, binomial/multinomial/permutations, `next_perm`/`next_comb`, rank/unrank, enumeration (`all_permutations`, `all_subsets`, `all_partitions`), Stirling/Bell/Catalan/Motzkin numbers |

## Information Theory — Wave 57 (`include/ms/info/`)

| Header | Description |
|--------|-------------|
| `info/info.hpp` | Shannon/joint/conditional entropy, mutual information, cross-entropy, KL/JS/Rényi/Tsallis divergence, Hellinger and total-variation distance, channel capacity, rate–distortion, LZ complexity, sample entropy |

## Finance — Wave 57 (`include/ms/finance/`)

| Header | Description |
|--------|-------------|
| `finance/finance.hpp` | Black–Scholes call/put and Greeks, implied vol, bond price/duration/convexity/YTM, NPV/IRR/PV/FV annuities, VaR/CVaR, Sharpe/Sortino, max drawdown, Kelly criterion, binomial option pricing, portfolio variance/return |

## Control — Wave 58 (`include/ms/control/`)

| Header | Description |
|--------|-------------|
| `control/control.hpp` | Transfer function and state-space models (`tf`, `ss`, `tf2ss`, `ss2tf`), interconnections, poles/zeros/stability, Bode/Nyquist/margins, step/impulse response, Lyapunov/Riccati (`lyap`, `riccati`, `dare`), LQR, controllability/observability, pole placement, PID tuning |

## Graph — Wave 58 (`include/ms/graph/`)

| Header | Description |
|--------|-------------|
| `graph/graph.hpp` | `Graph` adjacency list, BFS/DFS/topological sort, Dijkstra/Bellman-Ford/Floyd-Warshall/A*, connectivity and SCC, MST (Kruskal/Prim), PageRank/betweenness/closeness centrality, max flow, bipartite matching, greedy colouring |

## Complex Analysis — Wave 58 (`include/ms/cplx/`)

| Header | Description |
|--------|-------------|
| `cplx/cplx.hpp` | Residues, winding number, contour/Cauchy integrals, argument principle, Möbius transforms, Joukowski map, Poisson kernel, harmonic conjugate, hyperbolic distance, Laurent coefficients, Blaschke products |

## Quantum — Wave 58 (`include/ms/quantum/`)

| Header | Description |
|--------|-------------|
| `quantum/quantum.hpp` | Kets and density matrices, Pauli and standard gates, tensor products, QFT, Bell/GHZ/W/coherent/Fock states, von Neumann entropy, fidelity, partial trace, entanglement entropy, Schrödinger time evolution |

## Computational Geometry — Wave 59 (`include/ms/geo/`)

| Header | Description |
|--------|-------------|
| `geo/geo.hpp` | 2D/3D points, segments, rays, polygons; convex hull, Delaunay/Voronoi; `KDTree2D`/`KDTree3D` nearest/range queries; ray–triangle/sphere/AABB intersections; Bezier/B-spline/Hermite curves; area, centroid, moments |

## Differential Geometry — Wave 59 (`include/ms/diffgeo/`)

| Header | Description |
|--------|-------------|
| `diffgeo/diffgeo.hpp` | Metric tensor and inverse, Christoffel symbols, Riemann/Ricci/Einstein tensors, geodesics, surface first/second fundamental forms, Gaussian/mean/principal curvature, Lie bracket |

## Topology / TDA — Wave 59 (`include/ms/topo/`)

| Header | Description |
|--------|-------------|
| `topo/topo.hpp` | `SimplicialComplex`, Vietoris–Rips filtration, persistent homology diagrams, Betti numbers and Euler characteristic, bottleneck/Wasserstein distances between diagrams, Betti curves |

## Tensor Operations — Wave 59 (`include/ms/tensorops/`)

| Header | Description |
|--------|-------------|
| `tensorops/tensorops.hpp` | Row-major `Tensor`, contractions, `einsum`, mode products, Khatri-Rao/Kronecker, symmetrise/antisymmetrise, CP and Tucker/HOSVD decompositions, Frobenius norm |

## Machine Learning — Wave 60 (`include/ms/ml/`)

| Header | Description |
|--------|-------------|
| `ml/ml.hpp` | Linear/ridge/lasso/logistic regression, KNN, naive Bayes, decision trees; KMeans, DBSCAN, agglomerative clustering; PCA, t-SNE; autodiff, feedforward nets; loss/metrics, scalers, cross-validation |

## Image Processing — Wave 60 (`include/ms/image/`)

| Header | Description |
|--------|-------------|
| `image/image.hpp` | `Image` buffer, RGB/HSV conversion, resize/crop/flip/rotate, Gaussian/median/bilateral filters, Sobel/Canny/Laplacian edges, morphology, Otsu/adaptive threshold, histogram equalisation, Harris corners, connected components |

## Compression — Wave 60 (`include/ms/compress/`)

| Header | Description |
|--------|-------------|
| `compress/compress.hpp` | RLE, Huffman, LZ77, LZW, BWT/MTF, delta coding, bzip2-like pipeline, bit-pack helpers |

## Bignum — Wave 60 (`include/ms/bignum/`)

| Header | Description |
|--------|-------------|
| `bignum/bignum.hpp` | `BigInt` and `Rational` arbitrary-precision arithmetic, `bigint_gcd`/`lcm`/`pow`/`pow_mod`, factorial, Fibonacci, Miller–Rabin primality |

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

Plot commands: `plot`, `scatter`, `hist`, `imshow`, `spy`, `surf`; `show` redisplays ASCII preview; `saveplot <file>` writes ASCII preview to disk (GUI **Export Plot as PNG** when `MS_BUILD_GUI=ON`). CLI: `mathscriptc` script runner (executes .ms files as REPL command sequences); `mathscript-repl -e`, `--load`, `--jit`. Matrix assignment: `C = matmul(A, B)`, `x = solve(A, b)`, `T = transpose(A)`, `L = chol(A)`. Multi-target: `L, U, P = lu(A)`, `Q, R = qr(A)`, `U, S, V = svd(A)`, `D, V = eig_sym(A)`. Scalar from matrix: `d = det(A)`, etc. Session `save`/`load` persists scalars, matrices, and plot state.

### REPL bindings — Waves 61–62

C++ library modules from Waves 57–59 are header-only API; Waves 61–62 extend the REPL with matrix/scalar call bindings (no Wave 57–59 REPL surface yet).

**Wave 61 — linalg (matrix assignment):**

| Call | Description |
|------|-------------|
| `pinv(A)` | Moore–Penrose pseudo-inverse |
| `null(A)` | Null-space basis (wide-matrix safe via `A^T A` eigenvectors) |
| `orth(A)` | Orthonormal column basis (QR) |
| `kron(A, B)` | Kronecker product; nested operands supported (e.g. `kron(eye(2), eye(2))`) |
| `repmat(A, p, q)` | Tile matrix `p×q` |
| `linspace(a, b, n)` | Equally spaced `n×1` column vector |

**Wave 62 — image, compress, ML, bignum (matrix/scalar assignment):**

| Call | Description |
|------|-------------|
| `rgb2gray(M)` | `(H·W)×3` RGB rows → grayscale column vector |
| `sobel(M)`, `imgaussfilt(M, σ)`, `laplacian(M)`, `histeq(M)`, `sharpen(M)`, `threshold_otsu(M)`, `imresize(M, r, c)` | Grayscale image ops on `H×W` matrices |
| `rle_encode_vec(M)`, `rle_decode_vec(M)` | RLE on flattened matrix bytes |
| `delta_encode_vec(M)`, `delta_decode_vec(M)` | Delta coding on flattened bytes |
| `ml_accuracy(p, t)`, `ml_rmse(p, t)`, `ml_mse(p, t)`, `ml_r2(p, t)`, `ml_f1(p, t)`, `ml_precision(p, t)`, `ml_recall(p, t)`, `ml_mae(p, t)` | ML metrics on matching `N×1` vectors |
| `bigint("495")`, `bigint_factorial(n)`, `bigint_fib(n)`, `bigint_gcd("a", "b")` | Bignum parse/ops; results as scalars when representable in `double` |

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
