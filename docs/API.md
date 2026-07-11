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
| `core/sym.hpp` | String-based `Sym` with parse/eval: `+ - * /`, unary `-`, parentheses, literals, variables (`x0`, `x1`, …), and `sin`/`cos`/`exp`/`log`/`sqrt`/`tanh`; unbound variables default to 0 |
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
| `stats/stats.hpp` | Mean, variance, percentiles, t/z tests, correlation, linear regression; one-way ANOVA (`one_way_anova`), Mann-Whitney U (`mann_whitney_u`), two-sample KS (`ks_test_2sample`); bootstrap resampling (`bootstrap_mean`, mean-only `bootstrap_ci`, general `bootstrap_ci` with percentile-method CI for arbitrary statistics via `BootstrapResult`) |
| `prob/prob.hpp` | PDF/CDF/PPF for normal, exponential, binomial, Poisson, chi-square, t, gamma, uniform |
| `optim/optim.hpp` | `gradient_descent`, `newton_raphson`, `broyden`, `golden_section`, `newton_1d`, `simplex_solver`, `minimize_with_constraints` |
| `signal/signal.hpp` | Butterworth/low/high/band-pass filters, convolution, moving average, window functions |
| `special/special.hpp` | Broad special-function catalog: gamma, Bessel, elliptic, hypergeometric, Painlevé, etc.; DLMF additions: `zeta`, `zeta_hurwitz`, `eta_dirichlet`, `beta_dirichlet`, `polylog`, `clausen`, `debye` |
| `ode/ode.hpp` | Scalar/vector IVP solvers: `ode_euler`, `ode_rk4`, `ode_midpoint`, `ode_rk2`, `ode_rk45`, `ode_rk23`, `ode_adams_bashforth2`, `ode_euler_vec`, `ode_rk4_vec`, `ode_rk45_vec`; symplectic: `ode_verlet`, `ode_verlet_vec`; stiff/implicit: `ode_backward_euler`, `ode_backward_euler_vec`, `ode_bdf2` (BDF2 multistep, A-stable, bootstraps first step via BDF1); BVP: `ode_bvp_shooting`; DDE: `ode_dde_fixed_step`; events: `ode_event_detect`; DAE: `ode_dae_index1` |
| `pde/pde.hpp` | `pde_heat_1d`, `pde_heat_1d_cn`, `pde_heat_2d`, `pde_heat_2d_cn_adi` (Peaceman-Rachford ADI Crank-Nicolson, unconditionally stable), `pde_wave_1d`, `pde_wave_2d`, `pde_advection_1d`, `pde_poisson_1d`, `pde_poisson_2d`, `pde_burgers_1d`, `pde_reaction_diffusion_1d` (Fisher-KPP, zero-flux Neumann BC) with CFL/stability guards |
| `poly/poly.hpp` | Polynomial eval, add/sub/mul, derivative on coefficient vectors |
| `symbolic/symbolic.hpp` | AST `SymExpr` with `sym_add`/`sym_sub`/`sym_mul`/`sym_div`/`sym_neg`, `sym_sin`/`sym_cos`/`sym_tan`/`sym_exp`/`sym_log`/`sym_sqrt`/`sym_pow`, `sym_deriv`/`sym_diff`, `sym_simplify`, `sym_integrate`, `sym_substitute`, `sym_eval`, `sym_to_string`, `sym_parse` (recursive-descent text→`SymExpr` parser for `^` `*` `/` `+` `-`, unary minus, parens, functions, variables, literals) |
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
| `ml/ml.hpp` | Linear/ridge/lasso/logistic regression, KNN, naive Bayes, decision trees; `SVM` (binary classifier via SMO with linear/RBF kernels); `RandomForest` (bootstrap-aggregated ensemble with feature subsampling) and `GradientBoosting` (functional-gradient-boosted shallow trees for regression); KMeans, DBSCAN, agglomerative clustering; PCA, t-SNE; autodiff, feedforward nets; loss/metrics, scalers, cross-validation |

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

**Wave 178 — symbolic (string assignment):**

| Call | Description |
|------|-------------|
| `sym_diff("expr", "var")` | Symbolic differentiation of parsed expression w.r.t. variable; returns simplified string |
| `sym_simplify("expr")` | Constant-fold and algebraic simplification of parsed expression |
| `sym_integrate("expr", "var")` | Symbolic integration (power rule, linearity, sin/cos; unsupported forms documented) |
| `sym_eval("expr", "var=value")` | Numeric evaluation of parsed expression with one variable binding |

**Wave 179 — ODE formula-string bindings (scalar IVP):**

| Call | Description |
|------|-------------|
| `ode_euler("expr", t0, y0, t_end, steps)` | Forward Euler; `expr` parsed once via `sym_parse`, evaluated per step with `{t, y}` bindings |
| `ode_rk4("expr", t0, y0, t_end, steps)` | Classical RK4 with parsed formula RHS |
| `ode_midpoint("expr", t0, y0, t_end, steps)` | Explicit midpoint (RK2) with parsed formula RHS |
| `ode_rk45("expr", t0, y0, t_end, rtol, atol)` | Adaptive Dormand-Prince RK45 with parsed formula RHS |
| `ode_backward_euler("expr", t0, y0, t_end, steps)` | Implicit backward Euler (Newton iteration) with parsed formula RHS |

**Wave 180 — extended ODE formula-string bindings:**

| Call | Description |
|------|-------------|
| `ode_bdf2("expr", t0, y0, t_end, steps)` | A-stable BDF2 multistep stiff scalar solver; first step bootstrapped via BDF1; parsed formula RHS |
| `ode_verlet("accel_expr", t0, q0, v0, t_end, steps)` | Velocity Verlet for second-order q''=a(t,q); `accel_expr` evaluated with `{t, q}` bindings |
| `ode_verlet_vec("a0; a1; ...", t0, q0_vec, v0_vec, t_end, steps)` | Vector velocity Verlet; semicolon-separated acceleration formulas with `{t, q0..qN-1}` env |
| `ode_euler_vec("f0; f1; ...", t0, y0_vec, t_end, steps)` | Forward Euler on vector systems; per-component formulas with shared `{t, y0..yN-1}` env |
| `ode_rk4_vec("f0; f1; ...", t0, y0_vec, t_end, steps)` | Classical RK4 on vector systems with parsed per-component formulas |
| `ode_rk45_vec("f0; f1; ...", t0, y0_vec, t_end, rtol, atol)` | Adaptive Dormand-Prince RK45 on vector systems with parsed per-component formulas |

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
| `frameworks/axiom/axiom.hpp` | Evolutionary `Axiom` GP search; `Algorithm` holds `Sym` representation; `evaluate` genuinely evaluates `algo.representation` row-by-row against input data columns (`x0`, `x1`, …) via `Sym::eval` |
| `frameworks/cellai/cellai.hpp` | `CellMemory` temporal memory, `hebbian_update`, `energy` (Boltzmann -vᵀWh), `CellMemory::consolidate` short→long-term decay, `cell_to_cypha_features` |
| `frameworks/cypha/cypha.hpp` | `DifModel` mixture-of-experts with NIG uncertainty; `nig_fit`/`nig_pdf`/`nig_cdf`/`nig_sample`, `predict`, `predict_interval`, `ood_score`, `gh_gate` |
| `frameworks/gria/gria.hpp` | Information-theoretic `alpha`, GF(2^n), cellular automata, LFSR utilities |
| `frameworks/izaac/izaac.hpp` | VRF `CSPRNG`, session seeding, `rand_matrix`/`randn_matrix`, `mc::estimate_pi`; `bloom::BloomFilter`, `ratelimit::TokenBucket`, `diffpriv::laplace_mechanism`/`gaussian_mechanism`, `backtest::simulate_gbm_path`/`run_backtest`; `crypto::encrypt`/`decrypt` (`CipherText` CSPRNG keystream XOR + keyed tag, demo/internal use only), `mpc::split_secret`/`reconstruct_secret` (`Share`, Shamir k-of-n over prime field `PRIME`); `consensus::Cluster` (in-memory Raft-style election/replication simulation: `run_election`, `replicate`, `current_leader`; demo/testing only, not networked production consensus) |
