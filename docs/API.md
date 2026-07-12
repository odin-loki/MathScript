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
| `core/checked_arith.hpp` | Overflow-safe `checked_add`/`sub`/`mul`/`div`/`mod`/`neg`/`abs`/`pow` (`Result<T>`), `saturating_add`/`sub`/`mul`, `wrapping_add`/`sub`/`mul`, float introspection (`is_nan`/`is_inf`/`is_finite`/`is_normal`/`signbit`/`ulp`/`eps`/`huge`/`tiny`), checked `narrow`/`widen` casts — Wave 191 |
| `core/units.hpp` | Compile-time SI unit-dimension system: `Units` structural-type NTTP (7 base-dimension exponents), `TypedScalar<T, Units>` zero-runtime-overhead dimensional analysis (dimension-mismatched ops are compile errors) — Wave 203 |

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
| `fft/fft.hpp` | `fft`, `ifft`, `rfft`/`irfft`, `dft`, `fft2`/`ifft2` (2D FFT with divisor-based dimension inference), `dct2`/`idct2`, `dst2`/`idst2` (self-inverse orthonormal DST-I), shift helpers, and `goertzel` (O(n) single-frequency-bin DFT extraction) |
| `stats/stats.hpp` | Mean, variance, percentiles, t/z tests, correlation, linear regression; one-way ANOVA (`one_way_anova`, F-distribution p-value via `ms::prob::f_cdf`), Mann-Whitney U (`mann_whitney_u`), Kruskal-Wallis H-test (`kruskal_wallis`, tie-corrected, chi-squared p-value — non-parametric multi-group generalization of `mann_whitney_u`), Friedman test (`friedman`, tie-corrected non-parametric repeated-measures alternative to one-way ANOVA, reusing `average_ranks` and `chi2_cdf`), two-sample KS (`ks_test_2sample`); Jarque-Bera normality test (`jarque_bera`, via existing `skewness`/`kurtosis`), Ljung-Box autocorrelation test (`ljung_box`, via existing `acf`), Levene's, Bartlett's, and Fligner-Killeen equal-variance tests (`levene_test`/`bartlett_test`/`fligner_test` — `levene_test` delegates to `one_way_anova` on median-absolute-deviation-transformed data; `fligner_test` is the rank/normal-scores-based non-parametric alternative, generally the most outlier/non-normality-robust of the three — Wave 203); bootstrap resampling (`bootstrap_mean`, mean-only `bootstrap_ci`, general `bootstrap_ci` with percentile-method CI for arbitrary statistics via `BootstrapResult`) |
| `prob/prob.hpp` | PDF/CDF/PPF for normal (`norm_ppf`), exponential (`exp_ppf`), binomial, Poisson, chi-square (`chi2_ppf`), t (`t_ppf`), gamma (`gamma_cdf`/`gamma_ppf`), beta (`beta_pdf`/`beta_cdf`/`beta_ppf`), F (`f_pdf`/`f_cdf`/`f_ppf`), uniform, lognormal (`lognormal_pdf`/`lognormal_cdf`/`lognormal_ppf`, via the normal ppf through a log transform), Weibull (`weibull_pdf`/`weibull_cdf`/`weibull_ppf`, closed-form quantile), Laplace (`laplace_pdf`/`laplace_cdf`/`laplace_ppf`, closed-form quantile) |
| `optim/optim.hpp` | `gradient_descent`, `newton_raphson`, `broyden`, `golden_section`, `newton_1d`, `simplex_solver`, `minimize_with_constraints`; N-D unconstrained: `nelder_mead`, `bfgs`, `lbfgs`, `adam`; global/derivative-free: `simulated_annealing`, `differential_evolution`, `particle_swarm`, `cmaes` (Covariance Matrix Adaptation Evolution Strategy — rank-1/rank-mu covariance updates, best on ill-conditioned/rotated objectives); nonlinear equation solvers: `bisection`, `brentq`, `secant`, `halley`, `fixed_point`, `illinois` (anti-stagnation regula falsi) |
| `signal/signal.hpp` | Butterworth/low/high/band-pass filters, convolution, moving average, window functions; `welch_psd` (Welch PSD via segmented FFT with overlap) and `spectrogram` (STFT magnitude spectrogram, same windowing convention); analytic-signal family `hilbert`/`envelope`/`instantaneous_phase`/`instantaneous_freq` (FFT-based Hilbert transform with phase unwrapping); lag-windowed `xcorr`/`xcov`/`autocorr`; 2D `conv2` and polynomial `deconv` (via `ms::poly::poly_div_quot`); resampling family `upsample`/`downsample`/`decimate`/`interpolate`/`resample` (zero-stuffing and lowpass-filtered rational rate conversion); generic `filter`/`filtfilt` (direct-form IIR/FIR difference-equation application and zero-phase double-pass filtering); `firwin`/`firwin_highpass` (windowed-sinc FIR design with Rectangular/Hamming/Hann/Blackman windows); `savgol` (Savitzky-Golay smoothing via per-window local polynomial least-squares fits, preserves polynomial trends unlike a plain moving average — Wave 203) |
| `special/special.hpp` | Broad special-function catalog: gamma, Bessel, elliptic, hypergeometric, Painlevé, etc.; DLMF additions: `zeta`, `zeta_hurwitz`, `eta_dirichlet`, `beta_dirichlet`, `polylog`, `clausen`, `debye`; `erfinv`/`erfcinv`, `trigamma`/`polygamma`, `pochhammer`/`falling_factorial`, `rgamma`, and the public canonical `gamma_inc_reg`/`gamma_inc_reg_upper`/`gamma_inc`/`beta_inc_reg`/`beta_inc` (also used internally by `ms::prob`'s `gamma_cdf`/`beta_cdf`/`f_cdf`/`chi2_cdf`); Voigt/pseudo-Voigt spectroscopy line shapes (`voigt`, `pseudo_voigt`, `pseudo_voigt_auto`, built on a from-scratch Humlicek w4 Faddeeva-function approximation) |
| `ode/ode.hpp` | Scalar/vector IVP solvers: `ode_euler`, `ode_rk4`, `ode_midpoint`, `ode_rk2`, `ode_rk45`, `ode_rk23`, `ode_cashkarp` (embedded Cash-Karp RK(4,5)), `ode_adams_bashforth2`, `ode_euler_vec`, `ode_rk4_vec`, `ode_rk45_vec`; symplectic: `ode_verlet`, `ode_verlet_vec`; stiff/implicit: `ode_backward_euler`, `ode_backward_euler_vec`, `ode_bdf2` (BDF2 multistep, A-stable, bootstraps first step via BDF1); BVP: `ode_bvp_shooting`; DDE: `ode_dde_fixed_step`; events: `ode_event_detect`; DAE: `ode_dae_index1` |
| `pde/pde.hpp` | `pde_heat_1d`, `pde_heat_1d_cn`, `pde_heat_2d`, `pde_heat_2d_cn_adi` (Peaceman-Rachford ADI Crank-Nicolson, unconditionally stable), `pde_wave_1d`, `pde_wave_2d`, `pde_advection_1d` (first-order upwind), `pde_advection_1d_lax_wendroff` (second-order Lax-Wendroff), `pde_poisson_1d`, `pde_poisson_2d`, `pde_burgers_1d`, `pde_reaction_diffusion_1d` (Fisher-KPP, zero-flux Neumann BC) with CFL/stability guards |
| `poly/poly.hpp` | Polynomial eval, add/sub/mul/div/mod, derivative/integral, composition, GCD/LCM on coefficient vectors; `poly_pow`/`monic`/`reverse`/`shift`/`scale`; `poly_sylvester`/`resultant`/`discriminant`/`squarefree`; `bernstein` basis evaluation; `interp_newton` (Newton divided-difference form) and `interp_hermite` (derivative-constrained interpolation via doubled-node divided differences); `poly_rational_roots`/`poly_factor_rational` (Rational Root Theorem factorization over Q for near-integer-coefficient polynomials); `poly_cheb_eval`/`poly_cheb_expand` (Chebyshev series evaluation and its dual/inverse expansion via Chebyshev-node sampling and discrete orthogonality — Wave 203) |
| `symbolic/symbolic.hpp` | AST `SymExpr` with `sym_add`/`sym_sub`/`sym_mul`/`sym_div`/`sym_neg`, `sym_sin`/`sym_cos`/`sym_tan`/`sym_exp`/`sym_log`/`sym_sqrt`/`sym_pow`, `sym_deriv`/`sym_diff`, `sym_simplify`, `sym_integrate`, `sym_substitute`, `sym_eval`, `sym_to_string`, `sym_parse` (recursive-descent text→`SymExpr` parser for `^` `*` `/` `+` `-`, unary minus, parens, functions, variables, literals) |
| `domain/domain.hpp` | `factorial`, `nchoosek`, `gcd`, and `Graph` edge counting |

## Number Theory — Wave 57 (`include/ms/numthy/`)

| Header | Description |
|--------|-------------|
| `numthy/numthy.hpp` | Primality (`isprime`, `primes`, `prime_pi`), factorisation (`factor`, `factor_exp`), divisor functions (`divisors`, `euler_phi`, `mobius`, `jordan_totient`, `von_mangoldt`), modular arithmetic (`mod_inv`, `mod_pow`, `crt`), Legendre/Jacobi/Kronecker symbols (`quadratic_residues`), discrete log, Tonelli–Shanks, continued fractions, `farey`, `pell_solve`, `partition`, `cornacchia` (sum-of-two-squares `x^2+d*y^2=p` via Tonelli-Shanks + Euclidean reduction), `stern_brocot` (BFS mediant enumeration) |

## Combinatorics — Wave 57 (`include/ms/combo/`)

| Header | Description |
|--------|-------------|
| `combo/combo.hpp` | Factorials and derangements, binomial/multinomial/permutations, `next_perm`/`next_comb`, rank/unrank, enumeration (`all_permutations`, `all_subsets`, `all_partitions`), Stirling/Bell/Catalan/Motzkin numbers, plus `set_partitions`/`involutions`/`dyck_paths`/`motzkin_paths`/`necklaces`/`lyndon_words` — Wave 191, `bracelets` (dihedral-equivalence necklaces) — Wave 200 |

## Information Theory — Wave 57 (`include/ms/info/`)

| Header | Description |
|--------|-------------|
| `info/info.hpp` | Shannon/joint/conditional entropy, mutual information, cross-entropy, KL/JS/Rényi/Tsallis divergence, Hellinger and total-variation distance, channel capacity (closed-form BSC/BEC plus general discrete-memoryless-channel capacity via the Blahut-Arimoto algorithm: `blahut_arimoto`/`channel_capacity`), rate–distortion, LZ complexity, sample entropy |

## Finance — Wave 57 (`include/ms/finance/`)

| Header | Description |
|--------|-------------|
| `finance/finance.hpp` | Black–Scholes call/put and Greeks, implied vol, bond price/duration/convexity/YTM, NPV/IRR/PV/FV annuities, VaR/CVaR, Sharpe/Sortino/Information/Treynor ratios, CAPM, forward rate, max drawdown, Kelly criterion, binomial/American/trinomial (`trinomial_option`, Boyle's method) option pricing, digital/Black-76/barrier option pricing, Bachelier (normal) model forward-option pricing (`bachelier_call`/`bachelier_put`, arithmetic Brownian motion — allows negative forward/strike unlike Black-76/Black-Scholes — Wave 203), portfolio variance/return, Monte Carlo option pricing (`mc_european_call`/`mc_european_put` with antithetic-variate GBM sampling, `mc_asian_call`/`mc_asian_put` via discretized path simulation, `mc_lookback_floating_call`/`mc_lookback_floating_put`/`mc_lookback_fixed_call`/`mc_lookback_fixed_put` tracking running path min/max), closed-form geometric-average Asian options (`geo_asian_call`/`geo_asian_put`, Kemna-Vorst formula), Markowitz portfolio optimization (`min_variance_portfolio`, `efficient_frontier_portfolio`, `max_sharpe_portfolio`, via a private Gaussian-elimination solve), Black-Litterman model (`bl_implied_returns`, `bl_posterior_returns`, `bl_posterior_returns_default_omega`) |

## Control — Wave 58 (`include/ms/control/`)

| Header | Description |
|--------|-------------|
| `control/control.hpp` | Transfer function and state-space models (`tf`, `ss`, `tf2ss`, `ss2tf`), continuous↔discrete conversion (`c2d`/`d2c`, ZOH/Tustin/Euler), interconnections, poles/zeros/stability, Bode/Nyquist/margins, step/impulse response, Lyapunov/Riccati (`lyap`, `riccati`, `dare`), LQR, controllability/observability, controllability/observability Gramians (`gram`/`ctrb_gram`/`obsv_gram`, via the Lyapunov equation), pole placement, PID tuning |

## Graph — Wave 58 (`include/ms/graph/`)

| Header | Description |
|--------|-------------|
| `graph/graph.hpp` | `Graph` adjacency list, BFS/DFS/topological sort, Dijkstra/Bellman-Ford/Floyd-Warshall/A*, connectivity and SCC, articulation points/bridges, biconnected components (`biconnected_components`, Tarjan DFS low-link with an explicit edge stack — Wave 203), MST (Kruskal/Prim), PageRank/betweenness/closeness/eigenvector/Katz centrality, Laplacian/normalised Laplacian/algebraic connectivity, max flow, min cut, bipartite matching, greedy colouring, `is_isomorphic` (VF2-style backtracking vertex-bijection search with degree-sequence pre-check, for small graphs), `louvain`/`modularity` (two-phase greedy modularity-optimization community detection) |

## Complex Analysis — Wave 58 (`include/ms/cplx/`)

| Header | Description |
|--------|-------------|
| `cplx/cplx.hpp` | Residues, winding number, contour/Cauchy integrals, argument principle, Möbius transforms, Joukowski map, Poisson kernel, Green's function on a disk (`green_function_disk`), harmonic conjugate, hyperbolic distance, Laurent coefficients, Blaschke products |

## Quantum — Wave 58 (`include/ms/quantum/`)

| Header | Description |
|--------|-------------|
| `quantum/quantum.hpp` | Kets and density matrices, Pauli and standard gates, tensor products, QFT, Bell/GHZ/W/coherent/Fock states, von Neumann entropy, fidelity, partial trace, entanglement entropy, Schrödinger time evolution (`schrodinger`), uncertainty products (`uncertainty`), Hamiltonian spectra and ground states (`eigenspectrum`/`ground_state`), phase-space quasi-probability distributions (`wigner_function` via the Cahill-Glauber displaced-parity trace formula, `husimi_Q` via coherent-state overlap) |

## Computational Geometry — Wave 59 (`include/ms/geo/`)

| Header | Description |
|--------|-------------|
| `geo/geo.hpp` | 2D/3D points, segments, rays, polygons; convex hull (2D via Graham scan, 3D via brute-force face enumeration with coplanar-face fan-triangulation), Delaunay/Voronoi; `KDTree2D`/`KDTree3D` nearest/range queries; ray–triangle/sphere/AABB intersections; Bezier/B-spline/Hermite curves; area, centroid, moments |

## Differential Geometry — Wave 59 (`include/ms/diffgeo/`)

| Header | Description |
|--------|-------------|
| `diffgeo/diffgeo.hpp` | Metric tensor and inverse, Christoffel symbols, Riemann/Ricci/Einstein tensors, geodesics, surface first/second fundamental forms, Gaussian/mean/principal curvature, Lie bracket, Gauss-Bonnet theorem verification (`gauss_bonnet_integral`/`gauss_bonnet_residual`) |

## Topology / TDA — Wave 59 (`include/ms/topo/`)

| Header | Description |
|--------|-------------|
| `topo/topo.hpp` | `SimplicialComplex`, Vietoris–Rips and Čech filtrations (`vietoris_rips`, `cech_complex` — the latter via true minimum-enclosing-ball radii rather than the flag-complex shortcut), alpha complex (`alpha_complex`, the MEB-radius-filtered subcomplex of a 2D point cloud's Delaunay triangulation — always a subcomplex of `cech_complex` on the same points — Wave 203), persistent homology diagrams, Betti numbers and Euler characteristic, bottleneck/Wasserstein distances between diagrams, Betti curves |

## Tensor Operations — Wave 59 (`include/ms/tensorops/`)

| Header | Description |
|--------|-------------|
| `tensorops/tensorops.hpp` | Row-major `Tensor`, contractions, `einsum`, mode products, Khatri-Rao/Kronecker, symmetrise/antisymmetrise, CP and Tucker/HOSVD decompositions, `reconstruct_cp`/`reconstruct_tucker` (rebuild a dense tensor from decomposition factors), Tensor-Train decomposition (`decompose_tt`/`reconstruct_tt`, TT-SVD for order-4+ tensors, via a self-contained Jacobi SVD), Frobenius norm |

## Machine Learning — Wave 60 (`include/ms/ml/`)

| Header | Description |
|--------|-------------|
| `ml/ml.hpp` | Linear/ridge/lasso/`ElasticNet` (combined L1+L2 penalty)/logistic regression, KNN, naive Bayes, decision trees; `SVM` (binary classifier via SMO with linear/RBF kernels); `RandomForest` (bootstrap-aggregated ensemble with feature subsampling) and `GradientBoosting` (functional-gradient-boosted shallow trees for regression); `GaussianMixture` (diagonal-covariance GMM via EM); `LDA` (shared-covariance linear discriminant classifier, with supervised `transform()` dimensionality reduction) and `QDA` (per-class-covariance quadratic discriminant classifier); KMeans, DBSCAN, agglomerative clustering; PCA, t-SNE; autodiff, feedforward nets; loss/metrics, scalers, cross-validation |

## Image Processing — Wave 60 (`include/ms/image/`)

| Header | Description |
|--------|-------------|
| `image/image.hpp` | `Image` buffer, RGB/HSV conversion, resize/crop/flip/rotate, Gaussian/median/bilateral filters, Sobel/Canny/Laplacian/Roberts/LoG edges, morphology (incl. `imgradient_morph`), Otsu/adaptive threshold, histogram equalisation (`histeq`) and CLAHE (`adapthisteq`), Harris/Shi-Tomasi corners, Radon/inverse-Radon (`iradon`), connected components |

## Compression — Wave 60 (`include/ms/compress/`)

| Header | Description |
|--------|-------------|
| `compress/compress.hpp` | RLE, Huffman, LZ77, LZW, BWT/MTF, delta coding, bzip2-like pipeline, bit-pack helpers, `arithmetic_encode`/`arithmetic_decode` range coding — Wave 191, plus `wavelet_compress`/`wavelet_decompress` (single-level Haar DWT lossy coding) — Wave 202 |

## Bignum — Wave 60 (`include/ms/bignum/`)

| Header | Description |
|--------|-------------|
| `bignum/bignum.hpp` | `BigInt` and `Rational` arbitrary-precision arithmetic, `bigint_gcd`/`extended_gcd`/`lcm`/`pow`/`pow_mod`, `bigint_divmod` (combined quotient+remainder, shared long-division helper with `operator/`/`operator%`), `bit_length`/`is_even`/`is_odd`, base-2/8/10/16 string I/O, factorial, Fibonacci, Miller–Rabin primality, `Rational::floor`/`ceil`/`round` |

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

**Wave 181 — framework and spectral REPL bindings:**

| Call | Description |
|------|-------------|
| `gria_entropy([data], bins)` | Shannon entropy of histogram-binned data (default `bins=16`) |
| `gria_matrix_alpha(X, FX)` | Information-theoretic alpha between input matrix `X` and transformed `FX` |
| `gria_is_critical(a, tol)` | Whether alpha is within tolerance of the critical threshold (default `tol=0.05`) |
| `gria_classify(a)` | Classify compute regime from alpha: reversible / critical / irreversible |
| `cypha_nig_fit([data])` | Fit Normal-Inverse-Gaussian parameters to a data vector |
| `cypha_nig_pdf(x, mu, a, b, d)` | NIG probability density at `x` |
| `cypha_nig_cdf(x, mu, a, b, d)` | NIG cumulative distribution at `x` |
| `cypha_nig_sample(mu, a, b, d, n)` | Draw `n` samples from an NIG distribution |
| `cellai_hebbian_update(W, v, h, lr)` | Hebbian weight update for Boltzmann energy model |
| `cellai_energy(W, v, h)` | Boltzmann energy \(-v^T W h\) for weight matrix `W` and state vectors `v`, `h` |
| `izaac_estimate_pi(n)` | Monte Carlo \(\pi\) estimate using `n` samples (requires prior `izaac seed N`) |
| `izaac_laplace_noise(value, epsilon, sensitivity)` | Laplace mechanism differential-privacy noise |
| `izaac_gaussian_noise(value, epsilon, delta, sensitivity)` | Gaussian mechanism differential-privacy noise |

**Wave 182 — session-object registry (first REPL exposure of stateful framework classes):**

The `Interpreter` now holds a `std::variant`-backed named-handle registry so users can create a persistent stateful object once and invoke methods on it across multiple REPL commands. `session_objects()` lists all live handles and their types; `session_object_clear(handle)` frees one.

| Call | Description |
|------|-------------|
| `bloom_new(handle, expected_items, fp_rate)` | Create a `ms::izaac::bloom::BloomFilter` (requires prior `izaac seed N`) |
| `bloom_insert(handle, "item")` | Insert a string item into the named bloom filter |
| `bloom_check(handle, "item")` | Test membership (`true`/`false`, may false-positive per the filter's FP rate) |
| `tokenbucket_new(handle, capacity, refill_rate)` | Create a `ms::izaac::ratelimit::TokenBucket` |
| `tokenbucket_consume(handle, tokens, now)` | Attempt to consume tokens at time `now` (`true`/`false`) |
| `tokenbucket_available(handle, now)` | Tokens currently available at time `now` |
| `cellmemory_new(handle, input_dim, memory_dim, [time_scales])` | Create a `ms::cellai::CellMemory` |
| `cellmemory_step(handle, input)` | Advance memory state with an input column vector |
| `cellmemory_recall(handle, time_scale)` | Recall memory content at a given time scale |
| `cellmemory_consolidate(handle)` | Consolidate short-term into long-term memory |

**Wave 183 — session-object registry extended to `DifModel`/`consensus::Cluster`:**

| Call | Description |
|------|-------------|
| `difmodel_new(handle, input_dim, output_dim, n_experts, learning_rate)` | Create a `ms::cypha::DifModel` online learner |
| `difmodel_update(handle, x, y)` | Online update on one `(x, y)` observation |
| `difmodel_predict(handle, x)` | Point prediction for input `x` |
| `difmodel_predict_interval(handle, x)` | Prediction with NIG-based uncertainty interval (mean/lower/upper) |
| `difmodel_ood_score(handle, x)` | Out-of-distribution score for input `x` |
| `difmodel_gh_gate(handle, x)` | Whether `x` passes the generalized-hyperbolic protection gate |
| `cluster_new(handle, n_nodes, seed)` | Create a simulated Raft `ms::izaac::consensus::Cluster` |
| `cluster_run_election(handle)` | Run one election round; returns elected leader id or -1 |
| `cluster_replicate(handle, leader_id, "cmd")` | Replicate a command to a quorum via the given leader |
| `cluster_current_leader(handle)` | Current leader id, or -1 if none/ambiguous |
| `cluster_status(handle)` | Per-node role/term/log-size/commit-index diagnostic dump |

**Wave 184 — final ODE formula-bridge bindings (closes the REPL formula-bridge gap):**

| Call | Description |
|------|-------------|
| `ode_backward_euler_vec("f0; f1; ...", t0, [y0], t_end, steps)` | Vector implicit backward Euler; semicolon-separated per-component formulas, `{t, y0..yN-1}` env |
| `ode_dae_index1("f0; ...", "g0; ...", t0, [y0], [z0], t_end, steps)` | Semi-explicit index-1 DAE: differential formulas `f` and algebraic-constraint formulas `g`, both see `{t, y0..yN-1, z0..zM-1}` |
| `ode_bvp_shooting("f_expr", t0, y_a, t_end, y_b, steps)` | Second-order BVP `y''=f(t,y,y')` via shooting; formula env `{t, y, yp}` |
| `ode_dde_fixed_step("f_expr", "history_expr", t0, t_end, tau, steps)` | Scalar DDE; `f_expr` env `{t, y, ydelay}`, `history_expr` env `{t}` |
| `ode_event_detect("f_expr", "event_expr", t0, y0, t_end, steps)` | Scalar IVP with root-crossing event detection; both formulas env `{t, y}` |

**Wave 186 — pure-function REPL backlog (Waves 182–185):**

| Call | Description |
|------|-------------|
| `gamma_cdf(x, shape, scale)` | Gamma distribution CDF |
| `beta_pdf(x, alpha, beta)` / `beta_cdf(x, alpha, beta)` | Beta distribution PDF/CDF |
| `f_pdf(x, d1, d2)` / `f_cdf(x, d1, d2)` | F distribution PDF/CDF |
| `kruskal_wallis(groups)` | Non-parametric H-test; `groups` matrix rows = groups (semicolon-separated) |
| `cmaes("formula", x0, sigma0, max_iter[, seed])` | CMA-ES global optimizer via the `sym_parse` formula bridge; env `{x0, x1, ...}` sized to `x0` |
| `ifft2(M)` | Inverse 2D FFT; result matrix follows the existing `fft_fft2`/`fft_ifft` `[re, im]`-pair convention |
| `idst2(M)` | Inverse 2D DST; real-valued result, same layout as `fft_dst2` |

`reconstruct_cp`/`reconstruct_tucker` (Wave 185) remain unbound: their `CPDecomposition`/`TuckerDecomposition` struct inputs have no REPL representation since `decompose_cp`/`decompose_hosvd`/`decompose_tucker` are not yet REPL-bound either.

**Wave 187 — tensor decomposition REPL exposure via the session-object registry (closes the item above):**

| Call | Description |
|------|-------------|
| `tensorops_decompose_cp(handle, T, rank[, max_iter, tol])` | CP/CANDECOMP-PARAFAC decompose a 2D matrix `T`; stores the result under `handle` |
| `tensorops_decompose_tucker(handle, T, ranks[, max_iter, tol])` | Tucker (ALS) decompose; `ranks` is a bracket vector, e.g. `[2, 2]` |
| `tensorops_decompose_hosvd(handle, T, ranks)` | Tucker via HOSVD decompose |
| `tensorops_reconstruct_cp(handle)` | Rebuild the approximated matrix from a stored CP handle |
| `tensorops_reconstruct_tucker(handle)` | Rebuild the approximated matrix from a stored Tucker/HOSVD handle |

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
| `frameworks/axiom/axiom.hpp` | Evolutionary `Axiom` GP search; `Algorithm` holds `Sym` representation; `evaluate` genuinely evaluates `algo.representation` row-by-row against input data columns (`x0`, `x1`, …) via `Sym::eval`; `evolve` performs real GP tree evolution (internal `GPNode` population with random initialization, tournament selection, subtree crossover/mutation, elitism, synced to `representation` each generation); `PrimitiveRegistry::build_from_ms_namespace()` supplies the GP tree's unary-function pool, scoped to `ms::Sym`'s scalar grammar (`sin`/`cos`/`exp`/`log`/`sqrt`/`tanh`) and consulted at every tree-generation site |
| `frameworks/cellai/cellai.hpp` | `CellMemory` temporal memory, `hebbian_update`, `energy` (Boltzmann -vᵀWh), `CellMemory::consolidate` short→long-term decay, `cell_to_cypha_features` |
| `frameworks/cypha/cypha.hpp` | `DifModel` mixture-of-experts with NIG uncertainty; `nig_fit`/`nig_pdf`/`nig_cdf`/`nig_sample`, `predict`, `predict_interval`, `ood_score`, `gh_gate` |
| `frameworks/gria/gria.hpp` | Information-theoretic `entropy`/`compute_alpha`/`matrix_alpha`/`is_critical`/`classify`, GF(2^n), cellular automata, LFSR utilities |
| `frameworks/izaac/izaac.hpp` | VRF `CSPRNG` (xoshiro256**), session seeding, `rand_matrix`/`randn_matrix`, `mc::estimate_pi`, `estimate_pi`; `bloom::BloomFilter`, `ratelimit::TokenBucket`, `diffpriv::laplace_mechanism`/`gaussian_mechanism`, `backtest::simulate_gbm_path`/`run_backtest`; `crypto::encrypt`/`decrypt` (`CipherText` CSPRNG keystream XOR + keyed tag, demo/internal use only), `mpc::split_secret`/`reconstruct_secret` (`Share`, Shamir k-of-n over prime field `PRIME`); `consensus::Cluster` (in-memory Raft-style election/replication simulation: `run_election`, `replicate`, `current_leader`; demo/testing only, not networked production consensus); `fuzz::mutate` (deterministic CSPRNG-seeded bounded byte-buffer mutation for fuzz-corpus seed expansion) |
