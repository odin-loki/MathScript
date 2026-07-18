# MathScript Public API Index

Per-module public API reference for headers under `include/ms/`, grouped by category. Include paths use the `ms/...` prefix (e.g. `#include "ms/core/matrix.hpp"`). For usage examples and REPL calling conventions see [`docs/USER_GUIDE.md`](USER_GUIDE.md); for build layout and module structure see [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

## Umbrella

Single include that pulls in most library modules.

| Header | Description |
|--------|-------------|
| `ms/ms.hpp` | Aggregates the majority of public modules (core, linalg, memory, error, fft, stats, prob, optim, signal, special, simd, cuda, distributed, runtime, frameworks, ode, pde, poly, symbolic, domain, and extended modules through bignum). Modules not pulled in here (e.g. `core/expr.hpp`, `cpu/blas.hpp`, `interp/`) must be included individually. |

## Numerical core

Foundation types, linear algebra, BLAS/LAPACK, numerics, memory, error handling, runtime dispatch, and SIMD.

| Header | Description |
|--------|-------------|
| `core/matrix.hpp` | `Matrix<S, Order, Alloc>` dense matrix type with col/row-major storage |
| `core/tensor.hpp` | Fixed-rank `Tensor<S, N>` multi-dimensional array wrapper; `reshape` (layout change preserving row-major data order) |
| `core/sparse.hpp` | COO `Sparse<S>` matrix with `spmv`, `sparse_add`, and dense conversion |
| `core/scalar.hpp` | `Scalar` value with physical units and arithmetic |
| `core/sym.hpp` | String-based `Sym` with parse/eval: `+ - * /`, unary `-`, parentheses, literals, variables (`x0`, `x1`, …), and `sin`/`cos`/`exp`/`log`/`sqrt`/`tanh`; unbound variables default to 0 |
| `core/expr.hpp` | CRTP expression templates for lazy matrix evaluation (`MatAdd`, `MatScale`, `MatMul`) |
| `core/operations.hpp` | High-level `matmul`, `lu`, `qr`, `solve`, and related `Result<>` wrappers |
| `core/rng.hpp` | Session RNG hooks (`set_session_rng`, `session_uniform`, `session_normal`, `session_exponential` via inverse-CDF transform) |
| `core/checked_arith.hpp` | Overflow-safe `checked_add`/`sub`/`mul`/`div`/`mod`/`neg`/`abs`/`pow` (`Result<T>`), `saturating_add`/`sub`/`mul`, `wrapping_add`/`sub`/`mul`, float introspection (`is_nan`/`is_inf`/`is_finite`/`is_normal`/`signbit`/`ulp`/`eps`/`huge`/`tiny`/`nextafter`), checked `narrow`/`widen` casts |
| `core/units.hpp` | Compile-time SI unit-dimension system: `Units` structural-type NTTP (7 base-dimension exponents), `TypedScalar<T, Units>` zero-runtime-overhead dimensional analysis (dimension-mismatched ops are compile errors); `is_dimensionless` and a compile-time-checked `sqrt` (halves the unit exponents, `static_assert` rejects quantities with any odd exponent) |
| `linalg/linalg.hpp` | Eig/SVD/LDL/Schur result types; `rand`, `eig_sym`, `svd`, `expm`, `trace`, `det`, `norm`, decompositions; `matrix_rank` (numerical rank via SVD singular-value threshold); `solve_sylvester` (general Sylvester equation `A*X + X*B = C` via Kronecker-sum vectorization through the existing dense `solve()` — the non-symmetric counterpart to `ms::control::lyap`'s Lyapunov special case `B=A^T`) |
| `linalg/matrix_operations.hpp` | `expected`-based matmul, LU, QR, solve, and matrix utilities |
| `linalg/transpose.hpp` | `transpose()` returning row-major aligned matrix |
| `cpu/blas.hpp` | Column-major BLAS: `ddot`/`daxpy`/`dgemv` (Level-1/2), `dgemm`/`dsyrk`/`dtrsm`/`dger` (Level-3) |
| `cpu/lapack.hpp` | LAPACK-style factorizations and solvers: Cholesky, LU, QR, SVD, least squares, symmetric eigen |
| `cpu/blas_kernel.hpp` | AVX-512 micro-kernels for internal BLAS paths |
| `memory/policy.hpp` | `PolicyFlag` alignment/pinning/NUMA/pool allocation flags |
| `memory/aligned_allocator.hpp` | Cross-platform aligned `allocate`/`deallocate` helpers |
| `memory/pinned_allocator.hpp` | Host pinned memory allocator (CPU builds) |
| `memory/arena.hpp` | Per-thread monotonic PMR arena allocator; `bytes_used()` occupancy tracking |
| `memory/pool_allocator.hpp` | Slab-based pool for small fixed-size blocks |
| `memory/numa_allocator.hpp` | NUMA topology-aware local allocation |
| `error/error_types.hpp` | `DimensionMismatch`, `SingularMatrix`, `ValueOutOfRange`, and other `std::expected` error variants |
| `error/expected.hpp` | Re-exports error types; `Result<T>` alias for `std::expected<T, Error>` |
| `runtime/dispatch.hpp` | `ExecPolicy`, `Backend`, and `DispatchDecision` for CPU/GPU routing |
| `runtime/topology.hpp` | `SystemTopology` CPU/GPU/NUMA discovery |
| `runtime/thread_pool.hpp` | Configurable worker thread pool with futures; `parallel_for` range-parallel dispatch |
| `runtime/load_balancer.hpp` | `balance()` picks backend, device, and thread count for a workload |
| `simd/isa.hpp` | `IsaFeatures` detection and human-readable `isa_summary` |
| `simd/simd.hpp` | Vectorized `add`, `mul`, `dot`, `axpy`, `exp_map`, `gemv` with kernel dispatch; `sum` (reduction, mirroring `dot`'s ISA dispatch tiers and horizontal-reduction technique) |
| `poly/poly.hpp` | Polynomial eval, add/sub/mul/div/mod, derivative/integral, composition, GCD/LCM on coefficient vectors; `poly_pow`/`monic`/`reverse`/`shift`/`scale`; `poly_sylvester`/`resultant`/`discriminant`/`squarefree`; `bernstein` basis evaluation; `interp_newton` (Newton divided-difference form) and `interp_hermite` (derivative-constrained interpolation via doubled-node divided differences); `poly_rational_roots`/`poly_factor_rational` (Rational Root Theorem factorization over Q for near-integer-coefficient polynomials); `poly_cheb_eval`/`poly_cheb_expand` (Chebyshev series evaluation and its dual/inverse expansion via Chebyshev-node sampling and discrete orthogonality); `poly_partial_fractions` (partial fraction decomposition over simple/repeated real poles and complex-conjugate quadratic terms, via a coefficient-matching linear system); `poly_roots` (all real/complex roots via companion-matrix eigenvalues — Wilkinson-shifted QR on the upper-Hessenberg companion matrix, the same general approach `MATLAB`'s `roots()` uses); `poly_factor` (real polynomial factorization via `poly_roots`, grouping conjugate pairs into quadratics) |
| `ode/ode.hpp` | Scalar/vector IVP solvers: `ode_euler`, `ode_rk4`, `ode_midpoint`, `ode_rk2`, `ode_rk45`, `ode_rk23`, `ode_cashkarp` (embedded Cash-Karp RK(4,5)), `ode_adams_bashforth2`, `ode_euler_vec`, `ode_rk4_vec`, `ode_rk45_vec`; symplectic: `ode_verlet`, `ode_verlet_vec`; stiff/implicit: `ode_backward_euler`, `ode_backward_euler_vec`, `ode_bdf2` (BDF2 multistep, A-stable, bootstraps first step via BDF1), `ode_rosenbrock23`/`ode_rosenbrock23_vec` (linearly-implicit 2-stage Rosenbrock-Wanner method, one linear solve per stage per step rather than full Newton iteration); BVP: `ode_bvp_shooting`; DDE: `ode_dde_fixed_step`; events: `ode_event_detect`; DAE: `ode_dae_index1`; `ode_exponential_euler` (ETD1 exponential time-differencing for semi-linear stiff `y'=lambda*y+g(t,y)`, treating the linear term exactly via `expm1`); `ode_trapezoidal` (implicit trapezoidal rule, second-order accurate complement to `ode_euler`) |
| `pde/pde.hpp` | `pde_heat_1d`, `pde_heat_1d_cn`, `pde_heat_2d`, `pde_heat_2d_cn_adi` (Peaceman-Rachford ADI Crank-Nicolson, unconditionally stable), `pde_wave_1d`, `pde_wave_2d`, `pde_advection_1d` (first-order upwind), `pde_advection_1d_lax_wendroff` (second-order Lax-Wendroff), `pde_poisson_1d`, `pde_poisson_2d`, `pde_burgers_1d`, `pde_reaction_diffusion_1d` (Fisher-KPP, zero-flux Neumann BC) with CFL/stability guards; `pde_helmholtz_2d` (2D Helmholtz equation `Laplacian(u)+k^2*u=f` via the same dense finite-difference approach as `pde_poisson_2d` plus a reaction term — documents the near-resonance ill-conditioning inherent to Helmholtz problems); `pde_laplace_2d` (steady 2D Laplace equation ∇²u=0 via Jacobi iteration with Dirichlet boundary values) |
| `fem/fem.hpp` | 1D/2D P1 finite-element Poisson solvers: `mesh1d`/`mesh2d_rectangular`, `lagrange_basis`, `assemble_stiffness_1d`/`assemble_stiffness_2d`, `assemble_load_1d`/`assemble_load_2d` (Gauss quadrature), `apply_dirichlet`, `solve_fem` (via `ms::linalg::solve`); 2D uses structured triangular meshes on rectangles |
| `cfd/cfd.hpp` | 1D/2D finite-volume advection: `grid1d`/`grid2d`, `square_pulse`/`square_pulse_2d`, `constant_velocity`, explicit Euler upwind `upwind_fvm_advection`/`upwind_fvm_advection_2d`, time integration `run_advection`/`run_advection_2d`, mass integrals `integrated_mass`/`integrated_mass_2d`; `BoundaryCondition` periodic or zero-flux per axis; CFL guards reject unstable steps |
| `crypto/crypto.hpp` | Pure-C++ digests and ciphers (include individually — not in `ms/ms.hpp`): `sha256`/`sha512`/`hmac_sha256`/`hmac_sha512` with hex helpers; `hkdf_sha256`/`hkdf_sha512`; `pbkdf2_hmac_sha256`/`pbkdf2_hmac_sha512`; AES-128/256 ECB/CBC/GCM; ChaCha20 / ChaCha20-Poly1305; X25519; Ed25519; `constant_time_eq`; `random_bytes` (MVP) |
| `optim/optim.hpp` | `gradient_descent`, `newton_raphson`, `broyden`, `golden_section`, `newton_1d`, `simplex_solver`, `minimize_with_constraints`; N-D unconstrained: `nelder_mead`, `bfgs`, `lbfgs`, `adam`; global/derivative-free: `simulated_annealing`, `differential_evolution`, `particle_swarm`, `cmaes` (Covariance Matrix Adaptation Evolution Strategy — rank-1/rank-mu covariance updates, best on ill-conditioned/rotated objectives); nonlinear equation solvers: `bisection`, `brentq`, `secant`, `halley`, `fixed_point`, `illinois` (anti-stagnation regula falsi); `levenberg_marquardt` (damped Gauss-Newton nonlinear least squares via a finite-difference Jacobian and adaptive damping); `conjugate_gradient` (nonlinear Fletcher-Reeves/Polak-Ribière+ CG with Armijo backtracking line search — converges in near-N steps on exact quadratics); `rmsprop` (adaptive gradient optimizer via EMA of squared gradients); `adadelta` (adaptive gradient optimizer without manual learning-rate schedule) |
| `symbolic/symbolic.hpp` | AST `SymExpr` with `sym_add`/`sym_sub`/`sym_mul`/`sym_div`/`sym_neg`, `sym_sin`/`sym_cos`/`sym_tan`/`sym_exp`/`sym_log`/`sym_sqrt`/`sym_pow`, `sym_deriv`/`sym_diff`, `sym_simplify`, `sym_integrate`, `sym_substitute`, `sym_eval`, `sym_to_string`, `sym_parse` (recursive-descent text→`SymExpr` parser for `^` `*` `/` `+` `-`, unary minus, parens, functions, variables, literals); `sym_expand` (distributes multiplication over addition/subtraction, with bounded small-integer-power expansion); `sym_collect` (combine like terms in a variable); `sym_limit`, `sym_series`, `sym_solve_linear`; table-driven `sym_laplace`/`sym_ilaplace`, `sym_mellin`/`sym_imellin`, `sym_hankel`/`sym_ihankel`, `sym_fourier`/`sym_ifourier`, `sym_ztransform`/`sym_iztransform` (MVP: polynomials, `exp`, `sin/cos`, rationals, geometric sequences; unsupported forms return `sym_deriv` sentinel); `sym_dsolve` (separable first-order ODE MVP — general `dsolve` deferred) |
| `special/special.hpp` | Broad special-function catalog: gamma, Bessel, elliptic, hypergeometric, Painlevé, etc.; DLMF additions: `zeta`, `zeta_hurwitz`, `eta_dirichlet`, `beta_dirichlet`, `polylog`, `clausen`, `debye`; `erfinv`/`erfcinv`, `trigamma`/`polygamma`, `pochhammer`/`falling_factorial`, `rgamma`, and the public canonical `gamma_inc_reg`/`gamma_inc_reg_upper`/`gamma_inc`/`beta_inc_reg`/`beta_inc` (also used internally by `ms::prob`'s `gamma_cdf`/`beta_cdf`/`f_cdf`/`chi2_cdf`); Voigt/pseudo-Voigt spectroscopy line shapes (`voigt`, `pseudo_voigt`, `pseudo_voigt_auto`, built on a from-scratch Humlicek w4 Faddeeva-function approximation); `sph_bessel_j`/`sph_bessel_y` (spherical Bessel functions via closed-form base cases plus a stable upward recurrence); `assoc_legendre_p`/`sph_harmonic_y` (associated Legendre polynomials and complex spherical harmonics `Y_l^m(theta,phi)`, verified via sphere orthonormality); `kummer_u` (Kummer's confluent hypergeometric function of the second kind, via the standard connection formula in terms of `kummer_m`); `lambert_w` (Lambert W function, branches 0 and −1) |
| `domain/domain.hpp` | `factorial`, `nchoosek`, `gcd`, `lcm`, `extended_gcd` (Bezout coefficients via `ExtGcdResult`), and `Graph` edge counting |

## Statistics & data

Probability, inference, spectral transforms, signal processing, information theory, combinatorics, tensor decompositions, and arbitrary-precision arithmetic.

| Header | Description |
|--------|-------------|
| `stats/stats.hpp` | Mean, variance, percentiles, t/z tests, correlation, linear regression; one-way ANOVA (`one_way_anova`, F-distribution p-value via `ms::prob::f_cdf`), Mann-Whitney U (`mann_whitney_u`), Kruskal-Wallis H-test (`kruskal_wallis`, tie-corrected, chi-squared p-value — non-parametric multi-group generalization of `mann_whitney_u`), Friedman test (`friedman`, tie-corrected non-parametric repeated-measures alternative to one-way ANOVA, reusing `average_ranks` and `chi2_cdf`), two-sample KS (`ks_test_2sample`); Jarque-Bera and Shapiro-Wilk normality tests (`jarque_bera` via existing `skewness`/`kurtosis`; `shapiro_wilk`, order-statistic correlation via normalized weights and Royston's normalizing p-value transformation through `norm_cdf`), Ljung-Box autocorrelation test (`ljung_box`, via existing `acf`), Levene's, Bartlett's, and Fligner-Killeen equal-variance tests (`levene_test`/`bartlett_test`/`fligner_test` — `levene_test` delegates to `one_way_anova` on median-absolute-deviation-transformed data; `fligner_test` is the rank/normal-scores-based non-parametric alternative, generally the most outlier/non-normality-robust of the three); bootstrap resampling (`bootstrap_mean`, mean-only `bootstrap_ci`, general `bootstrap_ci` with percentile-method CI for arbitrary statistics via `BootstrapResult`); `wilcoxon_signed_rank` (paired non-parametric alternative to a paired t-test, tie-corrected rank-sum of signed differences with the same continuity-correction convention as `mann_whitney_u`); `partial_correlation` (correlation between two variables controlling for a third, via the standard `r_xy.z` formula over pairwise `correlation`) and `variance_inflation_factor` (multicollinearity diagnostic, `1/(1-R^2)` of a column-on-the-rest regression); `weighted_mean`/`weighted_variance`/`weighted_correlation` (weighted analogues of the unweighted descriptive statistics, reliability-weights bias correction, exactly reducing to their unweighted counterparts under uniform weights); `kde` (1D kernel density estimation over a user grid, Gaussian/Epanechnikov kernels); `mad` (median absolute deviation, robust scale estimate) |
| `prob/prob.hpp` | PDF/CDF/PPF for normal (`norm_ppf`), exponential (`exp_ppf`), binomial, Poisson, chi-square (`chi2_ppf`), t (`t_ppf`), gamma (`gamma_cdf`/`gamma_ppf`), beta (`beta_pdf`/`beta_cdf`/`beta_ppf`), F (`f_pdf`/`f_cdf`/`f_ppf`), uniform, lognormal (`lognormal_pdf`/`lognormal_cdf`/`lognormal_ppf`, via the normal ppf through a log transform), Weibull (`weibull_pdf`/`weibull_cdf`/`weibull_ppf`, closed-form quantile), Laplace (`laplace_pdf`/`laplace_cdf`/`laplace_ppf`, closed-form quantile), Gumbel (`gumbel_pdf`/`gumbel_cdf`/`gumbel_ppf`, Type-I extreme-value distribution for maxima), Cauchy (`cauchy_pdf`/`cauchy_cdf`/`cauchy_ppf`, heavy-tailed with undefined mean/variance), Pareto (`pareto_pdf`/`pareto_cdf`/`pareto_ppf`, power-law with support `x>=x_m`), Rayleigh (`rayleigh_pdf`/`rayleigh_cdf`/`rayleigh_ppf`, the 2D-vector-magnitude distribution used for wind speed/wave height/signal-magnitude modeling), Logistic (`logistic_pdf`/`logistic_cdf`/`logistic_ppf`, sigmoid CDF used in GLM/robust stats) |
| `fft/fft.hpp` | `fft`, `ifft`, `rfft`/`irfft`, `dft`, `fft2`/`ifft2` (2D FFT with divisor-based dimension inference), `dct2`/`idct2`, `dst2`/`idst2` (self-inverse orthonormal DST-I), shift helpers, and `goertzel` (O(n) single-frequency-bin DFT extraction); `fftfreq`/`rfftfreq` (NumPy-compatible frequency-axis helpers labeling each `fft`/`rfft` output bin with its actual frequency) |
| `signal/signal.hpp` | Butterworth/low/high/band-pass filters, convolution, moving average, window functions; `welch_psd` (Welch PSD via segmented FFT with overlap) and `spectrogram` (STFT magnitude spectrogram, same windowing convention); analytic-signal family `hilbert`/`envelope`/`instantaneous_phase`/`instantaneous_freq` (FFT-based Hilbert transform with phase unwrapping); lag-windowed `xcorr`/`xcov`/`autocorr`; 2D `conv2` and polynomial `deconv` (via `ms::poly::poly_div_quot`); resampling family `upsample`/`downsample`/`decimate`/`interpolate`/`resample` (zero-stuffing and lowpass-filtered rational rate conversion); generic `filter`/`filtfilt` (direct-form IIR/FIR difference-equation application and zero-phase double-pass filtering); `firwin`/`firwin_highpass` (windowed-sinc FIR design with Rectangular/Hamming/Hann/Blackman windows); `savgol` (Savitzky-Golay smoothing via per-window local polynomial least-squares fits, preserves polynomial trends unlike a plain moving average); `median_filter` (nonlinear sliding-window median filter, robust to outliers/impulsive noise); `lms_adaptive_filter` (online LMS adaptive FIR filter via stochastic gradient descent on the instantaneous squared error, for noise cancellation/system identification/equalization); `coherence` (magnitude-squared coherence `|Pxy|^2/(Pxx*Pyy)` between two signals via the same segmented/windowed Welch infrastructure as `welch_psd`, extended to also produce the complex cross-spectral density); `czt`/`czt_zoom_fft` (Chirp Z-Transform via Bluestein's algorithm — evaluates the Z-transform along an arbitrary spiral contour, generalizing the DFT; `czt_zoom_fft` is a "zoom FFT" convenience for narrowband high-resolution spectral analysis); `sosfilt` (second-order-sections IIR filtering via cascaded `filter()` — numerically stable for high-order IIR); `cheby1` (Chebyshev Type I IIR filter design via bilinear zpk transform, scipy-compatible); `cheby2` (Chebyshev Type II IIR filter design with stopband attenuation, scipy-compatible) |
| `info/info.hpp` | Shannon/joint/conditional entropy, mutual information, cross-entropy, KL/JS/Rényi/Tsallis divergence, Hellinger and total-variation distance, channel capacity (closed-form BSC/BEC plus general discrete-memoryless-channel capacity via the Blahut-Arimoto algorithm: `blahut_arimoto`/`channel_capacity`), rate–distortion, LZ complexity, sample entropy, `permutation_entropy` (Bandt-Pompe ordinal-pattern Shannon entropy of a time series, optionally normalized by `log2(order!)`); `normalized_entropy` (Shannon entropy scaled to [0, 1] by dividing by log₂(n)); `transfer_entropy` (directed information flow `TE(X->Y)` between time series via histogram-based conditional-entropy estimation, verified via the defining direction-asymmetry property) |
| `numthy/numthy.hpp` | Primality (`isprime`, `primes`, `prime_pi`), factorisation (`factor`, `factor_exp`), divisor functions (`divisors`, `euler_phi`, `mobius`, `jordan_totient`, `von_mangoldt`), modular arithmetic (`mod_inv`, `mod_pow`, `crt`), Legendre/Jacobi/Kronecker symbols (`quadratic_residues`), discrete log, Tonelli–Shanks, continued fractions, `farey`, `pell_solve`, `partition`, `cornacchia` (sum-of-two-squares `x^2+d*y^2=p` via Tonelli-Shanks + Euclidean reduction), `stern_brocot` (BFS mediant enumeration), `is_carmichael` (Korselt's-criterion Carmichael-number test) and `lucas_sequence` (generalized Fibonacci/Lucas numbers via O(log k) matrix exponentiation); `carmichael_lambda` (Carmichael function λ(n), the multiplicative-group-exponent generalization of Euler's totient, with the non-cyclic power-of-2 special case handled); `multiplicative_order` (smallest `k` with `a^k ≡ 1 mod n`, via Euler-totient prime-factor exponent reduction — always divides both `euler_phi(n)` and `carmichael_lambda(n)`); `primitive_root` (smallest primitive root modulo a prime) |
| `combo/combo.hpp` | Factorials and derangements, binomial/multinomial/permutations, `next_perm`/`next_comb`, rank/unrank, enumeration (`all_permutations`, `all_subsets`, `all_partitions`), Stirling/Bell/Catalan/Motzkin numbers, plus `set_partitions`/`involutions`/`dyck_paths`/`motzkin_paths`/`necklaces`/`lyndon_words`, `bracelets` (dihedral-equivalence necklaces), `gray_code` (closed-form binary reflected Gray code) and `de_bruijn_sequence` (FKM algorithm via Lyndon-word concatenation); `eulerian_number` (Eulerian numbers A(n,k), permutations of n elements with exactly k ascents, via DP recurrence); `restricted_partitions` (partitions of n into exactly k positive parts) |
| `bignum/bignum.hpp` | `BigInt` and `Rational` arbitrary-precision arithmetic, `bigint_gcd`/`extended_gcd`/`lcm`/`pow`/`pow_mod`, `bigint_divmod` (combined quotient+remainder, shared long-division helper with `operator/`/`operator%`), `bigint_isqrt` (exact BigInt-native Newton's-method integer square root, no floating point involved), `bit_length`/`is_even`/`is_odd`, base-2/8/10/16 string I/O, factorial, Fibonacci, Miller–Rabin primality, `Rational::floor`/`ceil`/`round`; `bigint_mod_inv` (modular inverse via `extended_gcd`); `bigint_next_prime` (smallest prime ≥ n) |
| `tensorops/tensorops.hpp` | Row-major `Tensor`, contractions, `einsum`, mode products, Khatri-Rao/Kronecker, symmetrise/antisymmetrise, CP and Tucker/HOSVD decompositions, `reconstruct_cp`/`reconstruct_tucker` (rebuild a dense tensor from decomposition factors), Tensor-Train decomposition (`decompose_tt`/`reconstruct_tt`, TT-SVD for order-4+ tensors, via a self-contained Jacobi SVD), Frobenius norm; `decompose_nmf`/`reconstruct_nmf` (non-negative matrix factorization via the Lee-Seung multiplicative-update rule) |

## Applied domains

Domain-specific algorithms spanning finance, control, graphs, geometry, machine learning, imaging, compression, quantum, complex analysis, differential geometry, and topology.

| Header | Description |
|--------|-------------|
| `finance/finance.hpp` | Black–Scholes call/put and Greeks, implied vol, bond price/duration/convexity/YTM, NPV/IRR/PV/FV annuities, VaR/CVaR, Sharpe/Sortino/Information/Treynor ratios, CAPM, forward rate, max drawdown, Kelly criterion, binomial/American/trinomial (`trinomial_option`, Boyle's method) option pricing, digital/Black-76/barrier option pricing, Bachelier (normal) model forward-option pricing (`bachelier_call`/`bachelier_put`, arithmetic Brownian motion — allows negative forward/strike unlike Black-76/Black-Scholes), portfolio variance/return, Monte Carlo option pricing (`mc_european_call`/`mc_european_put` with antithetic-variate GBM sampling, `mc_asian_call`/`mc_asian_put` via discretized path simulation, `mc_lookback_floating_call`/`mc_lookback_floating_put`/`mc_lookback_fixed_call`/`mc_lookback_fixed_put` tracking running path min/max), closed-form geometric-average Asian options (`geo_asian_call`/`geo_asian_put`, Kemna-Vorst formula), Markowitz portfolio optimization (`min_variance_portfolio`, `efficient_frontier_portfolio`, `max_sharpe_portfolio`, via a private Gaussian-elimination solve), Black-Litterman model (`bl_implied_returns`, `bl_posterior_returns`, `bl_posterior_returns_default_omega`), Vasicek and Cox-Ingersoll-Ross mean-reverting short-rate model zero-coupon bond pricing (`vasicek_bond_price`/`cir_bond_price`, closed-form affine `P(t,T)=A(tau)*exp(-B(tau)*r)` formulas); historical-simulation risk (`historical_var`/`historical_cvar`, empirical-percentile method complementing the parametric variance-covariance `var`/`cvar`); Merton structural credit-risk model (`merton_distance_to_default`, direct Black-Scholes `d2` formula with assets/debt substituted; `merton_implied_asset_params`, backing out unobservable asset value/volatility from observable equity value/volatility via fixed-point iteration on the coupled Black-Scholes equations); `heston_call` (Heston stochastic-volatility European call via Lewis/Fourier integration, σ_v→0 limit matches Black–Scholes); `sabr_call` (SABR stochastic-volatility European call via Hagan et al. asymptotic formula) |
| `control/control.hpp` | Transfer function and state-space models (`tf`, `ss`, `tf2ss`, `ss2tf`), continuous↔discrete conversion (`c2d`/`d2c`, ZOH/Tustin/Euler), interconnections, poles/zeros/stability, Bode/Nyquist/margins, step/impulse response, Lyapunov/Riccati (`lyap`, `riccati`, `dare`), LQR, controllability/observability, controllability/observability Gramians (`gram`/`ctrb_gram`/`obsv_gram`, via the Lyapunov equation), pole placement, PID tuning, discrete-time Kalman filter (`kalman_predict`/`kalman_update` on a `KalmanState{x,P}` pair); `step_info` (rise/settling time, percent overshoot, peak time/value extracted from a step-response trace); `lqe` (linear quadratic estimator — dual of LQR for observer gain design) |
| `graph/graph.hpp` | `Graph` adjacency list, BFS/DFS/topological sort, Dijkstra/Bellman-Ford/Floyd-Warshall/A*, connectivity and SCC, articulation points/bridges, biconnected components (`biconnected_components`, Tarjan DFS low-link with an explicit edge stack), MST (Kruskal/Prim), PageRank/betweenness/closeness/eigenvector/Katz centrality, Laplacian/normalised Laplacian/algebraic connectivity, max flow, min cut, bipartite matching, greedy colouring, `is_isomorphic` (VF2-style backtracking vertex-bijection search with degree-sequence pre-check, for small graphs), `louvain`/`modularity` (two-phase greedy modularity-optimization community detection), `eulerian_path` (Eulerian path/circuit detection and construction via Hierholzer's algorithm, undirected graphs), `tsp_heuristic` (Traveling Salesman heuristic — nearest-neighbor construction plus 2-opt local search over a dense pairwise distance matrix); `min_arborescence` (minimum spanning arborescence for directed graphs via the Chu-Liu/Edmonds algorithm, with cycle contraction/expansion — the directed analogue of Kruskal/Prim MST); `k_core_decomposition`/`k_core_subgraph` (core numbers via O(V+E) bucket-queue degree-peeling, and extraction of the maximal subgraph where every vertex has degree >= k); `transitive_closure` (Floyd-Warshall-style boolean reachability closure, complementing the existing weighted Floyd-Warshall shortest-path function); `eccentricity` (per-node maximum shortest-path distance; on connected graphs max eccentricity equals `diameter`) |
| `cplx/cplx.hpp` | Residues, winding number, contour/Cauchy integrals, argument principle, Möbius transforms, Joukowski map, Poisson kernel, Green's function on a disk (`green_function_disk`), harmonic conjugate, hyperbolic distance, Laurent coefficients, Blaschke products; `cauchy_principal_value` (PV of a real integral with a simple pole, via symmetric exclusion of a shrinking interval around the singularity) |
| `quantum/quantum.hpp` | Kets and density matrices, Pauli and standard gates, tensor products, QFT, Bell/GHZ/W/coherent/Fock states, von Neumann entropy, fidelity, partial trace, entanglement entropy, Schrödinger time evolution (`schrodinger`), uncertainty products (`uncertainty`), Hamiltonian spectra and ground states (`eigenspectrum`/`ground_state`), phase-space quasi-probability distributions (`wigner_function` via the Cahill-Glauber displaced-parity trace formula, `husimi_Q` via coherent-state overlap), Grover's search algorithm (`grover_search`/`grover_optimal_iterations`, explicit dense oracle/diffusion operator matrices); Schmidt decomposition (`schmidt_decomposition`/`schmidt_rank`, SVD of a bipartite state's reshaped coefficient matrix — Schmidt coefficients squared are the reduced density matrix's eigenvalues, cross-checked against `von_neumann_entropy`); `purity` (Tr(ρ²) purity measure for density matrices) |
| `geo/geo.hpp` | 2D/3D points, segments, rays, polygons; convex hull (2D via Graham scan, 3D via brute-force face enumeration with coplanar-face fan-triangulation), Delaunay/Voronoi; `KDTree2D`/`KDTree3D` nearest/range queries; ray–triangle/sphere/AABB intersections; Bezier/B-spline/Hermite curves; area, centroid, moments; minimum-area oriented bounding rectangle (`min_bounding_rect`, rotating calipers over `convex_hull_2d`); `minkowski_sum_convex` (Minkowski sum of two convex polygons via linear-time angular edge-merge, with a brute-force pairwise-sum-plus-hull fallback for degenerate inputs); `triangulate_polygon` (simple-polygon, convex or concave, triangulation via ear clipping); `clip_polygon` (Sutherland-Hodgman convex-clip-window polygon clipping — subject polygon may be non-convex, only the clip window must be); `poly_union` (convex-polygon union MVP via combined-vertex convex hull — documented limitation for non-convex inputs); `poly_intersect` (convex-polygon intersection MVP via Sutherland-Hodgman clipping) |
| `diffgeo/diffgeo.hpp` | Metric tensor and inverse, Christoffel symbols, Riemann/Ricci/Einstein tensors, geodesics, surface first/second fundamental forms, Gaussian/mean/principal curvature, Lie bracket, Gauss-Bonnet theorem verification (`gauss_bonnet_integral`/`gauss_bonnet_residual`), `parallel_transport` (integrates the parallel transport equation along an explicit parametrized curve, demonstrating holonomy around closed loops on curved surfaces); `torsion` (Frenet-Serret torsion of a 3D space curve, zero on planar curves, matches the closed-form constant torsion of a circular helix) |
| `topo/topo.hpp` | `SimplicialComplex`, Vietoris–Rips and Čech filtrations (`vietoris_rips`, `cech_complex` — the latter via true minimum-enclosing-ball radii rather than the flag-complex shortcut), alpha complex (`alpha_complex`, the MEB-radius-filtered subcomplex of a 2D point cloud's Delaunay triangulation — always a subcomplex of `cech_complex` on the same points), persistent homology diagrams, Betti numbers and Euler characteristic, bottleneck/Wasserstein distances between diagrams, Betti curves, `persistence_landscape` (Bubenik's vectorized functional summary of a persistence diagram — k-th-largest-tent-value-at-each-t, giving a genuine vector space unlike a raw diagram); `witness_complex`/`select_landmarks_maxmin` (landmark-based approximate complex for large point clouds, with max-min farthest-point landmark selection) |
| `ml/ml.hpp` | Linear/ridge/lasso/`ElasticNet` (combined L1+L2 penalty)/logistic regression, KNN, naive Bayes, decision trees; `SVM` (binary classifier via SMO with linear/RBF kernels); `RandomForest` (bootstrap-aggregated ensemble with feature subsampling) and `GradientBoosting` (functional-gradient-boosted shallow trees for regression); `GaussianMixture` (diagonal-covariance GMM via EM); `LDA` (shared-covariance linear discriminant classifier, with supervised `transform()` dimensionality reduction) and `QDA` (per-class-covariance quadratic discriminant classifier); KMeans, DBSCAN, agglomerative clustering; PCA, t-SNE; autodiff, feedforward nets; loss/metrics, scalers, cross-validation; `confusion_matrix`/`roc_curve`/`roc_auc` (threshold-swept binary-classification evaluation reusing `precision`/`recall`'s thresholding convention, AUC via the trapezoidal rule); `precision_recall_curve`/`average_precision` (PR-AUC via the same threshold sweep, step-function-integrated — more informative than ROC-AUC on imbalanced data); `IsolationForest` (unsupervised anomaly detection via an ensemble of random isolation trees, normalized path-length scoring `s(x,n)=2^(-E[h(x)]/c(n))`); `spectral_clustering` (graph-Laplacian eigendecomposition via `ms::linalg::eig_sym` followed by `KMeans` on the eigenvector embedding, for non-convex cluster shapes plain KMeans/DBSCAN/agglomerative clustering can't separate); `AdaBoost` (SAMME binary classifier with decision stumps or shallow trees) |
| `image/image.hpp` | `Image` buffer, RGB/HSV conversion, resize/crop/flip/rotate, Gaussian/median/bilateral filters, Sobel/Canny/Laplacian/Roberts/LoG edges, morphology (incl. `imgradient_morph`), Otsu/adaptive threshold, histogram equalisation (`histeq`) and CLAHE (`adapthisteq`), Harris/Shi-Tomasi corners, Radon/inverse-Radon (`iradon`), connected components, `hough_lines` (standard Hough transform for straight-line detection via a `(rho,theta)` accumulator with local-maximum peak detection, composable with `canny`); `hough_circles` (circle Hough transform via a `(cx,cy,r)` accumulator with peak non-max-suppression); `watershed` (classic immersion watershed segmentation on grayscale with marker seeds); `slic` (SLIC superpixel segmentation) |
| `compress/compress.hpp` | `run_length_encode`/`run_length_decode`, Huffman, LZ77, LZW, BWT/MTF, delta coding, bzip2-like pipeline, bit-pack helpers, `arithmetic_encode`/`arithmetic_decode` range coding, plus `wavelet_compress`/`wavelet_decompress` (single-level Haar DWT lossy coding); `golomb_rice_encode`/`golomb_rice_decode` (Golomb-Rice entropy coding for geometric-like-distributed non-negative integers); `ans_encode`/`ans_decode` (asymmetric numeral systems entropy coding) |

## GPU & distributed (optional)

CUDA GPU kernels (when `MS_ENABLE_CUDA=ON`) and MPI distributed linear algebra (when `MS_ENABLE_MPI=ON`).

| Header | Description |
|--------|-------------|
| `cuda/buffer.hpp` | `DeviceBuffer`, `PinnedBuffer`, and `StreamPool` GPU memory management; `StreamPool::acquire()`/`release()` return non-null CUDA streams when `MS_HAS_CUDA=1` |
| `cuda/blas.hpp` | GPU `matmul` for dense `Matrix<double>` |
| `cuda/fft.hpp` | GPU `fft` / `ifft` |
| `cuda/solver.hpp` | GPU `solve` via cuSOLVER GETRF/GETRS; `lu()` via cuSOLVER GETRF (returns `(L, U, P)`) |
| `cuda/elementwise.hpp` | In-place `add_inplace` and `fill` on device spans |
| `cuda/sparse.hpp` | COO sparse matrix-vector multiply on GPU |
| `cuda/nvml.hpp` | NVML `DeviceStats` and utilization queries; `device_stats()` populates `memory_total_bytes`/`memory_used_bytes` from `cudaMemGetInfo` (NVML optional for utilization/name); `device_memory_free` (derived `total - used` with underflow guard) |
| `cuda/nccl.hpp` | NCCL availability and communicator size helpers; `allreduce_sum`/`max`/`min`/`prod`/`avg`, `broadcast`, `reduce` — stub-safe identity when `MS_HAS_NCCL=0` or comm size 1 |
| `distributed/mpi_context.hpp` | `MPIContext` init/finalize, rank/size, `allreduce_sum`/`allreduce_max`/`allreduce_min`, `barrier` |
| `distributed/dist_matrix.hpp` | `DistMatrix` local shards; `scatter`, `gather`, `combine_gather` |
| `distributed/block.hpp` | Block and block-cyclic row partitioning helpers |
| `distributed/matmul.hpp` | Distributed GEMM `matmul(A, B, ctx)` — row-block scatter + local GEMM (stub-safe single-rank) |
| `distributed/linalg.hpp` | Distributed wrappers for `eig_sym`, `svd`, `lu` via gather-on-root |
| `distributed/solve.hpp` | Distributed linear solve `A x = b` (gather-on-root → CPU `ms::solve` today) |

**REPL bindings (MPI / distributed):** `mpi` (status dump), `mpi_rank()`, `mpi_size()`, `mpi_allreduce_sum(x)`, `dist_solve(A,b)`, `dist_matmul(A,B)` — stub-safe when `MS_ENABLE_MPI=OFF` (rank 0, size 1, local solve/matmul).

**REPL bindings (CUDA / NCCL):** `cuda_allreduce_sum` / `max` / `min` / `prod` / `avg`, `cuda_broadcast`, `cuda_reduce` — stub-safe identity when `MS_HAS_NCCL=0` or comm size 1.

## Interpreter & tooling

REPL interpreter, plotting, JIT backends, and compile-time unsafe-site profiling.

| Header | Description |
|--------|-------------|
| `interp/repl_engine.hpp` | `Interpreter` REPL session, matrix/scalar state, plot series, save/load; `list_session_objects()` public handle enumeration |
| `interp/plot_console.hpp` | ASCII plot previews for console REPL (`format_plot_preview`) |
| `interp/jit_backend.hpp` | `JitBackend` abstraction (`Repl` / `OrcJit`); `create_backend()` factory; optional `-DMS_BUILD_JIT=ON` links LLVM ORC LLJIT |
| `interp/jit_backend_impl.hpp` | ORC JIT factory implementations (`create_orc_jit_stub_backend`, `create_orc_jit_llvm_backend` when `MS_JIT_LLVM` is defined) and `JitDispatchStats` counters; included by JIT-enabled builds. |
| `unsafe/unsafe.hpp` | `MS_UNSAFE(reason)` attribute macro for justified unsafe sites; full enforcement when built with `MS_BUILD_PLUGIN`. `UnsafeRegistry` collects annotated sites at compile time and emits `${CMAKE_BINARY_DIR}/ms-unsafe-audit.json` (and `.jsonl`) when `MS_BUILD_PLUGIN=ON` |

### REPL scalar expressions

Assignments of the form `name = <expr>` support:

- Literals and scalar variables (`x = 3.14`, `y = x`)
- Binary operators with standard precedence: `+`, `-`, `*`, `/` (e.g. `1 + 2 * 3` → 7)
- Parentheses: `(1 + 2) * 3` → 9
- Unary libm calls: `sin`, `cos`, `sqrt`, `exp`, `log`, …
- Two-argument libm calls: `pow(x, 2)`, `min(a, b)`, `max(a, b)`, `atan2(y, x)`

Plot commands: `plot`, `scatter`, `hist`, `imshow`, `spy`, `surf`; `show` redisplays ASCII preview; `saveplot <file>` writes ASCII preview to disk (GUI **Export Plot as PNG** when `MS_BUILD_GUI=ON`; GUI REPL input supports **Up-arrow / Down-arrow command history** with draft recall; **Wave 233 GUI**: script-editor syntax highlighting, window/splitter layout persistence, variable inspector panel, red error output, **Stop** cooperative cancel, status-bar GPU name and free/total memory; **Wave 238 GUI**: **Find in Output** (**Ctrl+F** / **F3**), **View → Show Plot Panel** toggle, **File → Export Command History…**). Session meta-commands: `export history <file>`, `save_history <file>` (Wave 238). CLI: `mathscriptc` script runner (executes .ms files as REPL command sequences); `mathscript-repl -e`, `--load`, `--jit`. Matrix assignment: `C = matmul(A, B)`, `x = solve(A, b)`, `T = transpose(A)`, `L = chol(A)`. Multi-target: `L, U, P = lu(A)`, `Q, R = qr(A)`, `U, S, V = svd(A)`, `D, V = eig_sym(A)`. Scalar from matrix: `d = det(A)`, etc. Session `save`/`load` persists scalars, matrices, plot state, and command history.

### REPL bindings

Most C++ library modules are header-only; the REPL exposes a subset as matrix/scalar call bindings below.

**Linear algebra (matrix assignment):**

| Call | Description |
|------|-------------|
| `pinv(A)` | Moore–Penrose pseudo-inverse |
| `null(A)` | Null-space basis (wide-matrix safe via `A^T A` eigenvectors) |
| `orth(A)` | Orthonormal column basis (QR) |
| `kron(A, B)` | Kronecker product; nested operands supported (e.g. `kron(eye(2), eye(2))`) |
| `repmat(A, p, q)` | Tile matrix `p×q` |
| `linspace(a, b, n)` | Equally spaced `n×1` column vector |

**Image, compression, ML, and bignum (matrix/scalar assignment):**

| Call | Description |
|------|-------------|
| `rgb2gray(M)` | `(H·W)×3` RGB rows → grayscale column vector |
| `rgb2hsv(M)` | `(H·W)×3` RGB rows → `H×3` HSV matrix (Wave 234) |
| `sobel(M)`, `prewitt(M)`, `scharr(M)`, `roberts(M)`, `imgaussfilt(M, σ)`, `laplacian(M)`, `histeq(M)`, `sharpen(M)`, `threshold_otsu(M)`, `imresize(M, r, c)` | Grayscale image ops on `H×W` matrices |
| `imdilate(M, k)`, `imerode(M, k)`, `imopen(M, k)`, `imclose(M, k)` | Binary/grayscale morphology on `H×W` matrices (Wave 234) |
| `rle_encode_vec(M)`, `rle_decode_vec(M)` | RLE on flattened matrix bytes |
| `delta_encode_vec(M)`, `delta_decode_vec(M)` | Delta coding on flattened bytes |
| `ml_linear_fit(X, y)`, `ml_linear_predict(X, model)` | Ordinary least-squares regression fit/predict (Wave 234) |
| `ml_ridge_fit(X, y, alpha)`, `ml_ridge_predict(X, model)` | Ridge regression fit/predict (Wave 234) |
| `ml_logistic_fit(X, y)`, `ml_logistic_predict(X, model)` | Binary logistic regression fit/predict (Wave 234) |
| `ml_accuracy(p, t)`, `ml_rmse(p, t)`, `ml_mse(p, t)`, `ml_r2(p, t)`, `ml_f1(p, t)`, `ml_precision(p, t)`, `ml_recall(p, t)`, `ml_mae(p, t)` | ML metrics on matching `N×1` vectors |
| `bigint("495")`, `bigint_factorial(n)`, `bigint_fib(n)`, `bigint_gcd("a", "b")` | Bignum parse/ops; results as scalars when representable in `double` |

**Symbolic (string assignment):**

| Call | Description |
|------|-------------|
| `sym_diff("expr", "var")` | Symbolic differentiation of parsed expression w.r.t. variable; returns simplified string |
| `sym_simplify("expr")` | Constant-fold and algebraic simplification of parsed expression |
| `sym_integrate("expr", "var")` | Symbolic integration (power rule, linearity, sin/cos; unsupported forms documented) |
| `sym_eval("expr", "var=value")` | Numeric evaluation of parsed expression with one variable binding |
| `sym_expand("expr")` | Distribute multiplication over addition/subtraction |
| `sym_collect("expr", "var")` | Combine like terms in a variable |
| `sym_substitute("expr", "var", "replacement")` | Replace a variable with another symbolic expression |
| `sym_limit("expr", "var", point)` | Symbolic/numeric limit at `point` |
| `sym_series("expr", "var", point, order)` | Taylor series expansion to given order |
| `sym_solve_linear("eqs", "vars")` | Solve linear equation(s) for variable(s); semicolon-separated for systems |
| `sym_laplace("expr", "t", "s")` | Laplace transform (time → s-domain) |
| `sym_ilaplace("expr", "s", "t")` | Inverse Laplace transform (s-domain → time) |
| `sym_mellin("expr", "t", "s")` | Mellin transform (t-domain → s-domain; table: `c`, `t^a`, `exp(-a*t)`, `t^n*exp(-a*t)`, `1/(1+t)`) |
| `sym_hankel("expr", "r", "k")` / `sym_ihankel(...)` | Hankel transform MVP (Wave 241) |
| `sym_imellin("expr", "s", "t")` | Inverse Mellin transform (s-domain → t); unsupported → `sym_deriv` sentinel |
| `sym_hankel("expr", "r", "k")` | Hankel transform (r-domain → k-domain; table: `exp(-a*r)`, `r^n*exp(-a*r)`, `1/sqrt(r^2+a^2)`) |
| `sym_ihankel("expr", "k", "r")` | Inverse Hankel transform (k-domain → r); unsupported → `sym_deriv` sentinel |
| `sym_fourier("expr", "t", "omega")` | Fourier transform (time → frequency) |
| `sym_ifourier("expr", "omega", "t")` | Inverse Fourier transform (frequency → time) |
| `sym_ztransform("expr", "n", "z")` | Z-transform (discrete n → z-domain) |
| `sym_iztransform("expr", "z", "n")` | Inverse Z-transform (z-domain → discrete n) |
| `sym_dsolve("rhs", "x", "y")` | Separable first-order ODE `dy/dx = rhs`; returns symbolic solution string (unsupported → `sym_deriv` sentinel) |

**Optimization (scalar assignment, formula-string bridge):**

| Call | Description |
|------|-------------|
| `bfgs("formula", x0)` | BFGS quasi-Newton minimization; env `{x0, x1, ...}` sized to `x0` |
| `lbfgs("formula", x0)` | Limited-memory BFGS |
| `nelder_mead("formula", x0)` | Nelder–Mead simplex |
| `adam("formula", x0[, lr, max_iter])` | Adam adaptive gradient |
| `golden_section("formula", a, b)` | 1-D golden-section search on `[a, b]` |
| `levenberg_marquardt("formula", x0)` | Nonlinear least squares (returns residual norm) |

**Control analysis (scalar/matrix assignment):**

| Call | Description |
|------|-------------|
| `control_poles(num, den)` | Transfer-function pole list |
| `control_zeros(num, den)` | Transfer-function zero list |
| `control_step_info(num, den)` | Step-response metrics (rise/settling/overshoot) |
| `control_step_response(num, den[, t_end[, n_pts]])` | Step response as `N×2` `[t, y]` |
| `control_impulse_response(num, den[, t_end[, n_pts]])` | Impulse response as `N×2` `[t, y]` |
| `control_nyquist(num, den)` | Nyquist curve as `N×2` real/imag matrix |
| `control_lqe(A, C, Q, R)` | Linear quadratic estimator (Kalman) gain matrix (Wave 237) |

**Quantum information (scalar assignment):**

| Call | Description |
|------|-------------|
| `quantum_purity(rho)` | Density-matrix purity Tr(ρ²) |
| `quantum_schmidt_rank(psi, dim_a, dim_b)` | Schmidt rank of bipartite state vector |
| `quantum_uncertainty(psi, A, B)` | Uncertainty product ⟨AB⟩ − ⟨A⟩⟨B⟩ |
| `quantum_grover_optimal_iterations(n_qubits, n_marked)` | Optimal Grover iteration count |

**Finance portfolio/pricing (scalar/matrix assignment, Wave 234):**

| Call | Description |
|------|-------------|
| `finance_min_variance_portfolio(cov)` | Global minimum-variance portfolio weights from `N×N` covariance |
| `finance_max_sharpe_portfolio(cov, mu, risk_free)` | Maximum Sharpe-ratio portfolio weights |
| `finance_efficient_frontier(cov, mu, target_return)` | Efficient-frontier portfolio weights for a target return |
| `finance_max_sharpe(cov, mu, risk_free)` | Maximum Sharpe (tangency) portfolio weights |
| `finance_portfolio_return(weights, returns)` | Portfolio expected return from `N×1` weights and returns |
| `finance_heston_call(S, K, T, r, v0, kappa, theta, sigma_v, rho)` | Heston stochastic-volatility European call price |
| `finance_sabr_call(S, K, T, r, alpha, beta, rho, nu)` | SABR stochastic-volatility European call price (Wave 237) |
| `finance_sabr_put(S, K, T, r, alpha, beta, rho, nu)` | SABR stochastic-volatility European put price (Wave 241) |

**Graph community/centrality (matrix assignment, Wave 234):**

| Call | Description |
|------|-------------|
| `graph_louvain(A)` | Louvain community partition as `K×M` vertex-index matrix |
| `graph_eigenvector_centrality(A)` | Power-iteration eigenvector centrality column |
| `graph_articulation_points(A)` | Articulation points of undirected adjacency `A` as `N×1` column |
| `graph_bridges(A)` | Bridge edges of undirected graph as `E×2` endpoint matrix (Wave 237) |
| `graph_min_cut(A, source, sink)` | Minimum s–t cut value on directed capacity matrix (Wave 237) |
| `graph_transitive_closure(A)` | Boolean reachability closure as `N×N` matrix (Wave 237) |

**CUDA matrix ops (matrix assignment, stub-safe when `MS_ENABLE_CUDA=OFF`):**

| Call | Description |
|------|-------------|
| `cuda_lu(A)` | GPU LU factorization summary (when CUDA available) |
| `cuda_add(A, B)` | Element-wise matrix add on GPU (falls back to CPU) |
| `cuda_allreduce_sum(x)` | NCCL all-reduce sum of scalar `x` (stub: identity when NCCL off or size 1; Wave 237) |

**Crypto, FEM, and CFD (string/matrix assignment):**

| Call | Description |
|------|-------------|
| `crypto_aes128_encrypt_block(key_hex, block_hex)` | AES-128 ECB single-block encrypt; returns hex ciphertext |
| `crypto_aes128_decrypt_block(key_hex, block_hex)` | AES-128 ECB single-block decrypt; returns hex plaintext (Wave 253) |
| `crypto_aes256_encrypt_block(key_hex, block_hex)` | AES-256 ECB single-block encrypt; returns hex ciphertext (Wave 251) |
| `crypto_aes128_cbc_encrypt(key_hex, iv_hex, plain_hex)` | AES-128 CBC encrypt; returns hex ciphertext |
| `crypto_aes128_cbc_decrypt(key_hex, iv_hex, cipher_hex)` | AES-128 CBC decrypt; returns hex plaintext |
| `crypto_aes128_gcm_encrypt(key_hex, iv_hex, aad_hex, plaintext_hex)` | AES-128-GCM seal; returns hex `ciphertext:tag` (Wave 234) |
| `crypto_aes128_gcm_decrypt(key_hex, iv_hex, aad_hex, ciphertext_hex, tag_hex)` | AES-128-GCM open; returns hex plaintext (Wave 234) |
| `crypto_chacha20(key_hex, nonce_hex, counter, data_hex)` | ChaCha20 stream cipher (XOR self-inverse); returns hex output |
| `crypto_chacha20_poly1305_encrypt(key_hex, nonce_hex, aad_hex, plaintext_hex)` | ChaCha20-Poly1305 seal; returns hex `ciphertext:tag` (Wave 235) |
| `crypto_chacha20_poly1305_decrypt(key_hex, nonce_hex, aad_hex, ciphertext_hex, tag_hex)` | ChaCha20-Poly1305 open; returns hex plaintext (Wave 235) |
| `crypto_x25519_shared(priv_hex, pub_hex)` | X25519 ECDH shared secret as hex (Wave 236) |
| `crypto_x25519_keypair(hex_priv32)` | X25519 public key from 32-byte private key hex (Wave 255) |
| `mpi_bcast(x)` | MPI stub broadcast from root 0 (identity when stub; Wave 254) |
| `geo_triangulate_polygon(P)` / `geo_convex_hull_3d(P)` | Polygon triangulation / 3D hull face indices (Wave 255) |
| `geo_kdtree_knn(P,x,y,k)` / `geo_kdtree_range(P,x,y,r)` | 2D KD-tree kNN / range query index columns (Wave 255) |
| `geo_intersect_seg_seg(...)` / `geo_intersect_ray_sphere(...)` / `geo_intersect_ray_aabb(...)` | Intersection tests → 1/0 (Wave 255) |
| `graph_katz_centrality(A)` / `graph_laplacian(A)` / `graph_adjacency_spectrum(A)` | Katz / Laplacian / spectral radius (Wave 255) |
| `graph_algebraic_connectivity(A)` | Fiedler value scalar (Wave 255) |
| `stats_linear_regression(x,y)` / `stats_pacf` / `stats_kde` / `stats_bootstrap_ci` | Regression / PACF / KDE / bootstrap CI (Wave 256) |
| `stats_shapiro_wilk` / `stats_mann_whitney_u` / `stats_one_way_anova` / `stats_wilcoxon_signed_rank` | Hypothesis tests (Wave 256) |
| `graph_normalised_laplacian(A)` / `graph_modularity(A,C)` / `graph_eccentricity(A)` / `graph_is_strongly_connected(A)` | Structure metrics (Wave 256) |
| `geo_kdtree_3d_nearest` / `geo_intersect_ray_tri` / `geo_dist_point_plane` / `geo_dist_point_seg3d` | 3D geo queries (Wave 256) |
| `imflip` / `imrotate90` / `threshold_binary` / `adapthisteq` | Image transforms (Wave 256) |
| `stats_friedman` / `stats_ks_2sample` / `stats_jarque_bera` / `stats_ljung_box` | Nonparametric / normality / residual tests (Wave 257) |
| `stats_levene` / `stats_bartlett` / `stats_fligner` | Variance homogeneity tests (Wave 257) |
| `label_components` / `watershed` / `slic` | Segmentation (Wave 257) |
| `hough_lines` / `hough_circles` / `harris` / `shi_tomasi` | Hough + corners (Wave 257) |
| `prob_*_pdf/cdf/ppf` extensions | Lognormal/weibull/etc. + beta/gamma/f CDF (Wave 257) |
| `imtophat` / `imbothat` / `imadjust` / `imhist` | Morphology + histogram (Wave 258) |
| `radon` / `iradon` / `gray2rgb` / `impad` | Radon + color/pad (Wave 258) |
| `sqrtm` / `logm` / `tril` / `triu` / `cosm` / `sinm` | Matrix functions (Wave 258) |
| `graph_dijkstra` / `graph_bellman_ford` | Shortest paths → Nx2 [dist,parent] (Wave 258) |
| `info_permutation_entropy` / `info_transfer_entropy` | Time-series information measures (Wave 258) |
| `stats_partial_correlation` / `stats_weighted_mean` / `stats_trimmed_mean` / `stats_arfit` / `stats_multiple_regression` | Correlation / means / regression (Wave 258) |
| `stats_vif` / `stats_variance_inflation_factor` | VIF for column `j` of design matrix `X` (Wave 260) |
| `hess(A)` / `schur(A)` / `T, Q = schur(A)` | Hessenberg / Schur (Wave 259) |
| `finance_efficient_frontier` / `finance_max_sharpe` | Mean-variance portfolios (Wave 259) |
| `geo_bezier_eval` / `geo_catmull_rom` / `geo_bspline_eval` | Curve evaluation → 1×2 (Wave 259) |
| `geo_bezier_deriv` / `geo_hermite_curve` | Bezier derivative / Hermite point → 1×2 (Wave 260) |
| `combo_binomial` / `combo_bell_num` / `numthy_factor` | Combinatorics / factorization (Wave 259) |
| `numthy_divisors` / `numthy_divisors_vec` | Sorted divisors as `N×1` (Wave 260) |
| `fft_goertzel(x,f,fs)` | Single-bin Goertzel DFT as 1×2 `[re,im]` (Wave 260) |
| `bidiag(A)` / `U, B, V = bidiag(A)` | Bidiagonalization (Wave 260) |
| `solve_sylvester(A,B,C)` | Sylvester equation `A*X + X*B = C` (Wave 260) |
| `minres(A,b)` | MINRES iterative solve (Wave 260) |
| `cg(A,b)` | Conjugate gradient iterative solve (Wave 261) |
| `gmres(A,b)` | GMRES iterative solve (Wave 261) |
| `jacobi(A,b)` | Jacobi iterative solve (Wave 261) |
| `imgradient_morph` | Morphological gradient (Wave 259) |
| `geo_point_in_aabb(px,py,minx,miny,maxx,maxy)` | 1 if point inside 2D AABB else 0 (Wave 254) |
| `geo_overlap_aabb(...)` | 1 if 3D AABBs overlap else 0 (Wave 254) |
| `signal_deconv(y,b)` | Polynomial deconvolution of column vectors (Wave 254) |
| `signal_lms(x,d,filter_length,mu)` / `signal_lms_weights(...)` | LMS adaptive filter output/error or final weights (Wave 254) |
| `signal_czt(x,m,w_re,w_im,a_re,a_im)` / `signal_czt_zoom(...)` | Chirp Z-Transform / zoom-FFT as M×2 `[re,im]` (Wave 254) |
| `crypto_hkdf_sha256(hex_ikm, hex_salt, hex_info, len)` | HKDF-SHA256 extract/expand; returns `len` bytes as hex (Wave 238) |
| `crypto_pbkdf2_sha256(hex_pass, hex_salt, iter, dklen)` | PBKDF2-HMAC-SHA256; returns `dklen` bytes as hex (Wave 239) |
| `crypto_ed25519_keypair(hex_seed)` | Ed25519 public key from 32-byte seed (hex out; Wave 240) |
| `crypto_ed25519_sign(hex_seed_or_sk, hex_msg)` | Ed25519 signature (hex; Wave 240) |
| `crypto_ed25519_verify(hex_pub, hex_msg, hex_sig)` | Ed25519 verify → `1`/`0` (Wave 240) |
| `fem_poisson2d(nx, ny)` | 2D P1 Poisson solve on unit square (`f=1`, zero Dirichlet); returns solution matrix |
| `fem_poisson3d(nx, ny, nz)` | 3D P1 Poisson solve on unit cube (`f=1`, zero Dirichlet); returns solution column (Wave 236) |
| `cfd_advection2d(nx, ny, vx, vy, cfl, dt)` | 2D structured FVM upwind advection final field |
| `cfd_advection3d(nx, ny, nz, vx, vy, vz, t_end, dt)` | 3D structured FVM upwind advection final field (Wave 236) |

**MPI / distributed (scalar/matrix assignment):**

| Call | Description |
|------|-------------|
| `mpi` | Print MPI backend status (stub: `backend=stub rank=0 size=1`) |
| `mpi_rank()` | Current MPI rank (stub: 0) |
| `mpi_size()` | MPI world size (stub: 1) |
| `mpi_allreduce_sum(x)` | All-reduce sum of scalar `x` (stub: identity) |
| `dist_solve(A, b)` | Distributed linear solve `A x = b` (stub: local gather + `ms::solve`) |
| `dist_matmul(A, B)` | Distributed matrix multiply — row-block local GEMM (stub-safe single-rank; Wave 236) |
| `dist_cg(A, b)` | Distributed conjugate-gradient linear solve (stub: gather + CG on rank 0; Wave 238) |
| `dist_gmres(A, b)` | Distributed GMRES (stub gather or MPI block; Wave 239/240) |
| `dist_jacobi(A, b)` | Distributed Jacobi (stub gather or MPI block; Wave 240) |
| `dist_bicgstab(A, b)` | Distributed BiCGSTAB (stub gather; Wave 241) |
| `dist_minres(A, b)` | Distributed MINRES (stub gather; Wave 242) |
| `dist_qmr(A, b)` | Distributed QMR (stub gather; Wave 243) |
| `dist_tfqmr(A, b)` | Distributed TFQMR (stub gather; Wave 244) |
| `bicgstab(A, b)` / `qmr(A, b)` / `lsqr(A, b)` / `tfqmr(A, b)` / `lsmr(A, b)` | Local iterative solvers (Wave 240/243/244; `tfqmr` MVP via BiCGSTAB) |
| `cuda_allreduce_max(x)` / `cuda_allreduce_min(x)` | NCCL stub max/min (identity when stub; Wave 242) |
| `crypto_hmac_sha512(hex_key, hex_data)` | HMAC-SHA512 digest as hex (Wave 243) |
| `crypto_pbkdf2_hmac_sha512(hex_pass, hex_salt, iter, dklen)` | PBKDF2-HMAC-SHA512 (Wave 244) |
| `crypto_hkdf_sha512(hex_ikm, hex_salt, hex_info, len)` | HKDF-SHA512 (Wave 245) |
| `crypto_aes256_cbc_encrypt` / `crypto_aes256_cbc_decrypt` | AES-256-CBC hex I/O (Wave 245) |
| `crypto_aes256_gcm_encrypt` / `crypto_aes256_gcm_decrypt` | AES-256-GCM AEAD hex I/O (Wave 246) |
| `dist_lsmr(A, b)` | Distributed LSMR (stub gather; Wave 246) |
| `cuda_allreduce_prod(x)` | NCCL stub product (identity when stub; Wave 246) |
| `cuda_allreduce_avg(x)` | NCCL stub average (identity when stub; Wave 247) |
| `cuda_broadcast(x)` | NCCL stub broadcast from root 0 (identity when stub; Wave 248) |
| `cuda_reduce(x)` | NCCL stub reduce-to-root (identity when stub; Wave 249) |
| `crypto_constant_time_eq(hex_a,hex_b)` | Constant-time compare → `1`/`0` (Wave 248) |
| `crypto_random_bytes(n)` | Random bytes as hex (MVP; Wave 248) |
| `crypto_sha256(hex_data)` / `crypto_hmac_sha256(hex_key,hex_data)` | SHA-256 / HMAC-SHA256 hex digests (Wave 249) |
| `crypto_sha512(hex_data)` | SHA-512 hex digest (Wave 250) |
| `crypto_aes256_encrypt_block(key_hex,block_hex)` / `crypto_aes256_decrypt_block(...)` | AES-256 ECB single block (Wave 251/252) |
| `crypto_aes128_decrypt_block(key_hex,block_hex)` | AES-128 ECB single-block decrypt (Wave 253) |
| `cuda_allgather(x)` | NCCL stub allgather (Wave 253) |
| `mpi_barrier()` | MPI stub barrier (Wave 252) |
| `signal_sosfilt(sos,x)` | Second-order sections filter (Wave 252) |
| `signal_firwin` / `signal_firwin_highpass` | Windowed-sinc FIR design (Wave 252) |
| `signal_savgol(x,window_length,polyorder)` | Savitzky–Golay smooth (Wave 252) |
| `signal_xcorr` / `signal_xcov` / `signal_autocorr` | Lag-domain correlation (Wave 253) |
| `signal_median_filter(x,window_length)` | Odd-window median filter (Wave 253) |
| `signal_conv2(A,K)` | 2D convolution (Wave 253) |
| `signal_median_filter(x,window_length)` | Sliding-window median filter (Wave 253) |
| `cuda_nccl_available()` / `cuda_nccl_comm_size()` / `cuda_nccl_device_count()` | NCCL stub introspection (Wave 250) |
| `mpi_allreduce_max(x)` / `mpi_allreduce_min(x)` | MPI stub max/min (Wave 251) |
| `signal_coherence(x,y,fs,nperseg)` | Magnitude-squared coherence Nx2 (Wave 250) |
| `signal_filtfilt(b,a,x)` / `signal_filter(b,a,x)` | Zero-phase / causal IIR apply (Wave 250/251) |
| `signal_cheby1(order,rp_db,cutoff,fs[,type])` | Chebyshev I coeffs 2×N (Wave 251) |
| `geo_convex_hull(P)` | Convex hull vertices K×2 (Wave 251) |
| `dist_lsqr(A, b)` | Distributed LSQR (stub gather; Wave 249) |
| `signal_resample(x,p,q)` / `signal_decimate(x,q)` / `signal_interpolate(x,p)` | Rational resampling (Wave 249) |
| `geo_clip_polygon(A,B)` | Clip Nx2 subject against Mx2 convex window → Kx2 (Wave 248) |
| `signal_upsample(x,n)` / `signal_downsample(x,n)` | Integer rate change (Wave 248) |
| `graph_maximum_matching(A)` | Edmonds blossom matching → Mx2 edges (Wave 246) |
| `signal_unwrap(x)` | NumPy-style phase unwrap (Wave 247) |
| `geo_minkowski_sum(A,B)` | Convex Minkowski sum → Mx2 (Wave 247) |
| `geo_min_bounding_rect(P)` | Oriented bounding rect `[cx;cy;w;h;angle_rad]` (Wave 247) |
| `signal_instantaneous_phase(x)` | Analytic-signal phase column (Wave 243) |
| `signal_spectrogram(x, fs)` | STFT magnitude matrix (Wave 244) |
| `finance_heston_put(S,K,T,r,v0,kappa,theta,sigma_v,rho)` | Heston put via put-call parity (Wave 244) |
| `geo_poly_union` / `geo_poly_intersect` / `geo_poly_diff` | Convex polygon boolean ops (Wave 245) |
| `graph_k_core_decomposition(A)` / `graph_k_core_subgraph(A,k)` / `graph_chromatic_number(A)` | Graph core / coloring (Wave 245) |
| `signal_cheby2(order, rs_db, cutoff, fs)` | Chebyshev Type II IIR design; returns `[b; a]` rows (Wave 241) |
| `signal_periodogram(x, fs)` / `signal_welch_psd(x, fs, nperseg)` | PSD helpers; freq/power columns (Wave 241) |
| `signal_envelope(x)` / `signal_hilbert(x)` / `signal_instantaneous_freq(x, fs)` | Analytic-signal helpers; envelope column, Hilbert N×2 [re,im], inst. freq column (Wave 242) |

**Session meta-commands (Wave 238):**

| Command | Description |
|---------|-------------|
| `export history <file>` | Write REPL command history to a text file (one line per command) |
| `save_history <file>` | Alias for `export history` |

**ODE formula-string bindings (scalar IVP):**

| Call | Description |
|------|-------------|
| `ode_euler("expr", t0, y0, t_end, steps)` | Forward Euler; `expr` parsed once via `sym_parse`, evaluated per step with `{t, y}` bindings |
| `ode_rk4("expr", t0, y0, t_end, steps)` | Classical RK4 with parsed formula RHS |
| `ode_midpoint("expr", t0, y0, t_end, steps)` | Explicit midpoint (RK2) with parsed formula RHS |
| `ode_rk45("expr", t0, y0, t_end, rtol, atol)` | Adaptive Dormand-Prince RK45 with parsed formula RHS |
| `ode_backward_euler("expr", t0, y0, t_end, steps)` | Implicit backward Euler (Newton iteration) with parsed formula RHS |

**ODE formula-string bindings (extended):**

| Call | Description |
|------|-------------|
| `ode_bdf2("expr", t0, y0, t_end, steps)` | A-stable BDF2 multistep stiff scalar solver; first step bootstrapped via BDF1; parsed formula RHS |
| `ode_verlet("accel_expr", t0, q0, v0, t_end, steps)` | Velocity Verlet for second-order q''=a(t,q); `accel_expr` evaluated with `{t, q}` bindings |
| `ode_verlet_vec("a0; a1; ...", t0, q0_vec, v0_vec, t_end, steps)` | Vector velocity Verlet; semicolon-separated acceleration formulas with `{t, q0..qN-1}` env |
| `ode_euler_vec("f0; f1; ...", t0, y0_vec, t_end, steps)` | Forward Euler on vector systems; per-component formulas with shared `{t, y0..yN-1}` env |
| `ode_rk4_vec("f0; f1; ...", t0, y0_vec, t_end, steps)` | Classical RK4 on vector systems with parsed per-component formulas |
| `ode_rk45_vec("f0; f1; ...", t0, y0_vec, t_end, rtol, atol)` | Adaptive Dormand-Prince RK45 on vector systems with parsed per-component formulas |

**Framework and spectral REPL bindings:**

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

**Session-object registry:**

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

**Session-object registry (`DifModel` / `consensus::Cluster`):**

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

**ODE formula-bridge bindings (remaining):**

| Call | Description |
|------|-------------|
| `ode_backward_euler_vec("f0; f1; ...", t0, [y0], t_end, steps)` | Vector implicit backward Euler; semicolon-separated per-component formulas, `{t, y0..yN-1}` env |
| `ode_dae_index1("f0; ...", "g0; ...", t0, [y0], [z0], t_end, steps)` | Semi-explicit index-1 DAE: differential formulas `f` and algebraic-constraint formulas `g`, both see `{t, y0..yN-1, z0..zM-1}` |
| `ode_bvp_shooting("f_expr", t0, y_a, t_end, y_b, steps)` | Second-order BVP `y''=f(t,y,y')` via shooting; formula env `{t, y, yp}` |
| `ode_dde_fixed_step("f_expr", "history_expr", t0, t_end, tau, steps)` | Scalar DDE; `f_expr` env `{t, y, ydelay}`, `history_expr` env `{t}` |
| `ode_event_detect("f_expr", "event_expr", t0, y0, t_end, steps)` | Scalar IVP with root-crossing event detection; both formulas env `{t, y}` |

**Additional pure-function bindings:**

| Call | Description |
|------|-------------|
| `gamma_cdf(x, shape, scale)` | Gamma distribution CDF |
| `beta_pdf(x, alpha, beta)` / `beta_cdf(x, alpha, beta)` | Beta distribution PDF/CDF |
| `f_pdf(x, d1, d2)` / `f_cdf(x, d1, d2)` | F distribution PDF/CDF |
| `kruskal_wallis(groups)` | Non-parametric H-test; `groups` matrix rows = groups (semicolon-separated) |
| `cmaes("formula", x0, sigma0, max_iter[, seed])` | CMA-ES global optimizer via the `sym_parse` formula bridge; env `{x0, x1, ...}` sized to `x0` |
| `ifft2(M)` | Inverse 2D FFT; result matrix follows the existing `fft_fft2`/`fft_ifft` `[re, im]`-pair convention |
| `idst2(M)` | Inverse 2D DST; real-valued result, same layout as `fft_dst2` |

`reconstruct_cp`/`reconstruct_tucker` remain unbound as pure functions: their `CPDecomposition`/`TuckerDecomposition` struct inputs have no REPL representation. Use the `tensorops_decompose_*` / `tensorops_reconstruct_*` session-handle bindings instead.

**Tensor decomposition (session-object registry):**

| Call | Description |
|------|-------------|
| `tensorops_decompose_cp(handle, T, rank[, max_iter, tol])` | CP/CANDECOMP-PARAFAC decompose a 2D matrix `T`; stores the result under `handle` |
| `tensorops_decompose_tucker(handle, T, ranks[, max_iter, tol])` | Tucker (ALS) decompose; `ranks` is a bracket vector, e.g. `[2, 2]` |
| `tensorops_decompose_hosvd(handle, T, ranks)` | Tucker via HOSVD decompose |
| `tensorops_reconstruct_cp(handle)` | Rebuild the approximated matrix from a stored CP handle |
| `tensorops_reconstruct_tucker(handle)` | Rebuild the approximated matrix from a stored Tucker/HOSVD handle |

## Research frameworks

Higher-level research frameworks built on the core library (GP search, temporal memory, uncertainty modeling, information theory, and cryptographic/MPC utilities).

| Header | Description |
|--------|-------------|
| `frameworks/axiom/axiom.hpp` | Evolutionary `Axiom` GP search; `Algorithm` holds `Sym` representation; `evaluate` genuinely evaluates `algo.representation` row-by-row against input data columns (`x0`, `x1`, …) via `Sym::eval`; `evolve` performs real GP tree evolution (internal `GPNode` population with random initialization, tournament selection, subtree crossover/mutation, elitism, synced to `representation` each generation); `PrimitiveRegistry::build_from_ms_namespace()` supplies the GP tree's unary-function pool, scoped to `ms::Sym`'s scalar grammar (`sin`/`cos`/`exp`/`log`/`sqrt`/`tanh`) and consulted at every tree-generation site; `mse_fitness`/`rmse_fitness` (supervised regression GP fitness over a dataset) |
| `frameworks/cellai/cellai.hpp` | `CellMemory` temporal memory, `hebbian_update`, `energy` (Boltzmann -vᵀWh), `boltzmann_weights` (numerically-stable Gibbs probabilities over an energy vector), `CellMemory::consolidate` short→long-term decay, `cell_to_cypha_features` |
| `frameworks/cypha/cypha.hpp` | `DifModel` mixture-of-experts with NIG uncertainty; `nig_fit`/`nig_pdf`/`nig_cdf`/`nig_sample`, `predict`, `predict_interval`, `ood_score`, `gh_gate`; `nig_mean`/`nig_variance` (closed-form NIG moments, O(1) alternative to numerical integration or Monte Carlo via `nig_sample`) |
| `frameworks/gria/gria.hpp` | Information-theoretic `entropy`/`compute_alpha`/`matrix_alpha`/`is_critical`/`classify`, GF(2^n), cellular automata (`ca::step`, `ca::langton_lambda`, `ca::hamming_distance`, `ca::divergence_trajectory`), LFSR utilities |
| `frameworks/izaac/izaac.hpp` | VRF `CSPRNG` (xoshiro256**), session seeding, `rand_matrix`/`randn_matrix`, `mc::estimate_pi`, `estimate_pi`; `bloom::BloomFilter`, `ratelimit::TokenBucket`, `diffpriv::laplace_mechanism`/`gaussian_mechanism`/`exponential_mechanism` (discrete DP selection), `backtest::simulate_gbm_path`/`run_backtest`; `crypto::encrypt`/`decrypt` (`CipherText` CSPRNG keystream XOR + keyed tag, demo/internal use only), `mpc::split_secret`/`reconstruct_secret` (`Share`, Shamir k-of-n over prime field `PRIME`); `consensus::Cluster` (in-memory Raft-style election/replication simulation: `run_election`, `replicate`, `current_leader`; demo/testing only, not networked production consensus); `fuzz::mutate` (deterministic CSPRNG-seeded bounded byte-buffer mutation for fuzz-corpus seed expansion) |
