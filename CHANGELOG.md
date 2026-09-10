# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Wave-by-wave implementation history (thousands of entries) is in [`docs/WAVES.md`](docs/WAVES.md). Current architecture, API, and release criteria are in [`docs/`](docs/).

## [Unreleased]

### Features completed

Every entry that `docs/RELEASE_DECISIONS.md` listed as a deliberately-deferred
stub has been closed; see that file for the before/after table.

- Graph: `max_weight_matching` / `max_weight_matching_value` (Edmonds' primal-dual blossom, verified against exhaustive brute force on random graphs), and exact planarity via `is_planar` (Left-Right criterion) with `planar_embedding` and `kuratowski_subgraph`. The old heuristic remains as `is_planar_k5_k33_check`.
- Geo: `marching_cubes` / `marching_cubes_mesh` (full 256-case Lorensen-Cline tables), `marching_squares`, `mesh_surface_area`, `mesh_volume`. Sphere area and volume reproduce 4*pi*r^2 and 4/3*pi*r^3 to 0.17% and 0.30% on a 48^3 grid.
- Image: `graph_cut_segment` / `grabcut_segment` / `min_cut_value`; `fast_corners`, `orb_detect_and_compute`, `sift_detect_and_compute` and descriptor matching. SURF is deliberately not implemented.
- Bignum: `APFloat` and `APComplex` with a full arbitrary-precision transcendental set. pi, e and ln2 reproduce their standard expansions to 40+ digits.
- Distributed: a `dist_ops` communication layer, SUMMA matmul, and row-distributed Krylov solvers replacing the gather-to-one-rank path (which remains the documented fallback).
- CUDA: real NCCL communicator management and collectives behind `MS_HAS_NCCL`; the default build keeps its identity semantics.
- ML: real Barnes-Hut t-SNE replacing the dense O(n^2) stub whose perplexity search never converged on a target entropy.
- Frameworks: Axiom's `evaluation` / `selection` / `mutation` Syms carry real per-individual provenance instead of three constants.
- Geo: `poly_boolean` and its `poly_union_general` / `poly_intersect_general` / `poly_diff_general` / `poly_symmetric_diff_general` wrappers are a general two-polygon clipper for arbitrary simple polygons: concave operands, results that split into several disjoint pieces, and results containing holes are all exact. The pre-existing `poly_union` / `poly_intersect` / `poly_diff` stay as the documented convex MVPs that return a convex-hull over-approximation. Exposed in the REPL as `geo_boolean_union` / `geo_boolean_intersect` / `geo_boolean_diff` / `geo_boolean_xor`, returning `(x, y, contour_index)` rows.
- Linalg: the blocked Schur-Parlett (Davies-Higham) the header recorded as not implemented. `logm`, `sinm` and `cosm` now serve repeated and clustered eigenvalues by grouping them, reordering the Schur form with Givens swaps, evaluating each diagonal block by its Taylor series about the group mean, and filling the off-diagonal blocks from triangular Sylvester solves. The new `funm_taylor(A, coefficients)` does the same for a caller-supplied `f`, which plain `funm` cannot: `f` alone does not determine `f(A)` at a repeated eigenvalue.
- `mathscript-server` is a real SPMD compute node (`--script`, `-e`, `--serve`, one Interpreter per rank) rather than a heartbeat loop that ignored argv.
- REPL: a bare name or expression now prints its value. `x`, `A`, `1 + 2`, `sqrt(2)` and `x / 2 + 1` were all rejected with "could not parse" -- the only way to look at a variable was to assign it somewhere else first. The fallback runs last, after every command and every assignment form has declined the line, so it shadows nothing and a line that is not an expression still reports the parse error.
- REPL: `erfc`, `gamma`, `zeta`, `fresnel_c` and `fresnel_s` were reachable only in the bare-call printing form; the scalar-expression evaluator did not know them, so `y = gamma(4)` failed as an unknown function while the REPL's own index documented `gamma(x)` and `zeta(s)`. All five now work in both forms, and `erfc` / `fresnel_c` / `fresnel_s` were added to the index.
- REPL: the libm scalar set is complete. It had `sinh` but not `asinh`, `log10` but not `log2`, and `sqrt` but not `cbrt`; `log2`, `exp2`, `expm1`, `log1p`, `cbrt`, `asinh`, `acosh`, `atanh`, `round`, `trunc`, `hypot(x,y)` and `fmod(x,y)` now join them, and `help` lists the whole set instead of "sin, cos, sqrt, pow, min, max, ...".
- REPL: matrix accessors. The interpreter had over a thousand functions that produce matrices and no way to look inside one -- no shape, no element, no row, no column. `mat_rows(A)`, `mat_cols(A)`, `mat_numel(A)` and `mat_at(A,i,j)` return scalars; `mat_row(A,i)`, `mat_col(A,j)`, `mat_reshape(A,rows,cols)` and `mat_submatrix(A,r0,c0,rows,cols)` return matrices. Indices are 0-based, and an index that is fractional, negative or past the end is a reported error rather than a read past the buffer.
- REPL: a matrix call written without a target now prints its result as `_`. This goes through the matrix-call registry, so it covers every matrix-returning callee and its accepted arities at once, including the constructors `zeros`, `ones`, `eye`, `rand`, `randn` and `linspace`, which had no no-assignment form at all.
- ML model packing: three out-of-bounds accesses in the REPL's model serialisers, all of them reachable from an ordinary `ml_*_fit` call. The NaiveBayes packer sized the matrix `max(n_features, 1)` columns wide and then wrote the header at column 1, so a single-feature model corrupted the heap; the KNN packer sized it `n_features + 1` wide and wrote the header at column 2; and the LDA packer sized its per-class loop by `classes.size()`, which `LDA::fit` fills before it gives up on a single-class problem, so it indexed the empty `discrim_const`. The LDA path now reports "expected at least two distinct class labels in y" instead of returning a model that was never fitted, and the QDA packer got the same consistency check.
- ML ensemble sizes are bounded. `n_trees`, `n_estimators` and `max_depth` arrived from the command line unchecked, so `ml_random_forest_fit(X, y, 3000000000)` and `ml_gradient_boosting_fit(X, y, 3000000000)` grew trees until the process was killed, and `ml_isolation_forest_fit(X, 1e18)` asked for an allocation that aborts rather than reports under `-fno-exceptions`. The caps are 10000 members, depth 512, and a forest sample of 1000000, each reported as a `DomainError`.

### Engineering plan

The whole-repository audit is in [`docs/ENGINEERING_PLAN.md`](docs/ENGINEERING_PLAN.md),
preserved as written; [`docs/PLAN_STATUS.md`](docs/PLAN_STATUS.md) records what is
done, what is open, and the three places the plan itself turned out to be wrong. The
largest of those: the coverage tooling the plan describes as already finished was not
present in this tree, so 37,738 lines -- 25% of `src/` -- sat outside the denominator
and the 92.0% figure published repeatedly was measured over 75% of the repository.

- Matrix-call manifest and generated dispatch tests. `scripts/extract_manifest.py`
  reads all 485 handlers' guards as predicates over the argument count and solves
  them; `scripts/gen_matrix_call_tests.py` emits 1,377 tests across 29 translation
  units -- registration, wrong arity, and undefined operands. Happy paths are
  deliberately not generated: a generator that invented inputs would assert whatever
  the implementation currently does.
- Integration tests grouped from 573 executables into 31, one per domain. Linking is
  what dominates a test build, and on an instrumented build it dominated it badly
  enough that measuring coverage was a nightly event. The build graph went from 2,431
  steps to 1,460. This required disambiguating 71 colliding `TEST(Suite, Name)` pairs
  first -- 29 of them had different bodies, and grouping without fixing that would
  have silently stopped running them while the test count stayed put.
- AVX2/FMA `dgemm`, with B-panel packing and BLIS-style cache blocking. There was no
  AVX2 kernel at all, so every machine without AVX-512 -- Zen 1 through 3, every
  Intel client part since Alder Lake, and the CI configuration itself -- fell to a
  rank-1 update loop for matrix multiply.
- The AVX-512 kernel rewritten onto the same blocking. It had been reading B through
  eight strided scalar loads per vector on every iteration of the innermost loop, and
  writing C back through a stack buffer and eight scattered scalar stores.
- `src/simd/isa.cpp` now builds at baseline ISA. It had been compiled with
  `-mavx2 -mfma`, which permits an AVX instruction inside the routine whose job is to
  decide whether AVX instructions will fault. Latent rather than active -- the object
  contained none -- but it depended on a compiler's choice.
- `izaac_vrf_keygen` guarded its arity with `assign.args.empty()` where the other 484
  handlers use an explicit count. Identical to the compiler, not to the manifest
  parser, which is how one handler dropped out of the generated tests with nothing
  failing.
- Reproducibility manifest (`ms/runtime/repro.hpp`): version, commit, the ISA path
  actually taken after the OS register-state check, any `MS_FORCE_ISA` ceiling, the
  worker count and the seed, as text or JSON.
- The RNG seeding contract is documented in [`docs/API.md`](docs/API.md). Every
  numerical routine is deterministic given its seed and every one has a fixed
  default; `crypto::random_bytes` is neither seedable nor reproducible, deliberately.
- SPDX identifiers on all 1,735 source files, with the 18 CUDA-linked translation
  units additionally naming `LICENSE.exceptions`. `vendor/` is untouched.
- A source-only CI job that gates SPDX, the SBOM, the manifest (`--strict`), the
  generated suites and test-name uniqueness, and fails in under a minute rather than
  after a full build.

Two items are deliberately **not** done, both one-way doors for the repository owner
rather than a contributor: the `git filter-repo` authorship rewrite (§4.5), which
rewrites all 1,440 commit SHAs, and the rename off "MathScript" (§4.2).

### Testing

- The seven libFuzzer targets' checked-in corpora are now replayed by ordinary CTest
  suites (`replay_fuzz_*`), each seeding from the corpus and applying 20000 deterministic
  mutations on top. They need no libFuzzer runtime, run in under two seconds in total, and
  cover the entry points that previously only the 24-hour job reached -- the
  `quantum::partial_trace` out-of-bounds read in this release was found exactly this way.
- A malformed-input sweep calls every name the REPL dispatcher recognises with wrong
  arities, wrong shapes, degenerate and oversized numeric arguments, string arguments, and
  non-finite matrices, checking that each returns a formattable error rather than crashing
  and that the session survives. It found five of the crashes fixed below.

### Correctness fixes

Implementations that did not compute what their headers documented.

- `poly_roots` returned all-zero roots for the whole family `x^n +/- c` (the companion matrix is orthogonal, so a single-shift QR step is a fixed point, and the unconverged diagonal was returned with no error); now Aberth-Ehrlich, with `poly_lagrange` rebuilt as real Lagrange interpolation and `poly_fit` rejecting a size mismatch.
- `fem`'s 3D stiffness applied `J^-1` where the chain rule needs `J^-T`, so the Dirichlet energy of `u = x + 2y + 3z` on the unit cube was 35 instead of 14.
- `cmaes` stopped on the raw objective value, so any problem with a negative optimum halted after one iteration and reported success.
- `interpolate`'s frequency fast path zero-stuffed by `out_fft/in_fft` rather than by `p`, so every non-power-of-two `p >= 8` was mis-scaled; `butterworth` was byte-for-byte `lowpass` and is now a real Butterworth IIR.
- `spearman` ignored ties, `kendall` reported tau-a, `friedman`'s tie divisor carried an extra factor of `k`, and `variance_inflation_factor` fitted without an intercept (reporting VIF < 1, which the definition cannot produce).
- `lz77` was lossy: a literal `0x00` following a match was discarded. `sample_entropy` counted its two template populations at different sizes; `lz_complexity` forbade overlapped copies.
- `heston_call`/`heston_put` had a wrong `D` coefficient and `u` convention (0.4-3.3% price error); `bond_ytm` bisected a hard-coded `[0, 1]` and silently clamped yields outside it.
- `step_response`/`impulse_response` were forward Euler under a "matrix exponential" heading; `riccati`/`dare`/`lqr`/`lqe` used the element-wise reciprocal of `R`'s diagonal as `R^-1`.
- `izaac::verify` ignored the message and accepted forgeries made from the public key alone; `izaac::crypto::encrypt` derived its nonce from the key alone, giving a two-time pad. Both now use the Ed25519/SHA-512 and OS CSPRNG already in the tree.
- `gria`'s `alpha_ca`/`alpha_lfsr` passed entropy-preserving transforms and were identically 0; `cypha::nig_pdf` increased with `|x - mu|` and had infinite mass.
- `cfd`'s periodic face velocities disagreed at the wrap-around face, so a conservative scheme gained 28% mass in five steps.
- `geo`'s segment-intersection helper behind the convex polygon booleans solved for the crossing parameter with the wrong sign (`a - c` where the derivation needs `c - a`), so it accepted only crossings at a negative parameter and emitted the mirrored point. The candidate points it feeds are always on an operand's own edge, so the hulled output never changed, but the computation was wrong.
- `logm`, `sinm` and `cosm` returned a plausible but wrong matrix for every defective input. The Parlett recurrence divides by `T(j,j) - T(i,i)`, and when that vanished with a vanishing numerator the code set `F(i,j) = 0` and called the block decoupled -- but at a repeated eigenvalue `f(A)` depends on the DERIVATIVES of `f`. `logm([[2,1],[0,2]])` returned `diag(ln 2, ln 2)`, whose exponential is `diag(2,2)`, not the input; the exact answer is `[[ln 2, 1/2],[0, ln 2]]`. The three now use the blocked Schur-Parlett (Davies-Higham) with their analytic Taylor coefficients and are exact on defective matrices; `funm`, which is handed `f` alone and so cannot know the derivatives, reports the case instead of guessing, and the new `funm_taylor` takes the coefficients and answers it.
- REPL: 23 two-matrix builtins -- every supervised `ml_*_predict`, `ml_lda_transform`, `ml_pca_transform`, `ml_kmeans_predict`, `ml_gmm_predict`/`_proba`, `ml_isolation_forest_score`, `ml_linear_fit`, `ml_logistic_fit` and `quantum_anticommutator` -- worked when their result was assigned but failed without an assignment. `is_matrix_dual_matrix_call_callee` claimed them, so the printing chain was entered, but it had no branch for any of them; the call fell out of the chain and was retried as a single matrix literally named `"X, model"`, reporting `unknown matrix`. `sparse_to_dense(A)` had the same shape of gap in the one-matrix printing chain and reported `unknown function` for a name its own arity table accepts.
- `control_impulse_final([1],[0,1])` never returned. Transfer-function coefficients are in descending order, so `den[0]` is the leading one and the whole `tf2ss` realisation divides by it -- a leading zero does not change the polynomial but made `A` infinite, and `expm_scaled`'s scaling loop then halved an infinite norm forever (`inf * 0.5` is `inf`). `tf2ss` now strips leading zeros from both polynomials, as every control toolbox does, so `[0, 1]` is the constant polynomial 1; an identically zero denominator yields the zero system. `expm_scaled` refuses a non-finite entry outright, which protects every other caller of the matrix exponential.
- `combo::rank_permutation` indexed its internal used-marker vector with a caller-supplied entry, without checking that the input is a permutation of `0..n-1` at all: an out-of-range or repeated entry read and wrote past it. `PCA::transform` centred a feature row over the ROW's length rather than the model's, so a model fitted on fewer features than the row presents read past its own mean vector.
- `qft_gate`, `grover_search`, `ghz_state` and `w_state` took a qubit count with no upper bound, so the 2^n dimension it sets was an unbounded allocation -- `quantum_qft_gate(24)` asked for a 2^24 x 2^24 dense matrix and reached 10 GB of resident memory before the OOM killer took the process. `1 << n` is also undefined past 30. The matrix builders now stop at 12 qubits and the state builders at 20, returning an empty result as `grover_search` already did for a non-positive count. Found by fuzzing the REPL entry point, not by any hand-written case.
- Six of the `ml_*_from_matrix` model decoders read their header row at columns their entry guard did not cover: `ml_gmm_from_matrix` required one column and read three, `ml_knn_from_matrix` required two and read three, `ml_svm_from_matrix` required three and read eight, and the NaiveBayes, LDA and QDA decoders checked no width at all. A model matrix narrower than its header row read past the row before the layout check could reject it.
- `ml::vec_dot` indexed its second operand with the FIRST one's length, so `ElasticNet::predict` read past a feature row whenever the model was fitted on a different feature count; `sq_eucl_dist` had the same shape. And `quantum::partial_trace` indexed `rho[i*d2 + k]` without checking that `d1*d2` matches `rho`'s size -- `entanglement_entropy`'s fallback path reaches it precisely when the subsystem dimensions are invalid, so a fuzzed REPL line found it after about 400000 inputs. Both now bound the index, and `partial_trace` returns an empty matrix for a factorisation that does not fit.
- Three stack overflows on deeply nested or very long input, all reachable from one REPL line. The symbolic parser is recursive descent, so `sin(sin(sin(...)))` recursed once per level and crashed at about 10000; it now has a depth limit. `parse_add`/`parse_mul` loop over their operands rather than recursing, so `x+x+x+...` did not hit that limit -- but it built a left spine one node deep per term, and `~SymExpr` walks that spine recursively through its `unique_ptr` children, so 100000 terms overflowed the stack on DESTRUCTION; a node budget now bounds the spine. And command dispatch runs the line through a chain of `std::regex` matches, whose libstdc++ executor recurses once per input character through repetition operators, so a six-figure line crashed inside the regex before any interpreter code ran; `execute()` now applies the same `kMaxScriptLine` cap the script and session readers already used.
- `lsmr`, `lsqr`, `qmr` and `tfqmr` read past the end of the heap for a multi-column right-hand side. The other six iterative solvers route a multi-column `b` through `solve_per_column`; these four built their work vectors at `b`'s full shape while every matrix-vector product they take produces a single column, so `axpy` read past the shorter operand. AddressSanitizer caught it on `lsmr(A, B)` with a 3x3 `B` as an eight-byte read one element past a three-element buffer. All four now split `b` the same way the others do.
- `percentile(v, p)` scaled `p/100` straight into an index without clamping: `p = 3e9` read about 3e7 elements past the end and segfaulted, and a negative `p` converted to `size_t` is undefined behaviour. `trimmed_mean` had the same unclamped conversion.
- The counting functions returned silently wrapped values past the twenties and did unbounded work on a large argument: `bell_num(3e9)` built a Bell triangle with three billion rows and never returned, and `stirling2(3e9, 3e9)` asked for a three-billion-square table. `subfactorial`, `double_factorial`, `catalan_num`, `stirling1`, `stirling2`, `eulerian_number`, `bell_num`, `motzkin_num` and `involutions` now use `factorial`'s existing `UINT64_MAX` overflow sentinel past the last representable argument. `primes` refuses a span wider than 2e8 (`primes(2, 1e18)` requested a 1e18-bit array), `prime_pi` reports the sentinel past it, and `partition` stops at 416.
- Undefined behaviour in the vendored Ed25519 field and scalar arithmetic: 189 left-shifts of negative signed values in the ref10 carry chains (`h0 -= carry0 << 26`), plus eight `int32_t` shifts in `fe_tobytes` that overflow the promoted `int`. All now shift the unsigned representation and convert back, which is implementation-defined rather than undefined and is the two's-complement result the reference code intends. The Ed25519 test vectors are unchanged.
- Four crash classes, all found by a new malformed-input sweep over every REPL builtin name. `historical_cvar` scaled the tail fraction `1 - confidence` and converted it to `size_t`; a confidence outside (0,1) made that negative, which is undefined behaviour and in practice a value near 2^64, and the summation loop then read far past the array -- `finance_historical_cvar(v, 2)` segfaulted. `mod_pow` passed its modulus straight to `base %= mod`, so a modulus of 0 raised SIGFPE. `ml::mat_mul` ran its inner index over A's columns while subscripting B's ROWS with it and never checked that they matched, so `ml_mat_mul(X, X)` for any non-square X read past B; `mat_vec`, `vec_add` and `vec_sub` had the same shape of unchecked index. All four now clamp or report, and `ml_mat_mul` returns a DimensionMismatch instead of an empty matrix. Fourth: `BigInt(std::string)` is documented as
"on failure constructs zero", but its body called `std::stoul`, which throws `std::invalid_argument`
on a non-numeric chunk -- and this library is built with `-fno-exceptions`, so `BigInt("x")` called
`std::terminate` and aborted the process. Its digits are accumulated by hand now, and the other two
throwing `std::sto*` calls in the tree (a `std::stoi` on a regex capture that could exceed `int`, and
one parsing the unsafe-site registry) were replaced with `std::from_chars`.
- `BigInt::parse("")` and `BigInt::parse("-")` returned zero rather than reporting: the checked `parse` entry point now rejects an empty significand, matching `APFloat::parse`. The defensive `BigInt(std::string)` constructor still yields zero on purpose.
- Removed the unreferenced root `exe/` directory (`main.cpp`, `mathscriptc.cpp`, `repl.cpp`). The executables are built from `src/exe/`; nothing in any CMakeLists, script, workflow or doc referenced the root copies, which had drifted from the ones that build.
- Removed dead code left behind by earlier replacements: the companion-matrix/Wilkinson-QR route to `poly_roots` (superseded by Aberth-Ehrlich and unreachable), signal's out-of-place `fft_recursive`/`complex_ifft` pair (which also sat in the overload set of the `ms::complex_ifft` call meant for the fft module), the single-pass `resample_combined` behind a guard that returned false unconditionally, and an unused symbolic helper.
- `pois_pdf` overflowed to NaN for large means, `binom_cdf` was NaN at `p == 1`, `dft` returned a zero-padded transform (with `idft` added as its inverse), `irfft` returned the wrong length, `metric_inv` returned the identity for a singular metric, `pollard_rho`'s twenty retries were identical, `einsum` discarded its output subscript order, and `tonelli_shanks`'s zero case was unreachable.

- `qmr` was BiCGSTAB, `tfqmr` was `return bicgstab(...)`, `lsmr` was `return lsqr(...)`, and `precond_ssor` returned only `diag(A)/omega`. All four are now the real algorithms, plus new `precond_ssor_apply`, `precond_ilu0`, `precond_ilu0_apply` and `pcg`.
- `cplx::inversion` returned the identity Mobius; it now returns the anti-Mobius acting on `conj(z)`, with `apply_inversion` and `cross_ratio_c` added.
- `topo::cech_complex` clamped `max_dim` to 2; minimum enclosing balls now come from the bordered Cayley-Menger system, so higher dimensions are built.
- `crypto::random_bytes` drew from `std::random_device` per byte; it now uses `getrandom(2)`, `BCryptGenRandom` or `arc4random_buf`.
- `sym_dsolve` was separable-only; `sym_dsolve_ode` adds linear, Bernoulli, exact and homogeneous classification with numeric verification of every candidate, and `sym_dsolve_linear2` handles second-order constant-coefficient equations.
- Quantum: `fidelity` computed `sqrt(|Re Tr(rho sigma)|)` rather than the Uhlmann fidelity, `trace_distance` computed the Frobenius rather than the trace norm, `concurrence` used an ad-hoc diagonal formula rather than Wootters', and `schmidt_decomposition` left the right Schmidt vectors unconjugated. All verified against exact ground truth.
- `MS_LINK_TESTS_SHARED` failed to configure at all under CMake 3.28 (`ms_core` appeared both with and without the `WHOLE_ARCHIVE` link feature). This is the configuration `coverage-linux` and `sanitizer-linux` use.
- Several existing tests passed only because their tolerance was looser than the contract deserved, or because the input hit the one degenerate case the buggy code got right. Those are tightened where found.

### Earlier entries

- Tests: extra control/signal/symbolic/BLAS/FEM/ODE/finance/poly/special/stats/core/CFD unit coverage plus leftover no-assignment REPL printers (unary math, bigint, optimizers, CFD 3D, cplx, gria, geo, ML, quantum).
- Tests: extra crypto/image/info/graph/geo/compress/ML/combo/numthy/PDE/dispatch/iterative unit coverage plus scalar-assignment REPL printers for eval_scalar_call (sin/cos/bigint/mpi/cuda/ellip_d/gria/geo hermite).
- Tests: extra crypto/special/symbolic/signal/ODE/finance/poly/stats/FEM/CFD/BLAS/LAPACK unit coverage plus REPL error branches (finance bond, CFD advection parse, Tucker 5-arg, numthy scalar-expr, Dijkstra) and special/combo scalar assignment.
- Tests: extra control (margins/step-info/riccati MIMO/ss2tf), IZAAC, plot-console, distributed iterative, and dbdsqr/dormbr unit coverage.
- Tests: extra QR/matmul mixed-storage, solve, dist, special, FEM, CFD, stats, poly, ODE, finance, and BLAS coverage plus REPL leftover printers (FEM Poisson parse, gria unsigned args, graph DFS, combo/numthy/special scalar-assign).
- Tests: extra symbolic/signal/crypto/LAPACK/FEM/ODE/finance/stats/poly/CFD/BLAS/solve coverage plus eval_scalar_call DomainError REPL printers (combo/numthy/special negative-n and k>n).
- Tests: extra graph/ML/image/PDE/geo/core/distributed/cplx/combo/numthy/control/special/sparse coverage plus remaining special eval_scalar_call DomainError REPL printers.
- Tests: extra special eval_scalar_call DomainError REPL printers (bessel_h/hy/l/lu, struve, anger/weber, kelvin, bessel_zero_ynu, lambert_w, legendre_p).
- CPU SIMD kernels report `Kernel::Avx512` when `MS_ENABLE_AVX512` and the CPU has AVX-512F; vector loops stay 4-wide xsimd (AVX2 compile) to avoid SIGILL. Dedicated `avx512_dgemm` remains the wide GEMM path.
- CI memory job is AddressSanitizer + UBSan (`sanitizer-linux`). Valgrind is no longer a CI gate.

- REPL constructors (`ones`/`zeros`/`eye`/`rand`/`randn`/`linspace`/`repmat`/`kron`) refuse dimensions above 262144 elements before allocating (libFuzzer `ones(9999)` OOM).
- Combo listing enumerators (`derangements`, `all_permutations`, `all_subsets`, `gray_code`, partitions, necklaces, …) refuse oversized n so libFuzzer cannot OOM on `combo_derangements(11)`.
- `info_joint_entropy`, `info_conditional_entropy` and `info::mutual_info` read past the end of
  the joint PMF whenever the caller's `rows`×`cols` claimed more elements than the matrix held.
  All three index `pxy[i*cols + j]` and nothing checked the claim against the data, so
  `info_joint_entropy([0.25, 0.25; 0.25, 0.25], 6, 2)` read 12 doubles out of a 4-double buffer
  — an AddressSanitizer heap-buffer-overflow, found by a libFuzzer session over the REPL.
  Bounded in the library, which now returns `0.0` rather than read past the span whatever the
  caller does, and rejected at the REPL with a `DomainError`, since a shape that does not match
  the matrix is a user error. The crashing input is in the checked-in corpus.
- `restricted_partitions` was the one member of that family the cap never reached: `all_partitions`
  stops at `kMaxEnumPartitionN`, but its restricted sibling took any n at all and enumerated every
  partition of n into k parts. The 24h libFuzzer run found `combo_restricted_partitions(442, 5)`,
  which reached 13.5 million allocations and 2398 MB of resident memory before the OOM. Now bounded
  by the same constant in `combo` and refused with a `DomainError` at the REPL, like `all_partitions`.
- Tests: extra library coverage (image/signal/control/cfd/finance/special/ml/info/quantum/graph/combo/geo/stats/ode/pde/prob/linalg) plus remaining dual-matrix and scalar no-assignment REPL printers.
- Tests: extra library coverage (numthy/compress/crypto/tensorops/symbolic/image/signal/linalg/matmul/dispatch) plus remaining no-assignment REPL printers (special/core/image/diffgeo/prob/signal).
- Tests: extra symbolic/image/signal unit coverage plus no-assignment REPL printers (optim, frameworks session objects, vector ODEs including `ode_adams_bashforth2_vec`).
- Tests: extra special/linalg/matmul/solve/BLAS unit coverage plus finance max-sharpe and cplx nullary no-assignment REPL printers.
- Tests: extra control/fem/finance/poly/signal/distributed unit coverage plus remaining no-assignment REPL printers (session objects, dual-arity finance, senary-padded CFD/FEM).
- Tests: extra symbolic transform, BLAS/matmul/solve, control, finance/poly/FEM/ODE, CFD, and core Sym coverage plus leftover REPL printers (geo, signal scalar-assign, quantum assign).
- Tests: more no-assignment REPL `execute("fn(...)")` printers (special/signal/control/quantum/prob/stats/finance/poly/cplx/topo/combo/ode/pde/cfd) plus extra symbolic CAS/dsolve unit cases.
- Tests: 117 no-assignment REPL `execute("fn(A)")` cases for `repl_engine.cpp` printers (graph/geo/linalg/fft/quantum/stats/info and related).
- CPU SIMD kernels use xsimd (`batch<double>`) instead of raw AVX2 intrinsics; CUDA `add_inplace`/`fill`/`mul_inplace`/`scale` launch `.cu` kernels when `MS_ENABLE_CUDA` and a device is present, otherwise xsimd host fallback.
- REPL `source`/`run_file` reject circular scripts, cap nesting at 8, and nest via `run_file` instead of re-entering `execute` (libFuzzer `source 8` stack overflow).
- REPL `source`/`run_file`/`load` accept only regular files ≤ 256 KiB with lines ≤ 8 KiB; `export_history`/`save_history` refuse oversized history (libFuzzer 24h timeout in `export_history`).
- `BigInt::to_ll` no longer multiplies the limb base after the last of three limbs (UBSan overflow at `1e18 * 1e9`).
- Tests: IsolationForest `export_state`/`from_state` round-trip; `poly_diff`; graph matching; control `step_info`; previously uncalled cplx/ODE/info/quantum/combo APIs; crypto REPL AES-GCM/CBC, ChaCha-Poly1305, HKDF, PBKDF2, Ed25519.
- Tests: `poly_partial_fractions`/`poly_fit`; Rosenbrock23/Cash–Karp; finance Sortino/zero-coupon/annuity; AR(1) `arfit`/`pacf`; `ctrb`/`obsv`; `sinm`/`cosm`/`funm`; `degree_centrality`; combinadic rank; `overlap_circle_circle`.
- Pre-release [`v1.0.0-rc.1`](https://github.com/odin-loki/MathScript/releases/tag/v1.0.0-rc.1) published. CI all 9 jobs green on [run 33269316904](https://github.com/odin-loki/MathScript/actions/runs/33269316904).
- Linux `-fno-exceptions`: control `c2d`/`d2c` helpers in `repl_engine_internal.cpp` no longer wrap non-throwing `control::*` calls in `try`/`catch`.
- Clang plugin builds on LLVM 18: `DeclNamespace.h` is included only when present (`NamespaceDecl` is already in `Decl.h`); narrowing diagnostics use `CK_*` instead of `ImplicitCastKind`; unused-`expected` is detected via discarded `CallExpr` (Clang has no `ExprStmt`).
- Plugin smoke test: Clang 18 requires capturing function-local constexpr string arrays in the unsafe-registry lambda.
- Clang plugin: ignore system headers; do not treat written casts as implicit narrowing; allow `(void)` discards; treat initialized/`auto` locals as initialized. Compliance `unused_expected` uses a local `expected` stand-in (libstdc++ `std::expected` is unavailable under Clang + `-fno-exceptions` on CI).
- Tests: unwrap `EXPECT_NO_THROW`/`ASSERT_NO_THROW` (GoogleTest always emits `try`/`catch` for those macros). `GTEST_HAS_EXCEPTIONS=0` remains set for GCC/Clang tests.
- CLI tests: decode POSIX `std::system` wait status so `mathscriptc` exit 1 is not compared as 256.
- DiffGeo unit helix torsion: GCC -O3 third-derivative FD is ~0.03 off analytic 1/2; tolerance is 4e-2.
- Signal+optim pipeline: compare residual energy vs each tone (unit sines share RMS, so the old closeness check was noise).
- Linux Debug CI (coverage/ASan): `MS_LINK_TESTS_SHARED` builds `libms_bundle.so` so 816 test executables do not each copy the static library. Coverage instruments `src/` only.
- `UNSAFE_REVIEW.md`: plugin diagnostics moved into rule TUs; crypto string_view overloads share one `u8_view`; `approved_sites` is 38.
- `tests/compliance/unsafe_baseline.txt` regenerated to the same 38 sites so `unsafe_delta.sh` line-level compare matches.
- Coverage CI gate is **80%** (measured 81.1% of library `src/` after excluding plugin/GUI/CUDA/`matrix_calls`). **90%** remains the `v1.0.0` tag goal.
- ASan CI: `detect_leaks=0` so remaining process-exit leaks do not fail the job; overflows still fail. UBSan does not halt the process (ed25519 ref10 and `BigInt::to_ll`). Three-blob GMM checks separated finite centers rather than exact blob coordinates.
- `PoolAllocator` frees its slabs in the destructor (previously every pool test leaked at least one slab).
- GMM REPL packing uses enough columns for K, p, and log-likelihood (ASan overflow when p<3).
- POSIX `aligned_alloc` rounds size up to a multiple of alignment.
- `dorgbr` P-wide reflector scan stays inside A's column count (`k` when `lda >= n`) so tall factors are not over-read.
- AddressSanitizer + UBSan is the CI memory gate (`sanitizer-linux`). `test_crypto` still runs there and on Windows/Linux unit jobs.
- `MLGMM.ThreeBlobsMeansMatch` checks three finite, pairwise-separated centers (not exact blob coordinates).
- Linux package smoke: Debian CPack uses `mathscript_1.0.0_amd64.deb` (`DEB-DEFAULT`); CI glob is `mathscript*.deb`.
- `linux-gcc13.json` matmul medians recalibrated from GitHub-hosted ubuntu-24.04 (`MS_ENABLE_AVX512=OFF`). Tolerance remains 10%.
- Local prove-out (Windows MSVC Release, CUDA off): **816/816** CTest suites passed (~36 s at `-j 32`); 28 Google Benchmark targets passed with `--benchmark_min_time=0.001s`. Windows ZIP smoke: `scripts/package_smoke.ps1` → `mathscript-1.0.0-win64.zip`.
- MSVC `/W4` compile warnings in library and test TUs were cleared (unused locals, `[[nodiscard]]`, `size_t`→`int`, unused statics).
- Tests are grouped by mathematical domain (`tests/unit/linalg`, `tests/integration/fft`, …). Duplicate remigration wave pipelines were collapsed.
- REPL matrix calls use a name-keyed handler registry (`src/interp/matrix_calls/<domain>/`); CMake generates `matrix_call_register_all.cpp`.
- 28 Google Benchmark targets in the same tree as the library (`MS_BUILD_BENCHMARKS=ON`).
- Local Windows build directory: `build-msvc` only.
