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
- `mathscript-server` is a real SPMD compute node (`--script`, `-e`, `--serve`, one Interpreter per rank) rather than a heartbeat loop that ignored argv.

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
