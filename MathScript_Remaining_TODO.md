# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-19 (Wave 269 ✅ COMPLETE — **428 CTest suites** on `main`)

---

## Performance profiling iteration ✅ FULLY COMPLETE (Waves 218–230)

Twelve waves optimized and **certified** all identified hot paths across every `src/` module (35 libraries), closed benchmark infra, and established baseline refresh workflows. Wave 228 was a certification pass; Waves 229–230 closed infra and baseline path only (no code profiling).

| Wave | Focus |
|------|--------|
| 218–224 | (see CHANGELOG.md) |
| **225** | tensorops einsum, imfilter/sharpen/LoG, Floyd-Warshall, CZT, hilbert buffers, smoke fix |
| **226** | topo/geo/ode-pde/ml benches, smoke-safe args, MSVC baseline populated, REPL micro-opts |
| **227** | quantum/bignum/compress benches, finance/symbolic/frameworks opts, Linux+MSVC baseline sync, `docs/PERFORMANCE.md` |
| **228** | Certification audit (`numthy`, `cplx`, `optim`, `diffgeo`, `domain`, `runtime/cpu`); `bench_finance`; Linux schema + smoke guard |
| **229** | Benchmark infra closure — Linux CI builds all **28** benches; `bench_cmake_targets.sh`; docs sync |
| **230** | Baseline refresh path — `bench-baseline-linux.yml` (`workflow_dispatch`); docs final sign-off |

**PROFILING ITERATION FULLY COMPLETE — no further profiling, benchmark-infra, or baseline-path waves.**

| Item | Status |
|------|--------|
| Linux CI bench build | ✅ `benchmark-linux` builds all 28 targets |
| Linux baseline refresh | ✅ `bench-baseline-linux.yml` — `workflow_dispatch`; artifact download → maintainer commit |
| Windows baseline refresh | ✅ `bench_write_msvc_baseline.ps1` (local) |

---

## Product profiling & adjustment ✅ CLOSED

Waves **218–230** certified hot paths, bench infra, and baseline refresh. **No further profiling waves.** Feature work is Waves **231+** — see `CHANGELOG.md`.

---

## Feature waves ✅ 231–256 COMPLETE (summary)

| Wave | Focus |
|------|--------|
| **231** | Crypto AES/ChaCha; FEM 2D; CFD 2D + benches |
| **232** | Symbolic transforms; CUDA LU/StreamPool; MPI REPL; plugin audit; GUI history |
| **233** | GUI polish; optim/control/quantum REPL; `sym_dsolve`; CUDA REPL |
| **234** | Finance/graph/image/ml REPL; AES-128-GCM |
| **235** | ChaCha20-Poly1305; FEM 3D mesh/stiffness; GUI Ctrl+Enter |
| **236** | X25519; CFD 3D; FEM 3D solve; `dist_matmul`; GUI Clear/About/Recent/zoom |
| **237** | NCCL stub; modular plugin rules; remaining REPL gaps |
| **238** | dist CG; GUI find/plot toggle; HKDF; 3D benches; history export/persist |
| **239** | PBKDF2; `dist_gmres`; GUI script find/goto-line; `run_file`; HKDF/X25519/`dist_cg` benches; README matrix |
| **240** | Ed25519; `jacobi`/`dist_jacobi` + GMRES MPI; `sym_mellin`; GUI line numbers/LaTeX export; REPL graph/finance/`bicgstab`; PBKDF2/`dist_gmres` benches |
| **241** | Hankel; `sabr_put`; `dist_bicgstab`; signal cheby2/periodogram/welch REPL; GUI completer; Ed25519/`dist_jacobi` benches |
| **242** | `dist_minres`; graph iso/hamiltonian/TSP REPL; hilbert/envelope REPL; GUI theme toggle; NCCL max/min stubs; SABR/`dist_bicgstab` benches |
| **243** | `dist_qmr`; HMAC-SHA512; `qmr`/`lsqr`/phase REPL; GUI word wrap; graph articulation tests; `dist_minres` bench |
| **244** | `dist_tfqmr`; PBKDF2-SHA512; `heston_put`; `tfqmr`/`lsmr`/spectrogram REPL; GUI replace; Wave 243 benches |
| **245** | AES-256-CBC; HKDF-SHA512; `poly_diff` + geo boolean REPL; graph k-core/chromatic; GUI find-prev; Wave 244 benches |
| **246** | AES-256-GCM; `dist_lsmr`; blossom matching; NCCL `allreduce_prod`; GUI Duplicate Line; Wave 245 benches |
| **247** | `unwrap`; geo minkowski/OBB REPL; NCCL `allreduce_avg`; Toggle Comment; wave247 pipeline; Wave 246 benches |
| **248** | NCCL `broadcast`; `constant_time_eq`/`random_bytes`; geo clip + upsample/downsample REPL; GUI Indent; Wave 247 benches |
| **249** | NCCL `reduce`; `dist_lsqr`; SHA-256 REPL; signal resample/decimate/interpolate; GUI Move Line; Wave 248 benches |
| **250** | `crypto_sha512`; NCCL introspect; coherence/filtfilt; GUI Delete Line; `BM_DistLsqr`; wave250 pipeline |
| **251** | `signal_filter`/`cheby1`; MPI max/min; `geo_convex_hull`; Select All/Undo; AES-256 block; wave251 pipeline |
| **252** | `sosfilt`/`firwin`/`savgol`; `mpi_barrier`; AES-256 decrypt; GUI Replace All; wave252 pipeline |
| **253** | xcorr/xcov/autocorr; median_filter; conv2; AES-128 decrypt; NCCL allgather; matching brace |
| **254** | deconv/LMS/CZT; `mpi_bcast`; geo AABB; GUI Add Selection; wave254 pipeline |
| **255** | geo triangulate/hull3d/kdtree/ray; graph spectral; X25519 keypair; Trim Whitespace |
| **256** | stats TS/inference; graph structure; geo 3D; image transforms; Remove Blank Lines |

See `CHANGELOG.md` for per-wave branch tables.

---

## Wave 237 ✅ COMPLETE (NCCL stub, modular plugin, remaining REPL gaps)

| Area | Deliverable |
|------|-------------|
| NCCL / CUDA | Stub `allreduce_sum`; REPL `cuda_allreduce_sum`; `test_nccl_stub` |
| Plugin | Modular rules: memory, exception, cast (`rule_context` + `rules/*`) |
| REPL | `finance_sabr_call`; `graph_bridges` / `graph_min_cut` / `graph_transitive_closure`; `control_lqe`; `scharr` / `roberts`; `integration_repl_wave237_pipeline` |

**387 CTest suites** — all passing on `main`. **Major planned product feature waves 231–237 are largely closed**; remaining work is incremental API/IDE polish.

---

## Wave 238 ✅ COMPLETE (dist CG, GUI find/plot, HKDF, 3D benches, history export/persist)

| Area | Deliverable |
|------|-------------|
| Distributed | `dist_cg` stub-safe conjugate-gradient solver; `test_dist_cg` |
| GUI | Find-in-output panel; plot panel show/hide toggle; export command history |
| Crypto | HKDF-SHA256 (RFC 5869); REPL binding |
| Benchmarks | FEM 3D and CFD 3D smoke benchmark cases |
| REPL / session | History lines in session save/load; `export history` / `save_history` |

**388 CTest suites** — all passing on `main`. Incremental post–Wave 237 polish closed.

---

## Wave 239 ✅ COMPLETE (PBKDF2, dist GMRES, run_file, GUI script search, benches, docs)

| Area | Deliverable |
|------|-------------|
| Crypto | PBKDF2-HMAC-SHA256; REPL `crypto_pbkdf2_sha256`; RFC 6070 vectors |
| Distributed | `dist_gmres` stub-safe GMRES; `test_dist_gmres`; REPL binding |
| GUI | Find-in-script search; goto-line |
| REPL | `run_file` / `source`; `test_repl_commands` |
| Benchmarks | HKDF / X25519 (`bench_crypto`); `dist_cg` (`bench_distributed_cellai`) |
| Docs | README Waves 231–239 matrix; USER_GUIDE / API sync |

**389 CTest suites** — all passing on `main`.

---

## Wave 240 ✅ COMPLETE (Ed25519, dist Jacobi, Mellin, GUI IDE lite, REPL bindings, benches)

| Area | Deliverable |
|------|-------------|
| Crypto | Ed25519 keygen/sign/verify (RFC 8032); REPL hex bindings |
| Distributed | `ms::jacobi`; `dist_jacobi`; `dist_gmres` MPI block path; `test_dist_jacobi` |
| Symbolic | `sym_mellin` / `sym_imellin` table MVP + REPL |
| GUI | Line-number gutter; Ln/Col status; Export Last Result as LaTeX |
| REPL | bipartite match / biconnected / eulerian path; geo-asian; `bicgstab`; wave240 pipeline |
| Benchmarks | PBKDF2 + `dist_gmres` smoke cases |

## Wave 241 ✅ COMPLETE (Hankel, sabr_put, dist_bicgstab, signal REPL, GUI completer, benches)

| Area | Deliverable |
|------|-------------|
| Symbolic | `sym_hankel` / `sym_ihankel` table MVP + REPL |
| Finance | `sabr_put` + REPL |
| Distributed | `dist_bicgstab` stub-gather; `test_dist_bicgstab` |
| Signal REPL | `signal_cheby2`, `signal_periodogram`, `signal_welch_psd`; wave241 pipeline |
| GUI | REPL command completer |
| Benchmarks | Ed25519 + `dist_jacobi` smoke |

## Wave 242 ✅ COMPLETE (dist MINRES, graph/signal REPL, GUI theme, NCCL max/min, benches)

| Area | Deliverable |
|------|-------------|
| Distributed | `dist_minres`; `test_dist_minres`; REPL |
| Graph REPL | isomorphic / hamiltonian path / TSP heuristic; wave242 pipeline |
| Signal REPL | envelope / hilbert / instantaneous_freq; wave242 signal pipeline |
| GUI | Dark/light theme toggle |
| NCCL | stub `allreduce_max` / `allreduce_min` + REPL |
| Benchmarks | `dist_bicgstab` + SABR smoke |

## Wave 243 ✅ COMPLETE (dist QMR, HMAC-SHA512, linalg/signal REPL, GUI wrap, benches)

| Area | Deliverable |
|------|-------------|
| Distributed | `dist_qmr`; `test_dist_qmr`; REPL |
| Crypto | HMAC-SHA512 + REPL |
| REPL | `qmr`, `lsqr`, `signal_instantaneous_phase`; wave243 pipeline |
| Graph tests | articulation / euler circuit pipeline coverage |
| GUI | Word Wrap toggle |
| Benchmarks | `dist_minres` smoke |

## Wave 244 ✅ COMPLETE (dist TFQMR, PBKDF2-SHA512, heston_put, REPL/GUI, benches)

| Area | Deliverable |
|------|-------------|
| Distributed | `dist_tfqmr`; `test_dist_tfqmr`; REPL |
| Crypto | PBKDF2-HMAC-SHA512 + REPL |
| Finance | `heston_put` + REPL + bench |
| REPL | `tfqmr`, `lsmr`, `signal_spectrogram`; wave244 pipeline |
| GUI | Replace in Script / Replace Next |
| Benchmarks | `dist_qmr` + `hmac_sha512` smoke |
| Fix | `tfqmr` MVP via BiCGSTAB kernel |

## Wave 245 ✅ COMPLETE (AES-256-CBC, HKDF-SHA512, poly_diff, graph k-core, GUI find-prev, benches)

| Area | Deliverable |
|------|-------------|
| Crypto | AES-256-CBC + HKDF-SHA512 + REPL |
| Geo | `poly_diff`; geo boolean REPL |
| Graph REPL | k-core decomposition/subgraph + chromatic number |
| GUI | Find Previous in Script/Output |
| Benchmarks | `dist_tfqmr` + PBKDF2-SHA512 smoke |

## Wave 246 ✅ COMPLETE (AES-256-GCM, dist LSMR, blossom matching, NCCL prod, GUI, benches)

| Area | Deliverable |
|------|-------------|
| Crypto | AES-256-GCM + REPL |
| Distributed | `dist_lsmr`; `test_dist_lsmr`; REPL |
| Graph | Edmonds blossom `maximum_matching` + REPL |
| NCCL | stub `allreduce_prod` + REPL |
| GUI | Duplicate Line |
| Benchmarks | AES-256-CBC + HKDF-SHA512 smoke |

## Wave 247 ✅ COMPLETE (unwrap, geo minkowski, NCCL avg, Toggle Comment, benches)

| Area | Deliverable |
|------|-------------|
| Signal | `unwrap` + REPL `signal_unwrap` |
| Geo REPL | minkowski sum + min bounding rect |
| NCCL | stub `allreduce_avg` + REPL |
| GUI | Toggle Comment |
| Tests | wave247 pipeline (geo/signal + Wave 246 APIs) |
| Benchmarks | AES-256-GCM + `dist_lsmr` smoke |

## Wave 248 ✅ COMPLETE (NCCL broadcast, crypto ct-eq, geo clip, signal resample REPL, GUI indent, benches)

| Area | Deliverable |
|------|-------------|
| NCCL | stub `broadcast` + REPL `cuda_broadcast` |
| Crypto | `constant_time_eq`, `random_bytes` + REPL |
| Geo / Signal REPL | `geo_clip_polygon`; `signal_upsample` / `signal_downsample` |
| GUI | Indent / Unindent |
| Tests | wave248 pipeline |
| Benchmarks | `BM_SignalUnwrap`, `BM_CudaAllreduceAvg` |

## Wave 249 ✅ COMPLETE (NCCL reduce, dist LSQR, SHA-256 REPL, signal resample, GUI move line, benches)

| Area | Deliverable |
|------|-------------|
| NCCL | stub `reduce` + REPL `cuda_reduce` |
| Distributed | `dist_lsqr` + `test_dist_lsqr` + REPL |
| Crypto REPL | `crypto_sha256` / `crypto_hmac_sha256` |
| Signal REPL | resample / decimate / interpolate + wave249 pipeline |
| GUI | Move Line Up/Down |
| Benchmarks | broadcast / constant_time_eq / upsample smoke |
| Fix | MSVC C1061 — assign_matrix_call arms → tail |

## Wave 250 ✅ COMPLETE (crypto SHA-512, NCCL introspect, coherence/filtfilt, GUI delete line, benches)

| Area | Deliverable |
|------|-------------|
| Crypto REPL | `crypto_sha512` + `BM_Sha512_1MB` |
| NCCL | introspect queries + `BM_CudaReduce` |
| Signal REPL | `signal_coherence`, `signal_filtfilt` + wave250 pipeline |
| GUI | Delete Line |
| Benchmarks | `BM_DistLsqr_2x2` |

## Wave 251 ✅ COMPLETE (signal filter/cheby1, MPI max/min, geo hull, GUI select/undo, AES-256 block)

| Area | Deliverable |
|------|-------------|
| Signal REPL | `signal_filter`, `signal_cheby1` |
| MPI REPL | `mpi_allreduce_max` / `mpi_allreduce_min` |
| Geo REPL | `geo_convex_hull` vertices |
| GUI | Select All, Undo, Redo |
| Crypto REPL | `crypto_aes256_encrypt_block` + wave251 pipeline |
| Benchmarks | `BM_CudaNcclCommSize` |

## Wave 252 ✅ COMPLETE (sosfilt/firwin/savgol, mpi_barrier, AES-256 decrypt, GUI Replace All)

| Area | Deliverable |
|------|-------------|
| Signal REPL | `signal_sosfilt`, `signal_firwin`/`highpass`, `signal_savgol` |
| MPI REPL | `mpi_barrier` |
| Crypto | `aes256_decrypt_block` + REPL + wave252 pipeline |
| GUI | Replace All in Script |

## Wave 253 ✅ COMPLETE (xcorr/median/conv2, AES-128 decrypt, NCCL allgather, matching brace)

| Area | Deliverable |
|------|-------------|
| Signal REPL | xcorr/xcov/autocorr, median_filter, conv2 |
| Crypto | `aes128_decrypt_block` + wave253 pipeline |
| NCCL | stub `allgather` + `BM_CudaAllgather` |
| GUI | Go to Matching Brace |

**411 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–253** closed for this incremental batch.

## Wave 254 ✅ COMPLETE (deconv/LMS/CZT, mpi_bcast, geo AABB, Add Selection)

| Area | Deliverable |
|------|-------------|
| Signal REPL | `signal_deconv`, `signal_lms`/`signal_lms_weights`, `signal_czt`/`signal_czt_zoom` |
| MPI REPL | `mpi_bcast` |
| Geo REPL | `geo_point_in_aabb`, `geo_overlap_aabb` |
| GUI | Add Selection for Next Occurrence (Ctrl+D) |
| Tests | `integration_repl_wave254_pipeline` |

**412 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–254** closed for this incremental batch.

## Wave 255 ✅ COMPLETE (geo queries, graph spectral, X25519 keypair, trim whitespace)

| Area | Deliverable |
|------|-------------|
| Geo REPL | triangulate, hull3d, kdtree knn/range, seg/ray intersect |
| Graph REPL | katz, algebraic_connectivity, adjacency_spectrum, laplacian |
| Crypto REPL | `crypto_x25519_keypair` |
| GUI | Trim Trailing Whitespace |
| Tests | `BM_MPIContext_Bcast`; `integration_repl_wave255_pipeline` |

**413 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–255** closed for this incremental batch.

## Wave 256 ✅ COMPLETE (stats TS/inference, graph structure, geo 3D, image, blank lines)

| Area | Deliverable |
|------|-------------|
| Stats REPL | linear_regression/pacf/kde/bootstrap_ci; shapiro/mann_whitney/anova/wilcoxon |
| Graph REPL | normalised_laplacian, modularity, eccentricity, is_strongly_connected |
| Geo REPL | kdtree_3d_nearest, ray_tri, dist_point_plane/seg3d |
| Image REPL | imflip, imrotate90, threshold_binary, adapthisteq |
| GUI | Remove Blank Lines |
| Tests | `BM_StatsKde`; `integration_repl_wave256_pipeline` |

**414 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–256** closed for this incremental batch.

## Wave 257 ✅ COMPLETE (stats inference/variance, image segment/Hough, prob ext, Sort Lines)

| Area | Deliverable |
|------|-------------|
| Stats REPL | friedman/ks_2sample/jarque_bera/ljung_box; levene/bartlett/fligner |
| Image REPL | label_components/watershed/slic; hough_lines/circles; harris/shi_tomasi |
| Prob REPL | extended PDF/CDF/PPF + beta/gamma/f CDF |
| GUI | Sort Lines |
| Interp | `assign_matrix_call_tail2` (MSVC C1061) |
| Tests | `integration_repl_wave257_pipeline` |

**415 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–257** closed for this incremental batch.

## Wave 258 ✅ COMPLETE (image morph/radon, linalg matfun, graph SP, info/stats, Join Lines)

| Area | Deliverable |
|------|-------------|
| Image REPL | imtophat/imbothat/imadjust/imhist; radon/iradon/gray2rgb/impad |
| Linalg REPL | sqrtm/logm/tril/triu (+ cosm/sinm) |
| Graph REPL | dijkstra / bellman_ford |
| Info REPL | permutation_entropy / transfer_entropy |
| Stats REPL | partial_correlation / weighted_mean / trimmed_mean / arfit / multiple_regression |
| GUI | Join Lines |
| Tests | `integration_repl_wave258_pipeline` |

**416 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–258** closed for this incremental batch.

## Wave 259 ✅ COMPLETE (hess/schur, finance frontier, geo curves, combo/numthy, Unique Lines)

| Area | Deliverable |
|------|-------------|
| Linalg REPL | hess; schur (+ multi T,Q) |
| Finance REPL | efficient_frontier / max_sharpe |
| Geo REPL | bezier_eval / catmull_rom / bspline_eval |
| Combo/Numthy | binomial / bell_num; factor |
| Image REPL | imgradient_morph |
| GUI | Unique Lines |
| Tests | `integration_repl_wave259_pipeline` |

**417 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–259** closed for this incremental batch.

## Wave 260 ✅ COMPLETE (VIF/Goertzel/control responses, linalg ext, geo curves, Upper/Lower Case)

| Area | Deliverable |
|------|-------------|
| Stats REPL | stats_vif / stats_variance_inflation_factor |
| Numthy REPL | numthy_divisors (+ divisors_vec) |
| FFT REPL | fft_goertzel → 1×2 [re,im] |
| Geo REPL | geo_bezier_deriv / geo_hermite_curve |
| Control REPL | control_step_response / control_impulse_response |
| Linalg REPL | bidiag (+ multi); solve_sylvester; minres |
| GUI | Upper/Lower Case Selection |
| Tests | `integration_repl_wave260_pipeline` |

**418 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–260** closed for this incremental batch.

## Wave 261 ✅ COMPLETE (eig/ldl, iterative solvers, FFT freq, combo/numthy/stats, Title Case)

| Area | Deliverable |
|------|-------------|
| Linalg REPL | eig (+ multi); ldl (+ multi); cg / gmres / jacobi |
| FFT REPL | fftfreq / rfftfreq / ifftshift |
| Combo REPL | eulerian / gray_code / dyck_paths |
| Numthy REPL | factor_exp / farey / is_carmichael |
| Stats REPL | weighted_variance / weighted_correlation / bootstrap_mean |
| GUI | Title Case Selection |
| Tests | `integration_repl_wave261_pipeline` |

**419 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–261** closed for this incremental batch.

---

## Wave 262 — combo enum, numthy, lsq/diag, finance rates, poly, Reverse Lines ✅ COMPLETE

| Area | Deliverable |
|------|-------------|
| Combo REPL | necklaces / de_bruijn_sequence / motzkin_paths / set_partitions |
| Numthy REPL | stern_brocot / lucas_sequence / multiplicative_order / carmichael_lambda |
| Linalg REPL | lsq / diag |
| Finance REPL | bachelier_call/put / vasicek_bond_price / cir_bond_price / trinomial_option |
| Poly REPL | poly_resultant / poly_discriminant |
| GUI | Reverse Lines (Ctrl+Shift+R) |
| Tests | `integration_repl_wave262_pipeline` |

**420 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–262** closed for this incremental batch.

---

## Wave 263 — combo enum, pell/QR, poly interp, BL/Merton, Kalman, voigt/airy ✅ COMPLETE

| Area | Deliverable |
|------|-------------|
| Combo REPL | bracelets / lyndon_words / restricted_partitions / involutions |
| Numthy REPL | pell_solve / quadratic_residues |
| Poly REPL | poly_lagrange / poly_interp_newton |
| Finance REPL | bl_implied/posterior returns; merton DD; historical VaR/CVaR |
| Control REPL | kalman_predict/_cov / kalman_update/_cov |
| Special REPL | voigt / pseudo_voigt_auto / airy_ai |
| Tests | `integration_repl_wave263_pipeline` |

**421 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–263** closed for this incremental batch.

---

## Wave 264 — poly algebra, cornacchia, ratios, ctrb/tf2ss, specials, capacity, Split Lines ✅ COMPLETE

| Area | Deliverable |
|------|-------------|
| Poly REPL | roots / fit / interp_hermite / gcd / squarefree |
| Numthy REPL | cornacchia |
| Finance REPL | treynor / information_ratio |
| Control REPL | ctrb / obsv / ctrb_gram / obsv_gram / tf2ss / c2d / c2d_b |
| Special REPL | bessel_y / bessel_i / lambert_w / kummer_u / special_airy_bi |
| Info REPL | blahut_arimoto / channel_capacity |
| GUI | Split Lines (Ctrl+Shift+J) |
| Tests | `integration_repl_wave264_pipeline` |

**423 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–264** closed for this incremental batch.

---

## Wave 265 — poly transforms/factor, control TF, Merton/BL, combo prev, specials, funm, graph/info, Capitalize ✅ COMPLETE

| Area | Deliverable |
|------|-------------|
| Poly REPL | shift/scale/monic/reverse/pow/lcm/div_quot/mod/eval_at/sylvester; factor/rational_roots/factor_rational/partial_fractions/root_count/cheb_eval |
| Control REPL | series / parallel / feedback / ss2tf / d2c / c2d_tf / d2c_tf |
| Finance REPL | merton_implied_asset_params / bl_posterior_returns_default_omega |
| Combo REPL | prev_perm / prev_comb |
| Special REPL | bessel_k / chebyshev_t/u / hermite_h / laguerre_l / sph_bessel_j/y / assoc_legendre_p / gegenbauer_c |
| Linalg REPL | matrix_rank / funm / precond_diag / precond_ssor |
| Graph / Info REPL | connected_components / is_planar; normalized_entropy / channel_capacity_input |
| GUI | Capitalize Words (Ctrl+Alt+W) |
| Tests | `integration_repl_wave265_pipeline` |

**424 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–265** closed for this incremental batch.

---

## Wave 266 — cheb expand, Tustin/Euler, specials, PPFs, arborescence, image filters, KS, Invert Case ✅ COMPLETE

| Area | Deliverable |
|------|-------------|
| Poly REPL | poly_cheb_expand |
| Control REPL | c2d/d2c Tustin & Euler (+ TF Tustin) |
| Special REPL | erfi/erfcx/dawson/gamma_inc/beta; orthog/sph_harm; hypergeo/Whittaker; mathieu/heun/painleve2 |
| Prob / Stats REPL | chi2_ppf / exp_ppf; stats_ks_norm |
| Graph / Image REPL | min_arborescence; imfilter / sobel_x/y / hsv2rgb / dft_magnitude / laplacian_of_gaussian |
| GUI | Invert Case (Ctrl+Alt+I) |
| Tests | `integration_repl_wave266_pipeline` |

**425 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–266** closed for this incremental batch.

---

## Wave 267 — specials ext, ODE solvers, ML, geo hull, stats_max, Snake Case ✅ COMPLETE

| Area | Deliverable |
|------|-------------|
| Special REPL | painleve3–6 / dawsonx; elliptic/Jacobi/theta; hypergeo/Meijer; bessel/struve/kelvin ext; mathieu_a/spheroidal/PCF; zeta/Airy/orthog |
| ODE REPL | trapezoidal / cashkarp / rk23 / exponential_euler / rosenbrock23 |
| ML REPL | lasso / elastic_net / knn / naive_bayes / lda; pca / kmeans |
| Geo REPL | upper/lower hull / bezier_subdivide / kdtree_3d knn+range |
| Stats / GUI | stats_max_value; Snake Case Selection |
| Tests | `integration_repl_wave267_pipeline` |

**426 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–267** closed for this incremental batch.

---

## Wave 268 — ML QDA/SVM/trees/GMM/scalers-ROC, PDE CN/elliptic/hyperbolic, optim, compress, FEM1d, Kebab Case ✅ COMPLETE

| Area | Deliverable |
|------|-------------|
| ML REPL | qda/svm; decision_tree/random_forest/adaboost; gmm/dbscan/spectral_clustering; standard/minmax scaler; train_test_split; roc_auc/average_precision; assign_matrix_call_tail7 |
| PDE REPL | heat_1d_cn / heat_2d_cn_adi; poisson_1d / laplace_2d / helmholtz_2d; wave_2d / advection_1d_lax_wendroff / reaction_diffusion_1d |
| Optim REPL | conjugate_gradient / rmsprop / adadelta; bisection/brentq/secant/halley/fixed_point/illinois; simulated_annealing / differential_evolution / particle_swarm |
| Compress REPL | arithmetic_encode_vec / ans_encode_vec (+ decode) |
| FEM / GUI | fem_poisson1d; Kebab Case Selection (Ctrl+Alt+K) |
| Tests | `integration_repl_wave268_pipeline` |

**427 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–268** closed for this incremental batch.

---

## Wave 269 — ML metrics/gboost/isolation/tsne, compress golomb/wavelet, sparse COO, tensorops NMF/TT, topo, quantum, cplx/ODE/CFD1d, diffgeo, GUI case/trim, run_file stack fix ✅ COMPLETE

| Area | Deliverable |
|------|-------------|
| ML REPL | confusion_matrix/roc_curve/precision_recall_curve; gradient_boosting fit/predict; isolation_forest fit/score; agglomerative_fit; tsne_fit; assign_matrix_call_tail8 |
| Compress REPL | golomb_rice_encode_vec/decode_vec; wavelet_compress_vec/decompress_vec |
| Sparse REPL | sparse_from_coo / sparse_spmv / sparse_to_dense |
| Tensorops REPL | tensorops_decompose_nmf/reconstruct_nmf; tensorops_decompose_tt/reconstruct_tt |
| Topo REPL | topo_alpha_complex / topo_witness_complex / topo_persistence_landscape |
| Quantum REPL | quantum_wigner / quantum_husimi / quantum_grover_search |
| Cplx / ODE / CFD REPL | cplx_green_function_disk; ode_adams_bashforth2; cfd_advection1d |
| Diffgeo REPL | diffgeo_helix_torsion; diffgeo_sphere_gauss_bonnet (+ residual) |
| GUI / Fix | Camel Case (Ctrl+Alt+C); Screaming Snake (Ctrl+Alt+Shift+S); Trim Leading Whitespace (Ctrl+Shift+B); run_file stack-overflow fix (`b543e26`) |
| Tests | `integration_repl_wave269_pipeline` |

**428 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–269** closed for this incremental batch.

---

## Next (Wave 270+) — deferred

| Item | Notes |
|------|--------|
| Scalable multi-node MPI LA | Beyond stub/block `dist_*` gather and block iterative solvers |
| Full IDE | clangd/LSP, debugger, rich rendered LaTeX (master plan §11) |
| Linux baseline medians | Refresh via `bench-baseline-linux.yml` when `gh` authenticated |
| Deeper API gaps | Full NCCL multi-GPU, weighted blossom, etc. |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
