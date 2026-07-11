# MathScript HPC Computer Algebra System

[![CI](https://github.com/odin-loki/MathScript/actions/workflows/ci.yml/badge.svg)](https://github.com/odin-loki/MathScript/actions/workflows/ci.yml)

A high-performance Computer Algebra System built in C++23 with CPU math libraries, runtime dispatch, and a console REPL.

## Project Status: v1.0.0 — Phase 10 (Hardening) + Wave 194 Complete

Phase 9 (own BLAS/LAPACK SVD pipeline) is complete. Phase 10 adds CI, coverage reporting, Valgrind memcheck, fuzz testing, and install/packaging. **Wave 56** added adaptive ODE solvers, global optimisers, advanced polynomial operations, time-series statistics, and new linalg functions. **Wave 57** adds `ms::numthy`, `ms::combo`, `ms::info`, `ms::finance`, plus QMR/LSQR/LSMR/TFQMR iterative solvers and preconditioners. **Wave 58** adds `ms::control` (control theory: TF/SS, Bode, LQR, Lyapunov, Riccati, pole placement), `ms::graph` (graph theory: BFS/DFS, Dijkstra, A*, Floyd-Warshall, SCC, MST, PageRank, betweenness centrality, max-flow, coloring), `ms::cplx` (complex analysis: residues, contour integrals, Möbius transforms, Joukowski, Blaschke products), and `ms::quantum` (quantum mechanics: gates, density matrices, QFT, Bell/GHZ/W states, von Neumann entropy, time evolution). **Wave 59** adds `ms::geo` (computational geometry: 2D/3D primitives, convex hull, Delaunay triangulation, Voronoi, KD-trees, ray intersections, Bezier/B-spline curves, polygon measurements), `ms::diffgeo` (differential geometry: metric tensor, Christoffel symbols, Riemann/Ricci curvature, Einstein tensor, geodesics, surface fundamental forms, Gaussian/mean curvature, principal curvatures, Lie brackets), `ms::topo` (topology/TDA: simplicial complexes, Vietoris-Rips, persistent homology, Betti numbers, Euler characteristic, bottleneck/Wasserstein distances), and `ms::tensorops` (tensor analysis: einsum, contractions, mode products, CP/Tucker/HOSVD decompositions, Khatri-Rao/Kronecker products, symmetrization). **Wave 60** adds `ms::ml` (machine learning: linear/ridge/lasso/logistic regression, KNN, naive Bayes, decision trees, KMeans/DBSCAN/agglomerative clustering, PCA/t-SNE, autodiff, neural nets, scalers, cross-validation), `ms::image` (image processing: color conversion, resize/crop/flip, Gaussian/median/bilateral filters, Sobel/Canny edge detection, morphology, Otsu thresholding, histogram equalisation, Harris corners, connected components), `ms::compress` (compression: RLE, Huffman, LZ77/LZW, BWT/MTF, delta coding, bzip2-like pipeline), and `ms::bignum` (arbitrary-precision integers and rationals: gcd/lcm, pow/mod, factorial, Fibonacci, Miller-Rabin primality). **Wave 61** adds REPL linalg bindings (`pinv`, `null`, `orth`, `kron`, `repmat`, `linspace`) with nested matrix operand evaluation, expanded Wave 60 unit tests, and fixes `null()` for wide matrices (replaced thin-SVD column indexing with an `A^T A` eigenvector basis and relative eigenvalue cutoff). **Wave 62** fixes `dgesvd`/`pinv` crashes on wide matrices (`dbdsqr` OOB guard, VT column-major, bidiagonal copy before `dorgbr`), expands REPL Wave 60 bindings (image filters, delta compress, ML metrics, bignum ops), and adds +31 unit tests across ml/image/compress/bignum. **Wave 63** adds a Wave 57–59 integration pipeline (geo/topo/graph/tensorops/diffgeo/control/quantum), REPL bindings for graph/geo/combo/numthy/control/quantum, `docs/API.md` coverage for Waves 57–62, and Windows CPack ZIP packaging smoke. **Wave 64** adds Wave 58 integration pipeline (control/graph/cplx/quantum/finance/info), REPL bindings for cplx/finance/info/tensorops, +20 unit tests, and Linux CPack TGZ packaging smoke. **Wave 65** adds mega Wave 57–60 integration pipeline, fixes `von_neumann_entropy` for pure states, REPL diffgeo/topo bindings, and +12 control/graph/quantum unit tests. **Wave 66** fixes `ss2tf` DC gain for strictly proper systems, adds full cross-module integration pipeline, REPL `tensorops_matmul`/`combo_factorial`/`numthy_partition`/`finance_bond_price`, and +48 diffgeo/topo/cplx unit tests. **Wave 67** adds REPL Wave 63–66 integration pipeline test, `tensorops_einsum`/`geo_polygon_area` bindings, Kleinman LQR fix for the double integrator, and CI package artifact uploads (Windows ZIP, Linux TGZ). **Wave 68** adds REPL `finance_irr`/`info_kl_divergence` bindings, REPL Wave 63–68 integration pipeline test, fuzz `fuzz_repl_input` corpus seeds for recent REPL bindings, expanded ml/image unit tests, and tag checklist update to **244** suites. **Wave 69** adds REPL `control_dcgain`/`info_cross_entropy` bindings, REPL Wave 63–69 integration pipeline test, fuzz seeds for Wave 68–69 REPL bindings, matrix literal negative-entry fix, and tag checklist update to **245** suites. **Wave 70** adds REPL `control_is_stable`/`finance_var`/`info_mutual_info`/`combo_nchoosek` bindings, REPL Wave 63–70 integration pipeline test, fuzz seeds for Wave 70 REPL bindings, and tag checklist update to **246** suites. **Wave 71** adds REPL `finance_cvar`/`info_js_divergence`/`quantum_von_neumann_entropy` bindings, REPL Wave 63–71 integration pipeline test, fuzz seeds for Wave 71 REPL bindings, and tag checklist update to **247** suites. **Wave 72** adds REPL `finance_sortino`/`info_tv_distance`/`quantum_fidelity` bindings, REPL Wave 63–72 integration pipeline test, fuzz seeds for Wave 72 REPL bindings, and tag checklist update to **248** suites. **Wave 73** adds REPL `finance_max_drawdown`/`info_hellinger_dist`/`quantum_trace_distance` bindings, REPL Wave 63–73 integration pipeline test, fuzz seeds for Wave 73 REPL bindings, and tag checklist update to **249** suites. **Wave 74** adds REPL `finance_kelly_fraction`/`info_renyi_entropy`/`quantum_concurrence` bindings, REPL Wave 63–74 integration pipeline test, fuzz seeds for Wave 74 REPL bindings, and tag checklist update to **250** suites. **Wave 75** adds REPL `finance_compound`/`info_redundancy`/`quantum_entanglement_entropy` bindings, REPL Wave 63–75 integration pipeline test, fuzz seeds for Wave 75 REPL bindings, and tag checklist update to **251** suites. **Wave 76** adds REPL `finance_continuous_compound`/`info_efficiency`/`quantum_expectation` bindings, REPL Wave 63–76 integration pipeline test, fuzz seeds for Wave 76 REPL bindings, and tag checklist update to **252** suites. **Wave 77** adds REPL `finance_pv`/`info_channel_capacity_bsc`/`quantum_expectation_dm` bindings, REPL Wave 63–77 integration pipeline test, fuzz seeds for Wave 77 REPL bindings, and tag checklist update to **253** suites. **Wave 78** adds REPL `finance_fv_annuity`/`info_channel_capacity_bec`/`quantum_inner` bindings, REPL Wave 63–78 integration pipeline test, fuzz seeds for Wave 78 REPL bindings, and tag checklist update to **254** suites. **Wave 79** adds REPL `finance_pmt_annuity`/`info_shannon_hartley`/`quantum_ket_normalise` bindings, REPL Wave 63–79 integration pipeline test, fuzz seeds for Wave 79 REPL bindings, and tag checklist update to **255** suites. **Wave 80** adds REPL `finance_binomial_call`/`info_differential_entropy_gaussian`/`quantum_partial_trace` bindings, REPL Wave 63–80 integration pipeline test, fuzz seeds for Wave 80 REPL bindings, and tag checklist update to **256** suites. **Wave 81** adds REPL `finance_binomial_put`/`info_differential_entropy_uniform`/`finance_bs_put` bindings, REPL Wave 63–81 integration pipeline test, fuzz seeds for Wave 81 REPL bindings, and tag checklist update to **257** suites. **Wave 82** adds REPL `finance_bs_gamma`/`finance_bond_duration`/`info_rate_distortion_gaussian` bindings, REPL Wave 63–82 integration pipeline test, fuzz seeds for Wave 82 REPL bindings, and tag checklist update to **258** suites. **Wave 83** adds REPL `finance_bs_delta`/`finance_bs_vega`/`finance_bond_modified_duration` bindings, REPL Wave 63–83 integration pipeline test, fuzz seeds for Wave 83 REPL bindings, and tag checklist update to **259** suites. **Wave 84** adds REPL `finance_bs_theta`/`finance_bs_rho`/`finance_bond_convexity` bindings, REPL Wave 63–84 integration pipeline test, fuzz seeds for Wave 84 REPL bindings, and tag checklist update to **260** suites.

**CI on `main`:** Windows/Linux build-test, coverage, fuzz smoke, Valgrind, plugin-linux, jit-linux, benchmark-linux, unsafe delta.

### Completed Components

- **Build system:** Native Windows MSVC + Ninja; Linux GCC 13 / Clang; GoogleTest via FetchContent
- **Core types:** Matrix, Tensor, Sparse, Scalar, Sym
- **Linear algebra:** matmul, LU, QR, Cholesky, solve, eig, SVD, expm, trace, det, norm, kron, pinv, null, orth, sinm, cosm, funm, CG, BiCGSTAB, GMRES, MINRES, QMR, LSQR, LSMR, TFQMR + preconditioners
- **Math modules:** fft, stats, prob, optim, signal, special (CPU implementations)
- **Wave 56–57 modules:** ode (adaptive RK45/RK23), poly (advanced), stats (time-series), numthy, combo, info, finance
- **Wave 58 modules:** control (tf/ss/bode/lqr/lyap/riccati), graph (traversal/paths/MST/centrality/flow), cplx (residues/contours/Möbius/conformal), quantum (gates/states/entropy/evolution)
- **Wave 59 modules:** geo (2D/3D geometry: convex hull, Delaunay, KD-tree, Bezier, intersections, measurements), diffgeo (Christoffel, Riemann, Ricci, geodesics, surface curvature), topo (simplicial complex, persistent homology, Betti numbers, Wasserstein), tensorops (einsum, CP/Tucker/HOSVD, Khatri-Rao, symmetrize)
- **Wave 60 modules:** ml (supervised/unsupervised learning, PCA, autodiff, neural nets), image (filters, edges, morphology, features), compress (RLE/Huffman/LZ/LZW/BWT), bignum (BigInt/Rational, factorial, primality)
- **Wave 61:** REPL linalg bindings (`pinv`, `null`, `orth`, `kron`, `repmat`, `linspace`); `null()` wide-matrix fix
- **Wave 62:** SVD/`pinv` wide-matrix fix; REPL Wave 60 bindings (image/compress/ml/bignum); +31 unit tests
- **Wave 63:** Wave 57–59 integration pipeline; REPL graph/geo/combo/numthy/control/quantum; API docs; Windows ZIP packaging
- **Wave 64:** Wave 58 integration pipeline; REPL cplx/finance/info/tensorops; +20 unit tests; Linux TGZ packaging
- **Wave 65:** Mega Wave 57–60 pipeline; `von_neumann_entropy` fix; REPL diffgeo/topo; +12 unit tests
- **Wave 66:** `ss2tf` fix; full integration pipeline; REPL matmul/partition/bond/factorial; +48 diffgeo/topo/cplx tests
- **Wave 67:** REPL integration pipeline; `tensorops_einsum`/`geo_polygon_area`; LQR integrator fix; CI ZIP/TGZ artifacts
- **Wave 68:** REPL `finance_irr`/`info_kl_divergence`; Wave 63–68 pipeline; fuzz repl seeds; ml/image tests; 244 suites
- **Wave 69:** REPL `control_dcgain`/`info_cross_entropy`; Wave 63–69 pipeline; fuzz repl seeds; matrix literal fix; 245 suites
- **Wave 70:** REPL `control_is_stable`/`finance_var`/`info_mutual_info`/`combo_nchoosek`; Wave 63–70 pipeline; fuzz repl seeds; 246 suites
- **Wave 71:** REPL `finance_cvar`/`info_js_divergence`/`quantum_von_neumann_entropy`; Wave 63–71 pipeline; fuzz repl seeds; 247 suites
- **Wave 72:** REPL `finance_sortino`/`info_tv_distance`/`quantum_fidelity`; Wave 63–72 pipeline; fuzz repl seeds; 248 suites
- **Wave 73:** REPL `finance_max_drawdown`/`info_hellinger_dist`/`quantum_trace_distance`; Wave 63–73 pipeline; fuzz repl seeds; 249 suites
- **Wave 74:** REPL `finance_kelly_fraction`/`info_renyi_entropy`/`quantum_concurrence`; Wave 63–74 pipeline; fuzz repl seeds; 250 suites
- **Wave 75:** REPL `finance_compound`/`info_redundancy`/`quantum_entanglement_entropy`; Wave 63–75 pipeline; fuzz repl seeds; 251 suites
- **Wave 76:** REPL `finance_continuous_compound`/`info_efficiency`/`quantum_expectation`; Wave 63–76 pipeline; fuzz repl seeds; 252 suites
- **Wave 77:** REPL `finance_pv`/`info_channel_capacity_bsc`/`quantum_expectation_dm`; Wave 63–77 pipeline; fuzz repl seeds; 253 suites
- **Wave 78:** REPL `finance_fv_annuity`/`info_channel_capacity_bec`/`quantum_inner`; Wave 63–78 pipeline; fuzz repl seeds; 254 suites
- **Wave 79:** REPL `finance_pmt_annuity`/`info_shannon_hartley`/`quantum_ket_normalise`; Wave 63–79 pipeline; fuzz repl seeds; 255 suites
- **Wave 80:** REPL `finance_binomial_call`/`info_differential_entropy_gaussian`/`quantum_partial_trace`; Wave 63–80 pipeline; fuzz repl seeds; 256 suites
- **Wave 81:** REPL `finance_binomial_put`/`info_differential_entropy_uniform`/`finance_bs_put`; Wave 63–81 pipeline; fuzz repl seeds; 257 suites
- **Wave 82:** REPL `finance_bs_gamma`/`finance_bond_duration`/`info_rate_distortion_gaussian`; Wave 63–82 pipeline; fuzz repl seeds; 258 suites
- **Wave 83:** REPL `finance_bs_delta`/`finance_bs_vega`/`finance_bond_modified_duration`; Wave 63–83 pipeline; fuzz repl seeds; 259 suites
- **Wave 84:** REPL `finance_bs_theta`/`finance_bs_rho`/`finance_bond_convexity`; Wave 63–84 pipeline; fuzz repl seeds; 260 suites
- **Wave 85:** REPL `finance_bond_ytm`/`info_source_coding_rate`/`info_tsallis_entropy`; Wave 63–85 pipeline; fuzz repl seeds; 261 suites
- **Wave 86:** REPL `finance_bs_implied_vol`/`finance_portfolio_return`/`finance_portfolio_variance`; Wave 63–86 pipeline; fuzz repl seeds; 262 suites
- **Wave 87:** REPL `info_joint_entropy`/`info_conditional_entropy`/`info_sample_entropy`; Wave 63–87 pipeline; fuzz repl seeds; 263 suites
- **Wave 88:** REPL `quantum_pauli_x`/`quantum_pauli_z`/`quantum_cnot_gate`; Wave 63–88 pipeline; fuzz repl seeds; 264 suites
- **Wave 89:** REPL `quantum_pauli_y`/`quantum_swap_gate`/`quantum_identity`; Wave 63–89 pipeline; fuzz repl seeds; 265 suites
- **Wave 90:** REPL `quantum_hadamard_gate`/`cplx_hyperbolic_distance`/`info_lz_complexity`; Wave 63–90 pipeline; fuzz repl seeds; 266 suites
- **Wave 91:** REPL `quantum_pauli_plus`/`quantum_pauli_minus`/`quantum_toffoli_gate`; Wave 63–91 pipeline; fuzz repl seeds; 267 suites
- **Wave 92:** REPL `quantum_rotation_z`/`quantum_rotation_x`/`quantum_rotation_y`; Wave 63–92 pipeline; fuzz repl seeds; 268 suites
- **Wave 93:** REPL `quantum_phase_gate`/`quantum_qft_gate`/`cplx_poisson_kernel`; Wave 63–93 pipeline; fuzz repl seeds; 269 suites
- **Wave 94:** REPL `tensorops_inner`/`geo_dist3d`/`numthy_isprime`; Wave 63–94 pipeline; fuzz repl seeds; 270 suites
- **Wave 95:** REPL `combo_stirling2`/`numthy_euler_phi`/`numthy_mobius`; Wave 63–95 pipeline; fuzz repl seeds; 271 suites
- **Wave 96:** REPL `combo_catalan`/`combo_bell`/`numthy_num_divisors`; Wave 63–96 pipeline; fuzz repl seeds; 272 suites
- **Wave 97:** REPL `combo_stirling1`/`numthy_lcm`/`numthy_mod_pow`; Wave 63–97 pipeline; fuzz repl seeds; 273 suites
- **Wave 98:** REPL `combo_motzkin`/`combo_permutations`/`numthy_sum_divisors`; Wave 63–98 pipeline; fuzz repl seeds; 274 suites
- **Wave 99:** REPL `numthy_nextprime`/`numthy_liouville`/`combo_subfactorial`; Wave 63–99 pipeline; fuzz repl seeds; 275 suites
- **Wave 100:** REPL `numthy_prime_pi`/`numthy_legendre_symbol`/`combo_combinations_with_rep`; Wave 63–100 pipeline; fuzz repl seeds; 276 suites
- **Wave 101:** REPL `numthy_prevprime`/`combo_double_factorial`/`numthy_jacobi_symbol`; Wave 63–101 pipeline; fuzz repl seeds; 277 suites
- **Wave 102:** REPL `numthy_prime_nth`/`numthy_kronecker_symbol`/`numthy_tonelli_shanks`; Wave 63–102 pipeline; fuzz repl seeds; 278 suites
- **Wave 103:** REPL `ml_precision`/`ml_recall`/`ml_mae`; Wave 63–103 pipeline; fuzz repl seeds; 279 suites
- **Wave 104:** REPL `ml_huber`/`ml_hinge`/`numthy_mod_inv`; Wave 63–104 pipeline; fuzz repl seeds; 280 suites
- **Wave 105:** REPL `ml_binary_crossentropy`/`numthy_is_primitive_root`/`numthy_discrete_log`; Wave 63–105 pipeline; fuzz repl seeds; 281 suites
- **Wave 106:** REPL `ml_vec_norm`/`numthy_factor_count`/`geo_polygon_perimeter`; Wave 63–106 pipeline; fuzz repl seeds; 282 suites
- **Wave 107:** REPL `ml_vec_dot`/`numthy_primitive_root`/`geo_triangle_area`; Wave 63–107 pipeline; fuzz repl seeds; 283 suites
- **Wave 108:** REPL `geo_dist_sq2d`/`geo_vec2d_length`/`geo_cross2d`; Wave 63–108 pipeline; fuzz repl seeds; 284 suites
- **Wave 109:** REPL `geo_signed_area`/`geo_moment_of_inertia`/`geo_dist_point_seg2d`; Wave 63–109 pipeline; fuzz repl seeds; 285 suites
- **Wave 110:** REPL `geo_point_in_polygon`/`ml_categorical_crossentropy`/`geo_overlap_circles`; Wave 63–110 pipeline; fuzz repl seeds; 286 suites
- **Wave 111:** REPL `geo_bezier_eval_x`/`geo_bezier_eval_y`; Wave 63–111 pipeline; fuzz repl seeds; 287 suites
- **Wave 112:** REPL `quantum_ket_basis`/`quantum_fock_state`/`quantum_identity_n`; Wave 63–112 pipeline; fuzz repl seeds; 288 suites
- **Wave 113:** REPL `control_is_controllable`/`quantum_ket_superposition`/`numthy_extended_gcd`; Wave 63–113 pipeline; fuzz repl seeds; 289 suites
- **Wave 114:** REPL `mtf_encode_vec`/`geo_centroid_x`/`quantum_ghz_state`; Wave 63–114 pipeline; fuzz repl seeds; 290 suites
- **Wave 115:** REPL `control_is_observable`/`mtf_decode_vec`/`numthy_crt`; Wave 63–115 pipeline; fuzz repl seeds; 291 suites
- **Wave 116:** REPL `geo_centroid_y`/`quantum_w_state`/`numthy_divisors_vec`; Wave 63–116 pipeline; fuzz repl seeds; 292 suites
- **Wave 117:** REPL `bwt_encode_vec`/`bwt_primary_index`/`bwt_decode_vec`; Wave 63–117 pipeline; fuzz repl seeds; 293 suites
- **Wave 118:** REPL `control_impulse_final`/`combo_multinomial`/`numthy_factor_vec`; Wave 63–118 pipeline; fuzz repl seeds; 294 suites
- **Wave 119:** REPL `lzw_encode_vec`/`lzw_decode_vec`/`quantum_coherent_state`; Wave 63–119 pipeline; fuzz repl seeds; 295 suites
- **Wave 120:** REPL `huffman_encode_vec`/`huffman_decode_vec`/`geo_volume_tetrahedron`; Wave 63–120 pipeline; fuzz repl seeds; 296 suites
- **Wave 121:** REPL `control_lyap`/`combo_rank_permutation`/`combo_unrank_permutation`; Wave 63–121 pipeline; fuzz repl seeds; 297 suites
- **Wave 122:** REPL `control_lqr`/`combo_rank_combination`/`lz77_encode_vec`/`lz77_decode_vec`; Wave 63–122 pipeline; fuzz repl seeds; 298 suites
- **Wave 123:** REPL `control_pidtune_kp`/`quantum_bell_state`/`bzip2_compress_vec`/`bzip2_decompress_vec`; Wave 63–123 pipeline; fuzz repl seeds; 299 suites
- **Wave 124:** REPL `control_place`/`diffgeo_principal_curvature_sphere`/`topo_euler_sphere_surface`; Wave 63–124 pipeline; fuzz repl seeds; 300 suites
- **Wave 125:** REPL `combo_unrank_combination`/`control_pidtune_ki`/`control_pidtune_kd`; Wave 63–125 pipeline; fuzz repl seeds; 301 suites
- **Wave 126:** REPL `cplx_power_series_eval`/`cplx_winding_number`/`quantum_schrodinger`; Wave 63–126 pipeline; fuzz repl seeds; 302 suites
- **Wave 127:** REPL `topo_vietoris_rips_betti0`/`diffgeo_gaussian_curvature_sphere`/`control_dlyap`; Wave 63–127 pipeline; fuzz repl seeds; 303 suites
- **Wave 128:** REPL `lz77_encode_vec(M,window,lookahead)`/`control_riccati`/`control_dare`; Wave 63–128 pipeline; fuzz repl seeds; 304 suites
- **Wave 129:** REPL `control_bode_mag_db`/`cplx_residue_inv`/`cplx_contour_integral_oneoverz_im`; Wave 63–129 pipeline; fuzz repl seeds; 305 suites
- **Wave 130:** REPL `quantum_time_evolution`/`topo_betti_curve`/`diffgeo_mean_curvature_sphere`; Wave 63–130 pipeline; fuzz repl seeds; 306 suites
- **Wave 131:** REPL `control_bode_phase`/`control_phase_margin`/`combo_derangements`; Wave 63–131 pipeline; fuzz repl seeds; 307 suites
- **Wave 132:** REPL `cplx_line_integral_one`/`quantum_density_matrix`/`topo_bottleneck_distance`; Wave 63–132 pipeline; fuzz repl seeds; 308 suites
- **Wave 133:** REPL `diffgeo_christoffel_sphere`/`finance_bond_price(c,y,n,fv)`/`control_gain_margin`; Wave 63–133 pipeline; fuzz repl seeds; 309 suites
- **Wave 134:** REPL `finance_compound(cpp)`/`combo_all_permutations`/`control_bode`; Wave 63–134 pipeline; fuzz repl seeds; 310 suites
- **Wave 135:** REPL `quantum_op_apply`/`topo_persistence_diagram`/`diffgeo_geodesic_euclidean`; Wave 63–135 pipeline; fuzz repl seeds; 311 suites
- **Wave 136:** REPL `compress_bits_to_bytes`/`cplx_blaschke_product`/`graph_diameter`; Wave 63–136 pipeline; fuzz repl seeds; 312 suites
- **Wave 137:** REPL `compress_bytes_to_bits`/`graph_radius`/`combo_all_subsets`; Wave 63–137 pipeline; fuzz repl seeds; 313 suites
- **Wave 138:** REPL `control_margins`/`topo_wasserstein_distance`/`diffgeo_ricci_scalar_sphere`; Wave 63–138 pipeline; fuzz repl seeds; 314 suites
- **Wave 139:** REPL `quantum_schrodinger_final`/`graph_betweenness`/`imcrop`; Wave 63–139 pipeline; fuzz repl seeds; 315 suites
- **Wave 140:** REPL `medfilt2`/`bilateral`/`canny`; Wave 63–140 pipeline; fuzz repl seeds; 316 suites
- **Wave 141:** REPL `combo_all_compositions`/`combo_all_partitions`/`graph_closeness`; Wave 63–141 pipeline; fuzz repl seeds; 317 suites
- **Wave 142:** REPL `graph_degree_centrality`/`diffgeo_einstein_scalar_sphere`/`cplx_joukowski_inv`; Wave 63–142 pipeline; fuzz repl seeds; 318 suites
- **Wave 143:** REPL `graph_max_flow`/`diffgeo_surface_normal_sphere`/`quantum_commutator`; Wave 63–143 pipeline; fuzz repl seeds; 319 suites
- **Wave 144:** REPL `stats_correlation`/`signal_moving_average`/`geo_delaunay_2d`; Wave 63–144 pipeline; fuzz repl seeds; 320 suites
- **Wave 170:** REPL `geo_kdtree_nearest`/`topo_pairwise_distances`/`numthy_continued_fraction`; Wave 63–170 pipeline; fuzz repl seeds; 346 suites
- **Wave 171:** REPL `combo_next_perm`/`cplx_mobius_re`/`boxfilter`; Wave 63–171 pipeline; fuzz repl seeds; 347 suites
- **Wave 172:** REPL `geo_voronoi`/`numthy_convergents`/`ml_mat_transpose`; Wave 63–172 pipeline; fuzz repl seeds; 348 suites
- **Wave 173:** REPL `combo_next_comb`/`numthy_primes`/`graph_scc`; Wave 63–173 pipeline; fuzz repl seeds; 349 suites
- **Wave 174:** REPL `geo_hermite_x`/`ml_mat_mul`/`stats_min_value`; Wave 63–174 pipeline; fuzz repl seeds; 350 suites
- **Wave 175:** REPL `count_components`/`prewitt`/`fftshift`; Wave 63–175 pipeline; fuzz repl seeds; 351 suites
- **Wave 169:** REPL `prob_chi2_pdf`/`stats_two_sample_ttest`/`stats_chi2_gof`; Wave 63–169 pipeline; fuzz repl seeds; 345 suites
- **Wave 168:** REPL `prob_t_cdf`/`stats_iqr`/`fft_fft2`; Wave 63–168 pipeline; fuzz repl seeds; 344 suites
- **Wave 167:** REPL `prob_gamma_pdf`/`stats_ztest`/`stats_acf`; Wave 63–167 pipeline; fuzz repl seeds; 343 suites
- **Wave 166:** REPL `prob_chi2_cdf`/`stats_mad`/`graph_astar`; Wave 63–166 pipeline; fuzz repl seeds; 342 suites
- **Wave 165:** REPL `prob_exp_pdf`/`stats_rms`/`fft_ifft`; Wave 63–165 pipeline; fuzz repl seeds; 341 suites
- **Wave 164:** REPL `prob_pois_cdf`/`stats_harmonic_mean`/`signal_bandpass`; Wave 63–164 pipeline; fuzz repl seeds; 340 suites
- **Wave 163:** REPL `prob_uniform_cdf`/`stats_ttest`/`graph_bellman_ford_dist`; Wave 63–163 pipeline; fuzz repl seeds; 339 suites
- **Wave 162:** REPL `prob_pois_pdf`/`stats_geometric_mean`/`signal_highpass`; Wave 63–162 pipeline; fuzz repl seeds; 338 suites
- **Wave 161:** REPL `prob_binom_cdf`/`signal_butterworth`/`graph_euler_circuit`; Wave 63–161 pipeline; fuzz repl seeds; 337 suites
- **Wave 160:** REPL `prob_exp_cdf`/`fft_dft`/`graph_greedy_colour`; Wave 63–160 pipeline; fuzz repl seeds; 336 suites
- **Wave 159:** REPL `prob_binom_pdf`/`graph_topological_sort`/`stats_mode`; Wave 63–159 pipeline; fuzz repl seeds; 335 suites
- **Wave 158:** REPL `graph_dfs`/`stats_percentile`/`signal_lowpass`; Wave 63–158 pipeline; fuzz repl seeds; 334 suites
- **Wave 157:** REPL `prob_norm_ppf`/`signal_triangular`/`graph_is_tree`; Wave 63–157 pipeline; fuzz repl seeds; 333 suites
- **Wave 156:** REPL `graph_bfs`/`poly_compose`/`stats_var`; Wave 63–156 pipeline; fuzz repl seeds; 332 suites
- **Wave 155:** REPL `stats_kurtosis`/`prob_norm_pdf`/`signal_parzen`; Wave 63–155 pipeline; fuzz repl seeds; 331 suites
- **Wave 154:** REPL `poly_sub`/`fft_dst2`/`prob_norm_cdf`; Wave 63–154 pipeline; fuzz repl seeds; 330 suites
- **Wave 153:** REPL `graph_mst_prim`/`signal_blackman`/`stats_skewness`; Wave 63–153 pipeline; fuzz repl seeds; 329 suites
- **Wave 152:** REPL `stats_stddev`/`fft_idct2`/`poly_mul`; Wave 63–152 pipeline; fuzz repl seeds; 328 suites
- **Wave 151:** REPL `stats_kendall`/`graph_mst_kruskal`/`signal_correlate`; Wave 63–151 pipeline; fuzz repl seeds; 327 suites
- **Wave 150:** REPL `fft_dct2`/`poly_add`/`quantum_tensor_product`; Wave 63–150 pipeline; fuzz repl seeds; 326 suites
- **Wave 149:** REPL `stats_median`/`graph_is_connected`/`signal_hanning`; Wave 63–149 pipeline; fuzz repl seeds; 325 suites
- **Wave 148:** REPL `poly_integ`/`stats_spearman`/`signal_hamming`; Wave 63–148 pipeline; fuzz repl seeds; 324 suites
- **Wave 147:** REPL `fft_irfft`/`signal_convolve`/`graph_floyd_warshall`; Wave 63–147 pipeline; fuzz repl seeds; 323 suites
- **Wave 146:** REPL `poly_eval`/`graph_is_dag`/`stats_mean`; Wave 63–146 pipeline; fuzz repl seeds; 322 suites
- **Wave 145:** REPL `fft_rfft`/`graph_is_bipartite`/`poly_deriv`; Wave 63–145 pipeline; fuzz repl seeds; 321 suites
- **Wave 176:** `ms::pde` heat2d/wave1d/advection1d/poisson2d/burgers1d solvers; `ms::symbolic` sub/div/neg/tan/sqrt ops + integration + substitution; `ms::special` zeta/eta/polylog/clausen accuracy fixes + new `debye` DLMF function; `ms::ode` backward_euler/bvp_shooting/dde/event_detect stiff & advanced solvers; REPL `clausen`/`eta_dirichlet`/`debye` bindings; 354 suites
- **Wave 177:** `ms::pde` Crank-Nicolson `heat_1d_cn`, direct `poisson_1d`, explicit `wave_2d` solvers; `ms::ode` vector `backward_euler_vec` + `dae_index1` (semi-explicit index-1 DAE); `ms::izaac` new `bloom`/`ratelimit`/`diffpriv`/`backtest` application namespaces; `ms::cypha` `nig_cdf`/`nig_sample`/`gh_gate`/`predict_interval`; `ms::cellai` `energy`/`consolidate`; REPL bindings + fuzz seeds + integration test for all 6 `pde_*` solvers (heat_1d/heat_2d/wave_1d/advection_1d/poisson_2d/burgers_1d); 355 suites
- **Wave 178:** `ms::pde` Fisher-KPP `reaction_diffusion_1d` + ADI `heat_2d_cn_adi`; `ms::ode` symplectic `ode_verlet`/`ode_verlet_vec` + adaptive vector `ode_rk45_vec`; `ms::core::Sym` upgraded from a string-concatenation stub to a real recursive-descent expression parser/evaluator (fixes `Axiom::evaluate`, previously a no-op passthrough); `ms::symbolic` string formula parser `sym_parse` + REPL `sym_diff`/`sym_simplify`/`sym_integrate`/`sym_eval` bindings (first REPL exposure of the symbolic engine); `docs/API.md` synced through Wave 177; 356 suites
- **Wave 179:** `ms::ode` A-stable `ode_bdf2` (2nd-order backward differentiation, bootstrapped via BDF1); `ms::izaac` `crypto` (CSPRNG stream cipher, honestly documented as non-production-grade) + `mpc` (Shamir's Secret Sharing over a prime field); `ms::ml` `RandomForest` + `GradientBoosting` ensembles built on the existing `DecisionTree`; REPL `ode_euler`/`ode_rk4`/`ode_midpoint`/`ode_rk45`/`ode_backward_euler` bindings via the `sym_parse` formula bridge — e.g. `ode_rk4("y", 0, 1, 1, 100)` — resolving the long-deferred callback-argument REPL gap; `docs/API.md` synced through Wave 178; 357 suites
- **Wave 180:** `ms::stats` `one_way_anova`/`mann_whitney_u`/`ks_test_2sample` hypothesis tests; `ms::izaac` `consensus` namespace — simulated, deterministic Raft-style leader election + log replication (honestly documented as in-memory simulation, not a real distributed system); `ms::ml` `SVM` classifier via SMO with linear/RBF kernels; REPL formula bridge extended to `ode_bdf2`, `ode_verlet`/`ode_verlet_vec`, `ode_euler_vec`/`ode_rk4_vec`/`ode_rk45_vec` — vector systems use semicolon-separated per-component formulas, e.g. `ode_rk4_vec("y1; -y0", 0, [1, 0], 6.283, 200)` for a harmonic oscillator; `docs/API.md` synced through Wave 179; 357 suites
- **Wave 181:** REPL bindings for 13 previously-unexposed pure functions across `ms::gria`/`ms::cypha`/`ms::cellai`/`ms::izaac` (entropy, matrix_alpha, NIG fit/pdf/cdf/sample, Hebbian update, RBM energy, Monte Carlo pi, differential-privacy noise); `ms::ml` `GaussianMixture` (diagonal-covariance EM soft clustering); `ms::signal` `welch_psd`/`spectrogram` (reusing the existing FFT module); `ms::stats` `bootstrap_ci` (percentile method); **critical fix:** `ms::izaac::CSPRNG` was silently producing badly-biased output (only 8 of 32 state bytes mutated per call, weak single-pass seed mixing) — caught via manual smoke-testing when `izaac_estimate_pi` returned 4.0 and 3.0 instead of ~3.14159; replaced with a proper xoshiro256**-based generator with SplitMix64 seed expansion, fixing every RNG-dependent feature across the Izaac framework (Monte Carlo estimation, differential privacy noise, secret sharing, sampling); `docs/API.md` synced through Wave 180; 358 suites
- **Wave 182:** REPL session-object registry (`std::variant`-backed named-handle store on `Interpreter`) exposing the first stateful framework classes to the REPL — `bloom_new`/`bloom_insert`/`bloom_check`, `tokenbucket_new`/`tokenbucket_consume`/`tokenbucket_available`, `cellmemory_new`/`cellmemory_step`/`cellmemory_recall`/`cellmemory_consolidate`, plus `session_objects()`/`session_object_clear()` introspection; statistical-quality regression tests (chi-square uniformity, two-sample KS, lag-1 serial correlation, seed avalanche, Monte Carlo pi accuracy) added for `ms::izaac::CSPRNG` and RNG-dependent sampling functions to guard against a repeat of the Wave 181 bias bug; `ms::prob` gains `gamma_cdf`/`beta_pdf`/`beta_cdf`/`f_pdf`/`f_cdf` (via existing incomplete-gamma/incomplete-beta primitives), verified against exact distributional identities (Beta(1,1)≡Uniform, Gamma(1,·)≡Exponential, F(1,df) vs. t²); `docs/API.md` synced through Wave 181; 360 suites
- **Wave 183:** REPL session-object registry (Wave 182) extended to `ms::cypha::DifModel` (`difmodel_new`/`update`/`predict`/`predict_interval`/`ood_score`/`gh_gate`) and `ms::izaac::consensus::Cluster` (`cluster_new`/`run_election`/`replicate`/`current_leader`/`status`), completing REPL exposure for all previously-deferred stateful framework classes; `ms::optim` gains `cmaes` — a full Covariance Matrix Adaptation Evolution Strategy global optimizer (rank-1/rank-mu covariance updates, step-size control, log-weighted recombination) that outperforms the existing DE/PSO optimizers on ill-conditioned/curved objectives like the Rosenbrock valley; `ms::stats` gains `kruskal_wallis` (non-parametric multi-group rank test with tie correction, the natural generalization of `mann_whitney_u`), and `one_way_anova`'s F-distribution p-value now reuses `ms::prob::f_cdf` instead of a duplicated incomplete-beta implementation; 361 suites
- **Wave 184:** REPL formula bridge (Waves 179-180) extended to the last 5 deferred ODE bindings — `ode_backward_euler_vec`, `ode_dae_index1` (two formula-string arguments for the differential/algebraic parts), `ode_bvp_shooting` (shooting method, verified against the exact `y''=-y` -> `sin(t)` solution), `ode_dde_fixed_step` (delay differential equations, with a separate history-function formula), and `ode_event_detect` (root-crossing event detection, verified against an exact analytic crossing time) — closing out the REPL formula-bridge gap tracked since Wave 179; `ms::axiom::Axiom::evolve` now performs genuine genetic-programming tree evolution (an internal `GPNode` tree population with diverse random initialization, tournament selection, subtree crossover/mutation, and elitism, synced to `Algorithm::representation` `Sym` strings) instead of the previous placeholder that directly perturbed fitness scores without ever touching the expression trees; `ms::fft` gains `ifft2` (inverse 2D FFT, matching `fft2`'s dimension-inference convention) and `idst2` (inverse DST, exploiting `dst2`'s self-inverse orthogonal structure), filling two previously-flagged API gaps; 362 suites
- **Wave 185:** `ms::axiom::PrimitiveRegistry` rescoped from a leftover list of matrix-level operations (`matmul`/`solve`/`lu`/...) that were never actually consulted by the Wave 184 GP-tree rewrite, to `ms::Sym`'s real 6-function scalar grammar (`sin`/`cos`/`exp`/`log`/`sqrt`/`tanh`), now wired through as the genuine single source of truth for unary-function selection at every GP tree-generation site (initial population, mutation, depth-limit enforcement) — closing the architecture-mismatch follow-up flagged in Wave 184; `ms::tensorops` gains `reconstruct_cp`/`reconstruct_tucker`, rebuilding an approximated dense tensor from `CPDecomposition`/`TuckerDecomposition` factors (the natural "inverse" of `decompose_cp`/`decompose_hosvd`/`decompose_tucker`, needed for compression/verification workflows), which also caught and fixed a stale-residual bug in `decompose_cp`'s own ALS loop; new Google Benchmark coverage (`bench_optim_ml`) for `ms::optim`'s global optimizers (`simulated_annealing`/`differential_evolution`/`particle_swarm`/`cmaes`) and `ms::ml`'s heavier ensemble/clustering methods (`RandomForest`/`GradientBoosting`/`SVM`/`GaussianMixture`), which previously had zero hot-path benchmark coverage despite the project's stated Phase 10 benchmarking policy; 362 suites
- **Wave 186:** `ms::control` gains `c2d`/`d2c` (continuous↔discrete conversion for `StateSpace` and `TransferFunction`, via Zero-Order-Hold/Tustin/Euler, using the augmented-matrix `expm`/`logm` trick so it works correctly even on singular/marginally-stable plants like a pure integrator) — a fundamental control-engineering operation that was previously entirely absent, with no on-ramp from a continuous-time design to the module's existing `dlyap`/`dare` discrete-time solvers; `ms::prob` gains `chi2_ppf`/`t_ppf`/`f_ppf`/`gamma_ppf`/`beta_ppf`/`exp_ppf` quantile functions (via bisection on the existing CDFs, or closed-form for exponential), completing the CDF↔PPF pairing started with `norm_ppf` and Wave 182's `gamma_cdf`/`beta_cdf`/`f_cdf`, and directly enabling critical-value/confidence-interval computations for the project's hypothesis tests, verified against standard statistics-textbook critical values (e.g. `chi2_ppf(0.95, 1) ≈ 3.841`); REPL bindings wired for nine previously-unexposed pure functions from Waves 182–185 (`gamma_cdf`, `beta_pdf`/`beta_cdf`, `f_pdf`/`f_cdf`, `kruskal_wallis`, `cmaes` via the `sym_parse` formula bridge, `ifft2`, `idst2`) — `reconstruct_cp`/`reconstruct_tucker` were investigated but deliberately deferred since their `decompose_cp`/`decompose_hosvd`/`decompose_tucker` decomposition-struct inputs aren't yet REPL-representable; 363 suites
- **Wave 187:** REPL exposure for `ms::tensorops`' CP/Tucker/HOSVD tensor decompositions via the Wave 182 session-object registry (`tensorops_decompose_cp`/`tensorops_decompose_tucker`/`tensorops_decompose_hosvd` create a named handle; `tensorops_reconstruct_cp`/`tensorops_reconstruct_tucker` rebuild the tensor from it, on 2D matrices, consistent with the module's existing 2D-only REPL surface) — closes the follow-up flagged at the end of Wave 186; `ms::special` gains 12 functions explicitly planned in the project's original design spec but never implemented — `erfinv`/`erfcinv` (Newton-refined), `trigamma`/`polygamma` (via the existing `zeta_hurwitz`), `pochhammer`/`falling_factorial`, and publicly-exposed `gamma_inc_reg`/`gamma_inc_reg_upper`/`gamma_inc`/`beta_inc_reg`/`beta_inc`/`rgamma` — with `ms::prob`'s `gamma_cdf`/`beta_cdf`/`f_cdf`/`chi2_cdf` refactored to delegate to the now-public `ms::special` implementations instead of duplicating them privately; `ms::quantum` gains `uncertainty`/`eigenspectrum`/`ground_state` (three more spec-planned-but-missing functions, using the module's existing Jacobi Hermitian diagonalizer), verified against the generalized uncertainty relation and known closed-form spectra. 364 suites
- **Wave 188:** `ms::signal` gains the analytic-signal family — `hilbert` (FFT-based discrete Hilbert transform), `envelope`, `instantaneous_phase`, `instantaneous_freq` (with phase unwrapping) — plus lag-windowed `xcorr`/`xcov`/`autocorr`, 2D `conv2`, and polynomial `deconv` (built on the existing `ms::poly::poly_div_quot`); `ms::stats` gains `jarque_bera` (normality test via skewness/kurtosis), `ljung_box` (autocorrelation test via `acf`), and `levene_test`/`bartlett_test` (equal-variance tests, the natural companions to `one_way_anova`'s equal-variance assumption); `ms::ml` gains `LDA` (shared-covariance linear discriminant classifier, with supervised dimensionality-reduction `transform`) and `QDA` (per-class-covariance quadratic discriminant classifier) — closing three more tractable spec-vs-implementation gaps identified by the systematic cross-check technique. 364 suites (63 new test cases across existing files, no new CTest registrations)
- **Wave 189:** `ms::graph` gains `articulation_points`/`bridges` (undirected DFS low-link, the natural connectivity-analysis companions to the existing directed `strongly_connected_components`), `eigenvector_centrality`/`katz_centrality` (power iteration, mirroring the existing `pagerank`), `laplacian`/`normalised_laplacian`/`algebraic_connectivity` (Fiedler value, reusing the existing adjacency eigensolver), and `min_cut` (extracted from a shared Dinic residual-graph helper alongside the existing `max_flow`, verified consistent with it by the max-flow min-cut theorem); `ms::finance` gains `information_ratio`/`treynor_ratio`/`capm`/`forward_rate` (portfolio/rate one-liners), `digital_option`/`black76`/`barrier_option` (closed-form exotic option pricing, with `black76` verified equivalent to spot Black-Scholes at the matching forward price, and `barrier_option` verified via the knock-in+knock-out=vanilla identity), and `american_option` (binomial tree with early exercise, verified equal to the European call absent dividends and strictly greater for a deep-ITM put); `ms::numthy` gains `jordan_totient` (verified `J_1 ≡ euler_phi`), `von_mangoldt`, `quadratic_residues`, `farey`, and `pell_solve` (fundamental solution via integer continued-fraction period arithmetic, verified against known D=2/3/5/7 solutions) — three more mature modules closed via the ongoing systematic spec-vs-implementation cross-check. 364 suites (85 new test cases across existing files, no new CTest registrations)
- **Wave 190:** `ms::poly` gains `poly_pow`/`poly_monic`/`poly_reverse`/`poly_shift`/`poly_scale`/`poly_lcm` and, most significantly, `poly_sylvester`/`poly_resultant`/`poly_discriminant`/`poly_squarefree` (resultant verified zero iff sharing a root, discriminant verified against the quadratic formula and zero-iff-repeated-root, squarefree verified to strip multiplicity) plus `bernstein` (verified via the partition-of-unity identity) — along with a numerical-robustness fix to `poly_mod`/`poly_div_quot`/`poly_gcd`'s termination logic for near-zero leading coefficients, needed to make the repeated-root discriminant/squarefree cases work correctly; `ms::image` gains `roberts`/`laplacian_of_gaussian`/`imgradient_morph` (compositions of existing filters/morphology) and `shi_tomasi`/`iradon` (min-eigenvalue corner detection sharing `harris`'s structure tensor, and inverse Radon via unfiltered backprojection matching the existing `radon`'s convention, verified via a round-trip reconstruction test); `ms::bignum` gains `bigint_extended_gcd` (Bezout coefficients), `bigint_bit_length`/`is_even`/`is_odd`, base-2/8/16 string I/O, and `Rational::floor`/`ceil`/`round` (with correct negative-number rounding behavior). 364 suites (60 new test cases across existing files, no new CTest registrations)
- **Wave 191:** New `ms::core` module (`include/ms/core/checked_arith.hpp`) — type-generic `checked_add`/`sub`/`mul`/`div`/`mod`/`neg`/`abs`/`pow` returning `Result<T>` on overflow or domain error (division/modulo by zero, `INT_MIN`-negation, etc.), `saturating_add`/`sub`/`mul` and `wrapping_add`/`sub`/`mul` for integers, float introspection helpers (`is_nan`/`is_inf`/`is_finite`/`is_normal`/`signbit`/`ulp`/`eps`/`huge`/`tiny`), and checked `narrow`/`widen` casts — closing the `ms::core` cross-check gap with MathScript's first dedicated overflow-safety primitives; `ms::combo` gains `set_partitions`/`involutions`/`dyck_paths`/`motzkin_paths`/`necklaces`/`lyndon_words` — explicit enumerations (not just counts) that complement the module's existing `bell_num`/`catalan_num`/`motzkin_num`, cross-verified by matching enumerated counts against those closed-form formulas; `ms::compress` gains `arithmetic_encode`/`arithmetic_decode` — a byte-oriented Subbotin/Schindler-style range coder (64-bit accumulator, 32-bit range/code, underflow-avoidance via range clamping on renormalization) that is the natural entropy-optimal companion to the existing Huffman coding, verified via round-trip on skewed/uniform/random-looking/all-256-symbol inputs and shown to match-or-beat Huffman's bit count on skewed English text. 365 suites (1 new CTest registration: `test_checked_arith`; 78 new test cases total)
- **Wave 192:** `ms::numthy` gains `cornacchia` (sum-of-two-squares `x^2+d*y^2=p` via Tonelli-Shanks plus partial Euclidean reduction, verified via the reconstruction identity rather than literal outputs since either root ordering is valid) and `stern_brocot` (BFS mediant enumeration of the Stern-Brocot tree, verified reduced-form/distinctness/root); `ms::signal` gains the resampling family `upsample`/`downsample`/`decimate`/`interpolate`/`resample` (zero-stuffing and lowpass-filtered rational rate conversion, `decimate`/`interpolate` deriving their cutoff from a 0.8-of-new-Nyquist margin) — building these surfaced and fixed a real latent bug in the shared `fft_lowpass` helper (used by `lowpass`/`butterworth`/`highpass`/`bandpass`): its high-frequency zeroing loop's mirror-index arithmetic wrapped around and zeroed nearly all non-DC spectral bins regardless of the requested cutoff, silently making "low-pass" filtering nearly a no-op beyond the mean; replaced with a direct wrapped-distance-from-DC check (also corrected the one pre-existing test that had accidentally been relying on the old bug); `ms::finance` gains Monte Carlo option pricing — `mc_european_call`/`mc_european_put` (antithetic-variate GBM terminal-value sampling, verified to converge to the existing closed-form `bs_call`/`bs_put` within Monte-Carlo-standard-error tolerance, plus put-call parity) and `mc_asian_call`/`mc_asian_put` (discretized-path arithmetic-average simulation, verified cheaper than the equivalent European option per the well-known volatility-averaging result). 365 suites (no new CTest registrations; 47 new test cases across existing `test_numthy`/`test_signal_filters`/`test_finance` binaries, plus 1 pre-existing test corrected for the `fft_lowpass` fix).
- **Wave 193:** `ms::geo` gains `convex_hull_3d` (brute-force O(n^4) supporting-plane enumeration, matching `convex_hull_2d`'s simple-over-optimal style; faces with more than 3 coplanar hull vertices, e.g. a cube's square faces, are fan-triangulated via a 2D convex hull of the projected coplanar points so interior/redundant coplanar points are correctly excluded — verified via tetrahedron/cube/octahedron/icosahedron face counts, interior-point exclusion, outward-normal + one-sidedness re-derivation on the output, and Euler-characteristic `V-E+F=2` structural checks); `ms::finance` gains Markowitz mean-variance portfolio optimization — `min_variance_portfolio`, `efficient_frontier_portfolio`, and `max_sharpe_portfolio`, each solving `Σy=b` via a private partial-pivoting Gaussian eliminator (mirroring the existing `gauss_solve` precedent in `ms::control`), verified against the closed-form uncorrelated-asset weighting, the Lagrange/KKT `Σw=λ·1` optimality condition, and the tangency portfolio's defining max-Sharpe property versus hand-picked alternative weightings; `ms::poly` gains `interp_newton` (Newton divided-difference interpolation) and `interp_hermite` (derivative-constrained interpolation via doubled-node divided differences), verified against the existing `poly_lagrange` (same unique interpolating polynomial) and against `poly_deriv`-checked derivative values respectively — closing the module's two remaining documented interpolation gaps. 365 suites (no new CTest registrations; 53 new test cases across existing `test_geo`/`test_finance`/`test_poly_ext` binaries).
- **Wave 194:** REPL scalar bindings batch 1 — 13 previously library-only functions exposed to the interactive shell (`numthy_von_mangoldt`, `numthy_jordan_totient`, `poly_bernstein`, `finance_capm`, `finance_forward_rate`, `finance_black76`, `finance_digital_option`, `finance_american_option`, `finance_mc_european_call`/`_put`, `finance_mc_asian_call`/`_put`, `finance_barrier_option`) bound at all four `repl_engine.cpp` registration sites (generic scalar dispatcher, regex literal-call dispatcher, help cheatsheet, function listing), including a brand-new 9-argument `nonary` regex tier added for `finance_barrier_option` since no 9-arg literal-dispatch tier previously existed; `ms::finance` gains Monte Carlo lookback option pricing — `mc_lookback_floating_call`/`_put` and `mc_lookback_fixed_call`/`_put`, reusing the existing antithetic-variate GBM path-simulation infrastructure from `mc_asian_call`/`_put` but tracking the running path minimum/maximum instead of the running average, verified against vanilla-option bounds and pathwise non-negativity; `ms::graph` gains `is_isomorphic` — a simplified VF2-style backtracking vertex-bijection search with fast vertex/edge-count/directedness rejection and a sorted-degree-sequence pre-check ahead of the exhaustive search, verified via permuted-label isomorphic pairs and a genuine backtracking-required negative case ($K_{3,3}$ vs. the triangular prism — both 3-regular on 6 vertices/9 edges with identical degree sequences but different triangle counts). 365 suites (no new CTest registrations; 46 new test cases across existing `test_repl_commands`/`test_finance`/`test_graph` binaries).
- **Runtime:** Topology detection, thread pool, dispatch layer, own BLAS/LAPACK CPU kernels
- **Executables:** `mathscriptc`, `mathscript-repl`, `mathscript-server`
- **Unit tests:** 365 CTest suites — all passing (CUDA disabled)
- **CI baseline:** ~91% line coverage (**90%** enforced in CI)

### Build Instructions (Native Windows)

#### Prerequisites
- Visual Studio 2022/2026 Build Tools with C++ workload
- CMake 3.28+
- Ninja (included with Strawberry Perl or VS)

#### Build
```powershell
.\build.ps1
```

Or manually:
```powershell
cmd /c "`"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat`" -arch=amd64 && cmake -S . -B build-msvc -G Ninja -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF && cmake --build build-msvc --config Release"
```

#### Run tests
```powershell
ctest --test-dir build-msvc --output-on-failure
```

#### Run REPL
```powershell
.\build-msvc\bin\mathscript-repl.exe
# one-shot (non-interactive):
.\build-msvc\bin\mathscript-repl.exe -e "surf([1, 2; 3, 4])"
# optional LLVM ORC backend when linked:
.\build-msvc\bin\mathscript-repl.exe --jit -e "x = 1 + 2"
# load saved session:
.\build-msvc\bin\mathscript-repl.exe --load session.ms
```

### Linux (headless CI-style build)

Ubuntu 24.04 with GCC 13 (matches CI):

```bash
cmake -S . -B build-linux -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build-linux
ctest --test-dir build-linux --output-on-failure
```

For libFuzzer targets, use Clang with the project’s Linux Clang flags (see `cmake/options.cmake`).

### Coverage (Linux, GCC/Clang)

```bash
cmake -S . -B build-cov -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF \
  -DMS_ENABLE_COVERAGE=ON
cmake --build build-cov
ctest --test-dir build-cov --output-on-failure
MS_COVERAGE_MIN=90 bash scripts/coverage_report.sh build-cov
```

The CI coverage job enforces **90%** minimum line coverage.

### Valgrind (Linux)

```bash
cmake -S . -B build-valgrind -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF
cmake --build build-valgrind
bash scripts/valgrind_tests.sh build-valgrind
```

### Install and packages

```bash
cmake --install build-linux --prefix /opt/mathscript
cd build-linux && cpack -G TGZ && cpack -G ZIP
```

### CI

GitHub Actions (`.github/workflows/ci.yml`) on every push/PR to `main`:

- Windows MSVC build + full test suite + install smoke
- Linux GCC 13 build + full test suite + install/package smoke + unsafe surface audit
- Coverage report (~91% line; **90% minimum**) + artifact upload
- libFuzzer smoke on all **7** fuzz targets (4096 runs / 30s each)
- Valgrind memcheck (excludes long `test_fuzz_stress`)
- Benchmark regression (`bench_matmul`, `bench_fft`) via `benchmark-linux` job with **10% tolerance gate**
- Vendor checksum verification (`scripts/verify_vendor.sh`)

Weekly extended fuzz runs: `.github/workflows/nightly.yml` — scheduled **15 min / target × 7** (120 min job); manual **workflow_dispatch** also runs a **fuzz-marathon** job (**60 min / target × 7**, 480 min timeout).

Manual **24 h fuzz campaign**: `.github/workflows/fuzz-24h.yml` (`workflow_dispatch` only) — **7 parallel jobs**, **86400 s (24 h) / target**, **1500 min** job timeout each (Phase 10 ≥ 24 h per target).

Optional benchmarks (Linux CI): `-DMS_BUILD_BENCHMARKS=ON`, then `bash scripts/bench_smoke.sh build-bench`.

### Unsafe surface audit

```bash
bash scripts/unsafe_report.sh
```

Review entries and the approved baseline live in `UNSAFE_REVIEW.md`.

## Phase Progress

| Phase | Name | Status |
|-------|------|--------|
| 0 | Foundation | Complete |
| 1 | Core Types + REPL | Complete |
| 2 | Math Library CPU | Complete |
| 3 | GPU / CUDA | Stub + optional backend |
| 4–8 | IDE, distributed, frameworks, special functions | Substantial progress |
| 9 | Own BLAS/LAPACK core | Complete |
| 10 | Hardening (CI, coverage, packaging) | Complete — **v1.0.0** |

### Phase 10 checklist

**Done**
- **244** CTest suites passing (CUDA off in CI)
- CI green: Windows MSVC + Linux GCC build/test, install/package smoke, package artifact uploads (ZIP/TGZ)
- Coverage **~91%** line (**90%** enforced in CI)
- Valgrind memcheck on test suite (excludes long `test_fuzz_stress`)
- libFuzzer: **7 targets** (`fuzz_special_fns`, `fuzz_matrix_ops`, `fuzz_repl_input`, `fuzz_sym_parser`, `fuzz_poly_ops`, `fuzz_bignum`, `fuzz_mpi_message`) — CI smoke + nightly 15 min each
- Fuzz corpus layout under `tests/fuzz/corpus/`
- Unsafe surface audit scripted + baseline in `UNSAFE_REVIEW.md`
- Architecture + API docs (`docs/ARCHITECTURE.md`, `docs/API.md`)
- ORC JIT v2 LLVM backend + `test_jit_backend`; scalar literal/expression/libm-call JIT + native dispatch for all REPL call forms when `-DMS_BUILD_JIT=ON` (`jit-linux` CI); general matrix LLVM IR lowering is post-1.0; REPL fallback for unsupported forms
- Optim: 1D Newton-Raphson and Broyden (secant) root finders implemented
- Linux CI: conditional DEB/RPM `cpack` + `scripts/package_smoke.sh` install verification
- Windows CI: conditional NSIS `cpack` + `scripts/package_smoke.ps1` install verification (skips if `makensis` unavailable)
- Performance benchmarks: `MS_BUILD_BENCHMARKS=ON` → `bench_matmul`, `bench_fft` + `benchmark-linux` CI job
- Benchmark regression gate in CI: `linux-gcc13.json` baseline (0.1s / 5 reps / median; `MS_BENCH_TOLERANCE=10`)
- Compliance **`plugin-linux`** CI job (`MS_BUILD_PLUGIN=ON`, LLVM/Clang 18 paths pinned)
- CUDA matmul dispatch tests: `test_cuda_matmul` (GPU/AUTO CPU fallback when CUDA off)
- Vendor SHA-256 policy: `vendor/CHECKSUMS.sha256`, `scripts/verify_vendor.sh`, `verify_vendor` CMake target
- `dormbr('P',...)` heap corruption fixed (`apply_p_left_tall` for tall P-left; `apply_p_right_wide` for non-square P-right)
- `MS_UNSAFE(reason)` macro in `include/ms/unsafe/unsafe.hpp` + `compliance_unsafe_annotation` (with `MS_BUILD_PLUGIN=ON`)
- `scripts/unsafe_delta.sh` + baseline in `tests/compliance/unsafe_baseline.txt`
- Optim: box-constraint `minimize_with_constraints`, 2D Nelder-Mead `simplex_solver`
- **`docs/RELEASE.md`** — 1.0.0 tag checklist; `scripts/pre_release.sh` / `tag_1.0.0_checklist.sh` (Linux) and `tag_1.0.0_checklist.ps1` (Windows); `build.ps1 -Test`
- Clang plugin compliance rules (**20 enforced**, all `DiagnosticID`s except partial `UnsafeAudit`): **`no_raw_new`**, **`no_malloc`**, **`no_cstyle_cast`**, **`no_throw`**, **`no_catch`**, **`no_const_cast`**, **`no_goto`**, **`no_raw_ptr_arithmetic`**, **`no_unsafe_reinterpret`**, **`no_detach`**, **`no_vla`**, **`narrowing`**, **`no_signed_unsigned_mix`**, **`no_raw_thread`**, **`no_raw_mutex_lock`**, **`no_uninit`**, **`no_stored_span`**, **`no_volatile_sync`**, **`no_owning_raw_ptr`**, **`unused_expected`** (+ fail/ok tests on `plugin-linux`; `UnsafeAudit` via `MS_UNSAFE` + `scripts/unsafe_report.sh`)
- **`mathscript-repl -e` / `--load` / `--jit`** — one-shot eval, session load, optional ORC backend
- Qt **`PlotWidget`** + OpenGL **`PlotSurfWidget`** (`MS_BUILD_GUI=ON`): 2D plots; shaded 3D surf with lighting, drag rotation, wheel zoom, GUI PNG export
- REPL **`saveplot <file>`** writes ASCII plot preview; scalar **`pow`/`min`/`max`/`atan2`** and unary libm calls in expressions
- REPL matrix-call + multi-target **`lu`/`qr`/`svd`/`eig_sym`** assignments (`matmul`, `solve`, `transpose`, `chol`, scalar `det`/…)
- **`docs/CONTRIBUTING.md`** — build, test, coverage, plugin (LLVM 18), compliance layout
- Symbolic simplify expansions; `poly_sub`; unsafe delta CI now **blocking**
- `tests/numerical/` — NIST DLMF accuracy regression tests: Bessel J/Y/I/K, LU/SVD/solve/chol/eig_sym, FFT, special functions erf/gamma/digamma (26 reference tests, 4 CTest targets)
- `tests/integration/` — Cross-module pipeline tests: REPL→plot→save, session roundtrip, mathscriptc multi-line, JIT/REPL parity (5 targets)
- Typed unit tests: float/double parameterized LU, solve, chol per design brief §14.2
- `mathscript-repl --debug` trace mode (per-line timing, variable state diff, parse category)
- `mathscript-repl --eval-file <path>` script execution flag
- `tests/unit/test_memory`, `test_error_types`, `test_runtime`, `test_data_driven` — allocator, error formatting, ThreadPool, parameterized data-driven coverage
- `test_core_matrix`, `test_server_cli` — direct Matrix API and server CLI coverage
- Real bugs fixed by tests: `det()` sign error, `trace()` non-square guard, `cg`/`gmres` ConvergenceFail, `bicgstab`/`gmres` dimension guard, `erf()` in scalar assignments
- `scripts/run_tests.ps1` — Windows test runner with -Filter/-Verbose
- **Wave 4 additions:** typed unit tests for `stats`, `signal`, `sparse`, `simd`, and `fft` modules (float/double parameterized); advanced REPL session tests (save/load/eval-file/debug-trace); symbolic simplification typed suite; integration tests for REPL→plot→save pipeline and `mathscriptc` multi-line script execution
- Version bumped to **1.0.0**

**Remaining (post-1.0)**
- Extended fuzz (**24 h × 7**) — manual campaign via `gh workflow run fuzz-24h.yml` (see `docs/RELEASE.md`)
- Full ORC JIT v2 matrix LLVM IR lowering (post-1.0 enhancement)
- Windows installer / Linux packages (post-1.0 packaging)

## Documentation

- [Architecture overview](docs/ARCHITECTURE.md) — module layout, build flags, test structure, CI pipeline
- [Public API index](docs/API.md) — headers under `include/ms/` grouped by module
- `mathscript-master-plan.md` — complete 10-phase delivery plan
- [1.0.0 release checklist](docs/RELEASE.md) — tag criteria for Phase 10 exit
