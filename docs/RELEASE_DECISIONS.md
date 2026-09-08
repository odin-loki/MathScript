# MathScript 1.0 scope

This file used to record gaps that were **decided** rather than remaining. Almost
all of them have since been closed; what is left is genuinely out of scope, and
is listed with the reason.

See [`RELEASE.md`](RELEASE.md) for tag criteria.

## Closed

Everything below shipped. Each entry names what replaced the stub.

| Was | Now |
|-----|-----|
| Weighted blossom matching — "unweighted matching ships" | `max_weight_matching` / `max_weight_matching_value`: Edmonds' primal-dual blossom algorithm for general graphs, cross-checked against exhaustive brute force |
| Boyer–Myrvold planarity — "`is_planar_k5_k33_check` is the honest partial" | `is_planar` (Left-Right criterion), `planar_embedding` (combinatorial embedding), `kuratowski_subgraph` (K5/K3,3 subdivision certificate). The heuristic remains as `is_planar_k5_k33_check` / REPL `graph_is_planar_heuristic` |
| SIFT/SURF/ORB, graph-cut, marching cubes — "not in tree" | `marching_cubes`, `marching_cubes_mesh`, `marching_squares`, `mesh_surface_area`, `mesh_volume`; `graph_cut_segment`, `grabcut_segment`, `min_cut_value`; `fast_corners`, `orb_detect_and_compute`, `sift_detect_and_compute`, descriptor matching. SURF is deliberately **not** implemented (patent-encumbered in some jurisdictions, and superseded by ORB/SIFT in this tree) |
| APFloat/APComplex transcendentals — "rewrite-scale" | `APFloat` on a BigInt mantissa with `ap_pi`/`ap_e`/`ap_ln2`/`ap_exp`/`ap_log`/`ap_pow`/`ap_sin`/`ap_cos`/`ap_tan`/`ap_asin`/`ap_acos`/`ap_atan`/`ap_sinh`/`ap_cosh`/`ap_tanh`/`ap_sqrt`/`ap_cbrt`, plus `APComplex`. Guard digits make the returned digits correct |
| Scalable multi-node MPI linear algebra — "block/gather `dist_*` only" | `dist_ops` communication layer (`DistVec`, `dist_dot`, `dist_norm`, `dist_axpy`, 2D Cartesian process grid), SUMMA matmul, and row-distributed Krylov solvers. The gather path remains the documented fallback for a 1×1 grid, an MPI-less build, or a problem below threshold |
| Full NCCL multi-GPU — "stubs" | Real `ncclCommInitRank`/`ncclCommDestroy` lifetime management and `ncclAllReduce`/`ncclBroadcast`/`ncclReduce`/`ncclAllGather` behind `MS_HAS_NCCL`. The default build has no NCCL and keeps the documented identity semantics |
| `axiom.cpp` placeholders | `Algorithm::evaluation` / `selection` / `mutation` record real per-individual provenance (the applied form actually evaluated, the selection operator with its parameter, the variation chain applied) instead of three constants |
| GUI placeholders in `MainWindow.cpp` | Stale entry: these were Qt `setPlaceholderText` calls on the editor/output/input widgets, not unimplemented features |
| Isolated `TODO` comments in `repl_engine.cpp` | Stale entry: the tree contains no `TODO` or `FIXME` markers |
| t-SNE "simplified Barnes-Hut stub" | Real Barnes-Hut: kNN-sparse P with a per-point bisection against the target entropy, a 2^d-tree for the repulsive term under the `cell_width/distance < theta` criterion, early exaggeration, adaptive gains |
| `qmr`, `tfqmr`, `lsmr`, `precond_ssor` | Real QMR (two-sided Lanczos + Givens QMR smoothing), TFQMR (Freund), LSMR (Fong & Saunders), and the true SSOR operator. Previously these called BiCGSTAB, BiCGSTAB, LSQR and returned `diag(A)/omega` respectively |
| `cplx::inversion` | Returned the identity Möbius; now returns the anti-Möbius acting on `conj(z)`, with `apply_inversion` performing the whole map and `cross_ratio_c` exposing the complex cross-ratio |
| `sym_dsolve` separable-only MVP | `sym_dsolve_ode` classifies separable, quadrature, linear, Bernoulli, exact and homogeneous first-order equations, verifying every candidate numerically before returning it; `sym_dsolve_linear2` handles second-order linear constant-coefficient. `sym_dsolve` itself keeps its frozen contract |
| `cech_complex` `max_dim` clamped to 2 | The MEB of any point subset now comes from the bordered Cayley–Menger system, so dimensions up to `kMaxCechDim` are built |
| `crypto::random_bytes` "MVP: std::random_device" | `getrandom(2)` with a `/dev/urandom` fallback, `BCryptGenRandom` on Windows, `arc4random_buf` on BSD/macOS |
| `mathscript-server` heartbeat loop | A real SPMD compute node: `--script` / `-e` / `--serve`, one `Interpreter` per rank, rank-0 stdout |

## Still out of scope

| Item | Reason |
|------|--------|
| Full IDE (LSP, debugger, rendered LaTeX) | Separate product, not a library gap |
| SURF | Patent-encumbered in some jurisdictions; ORB and SIFT ship instead |
| `src/interp/jit_orc_stub.cpp` | Not a stub in the pejorative sense — this *is* the non-LLVM JIT backend, selected when `MS_BUILD_JIT` is off |
| CUDA solver `"not implemented"` paths | Return `Result` errors rather than silent success; the real cuSOLVER path is compiled when CUDA is present |
