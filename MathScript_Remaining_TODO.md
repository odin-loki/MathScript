# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-18 (Wave 251 ✅ COMPLETE — **409 CTest suites** on `main`)

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

## Feature waves ✅ 231–251 COMPLETE (summary)

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

**409 CTest suites** — all passing on `main`. **28-bench smoke OK**. Feature waves **231–251** closed for this incremental batch.

---

## Next (Wave 252+) — deferred

| Item | Notes |
|------|--------|
| Scalable multi-node MPI LA | Beyond stub/block `dist_*` gather and block iterative solvers |
| Full IDE | clangd/LSP, debugger, rich rendered LaTeX (master plan §11) |
| Linux baseline medians | Refresh via `bench-baseline-linux.yml` when `gh` authenticated |
| Deeper API gaps | Full NCCL multi-GPU, Freund TFQMR loop, weighted blossom, etc. |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
