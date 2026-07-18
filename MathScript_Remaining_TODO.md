# MathScript — Remaining Work & Execution Plan

**Author:** Odin Loch  
**Updated:** 2026-07-18 (Wave 238 complete — **388 CTest suites** on `main` @ Wave 238)

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

## Feature waves ✅ 231–238 COMPLETE (summary)

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

## Next (Wave 239+) — deferred

| Item | Notes |
|------|--------|
| Scalable multi-node MPI LA | Beyond stub-safe `dist_matmul` / `dist_cg` / gather-to-root solvers |
| Full IDE | clangd/LSP, debugger, rich LaTeX output (master plan §11) |
| Linux baseline medians | Refresh via `bench-baseline-linux.yml` when `gh` authenticated |
| Deeper API gaps | Per `mathscript-master-plan.md` as needed |

See `mathscript-master-plan.md` and `CHANGELOG.md` for full history.
