# MathScript HPC Computer Algebra System

[![CI](https://github.com/odin-loki/MathScript/actions/workflows/ci.yml/badge.svg)](https://github.com/odin-loki/MathScript/actions/workflows/ci.yml)

A high-performance Computer Algebra System built in C++23 with CPU math libraries, runtime dispatch, and a console REPL.

## Project Status: Phase 10 (Hardening) — Complete — pending 24h fuzz marathon clean run before v1.0.0 tag

Phase 9 (own BLAS/LAPACK SVD pipeline) is complete. Phase 10 adds CI, coverage reporting, Valgrind memcheck, fuzz testing, and install/packaging. **CI on `main`:** Windows/Linux build-test, coverage, fuzz smoke, Valgrind, plugin-linux, jit-linux, benchmark-linux, unsafe delta.

### Completed Components

- **Build system:** Native Windows MSVC + Ninja; Linux GCC 13 / Clang; GoogleTest via FetchContent
- **Core types:** Matrix, Tensor, Sparse, Scalar, Sym
- **Linear algebra:** matmul, LU, QR, Cholesky, solve, eig, SVD, expm, trace, det, norm, and helpers
- **Math modules:** fft, stats, prob, optim, signal, special (CPU implementations)
- **Runtime:** Topology detection, thread pool, dispatch layer, own BLAS/LAPACK CPU kernels
- **Executables:** `mathscriptc`, `mathscript-repl`, `mathscript-server`
- **Unit tests:** 72 CTest suites — all passing (CUDA disabled)
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
| 10 | Hardening (CI, coverage, packaging) | Complete — pending 24h fuzz marathon before **v1.0.0** tag |

### Phase 10 checklist

**Done**
- **72** CTest suites passing (CUDA off in CI)
- CI green: Windows MSVC + Linux GCC build/test, install/package smoke
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

**Remaining**
- Extended fuzz (**24 h × 7**) — must pass with zero crashes before tagging (`gh workflow run fuzz-24h.yml`; see `docs/RELEASE.md`)
- Version bump to **1.0.0** after fuzz marathon completes (`docs/RELEASE.md`)

## Documentation

- [Architecture overview](docs/ARCHITECTURE.md) — module layout, build flags, test structure, CI pipeline
- [Public API index](docs/API.md) — headers under `include/ms/` grouped by module
- `mathscript-master-plan.md` — complete 10-phase delivery plan
- [1.0.0 release checklist](docs/RELEASE.md) — tag criteria for Phase 10 exit
