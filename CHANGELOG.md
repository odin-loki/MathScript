# Changelog

All notable changes to MathScript are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-06-10

### Added
- GitHub Actions CI: Windows MSVC and Linux GCC build/test, install smoke, packaging
- Coverage job with **90%** minimum gate (~**91%** line coverage achieved)
- libFuzzer smoke for **7** targets; nightly extended runs (`.github/workflows/nightly.yml`); manual **24 h × 7** campaign (`.github/workflows/fuzz-24h.yml`)
- Valgrind memcheck job (excludes long `test_fuzz_stress`)
- Benchmark regression CI gate (`bench_matmul`, `bench_fft`; 0.1s / 5 reps / median vs `linux-gcc13.json`)
- Linux conditional DEB/RPM `cpack`; Windows conditional NSIS and WiX `cpack` when tools present
- `scripts/package_smoke.sh` / `scripts/package_smoke.ps1` install verification
- Architecture, API, contributing, and release docs (`docs/ARCHITECTURE.md`, `docs/API.md`, `docs/CONTRIBUTING.md`, `docs/RELEASE.md`)
- Vendor SHA-256 policy (`vendor/CHECKSUMS.sha256`, `scripts/verify_vendor.sh`)
- Unsafe surface audit (`scripts/unsafe_report.sh`, `scripts/unsafe_delta.sh`, `UNSAFE_REVIEW.md`); `MS_UNSAFE(reason)` macro + `compliance_unsafe_annotation`
- Compliance test infra (`tests/compliance/`, `test_plugin_smoke`); Clang plugin rules (**20 enforced**, matching `diagnostics.hpp` except partial `UnsafeAudit`) + compile-fail/pass tests for each rule
- Fuzz corpus layout under `tests/fuzz/corpus/`; CI/nightly/fuzz-24h use `-corpus_dir` when present
- ORC JIT v2 LLVM ORC LLJIT backend + `test_jit_backend` (optional **`MS_BUILD_JIT`**); scalar literal/expression/libm-call JIT; native dispatch for matrix/scalar/multi-target REPL call assignments when LLVM linked
- REPL: **`mathscript-repl -e` / `--load` / `--jit` / `--jit-stats`**; unary `-` and parentheses in scalar expressions; scalar **`pow`**, **`min`**, **`max`**, **`atan2`**
- ORC JIT dispatch stats (`--jit-stats`): logs jit-compiled vs native-dispatch vs repl-fallback paths per executed line
- REPL matrix-call and multi-target assignments: **`matmul`**, **`solve`**, **`transpose`**, **`chol`**, **`lu`/`qr`/`svd`/`eig_sym`**, scalar **`det`/`trace`/`norm`/`rank`/`cond`**
- REPL plot stubs: **`scatter`**, **`imshow`**, **`spy`**, **`surf`** with console ASCII previews + **`show`**; **`saveplot <file>`** writes ASCII plot preview; plot state in session save/load
- Qt **`PlotWidget`** + OpenGL **`PlotSurfWidget`** (`MS_BUILD_GUI=ON`): 2D plots, shaded 3D surf, GUI **Export Plot as PNG**
- Optim: 1D Newton-Raphson and Broyden root finders; box-constraint **`minimize_with_constraints`**; 2D Nelder-Mead **`simplex_solver`**
- CUDA matmul dispatch tests: **`test_cuda_matmul`** (GPU/AUTO CPU fallback when CUDA off)
- Symbolic simplify expansions; **`poly_sub`**
- `scripts/pre_release.sh` local release checklist (warn-only); `scripts/tag_1.0.0_checklist.sh` / `.ps1` read-only pre-tag gate
- `mathscriptc` script runner: executes `.ms` files as REPL command sequences, prints results to stdout, errors to stderr
- Additional benchmarks: `bench_linalg` (LU/solve/SVD/Chol), `bench_repl` (scalar expr, matrix assign+det); all wired to CI `benchmark-linux`
- `build.ps1 -Configure` (configure only) and `-Clean` (wipe `build-msvc` first) switches
- `sanitizer-linux` CI job: Debug + AddressSanitizer + UBSan (GCC 13)
- `mathscript-repl --debug` trace mode: per-line timing, variable diff, parse category to stderr
- `mathscript-repl --eval-file <path>` executes script lines before entering interactive mode
- `tests/numerical/` NIST DLMF reference-value accuracy tests: Bessel J/Y/I/K, LU/SVD/solve/chol/eig_sym residuals, FFT impulse/Parseval/roundtrip, erf/erfc/gamma/lgamma/digamma (26 cases)
- `test_memory`, `test_error_types`, `test_runtime`, `test_data_driven` — memory allocators, all Error variant formatting, ThreadPool parallel correctness, parameterized matmul/erf/FFT data-driven suite (33 cases)
- **Wave 4 additions (76 CTest suites total):** typed unit tests for `stats`, `signal`, `sparse`, `simd`, and `fft` modules (float/double parameterized); advanced REPL session tests (save/load/eval-file/debug-trace); symbolic simplification typed suite; integration tests for REPL→plot→save pipeline and `mathscriptc` multi-line script execution

### Fixed
- `collect_scalar_expr_variables` now recurses through unary `+`/`-` prefixes (e.g. `z = -a + b` JIT variable collection)
- `dormbr('P','L',...)` heap corruption: guard when `m < n`; reflector-order bug in `apply_p_left_tall`
- `det()` permutation sign computation (was incorrect for non-trivial permutations)
- `trace()` now returns `DimensionMismatch` for non-square matrices
- `cg`, `gmres` return `ConvergenceFail` when max iterations exceeded without convergence
- `bicgstab`, `gmres` validate dimension match between A and b
- `erf()` is now available in REPL scalar assignment expressions (`y = erf(0.5)`)

### Changed
- ORC JIT capabilities note clarifies: scalar exprs via LLVM IR; matrix/scalar/multi-target calls via native C++ dispatch; other forms via REPL fallback
- Benchmark regression check now uses the same calibration profile as baselines (0.1s min time, 5 repetitions, median compare; default `MS_BENCH_TOLERANCE=10`)
- `Sym::eval()` returns numeric literals; NaN for unresolved symbols
- `mathscriptc` is now a full script runner (executes `.ms` files); `--help` / `--version` retained
- Unsafe delta CI gate is now **blocking** on `main`
- Project version bumped to **1.0.0** — Phase 10 (Hardening) complete; all 86 CTest suites passing

[1.0.0]: https://github.com/odin-loki/MathScript/releases/tag/v1.0.0
