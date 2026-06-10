# Changelog

All notable changes to MathScript are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- GitHub Actions CI: Windows MSVC and Linux GCC build/test, install smoke, packaging
- Coverage job with **90%** minimum gate (~**91%** line coverage achieved)
- libFuzzer smoke for **7** targets; nightly extended runs (`.github/workflows/nightly.yml`)
- Valgrind memcheck job (excludes long `test_fuzz_stress`)
- Benchmark regression CI gate (`bench_matmul`, `bench_fft`; 0.1s / 5 reps / median vs `linux-gcc13.json`)
- Linux conditional DEB/RPM `cpack`; Windows conditional NSIS and WiX `cpack` when tools present
- `scripts/package_smoke.sh` / `scripts/package_smoke.ps1` install verification
- Architecture and API docs (`docs/ARCHITECTURE.md`, `docs/API.md`)
- Vendor SHA-256 policy (`vendor/CHECKSUMS.sha256`, `scripts/verify_vendor.sh`)
- Unsafe surface audit (`scripts/unsafe_report.sh`, `UNSAFE_REVIEW.md`)
- Compliance test infra (`tests/compliance/`, `test_plugin_smoke`)
- Fuzz corpus layout under `tests/fuzz/corpus/`
- Optim: 1D Newton-Raphson and Broyden root finders
- ORC JIT v2 stub backend + `test_jit_backend`; optional **`MS_BUILD_JIT`** LLVM ORC LLJIT linkage; scalar literal/expression/libm-call JIT when LLVM linked
- Compliance: Clang plugin rules (**20 enforced**, matching `diagnostics.hpp` except partial `UnsafeAudit`) + compile-fail/pass tests for each rule
- REPL plot stubs: **`scatter`**, **`imshow`**, **`spy`**, **`surf`** with console ASCII previews + **`show`**
- `scripts/pre_release.sh` local release checklist (warn-only)

### Added (continued)
- REPL **`saveplot <file>`** writes ASCII plot preview; GUI **Export Plot as PNG** for 2D and OpenGL surf views
- Scalar **`pow`**, **`min`**, **`max`**, **`atan2`** in scalar expressions (REPL + LLVM JIT)
- Parentheses in scalar expressions (`(1 + 2) * 3`)
- Scalar matrix-call assignments: **`d = det(A)`**, **`trace`**, **`norm`**, **`rank`**, **`cond`**
- Multi-target matrix assignments extended: **`U, S, V = svd(A)`**, **`D, V = eig_sym(A)`**
- `mathscript-repl -e <command>` for non-interactive one-shot evaluation
- `mathscript-repl --load` / `--jit`; unary `-` in scalar expressions; plot state in session save/load
- Fuzz corpus seeds for all 7 targets; CI/nightly/fuzz-24h use `-corpus_dir` when present
- `scripts/tag_1.0.0_checklist.sh` read-only pre-tag gate

### Fixed
- `dormbr('P','L',...)` heap corruption: guard when `m < n`; reflector-order bug in `apply_p_left_tall`

### Changed
- Benchmark regression check now uses the same calibration profile as baselines (0.1s min time, 5 repetitions, median compare; default `MS_BENCH_TOLERANCE=10`)
- `Sym::eval()` returns numeric literals; NaN for unresolved symbols
- `mathscriptc` supports `--help` / `--version`; compile requests exit with clear message
- Project version **0.2.0** (pre-1.0; **1.0.0** reserved for full release)

### Remaining (Phase 10)
- Full 24 h fuzz per target (`fuzz-24h.yml` configured for 24 h × 7; must run clean)
- Full ORC JIT v2 matrix LLVM IR lowering (native dispatch for matrix/scalar/multi-target REPL calls today)
- Version bump to **1.0.0** after fuzz marathon and final checklist
