# Contributing to MathScript

Short guide for building, testing, and running compliance checks locally.

Looking to just use MathScript? See [`docs/USER_GUIDE.md`](USER_GUIDE.md) instead.

## Prerequisites

- **CMake** 3.28+
- **Ninja**
- **C++23** compiler: MSVC 2022+ (Windows) or GCC 13 / Clang (Linux)

GoogleTest is fetched automatically when `MS_BUILD_TESTS=ON`.

## Build

Local Windows work uses **one** tree: `build-msvc`. Do not keep extra `build-*` directories. Linux CI jobs use short-lived names (`build-linux`, `build-cov`, …) that should not accumulate on a workstation.

### Windows (MSVC)

```powershell
.\build.ps1
```

Configure only (no compile):

```powershell
.\build.ps1 -Configure
```

Clean rebuild:

```powershell
.\build.ps1 -Clean
```

Benchmarks in the same tree (`MS_BUILD_BENCHMARKS=ON`):

```powershell
.\build.ps1 -Benchmark
```

Or manually with Ninja after loading the VS developer environment (see `README.md`).

### Linux (GCC 13, CI-style)

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

On Windows, use `build-msvc` instead of `build`. Filter by domain:

```bash
ctest --test-dir build -R test_fft          # FFT unit tests
ctest --test-dir build -R int_linalg         # linear-algebra REPL pipelines
```

### Adding a matrix REPL callee

1. Create `src/interp/matrix_calls/<domain>/<callee>.cpp` implementing `handle_<name>` and `ms_register_matrix_call_<name>()` that calls `register_matrix_call("name", &handle_<name>)`. Domain folders match the library (`linalg`, `fft`, `signal`, `stats`, …).
2. Reconfigure CMake (`GLOB_RECURSE` regenerates the registrar).
3. Add a GTest in `tests/unit/repl/test_repl_commands_<domain>.cpp` (or `test_repl_commands.cpp` for session/meta) and/or `tests/integration/<domain>/`.

## Coverage

Linux Debug build with gcov instrumentation. CI enforces **80%** minimum line coverage.
Last measured: **91.2%** lines, **98.3%** functions, **57.3%** branches.

Read the branch figure before quoting the line figure. This tree's largest files are
dispatch chains, and a dispatch chain reaches high line coverage with one branch of
each test taken — the 34-point gap is that, not an accident. Branch coverage was not
measured at all until `coverage_report.sh` was fixed to use lcov 2.x's
`branch_coverage` RC name in place of the `lcov_branch_coverage` it renamed; the old
name is still accepted, still warns, and collects nothing.

What is excluded is declared in `scripts/coverage_exclusions.txt`, one glob per line
with the reason it cannot execute on a runner, and every run prints how many lines
the exclusions hid (currently 8%). The exclusions used to be inline in the script and
removed about a quarter of `src/`, which is how a figure covering 75% of the
repository came to be published as if it covered all of it.

```bash
cmake -S . -B build-cov -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF \
  -DMS_ENABLE_COVERAGE=ON -DMS_LINK_TESTS_SHARED=ON
cmake --build build-cov
ctest --test-dir build-cov --output-on-failure
MS_COVERAGE_MIN=80 bash scripts/coverage_report.sh build-cov
```

Or run the CMake target after configuring with coverage enabled:

```bash
cmake --build build-cov --target coverage_report
```

### The coverage ratchet

`scripts/coverage_ratchet.py` compares a run against `tests/coverage_baseline.json`
and fails if any metric fell by more than the recorded tolerance:

```bash
python3 scripts/coverage_ratchet.py build-cov            # check
python3 scripts/coverage_ratchet.py build-cov --update   # raise the baseline
```

It allows a small slack rather than failing on any decrease at all, because a gate
that fires on noise is a gate people learn to re-run until it passes — the benchmark
job in this repository is the cautionary example. `--update` raises the baseline and
refuses to lower it without `--force`, so a deliberate drop leaves a trace in the
diff.

## Generated sources

Two directories are generated and must not be hand-edited. CI regenerates both and
fails on a dirty tree, so a new handler cannot land without its dispatch tests:

```bash
python3 scripts/extract_manifest.py        # tests/unit/matrix_calls/matrix_calls_manifest.json
python3 scripts/gen_matrix_call_tests.py   # tests/unit/matrix_calls/test_matrix_calls_*.cpp
```

The manifest reads every handler's dispatch guard as a predicate over the argument
count and solves it. A guard it cannot read is an anomaly, and `--strict` makes that
a failure: a handler nothing can parse is a handler whose dispatch nothing tests.

Other source-only checks, all of which run in the `Compliance` CI job and need no
build:

```bash
python3 scripts/add_spdx.py --check        # every source file carries a licence
python3 scripts/gen_sbom.py --check        # the SBOM matches the tree
python3 scripts/check_test_names.py        # no duplicate test names within one binary
```

`check_test_names.py` exists because integration tests are grouped one executable per
domain. `TEST(Suite, Name)` expands to a class whose members are implicitly inline, so
two files in one binary declaring the same pair link without a diagnostic and one body
silently replaces the other — the test count does not move while a test stops running.

## Pushing while CI is running

The CI workflow sets `cancel-in-progress: true`, so **any push to a branch cancels
the run already in flight on it**. The Windows MSVC job takes around 65 minutes, so a
push made 50 minutes in throws away 50 minutes and restarts from zero.

If a Windows build is in flight and the change is documentation or anything else that
cannot affect the result, hold the push until it reports. This has cost several full
cycles.

## Clang plugin (Linux, LLVM 18)

The enforcement plugin requires LLVM/Clang dev packages. On Ubuntu 24.04:

```bash
sudo apt-get install -y clang lld ninja-build llvm-18-dev libclang-18-dev
```

Configure and build the plugin plus smoke/compliance tests:

```bash
cmake -S . -B build-plugin -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON \
  -DMS_BUILD_PLUGIN=ON \
  -DMS_ENABLE_CUDA=OFF \
  -DMS_ENABLE_AVX512=OFF \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang

cmake --build build-plugin --target ms_plugin test_plugin_smoke
ctest --test-dir build-plugin -R 'test_plugin_smoke|compliance_' --output-on-failure
```

## LLVM ORC JIT (Linux, LLVM 18)

Optional JIT backend (`-DMS_BUILD_JIT=ON`). Links ORC LLJIT and JIT-compiles scalar REPL assignments: literals, full arithmetic expressions (with parentheses and unary `-`), and libm calls (`sin`, `pow`, `min`, …). Matrix and scalar call assignments dispatch to native kernels; unsupported lines delegate to the interpreted REPL. Enable in the CLI with `mathscript-repl --jit`.

```bash
cmake -S . -B build-jit -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON \
  -DMS_BUILD_JIT=ON \
  -DMS_ENABLE_CUDA=OFF \
  -DMS_ENABLE_AVX512=OFF \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm

cmake --build build-jit --target test_jit_backend test_plot_console
ctest --test-dir build-jit -R 'test_jit_backend|test_plot_console' --output-on-failure
```

CI runs this via the **`jit-linux`** job in `.github/workflows/ci.yml`.

### Enforced compliance rules

Twenty rules are enforced on **`plugin-linux`** today (each has `tests/compliance/<rule>/fail.cpp` and `ok.cpp`):

| Rule | Diagnostic ID | Forbidden pattern |
|------|---------------|-------------------|
| `no_raw_new` | `NoRawNew` | `new` / `delete` outside `[[ms::unsafe]]` |
| `no_malloc` | `NoMalloc` | `malloc` / `free` / `realloc` |
| `no_cstyle_cast` | `NoCStyleCast` | C-style `(T)x` casts |
| `no_throw` | `NoThrow` | `throw` expressions |
| `no_catch` | `NoCatch` | `try` / `catch` |
| `no_const_cast` | `NoConstCast` | `const_cast` |
| `no_goto` | `NoGoto` | `goto` statements |
| `no_raw_ptr_arithmetic` | `NoRawPtrArithmetic` | Raw pointer arithmetic |
| `no_unsafe_reinterpret` | `NoUnsafeReinterpret` | Unsafe `reinterpret_cast` |
| `no_detach` | `NoDetach` | `std::thread::detach` |
| `no_vla` | `NoVLA` | Variable-length arrays |
| `narrowing` | `Narrowing` | Implicit narrowing conversions |
| `no_signed_unsigned_mix` | `NoSignedUnsignedMix` | Risky signed/unsigned comparisons |
| `no_raw_thread` | `NoRawThread` | Raw `std::thread` construction |
| `no_raw_mutex_lock` | `NoRawMutexLock` | Manual `mutex.lock()` without RAII |
| `no_uninit` | `NoUninit` | Uninitialized local variables |
| `no_stored_span` | `NoStoredSpan` | `std::span` stored as class member |
| `no_volatile_sync` | `NoVolatileSync` | `volatile` for synchronization |
| `no_owning_raw_ptr` | `NoOwningRawPtr` | Owning raw pointer class members |
| `unused_expected` | `UnusedExpected` | Discarded `std::expected` / `ms::Result` |

`UnsafeAudit` is partial: `MS_UNSAFE(reason)` macro + `scripts/unsafe_report.sh` / `unsafe_delta.sh` (not a compile-fail rule). See `tests/compliance/README.md` for the full profile table and CI layout.

### Compliance test layout

```
tests/compliance/
├── CMakeLists.txt              # add_compliance_test(), test_plugin_smoke
├── test_plugin_smoke.cpp       # Plugin target smoke check
├── test_unsafe_annotation.cpp  # [[ms::unsafe]] compile check (MS_UNSAFE macro)
├── unsafe_baseline.txt         # Approved unsafe-surface delta baseline
└── <rule>/
    ├── fail.cpp                # Must NOT compile with plugin active
    └── ok.cpp                  # Must compile cleanly
```

## Pre-release checklist

Before tagging **v1.0.0** (after the 24 h fuzz marathon completes with zero crashes):

- Linux: `bash scripts/tag_1.0.0_checklist.sh` then `bash scripts/pre_release.sh`
- Windows: `.\build.ps1 -Test` then `.\scripts\tag_1.0.0_checklist.ps1`

Local fuzz corpus: `tests/fuzz/corpus/<target>/` (used by CI when present). Dispatch 24 h marathon: `bash scripts/fuzz_24h_dispatch.sh`.

See `docs/RELEASE.md` for the full tag criteria.

## Version header

`include/ms/version.hpp` is generated at configure time from `cmake/version.hpp.in` and `project(MathScript VERSION …)` in the root `CMakeLists.txt`. Do not edit it by hand; re-run CMake after changing the project version.
