# Contributing to MathScript

Short guide for building, testing, and running compliance checks locally.

## Prerequisites

- **CMake** 3.28+
- **Ninja**
- **C++23** compiler: MSVC 2022+ (Windows) or GCC 13 / Clang (Linux)

GoogleTest is fetched automatically when `MS_BUILD_TESTS=ON`.

## Build

### Windows (MSVC)

```powershell
.\build.ps1
```

Or manually with Ninja after loading the VS developer environment (see `README.md`).

### Linux (GCC 13, CI-style)

```bash
cmake -S . -B build-linux -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build-linux
```

## Test

```bash
ctest --test-dir build-linux --output-on-failure
```

On Windows, use `build-msvc` instead of `build-linux`.

## Coverage

Linux Debug build with gcov instrumentation (CI enforces **90%** minimum line coverage):

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

Or run the CMake target after configuring with coverage enabled:

```bash
cmake --build build-cov --target coverage_report
```

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

Optional JIT backend (`-DMS_BUILD_JIT=ON`). Links ORC LLJIT and JIT-compiles scalar REPL assignments: literals, full arithmetic expressions (with parentheses), and libm calls (`sin`, `pow`, `min`, …). Matrix and scalar call assignments (`C = matmul(A,B)`, `d = det(A)`, multi-target `L, U = lu(A)`, `U, S, V = svd(A)`, `D, V = eig_sym(A)`, …) dispatch to native kernels; unsupported lines delegate to the interpreted REPL.

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

See `docs/RELEASE.md` for the full tag criteria.

## Version header

`include/ms/version.hpp` is generated at configure time from `cmake/version.hpp.in` and `project(MathScript VERSION …)` in the root `CMakeLists.txt`. Do not edit it by hand; re-run CMake after changing the project version.
