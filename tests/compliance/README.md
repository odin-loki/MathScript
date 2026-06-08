# Compliance Tests

Plugin enforcement tests for the MathScript C++ profile. These verify that the
Clang plugin (`src/plugin/MsPlugin.cpp`) rejects forbidden patterns and accepts
correct alternatives.

## Build flag: `MS_BUILD_PLUGIN`

| Setting | Behaviour |
|---------|-----------|
| `OFF` (default) | `ms_plugin` is an INTERFACE stub; rule compile tests are skipped; `test_plugin_smoke` reports `GTEST_SKIP`. |
| `ON` | Builds the real Clang plugin (requires LLVM/Clang). Enables compile-fail and compile-pass rule tests. |

Enable locally when LLVM is installed:

```bash
cmake -S . -B build -DMS_BUILD_TESTS=ON -DMS_BUILD_PLUGIN=ON \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang
cmake --build build --target ms_plugin
ctest --test-dir build -R 'test_plugin_smoke|compliance_' --output-on-failure
```

CI runs the **`plugin-linux`** job on every push/PR (`MS_BUILD_PLUGIN=ON`, Clang + LLVM 18).
Configure passes explicit `-DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm` and
`-DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang`; the job builds `ms_plugin` and runs
`test_plugin_smoke` plus all registered `compliance_*` tests (no `continue-on-error`).

`test_unsafe_annotation.cpp` is a compile-only check for `[[ms::unsafe]]` via the
`MS_UNSAFE` macro in `include/ms/unsafe/unsafe.hpp`; it is registered when that macro exists.

## Layout

```
tests/compliance/
├── CMakeLists.txt          # add_compliance_test() and test_plugin_smoke
├── test_plugin_smoke.cpp   # Plugin target / metadata smoke check
├── test_unsafe_annotation.cpp  # Compile-only [[ms::unsafe]] (needs MS_UNSAFE macro)
├── unsafe_baseline.txt     # Approved unsafe-surface delta baseline
└── <rule>/                 # Per-rule tests
    ├── fail.cpp            # MUST NOT compile with plugin active
    └── ok.cpp              # MUST compile cleanly (often via [[ms::unsafe]])
```

## Profile rules (`src/plugin/diagnostics.hpp`)

Each rule gets a `tests/compliance/<rule>/` pair (`fail.cpp` / `ok.cpp`) once
`add_compliance_test(<rule>)` is enabled in `CMakeLists.txt`.

| Rule folder | Diagnostic ID | Status | Notes |
|-------------|---------------|--------|-------|
| `no_raw_new` | `NoRawNew` | **Enforced** | `new`/`delete` prohibited outside `[[ms::unsafe]]`; use `ms::make_unique` / `ms::make_shared` |
| `no_malloc` | `NoMalloc` | **Enforced** | `malloc` / `free` / `realloc` |
| `no_owning_raw_ptr` | `NoOwningRawPtr` | **Enforced** | Owning raw pointer class members |
| `no_vla` | `NoVLA` | **Enforced** | Variable-length arrays |
| `no_cstyle_cast` | `NoCStyleCast` | **Enforced** | C-style `(T)x` casts |
| `no_const_cast` | `NoConstCast` | **Enforced** | `const_cast` |
| `no_unsafe_reinterpret` | `NoUnsafeReinterpret` | **Enforced** | Unsafe `reinterpret_cast` |
| `narrowing` | `Narrowing` | **Enforced** | Implicit narrowing conversions |
| `no_throw` | `NoThrow` | **Enforced** | `throw` expressions |
| `no_catch` | `NoCatch` | **Enforced** | `try`/`catch` |
| `unused_expected` | `UnusedExpected` | **Enforced** | Discarded `std::expected` / `ms::Result` |
| `no_uninit` | `NoUninit` | **Enforced** | Uninitialized local variables |
| `no_stored_span` | `NoStoredSpan` | **Enforced** | Storing `std::span` as class member |
| `no_raw_thread` | `NoRawThread` | **Enforced** | Raw `std::thread` construction |
| `no_raw_mutex_lock` | `NoRawMutexLock` | **Enforced** | Manual `mutex.lock()` without RAII |
| `no_volatile_sync` | `NoVolatileSync` | **Enforced** | `volatile` for synchronization |
| `no_detach` | `NoDetach` | **Enforced** | `std::thread::detach` |
| `no_signed_unsigned_mix` | `NoSignedUnsignedMix` | **Enforced** | Risky signed/unsigned comparisons |
| `no_raw_ptr_arithmetic` | `NoRawPtrArithmetic` | **Enforced** | Raw pointer arithmetic |
| `no_goto` | `NoGoto` | **Enforced** | `goto` statements |
| `unsafe_audit` | `UnsafeAudit` | Partial | `MS_UNSAFE(reason)` macro + `compliance_unsafe_annotation`; static audit via `scripts/unsafe_report.sh` |

### Enforced rules (`plugin-linux`)

Each enforced rule has `fail.cpp` (must be rejected) and `ok.cpp` (must compile, often via `[[ms::unsafe]]`):

| Rule | CTest names |
|------|-------------|
| `no_raw_new` | `compliance_no_raw_new_fail` (`WILL_FAIL`), `compliance_no_raw_new_ok` |
| `no_malloc` | `compliance_no_malloc_fail`, `compliance_no_malloc_ok` |
| `no_cstyle_cast` | `compliance_no_cstyle_cast_fail`, `compliance_no_cstyle_cast_ok` |
| `no_throw` | `compliance_no_throw_fail`, `compliance_no_throw_ok` |
| `no_catch` | `compliance_no_catch_fail`, `compliance_no_catch_ok` |
| `no_const_cast` | `compliance_no_const_cast_fail`, `compliance_no_const_cast_ok` |
| `no_goto` | `compliance_no_goto_fail`, `compliance_no_goto_ok` |
| `no_raw_ptr_arithmetic` | `compliance_no_raw_ptr_arithmetic_fail`, `compliance_no_raw_ptr_arithmetic_ok` |
| `no_unsafe_reinterpret` | `compliance_no_unsafe_reinterpret_fail`, `compliance_no_unsafe_reinterpret_ok` |
| `no_detach` | `compliance_no_detach_fail`, `compliance_no_detach_ok` |
| `no_vla` | `compliance_no_vla_fail`, `compliance_no_vla_ok` |
| `narrowing` | `compliance_narrowing_fail`, `compliance_narrowing_ok` |
| `no_signed_unsigned_mix` | `compliance_no_signed_unsigned_mix_fail`, `compliance_no_signed_unsigned_mix_ok` |
| `no_raw_thread` | `compliance_no_raw_thread_fail`, `compliance_no_raw_thread_ok` |
| `no_raw_mutex_lock` | `compliance_no_raw_mutex_lock_fail`, `compliance_no_raw_mutex_lock_ok` |
| `no_uninit` | `compliance_no_uninit_fail`, `compliance_no_uninit_ok` |
| `no_stored_span` | `compliance_no_stored_span_fail`, `compliance_no_stored_span_ok` |
| `no_volatile_sync` | `compliance_no_volatile_sync_fail`, `compliance_no_volatile_sync_ok` |
| `no_owning_raw_ptr` | `compliance_no_owning_raw_ptr_fail`, `compliance_no_owning_raw_ptr_ok` |
| `unused_expected` | `compliance_unused_expected_fail`, `compliance_unused_expected_ok` |

## Rule test mechanics (§7.6)

`add_compliance_test()` drives Clang `-fsyntax-only` with `-fplugin=ms_plugin` (no link step):

- **fail.cpp** — code that must be rejected by the plugin (CTest `WILL_FAIL` build step).
- **ok.cpp** — code that must compile without diagnostics.

Skipped when `MS_BUILD_PLUGIN=OFF` or on MSVC (plugin compile tests require Clang).

## CI pipeline (§14.7)

The **`plugin-linux`** job in `.github/workflows/ci.yml` is Stage 2 (Compliance Tests):

1. Configure with `MS_BUILD_PLUGIN=ON`, LLVM/Clang 18 paths pinned.
2. Build `ms_plugin` and `test_plugin_smoke`.
3. Run `ctest -R 'test_plugin_smoke|compliance_'` (all twenty enforced rules plus `compliance_unsafe_annotation` when registered).
