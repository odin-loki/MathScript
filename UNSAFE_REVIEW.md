# Unsafe Surface Review

MathScript restricts unchecked memory access, raw casts, and interop boundaries via `[[ms::unsafe("reason")]]` annotations. This file tracks the statically audited baseline. Full compile-fail enforcement is the Clang plugin (`MS_BUILD_PLUGIN=ON`, CI job `plugin-linux`).

## Baseline

```
approved_sites: 38
```

The count covers matches for `reinterpret_cast`, `const_cast`, `[[ms::unsafe`, and `UNSAFE_SITE(` under `src/` and `include/`. Last verified: 2026-08-29 (38 grep matches; row `Matches` sum = 38).

Regenerate the report:

```bash
bash scripts/unsafe_report.sh
```

## Reviewed sites

| Location | Matches | Category | Justification |
|----------|--------:|----------|---------------|
| `src/cuda/fft.cpp` | 4 | CUDA interop | cuFFT/cuBLAS C APIs require raw device pointers at the library boundary |
| `src/plugin/unsafe_registry.hpp` | 3 | Plugin infrastructure | Documents and registers unsafe annotation sites for the Clang plugin |
| `src/plugin/unsafe_registry.cpp` | 1 | Plugin infrastructure | File banner mentions `[[ms::unsafe]]` |
| `src/plugin/rules/cast_rules.cpp` | 11 | Plugin diagnostics | Diagnostic strings mention `[[ms::unsafe]]` / `const_cast` / `reinterpret_cast` (moved out of `MsPlugin.cpp`) |
| `src/plugin/rules/cast_rules.hpp` | 1 | Plugin diagnostics | Identifier `diag_const_cast_` matches the `const_cast` audit pattern |
| `src/plugin/rules/memory_rules.cpp` | 11 | Plugin diagnostics | Diagnostic strings mention `[[ms::unsafe]]` |
| `src/plugin/rules/exception_rules.cpp` | 2 | Plugin diagnostics | Diagnostic strings mention `[[ms::unsafe]]` |
| `src/interp/repl_engine_internal.cpp` | 1 | Byte view | `reinterpret_cast` exposes `std::string` storage as `std::span<const uint8_t>` |
| `src/crypto/crypto.cpp` | 1 | Hash/HMAC byte view | `u8_view` maps `string_view` to `span<const uint8_t>` for digest APIs |
| `src/frameworks/axiom/axiom.cpp` | 2 | GP tree mutation | `const_cast` selects mutable crossover/mutation points inside owned `GPNode` trees |
| `include/ms/unsafe/unsafe.hpp` | 1 | Macro header | `MS_UNSAFE` / `[[ms::unsafe]]` macro definition; comment line matches the audit grep pattern |

New unsafe sites must add a row here and bump `approved_sites` after review.
