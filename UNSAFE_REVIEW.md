# Unsafe Surface Review

MathScript restricts unchecked memory access, raw casts, and interop boundaries via `[[ms::unsafe("reason")]]` annotations. This file tracks the statically audited baseline. Full compile-fail enforcement is the Clang plugin (`MS_BUILD_PLUGIN=ON`, CI job `plugin-linux`).

## Baseline

```
approved_sites: 33
```

The count covers matches for `reinterpret_cast`, `const_cast`, `[[ms::unsafe`, and `UNSAFE_SITE(` under `src/` and `include/`, excluding whole-line comments: a comment explaining why a function avoids `const_cast` is not an unsafe site, and counting those made the gate fail for documenting itself. Last verified: 2026-09-09 (33 matches; row `Matches` sum = 33).

Regenerate the report:

```bash
bash scripts/unsafe_report.sh
```

## Reviewed sites

| Location | Matches | Category | Justification |
|----------|--------:|----------|---------------|
| `src/cuda/fft.cpp` | 4 | CUDA interop | cuFFT/cuBLAS C APIs require raw device pointers at the library boundary |
| `src/plugin/unsafe_registry.hpp` | 1 | Plugin infrastructure | The `UNSAFE_SITE(reason)` macro that records annotation sites for the Clang plugin |
| `src/plugin/rules/cast_rules.cpp` | 11 | Plugin diagnostics | Diagnostic strings mention `[[ms::unsafe]]` / `const_cast` / `reinterpret_cast` (moved out of `MsPlugin.cpp`) |
| `src/plugin/rules/cast_rules.hpp` | 1 | Plugin diagnostics | Identifier `diag_const_cast_` matches the `const_cast` audit pattern |
| `src/plugin/rules/memory_rules.cpp` | 11 | Plugin diagnostics | Diagnostic strings mention `[[ms::unsafe]]` |
| `src/plugin/rules/exception_rules.cpp` | 2 | Plugin diagnostics | Diagnostic strings mention `[[ms::unsafe]]` |
| `src/interp/repl_engine_internal.cpp` | 1 | Byte view | `reinterpret_cast` exposes `std::string` storage as `std::span<const uint8_t>` |
| `src/crypto/crypto.cpp` | 1 | Hash/HMAC byte view | `u8_view` maps `string_view` to `span<const uint8_t>` for digest APIs |
| `src/frameworks/axiom/axiom.cpp` | 1 | GP tree mutation | `const_cast` selects the mutable crossover point inside an owned `GPNode` tree; the mutation path now reaches its node through `collect_mutable_nodes` instead |

New unsafe sites must add a row here and bump `approved_sites` after review.
