# Unsafe Surface Review

MathScript restricts unchecked memory access, raw casts, and interop boundaries via `[[ms::unsafe("reason")]]` annotations (see `mathscript-master-plan.md` §7.4). This file tracks the statically audited baseline until the Clang plugin audit (`MS_BUILD_PLUGIN`) is enabled in CI.

## Baseline

```
approved_sites: 40
```

The count covers matches for `reinterpret_cast`, `const_cast`, `[[ms::unsafe`, and `UNSAFE_SITE(` under `src/` and `include/`.

Regenerate the report:

```bash
bash scripts/unsafe_report.sh
```

## Reviewed sites

| Location | Category | Justification |
|----------|----------|---------------|
| `src/cuda/fft.cpp` | CUDA interop | cuFFT/cuBLAS C APIs require raw device pointers at the library boundary |
| `src/plugin/unsafe_registry.hpp` | Plugin infrastructure | Documents and registers unsafe annotation sites for the Clang plugin |
| `src/interp/jit_orc_llvm.cpp` | LLVM JIT interop | ORC `LLJIT::lookup` returns a symbol address; `reinterpret_cast` to typed function pointers to invoke generated code |
| `src/interp/repl_engine.cpp` | Byte view | `reinterpret_cast` exposes `std::string` storage as `std::span<const uint8_t>` for binary item handling |
| `src/frameworks/axiom/axiom.cpp` | GP tree mutation | `const_cast` selects mutable crossover/mutation points inside owned `GPNode` trees built from `const` node lists |
| `src/plugin/MsPlugin.cpp` | Plugin diagnostics | Diagnostic message strings mention `[[ms::unsafe]]`, `const_cast`, and `reinterpret_cast`; static audit pattern matches only |
| `include/ms/unsafe/unsafe.hpp` | Macro header | `MS_UNSAFE` / `[[ms::unsafe]]` macro definition; comment line matches the audit grep pattern |

New unsafe sites must add a row here and bump `approved_sites` after review.
