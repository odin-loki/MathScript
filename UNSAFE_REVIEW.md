# Unsafe Surface Review

MathScript restricts unchecked memory access, raw casts, and interop boundaries via `[[ms::unsafe("reason")]]` annotations (see `mathscript-master-plan.md` §7.4). This file tracks the statically audited baseline until the Clang plugin audit (`MS_BUILD_PLUGIN`) is enabled in CI.

## Baseline

```
approved_sites: 6
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

New unsafe sites must add a row here and bump `approved_sites` after review.
