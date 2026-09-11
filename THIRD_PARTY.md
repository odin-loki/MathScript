# Third-party components

Every component in this tree that is not original work, with its licence and how
it reaches a build. This file and the SBOM at `sbom.cdx.json` are generated from
the same inventory; regenerate both together with `scripts/gen_sbom.py`.

MathScript itself is AGPL-3.0-or-later (see `LICENSE`, `COPYRIGHT`), with an
additional permission for the NVIDIA CUDA libraries in `LICENSE.exceptions`.

## Vendored — source is in this repository

| Component | Version | Licence | Path | Notes |
|---|---|---|---|---|
| GoogleTest | 1.14.0 | BSD-3-Clause | `vendor/googletest/` | Test-only; not linked into any shipped binary. Licence at `vendor/googletest/LICENSE`. |
| xsimd | 13.2.0 | BSD-3-Clause | `vendor/xsimd/` | Header-only SIMD abstraction. |

Both are permissive and one-way compatible with AGPL-3.0: their terms allow
inclusion in a copyleft work, and this project's terms do not alter theirs.

File-level SHA-256 checksums for everything under `vendor/` are in
`vendor/CHECKSUMS.sha256`, verified by the `Vendor checksum verify` CI step, so a
vendored file cannot be modified without the build noticing.

## Optional at build time — found on the host, not vendored

| Component | Licence | Enabled by | Compatibility |
|---|---|---|---|
| LLVM / ORC JIT | Apache-2.0 with LLVM exception | `MS_BUILD_JIT=ON` | Compatible. Apache-2.0 is one-way compatible with GPLv3 and later. |
| Qt 6 | LGPL-3.0 or commercial | `MS_BUILD_GUI=ON` | Compatible **when linked dynamically**. Static linking attaches LGPL relinking obligations; keep Qt dynamic. |
| MPI (OpenMPI / MPICH) | BSD-3-Clause / permissive | `MS_ENABLE_MPI=ON` | Compatible. |
| NVIDIA CUDA runtime, cuBLAS, cuSOLVER, NCCL | Proprietary (NVIDIA EULA) | `MS_ENABLE_CUDA=ON` | **Not compatible without the additional permission** in `LICENSE.exceptions`. See that file before distributing a CUDA build. |
| Clang / LLVM tooling | Apache-2.0 with LLVM exception | `MS_BUILD_PLUGIN=ON` | Compatible. Build-time only. |

The default configuration vendors nothing proprietary: CUDA, MPI, Qt, the JIT and
the compliance plugin are all off unless asked for.

## Not dependencies

`curve25519-donna` is named in some design notes as a reference for the X25519
implementation. No third-party curve25519 source is vendored; `src/crypto/` is
original work. The reference is acknowledged rather than incorporated.

## Reporting

Licence questions, or a component you believe is inventoried incorrectly:
see `SECURITY.md` for contact details.
