# MathScript 1.0.0 release

CMake already reports version **1.0.0**. The git tag `v1.0.0` is cut only when the list below is true. Local Windows prove-out (816/816 CTest, 28-bench smoke) is done; remaining work includes CI, coverage, Valgrind, fuzz-24h, and packaging. Scope that will not ship in 1.0 is in [`RELEASE_DECISIONS.md`](RELEASE_DECISIONS.md).

## Tag criteria

1. **CI green** on `main` with no `continue-on-error`. Linux GCC 13 `-fno-exceptions` syntax gate on `build-test-linux` must pass.
2. **Tests** — full CTest passing. Current catalogue: **816** suites (Windows MSVC, CUDA off), grouped by mathematical domain.
3. **Coverage** — CI gate **80%** (`coverage-linux`; measured **81.1%** of library `src/` excluding plugin, GUI, CUDA stubs, and `matrix_calls` registrars). **90%** remains a `v1.0.0` tag goal.
4. **Valgrind** memcheck clean (`valgrind-linux`; unit + numerical). Full **816** suites run on `build-test-linux` and `build-test-windows`.
5. **Fuzz** — 24 h × 7 libFuzzer jobs, zero crashes (`fuzz-24h.yml`).
6. **Unsafe surface** — `UNSAFE_REVIEW.md` matches `scripts/unsafe_report.sh`; no new unreviewed sites.
7. **Packaging** — smoke scripts plus extra CPack generators when tools are present. `scripts/package_smoke.sh` installs the prefix and runs `cpack -G TGZ`. `scripts/package_smoke.ps1` installs the prefix and runs `cpack -G ZIP`. CI also runs DEB/RPM (Linux) and NSIS/WiX (Windows) when those tools exist.
8. **Benchmarks** — within **10%** of `linux-gcc13.json` (`benchmark-linux` on GitHub-hosted ubuntu-24.04, AVX-512 off).
9. **Compliance** — `plugin-linux` green, twenty compile-fail rules.
10. **JIT** — `jit-linux` with `-DMS_BUILD_JIT=ON`.
11. **Documentation** — architecture, API, contributing, and this file match the tagged tree. Wave-by-wave history lives in [`WAVES.md`](WAVES.md).

## Windows

After `.\build.ps1` (tree `build-msvc`). The script defaults are `build-msvc` and `install-smoke`:

```powershell
pwsh -NoProfile -File scripts/package_smoke.ps1 build-msvc install-smoke
```

That installs into the prefix, checks `mathscript-repl` / `mathscriptc` / `mathscript-server`, `ms_core.lib`, and `include/ms/version.hpp`, then runs `cpack -G ZIP`.

## Linux

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build
ctest --test-dir build --output-on-failure
bash scripts/unsafe_report.sh build/unsafe_report.txt
bash scripts/unsafe_delta.sh build/unsafe_report.txt
bash scripts/package_smoke.sh build install-smoke
```

`package_smoke.sh` defaults to `build-linux` if you omit the first argument; pass `build` when that is your tree. The script installs into the prefix, checks the three binaries, `libms_core.a`, and `include/ms/version.hpp`, then runs `cpack -G TGZ`.

## Fuzz marathon

Nightly runs 15 min × 7. Tag requires 24 h × 7:

```bash
gh workflow run fuzz-24h.yml
gh run list --workflow=fuzz-24h.yml
```

Helper: `bash scripts/fuzz_24h_dispatch.sh`. Read-only pre-tag: `bash scripts/tag_1.0.0_checklist.sh`.

## Tag procedure

1. Confirm fuzz marathon zero crashes.
2. Confirm CMake `project(MathScript VERSION 1.0.0)` and a green CI run you watched.
3. Move `[Unreleased]` in `CHANGELOG.md` to `[1.0.0] - <date>`.
4. `bash scripts/pre_release.sh` (Linux).
5. Push `main`, then `git tag -a v1.0.0 -m "MathScript 1.0.0"` and push the tag.
