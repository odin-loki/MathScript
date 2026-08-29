# MathScript 1.0.0-rc.1 (pre-release)

This is a **pre-release**. It is **not** the `v1.0.0` tag.

CMake already reports version 1.0.0. The git tag `v1.0.0` waits until every gate in `docs/RELEASE.md` is honest, including a 24 h × 7 libFuzzer marathon (`fuzz-24h.yml`) and the **90%** coverage tag goal.

## Artifacts

- Windows: `mathscript-1.0.0-win64.zip`
- Linux: `mathscript-1.0.0-Linux.tar.gz`

## What this RC is claiming

Met on the green `main` CI run that produced these packages:

- Full CTest (816 suites) on Windows MSVC and Linux GCC 13 (`-fno-exceptions`)
- Coverage CI gate **80%** (library `src/` after excluding plugin, GUI, CUDA stubs, and `matrix_calls`; **90%** is still a `v1.0.0` goal)
- Valgrind memcheck on unit + numerical tests
- ASan/UBSan on the test suite (overflows fail the job; **leak detection is off** in CI — arena/pool process-exit leaks are not a ship blocker for this RC)
- libFuzzer smoke, Clang plugin / compliance, LLVM ORC JIT smoke
- Benchmark regression within 10% of `linux-gcc13.json`
- Unsafe-surface delta vs `tests/compliance/unsafe_baseline.txt`
- Package smoke: ZIP (Windows), TGZ + DEB/RPM when tools exist (Linux)

## What is still not `v1.0.0`

- Fuzz marathon: 24 h × 7 jobs, zero crashes
- Coverage **90%** of library `src/`
- ASan leak detection is not a CI fail today (`detect_leaks=0`)

See `CHANGELOG.md` `[Unreleased]` and `docs/RELEASE.md`.
