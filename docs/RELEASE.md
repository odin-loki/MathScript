# MathScript 1.0.0 release

CMake already reports version **1.0.0**. The git tag `v1.0.0` is cut only when the list below is true. Pre-release [`v1.0.0-rc.1`](https://github.com/odin-loki/MathScript/releases/tag/v1.0.0-rc.1) is published: CI green ([run 33269316904](https://github.com/odin-loki/MathScript/actions/runs/33269316904)), 816 CTest suites on Windows and Linux, AddressSanitizer + UBSan, and packaging smoke. Remaining for the tag is the 24 h fuzz marathon; the **90%** coverage goal is now met (CI gate is **80%**, last measured **92.0%** line / **97.9%** function over the full 873-suite run). What the deferred-stub list became, and the little that is still out of scope, is in [`RELEASE_DECISIONS.md`](RELEASE_DECISIONS.md).

## Tag criteria

1. **CI green** on `main` with no `continue-on-error`. Linux GCC 13 `-fno-exceptions` syntax gate on `build-test-linux` must pass.
2. **Tests** — full CTest passing. Current catalogue: **873** suites (Linux GCC 13, CUDA off), grouped by mathematical domain.
3. **Coverage** — CI gate **80%** (`coverage-linux`). Measured **92.0%** line and **97.9%** function coverage of library `src/`, excluding plugin, GUI, CUDA stubs, and `matrix_calls` registrars, over the full suite with `MS_BUILD_INTEGRATION=ON`. The **90%** `v1.0.0` tag goal is met.
4. **ASan + UBSan** clean (`sanitizer-linux`; overflows and UB fail the job). Leak detection stays off (`detect_leaks=0`) for process-exit pool/AD graphs. Last local run: **300/300 passing with zero sanitizer reports** — the sanitizer tree is configured `MS_BUILD_INTEGRATION=OFF`, so it carries the unit suites only, not the full catalogue. Full **873** suites run on `build-test-linux` and `build-test-windows`.
5. **Fuzz** — 24 h × 7 libFuzzer jobs, zero crashes (`fuzz-24h.yml`). Last local run: real libFuzzer under Clang 18, 7 targets × 10 min seeded from the checked-in corpora = **353 193 095 executions, zero crashes**. The corpus-replay harness (7 targets × 5 seeds × 200 000 mutations = 7 000 000 inputs) is also clean, and the corpora are replayed on every build by the `replay_fuzz_*` CTest suites, so a regression is caught even where no libFuzzer runtime is installed.

   Two ways to run the marathon itself. `scripts/fuzz_24h_dispatch.sh` sends it to Actions, one 2-core runner per target. `scripts/fuzz_24h_local.sh` runs all seven at once on a workstation and puts every spare core behind them (`MS_FUZZ_WORKERS` per target, defaulting to a split of the core count); it seeds from the checked-in corpora, writes new coverage back into them, holds the same 2048 MB RSS limit CI uses so an out-of-memory finding reproduces identically, and fails if any target leaves a `crash-`/`oom-`/`leak-`/`timeout-` artifact — including when libFuzzer's own exit code is 0, which happens when a worker rather than the parent finds the input.

   The marathon earns its place: it found an out-of-memory in `combo_restricted_partitions(442, 5)` that 353 million local executions had not, at 2398 MB peak RSS. That input is now in the corpus.
6. **Unsafe surface** — `UNSAFE_REVIEW.md` matches `scripts/unsafe_report.sh`; no new unreviewed sites. Last local run: **33 sites against a baseline of 33**, `unsafe_delta.sh` clean.
7. **Packaging** — smoke scripts plus extra CPack generators when tools are present. `scripts/package_smoke.sh` installs the prefix and runs `cpack -G TGZ`. `scripts/package_smoke.ps1` installs the prefix and runs `cpack -G ZIP`. CI also runs DEB/RPM (Linux) and NSIS/WiX (Windows) when those tools exist. Last local run: install prefix and `mathscript-1.0.0-Linux.tar.gz` both produced, **smoke OK**.
8. **Benchmarks** — within **10%** of `linux-gcc13.json` (`benchmark-linux` on GitHub-hosted ubuntu-24.04, AVX-512 off). Last local run: **check OK**, worst delta +9.3% (`BM_fft/256`). The gate is deliberately thin: only **5** of the 445 baseline entries carry a measured `median_time_ns`, and the other **440** are nulls the comparison skips.

   Filling them in was tried and rejected on evidence. `bench-baseline-linux.yml` was dispatched twice against this branch, and comparing the two artifacts — the same code, two `ubuntu-24.04` runners — shows the tolerance cannot survive the hardware:

   - **94%** of genuine benchmarks (327 of 345) differ by more than the 10% threshold between the two runs, the worst by **214%** (`BM_ConvexHull2D/4096`). 1003 of them came out *slower* on the run that contained a pure performance fix, which is only possible if the machine, not the code, is doing the talking.
   - **1944 of the 2430** entries `--write-baseline` emits are `_stddev`/`_cv`/`_mean`/`_median` repetition aggregates rather than distinct benchmarks. The committed file holds none of these by design; a run-to-run `_stddev` swing of 7000% is dispersion, not a regression, and gating on it means nothing.

   So a full baseline would make this job fail on almost every run, and a gate that always fails is one nobody reads. The five entries that are measured are loose enough to absorb runner variance and still caught a real **+140.2%** regression on this branch (`BM_fft/256`, an uncached sysfs walk on the dispatch path) with no false positive. Broadening it needs a fixed machine profile — a self-hosted or pinned runner — or a comparison that is statistical rather than a fixed percentage, not simply more numbers.
9. **Compliance** — `plugin-linux` green, twenty compile-fail rules. Last local run: **42/42 passing** (Clang 18 + LLVM 18).
10. **JIT** — `jit-linux` with `-DMS_BUILD_JIT=ON`. Last local run: **2/2 passing** (Clang 18 + LLVM 18).
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
