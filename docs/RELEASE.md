# MathScript 1.0.0 release

CMake already reports version **1.0.0**. The git tag `v1.0.0` is cut only when the list below is true. Pre-release [`v1.0.0-rc.1`](https://github.com/odin-loki/MathScript/releases/tag/v1.0.0-rc.1) is published: CI green ([run 33269316904](https://github.com/odin-loki/MathScript/actions/runs/33269316904)), 816 CTest suites on Windows and Linux, AddressSanitizer + UBSan, and packaging smoke. Coverage has since been measured honestly over the corrected denominator (criterion 3), and the two engineering-plan decisions that were outstanding — the rename off "MathScript" (§4.2) and the commit-authorship rewrite (§4.5) — have both been **declined by the repository owner**, so neither gates the tag; see [`PLAN_STATUS.md`](PLAN_STATUS.md). **Remaining for the tag: CI green on `main` (criterion 1) and the 24 h fuzz marathon (criterion 5).** What the deferred-stub list became, and the little that is still out of scope, is in [`RELEASE_DECISIONS.md`](RELEASE_DECISIONS.md).

## Tag criteria

1. **CI green** on `main` with no `continue-on-error`. Linux GCC 13 `-fno-exceptions` syntax gate on `build-test-linux` must pass.
2. **Tests** — full CTest passing. Current catalogue: **336** CTest suites (Linux GCC 13, CUDA off).
   That number fell from 873 without a single test being removed: the 573 integration executables were grouped
   into 31 per-domain binaries (plan §8.8), and the 1,377 generated matrix-call dispatch tests are one binary
   rather than 29. CTest suites are executables, not tests, and the two are worth not confusing —
   `test_matrix_calls` alone contains 1,377. Grouping cut the build graph from 2,431 steps to 1,460.
   Test *names* are now checked for uniqueness within each executable (`scripts/check_test_names.py`) because
   71 pairs collided during the grouping and 29 of them had different bodies: without that check the count
   would have stayed put while the tests silently stopped running.
3. **Coverage** — CI gate **80%** (`coverage-linux`). Measured over the corrected denominator:
   **91.2% lines** (76,687 of 84,077) and **98.3% functions** (5,492 of 5,588), on an instrumented Debug build
   with `MS_BUILD_INTEGRATION=ON`, 336/336 CTest suites passing. The **90%** tag goal is met.

   **The previously published 92.0% line / 97.9% function figures are withdrawn**, and the new number is not a
   correction of them so much as a different measurement. They were taken over a denominator that excluded
   `matrix_calls`, `plugin` and `ms_bundle` — about a quarter of `src/` — because
   `scripts/coverage_exclusions.txt` was not present in this tree and the exclusions were open-ended rather than
   limited to paths that cannot execute on a CI runner. That number covered 75% of the repository and was
   reported as if it covered all of it.

   The exclusions now hide **7,698 of 91,775 instrumented lines (8%)**, and `coverage_report.sh` prints that
   figure on every run, because a list nobody sees is a list that grows. What remains excluded is `/usr/*`,
   `vendor/`, GoogleTest, the tests themselves, `src/cuda` and `src/gui` — and in this configuration three of
   those patterns matched nothing at all, which the run reports rather than passing over.

   Two honest caveats. The figure barely moved despite a substantially larger denominator, which is not what
   was predicted; the likely reason is that the 1,377 generated dispatch tests on this branch exercise the
   `matrix_calls` code the old exclusions removed. That is an explanation, not a measurement — the two changes
   landed together and were not isolated from each other. What *is* measured is that `matrix_calls` handlers
   dominate the lowest-coverage entries in `build-cov/coverage-ranked.txt`, so the newly-included code remains
   the weakest in the tree. The old figure's denominator was never recorded in comparable units, so no ratio
   between the two numbers is quoted here.

   Branch coverage is measured separately — see criterion 3a, and read it before quoting the 91.2%.
3a. **Branch coverage** — **57.3%** (82,082 of 143,192 branches), over the same corrected denominator.
   No CI gate is set for it yet.

   This is 34 points below line coverage, and it had never been measured. `coverage_report.sh` asked lcov for
   branch data under `lcov_branch_coverage`, which lcov 2.x renamed to `branch_coverage`; the old name is still
   accepted, still warns that it is deprecated, and then collects nothing — so every run reported
   `branches...: no data found` while the script's header claimed branch coverage was being measured. The gate
   itself was honest (a requested `MS_COVERAGE_BRANCH_MIN` fails as "not measured" rather than passing
   vacuously), so nothing was ever silently green; the measurement simply never happened, and the branch gate
   was unusable by anyone who tried to set it.

   The gap is the point rather than an embarrassment. This tree's largest files are dispatch chains, and a
   dispatch chain reaches high line coverage with one branch of each test taken — which is exactly what the
   script's own header comment predicted and what nothing was able to confirm. **91.2% line coverage on this
   codebase means considerably less than it sounds like**, and 57.3% is the number that says where the work is.

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
