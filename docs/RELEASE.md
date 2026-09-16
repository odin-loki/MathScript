# MathScript 1.0.0 release

CMake already reports version **1.0.0**. The git tag `v1.0.0` is cut only when the list below is true. Pre-release [`v1.0.0-rc.1`](https://github.com/odin-loki/MathScript/releases/tag/v1.0.0-rc.1) is published: CI green ([run 33269316904](https://github.com/odin-loki/MathScript/actions/runs/33269316904)), 816 CTest suites on Windows and Linux as the catalogue was then (criterion 2 has the current count and why it moved), AddressSanitizer + UBSan, and packaging smoke. Coverage has since been measured honestly over the corrected denominator (criterion 3), and the two engineering-plan decisions that were outstanding — the rename off "MathScript" (§4.2) and the commit-authorship rewrite (§4.5) — have both been **declined by the repository owner**, so neither gates the tag; see [`PLAN_STATUS.md`](PLAN_STATUS.md). **Remaining for the tag: CI green on `main` (criterion 1) and the 24 h fuzz marathon (criterion 5).** What the deferred-stub list became, and the little that is still out of scope, is in [`RELEASE_DECISIONS.md`](RELEASE_DECISIONS.md).

## Tag criteria

1. **CI green** on `main` with no `continue-on-error`. Linux GCC 13 `-fno-exceptions` syntax gate on `build-test-linux` must pass.
2. **Tests** — full CTest passing. Current catalogue: **374** CTest suites with
   `MS_BUILD_INTEGRATION=ON`, **343** without; the difference is exactly the 31 per-domain
   integration executables, one per directory under `tests/integration/`. `MS_BUILD_INTEGRATION`
   defaults to **ON** (`cmake/options.cmake`), so 374 is what an ordinary `cmake -S . -B build`
   produces. Last local run: **374/374 passing** (Release, GCC 13, CUDA off, AVX-512 off), and
   **343/343** on a tree configured with integration off.

   **A CTest suite is an executable, not a test**, and the two are worth not confusing. Behind
   those suites are **27,733** `TEST` macros: 24,976 unit, 1,460 integration, 1,292 numerical.
   `test_matrix_calls` is one suite and contains **1,583** of them, generated from
   `matrix_calls_manifest.json` into 29 source files by `scripts/gen_matrix_call_tests.py`.

   The suite count fell from 873 without a single test being removed: the 573 integration
   executables were grouped into per-domain binaries (plan §8.8) and the generated dispatch
   tests became one binary rather than 29. Grouping cut the build graph from 2,431 steps to
   1,460. Test *names* are now checked for uniqueness within each executable
   (`scripts/check_test_names.py`) because 71 pairs collided during the grouping and 29 of them
   had different bodies: without that check the count would have stayed put while the tests
   silently stopped running.
3. **Coverage** — CI gate **80%** (`coverage-linux`). Re-measured on this branch's head:
   **91.5% lines** (84,496 of 92,302) and **97.9% functions** (6,000 of 6,126), on an instrumented
   Debug build with `MS_BUILD_INTEGRATION=ON`, `MS_LINK_TESTS_SHARED=ON` and **374/374** CTest
   suites passing. The **90%** tag goal is met, and `scripts/coverage_ratchet.py` passes against
   the 2026-09-10 baseline on all three metrics — lines −0.10, functions −0.40, branches +0.80,
   inside the 0.5-point tolerance the baseline file justifies from measurement.

   Function coverage is the one drifting in the wrong direction, and −0.40 is most of that
   slack. It is inside the gate, and it is worth watching rather than filing away.

   The previously published **91.2% / 98.3%** figures were taken over a smaller tree (84,077
   instrumented lines against 92,302 now) and are superseded rather than contradicted.

   **The previously published 92.0% line / 97.9% function figures are withdrawn**, and the new number is not a
   correction of them so much as a different measurement. They were taken over a denominator that excluded
   `matrix_calls`, `plugin` and `ms_bundle` — about a quarter of `src/` — because
   `scripts/coverage_exclusions.txt` was not present in this tree and the exclusions were open-ended rather than
   limited to paths that cannot execute on a CI runner. That number covered 75% of the repository and was
   reported as if it covered all of it.

   The exclusions now hide **8,049 of 100,351 instrumented lines (8%)**, and `coverage_report.sh` prints that
   figure on every run, because a list nobody sees is a list that grows. What remains excluded is `/usr/*`,
   `vendor/`, GoogleTest, the tests themselves, `src/cuda` and `src/gui` — and in this configuration three of
   those patterns matched nothing at all, which the run reports rather than passing over.

   Two honest caveats. The figure barely moved despite a substantially larger denominator, which is not what
   was predicted; the likely reason is that the generated dispatch tests on this branch exercise the
   `matrix_calls` code the old exclusions removed. That is an explanation, not a measurement — the two changes
   landed together and were not isolated from each other. What *is* measured is that `matrix_calls` handlers
   dominate the lowest-coverage entries in `build-cov/coverage-ranked.txt`, so the newly-included code remains
   the weakest in the tree. The old figure's denominator was never recorded in comparable units, so no ratio
   between the two numbers is quoted here.

   Branch coverage is measured separately — see criterion 3a, and read it before quoting the 91.5%.
3a. **Branch coverage** — **58.6%** (90,257 of 154,134 branches), over the same corrected
   denominator, re-measured with the rest. No CI gate is set for it yet; the ratchet does watch
   it, and it has moved up 0.8 points since 2026-09-10.

   This is 34 points below line coverage, and it had never been measured. `coverage_report.sh` asked lcov for
   branch data under `lcov_branch_coverage`, which lcov 2.x renamed to `branch_coverage`; the old name is still
   accepted, still warns that it is deprecated, and then collects nothing — so every run reported
   `branches...: no data found` while the script's header claimed branch coverage was being measured. The gate
   itself was honest (a requested `MS_COVERAGE_BRANCH_MIN` fails as "not measured" rather than passing
   vacuously), so nothing was ever silently green; the measurement simply never happened, and the branch gate
   was unusable by anyone who tried to set it.

   The gap is the point rather than an embarrassment. This tree's largest files are dispatch chains, and a
   dispatch chain reaches high line coverage with one branch of each test taken — which is exactly what the
   script's own header comment predicted and what nothing was able to confirm. **91.5% line coverage on this
   codebase means considerably less than it sounds like**, and 58.6% is the number that says where the work is.

4. **ASan + UBSan** clean (`sanitizer-linux`; overflows and UB fail the job). Leak detection stays off (`detect_leaks=0`) for process-exit pool/AD graphs. The job builds with `MS_BUILD_INTEGRATION=ON` and `MS_LINK_TESTS_SHARED=ON`, so it carries the whole catalogue, and excludes three suites by name (`test_fuzz_stress`, `test_cuda_matmul`, `test_cuda_stub`). Last local run: **371/371 passing with zero sanitizer reports**. An earlier edition of this file said the sanitizer tree was configured `MS_BUILD_INTEGRATION=OFF` and carried the unit suites only; that has not been true since 2026-08-29.
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

## What has been verified, and where

A tag criterion is only worth what its last measurement is worth, so this table says
when each one was last run and on what. **CI has not run on this branch since
2026-09-11** (run 237, commit `02dbdd1`, green): `.github/workflows/ci.yml` fires on
pushes to `main` and on pull requests targeting `main`, pull request #1 was closed
without merging, and 12 commits have landed on the branch since. Everything below
marked *local* was run on this branch's head on an Ubuntu 24.04 container with GCC
13.3.0, Clang 18.1.3 and LLVM 18 — the same toolchain versions the CI jobs install.

| Criterion | Job it mirrors | Last run | Result |
|---|---|---|---|
| 1. CI green on `main` | — | never | **The branch is not on `main`.** 178 commits ahead; nothing here has run in CI since 2026-09-11 |
| 2. Full CTest | `build-test-linux` | local, this branch | **374/374**, Release GCC 13, `MS_BUILD_INTEGRATION=ON`, CUDA and AVX-512 off |
| 2a. `-fno-exceptions` syntax gate | `build-test-linux` | local, this branch | the seven files the job names, plus `repl_engine.cpp`, `repl_engine_internal.cpp` and `finance.cpp` |
| 3. Coverage | `coverage-linux` | local, this branch | **91.5% lines** (84,496 of 92,302), **97.9% functions**, **58.6% branches**, 374/374 suites on the instrumented tree; the ratchet passes on all three |
| 4. ASan + UBSan | `sanitizer-linux` | local, this branch | **371/371 with zero sanitizer reports**, `MS_BUILD_INTEGRATION=ON`, `MS_LINK_TESTS_SHARED=ON`, `detect_leaks=0:halt_on_error=1`, the three CUDA/stress suites excluded as the job excludes them |
| 5. Fuzz, 24 h × 7 | `fuzz-24h.yml` | **not run on this branch** | the smoke is clean — 7 targets × 4096 runs seeded from the checked-in corpora, no crash — and the marathon needs GitHub Actions |
| 6. Unsafe surface | `build-test-linux` | local, this branch | **33 sites against a baseline of 33**, delta clean |
| 6a. Vendor checksums | `build-test-linux` | local, this branch | **OK, 5 files** |
| 7. Packaging | `build-test-linux` | local, this branch | install prefix and `mathscript-1.0.0-Linux.tar.gz`, **smoke OK** |
| 8. Benchmarks | `benchmark-linux` | **not run on this branch** | the gate compares against a baseline taken on a GitHub-hosted runner; running it here would measure this container. See criterion 8 above for why the gate is five entries wide |
| 9. Compliance (plugin) | `plugin-linux` | local, this branch | **42/42**, Clang 18 + LLVM 18 |
| 9a. Compliance (source) | `compliance` | local, this branch | SPDX **1806/1806**; SBOM current; matrix-call manifest current (**485 handlers**); generated dispatch tests current (**29 sources, 1,583 tests**); test names unique across **33** executables |
| 10. JIT | `jit-linux` | local, this branch | **2/2**, Clang 18 + LLVM 18 |
| 11. Documentation | — | local, this branch | this file's counts re-measured; see criterion 2 |

**Windows MSVC has not been built or tested on this branch at all.** It cannot be from
here, and `build-test-windows` is the job that covers it — packaging with NSIS and WiX,
`/bigobj` on the grouped integration binaries, and the MSVC `CMAKE_OBJECT_PATH_MAX`
limit the target names were shortened for. A green Linux run says nothing about any of
those.

So the honest summary is that **every criterion that can be checked without GitHub
Actions or a Windows machine has been checked on this branch's head and passes**, and
the three that cannot — CI on `main`, the fuzz marathon, and Windows — are the three
that gate the tag.

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
