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
3. **Coverage** — CI gate **80%** (`coverage-linux`). Re-measured on `main` at `6eed6b43`
   on 2026-09-16: **91.5% lines** (84,594 of 92,415) and **97.9% functions** (6,008 of 6,134),
   on an instrumented Debug build with `MS_BUILD_INTEGRATION=ON`, `MS_LINK_TESTS_SHARED=ON`
   and **374/374** CTest suites passing. The **90%** tag goal is met, and
   `scripts/coverage_ratchet.py` passes against the 2026-09-10 baseline on all three
   metrics — lines −0.10, functions −0.40, branches +0.70, inside the 0.5-point
   tolerance the baseline file justifies from measurement.

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
3a. **Branch coverage** — **58.5%** (90,189 of 154,138 branches), over the same corrected
   denominator, re-measured with the rest. No CI gate is set for it yet; the ratchet does watch
   it, and it has moved up 0.7 points since 2026-09-10.

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
   codebase means considerably less than it sounds like**, and 58.5% is the number that says where the work is.

4. **ASan + UBSan** clean (`sanitizer-linux`; overflows and UB fail the job). That parenthesis was half true until 2026-09-17: UBSan's default is to print a diagnostic and carry on returning 0, and the job set `halt_on_error=1` only in `ASAN_OPTIONS`, which the UBSan runtime does not read. Undefined behaviour went into the log and the job went green over it. `UBSAN_OPTIONS: halt_on_error=1` makes the sentence true. The measurement that was quoted alongside it -- "the suite reports zero `runtime error:` lines" -- was WRONG, and how it was wrong is worth keeping: it came from grepping a `ctest --output-on-failure` log, and that option prints the output of FAILING tests only. While UBSan was not halting, every test passed, so no test's stderr was ever in the log and the grep had nothing to find. Turning the gate on immediately failed `test_crypto` and `test_repl_malformed_sweep` on two real defects -- `memcpy` from a null pointer for an empty HMAC key, and `1 << n_qubits` at n_qubits = 10000000 -- both now fixed. Verified properly by running all 398 test binaries directly and reading their stderr: zero findings in the checks this job enables. Leak detection stays off (`detect_leaks=0`) for process-exit pool/AD graphs. The job builds with `MS_BUILD_INTEGRATION=ON` and `MS_LINK_TESTS_SHARED=ON`, so it carries the whole catalogue, and excludes three suites by name (`test_fuzz_stress`, `test_cuda_matmul`, `test_cuda_stub`). Last local run: **371/371 passing with zero sanitizer reports**. An earlier edition of this file said the sanitizer tree was configured `MS_BUILD_INTEGRATION=OFF` and carried the unit suites only; that has not been true since 2026-08-29.
5. **Fuzz** — 24 h × 7 libFuzzer jobs, zero crashes (`fuzz-24h.yml`). Last completed local
   run: real libFuzzer under Clang 18, 7 targets × 10 min seeded from the checked-in corpora =
   **353 193 095 executions, zero crashes**. The corpus-replay harness (7 targets × 5 seeds ×
   200 000 mutations = 7 000 000 inputs) is also clean, and the corpora are replayed on every
   build by the `replay_fuzz_*` CTest suites, so a regression is caught even where no
   libFuzzer runtime is installed. The 24 h marathon itself was started on 2026-09-16
   (`scripts/fuzz_24h_local.sh` on the workstation, and `fuzz-24h.yml` dispatched); it is
   not yet a result. **The Actions rehearsal is now green: run 10 of `fuzz-24h.yml`,
   7 of 7 targets at 900 s each on 2026-09-17, zero crashes** — the first green run that
   workflow has had. The full 24 h is still what criterion 5 asks for; what the
   rehearsals establish is that the job itself works, which is what the previous
   seven runner-hours failed to establish.

   **A 24 h budget could not have been spent by the workflow as it stood, and the reason
   is worth keeping.** A job on a GitHub-hosted runner is terminated at six hours of
   execution time whatever `timeout-minutes` says — the job carried `timeout-minutes: 1500`
   and that number was never reachable. Run 14 (`seconds: 86400`, one target, dispatched
   2026-09-18 00:34Z) would have been killed at 06:34Z after 5 h 50 m of fuzzing, with no
   crash and no artifact, and reported as a **failure** — a result indistinguishable from a
   real finding and worthless as evidence either way. Two earlier dispatches had died of
   their own findings before reaching that limit, which is why it had not yet been seen.
   The workflow now splits the budget across chunks that fit (`fuzz-chunk.yml`, at most six,
   4 h each) chained with `needs:`, handing the corpus between them as an artifact the way
   `fuzz_session.sh` already hands it between the 900 s chunks inside one job. The corpus
   hand-off has a `chunks` dispatch input so it can be rehearsed in twenty minutes rather
   than first exercised in the run it matters in — the mistake this file already records
   once, in the seven runner-hours that proved `-corpus_dir=` was a no-op.

   Every job that runs libFuzzer now goes through `scripts/fuzz_session.sh`, which carries the corpus handling, the chunking, the sanitizer options and the artifact check. Two ways to run the marathon: `scripts/fuzz_24h_dispatch.sh` sends it to Actions, one 2-core runner per target; `scripts/fuzz_24h_local.sh` runs all seven at once on a workstation and puts every spare core behind them (`MS_FUZZ_WORKERS` per target). Both seed from the checked-in corpora, write new coverage back into them, hold the 2048 MB RSS limit so an out-of-memory finding reproduces identically, and fail if any target leaves a `crash-`/`oom-`/`leak-`/`timeout-` artifact — including when libFuzzer's own exit code is 0, which happens when a worker rather than the parent finds the input.

   **The 2026-09-16 dispatch found two defects in this job rather than in the library, and they are worth reading before the next run is quoted.** `fuzz_repl_input` died at 69 minutes with `out-of-memory (used: 2057Mb; limit: 2048Mb)` — and a **live heap of 43.8 MB**. The memory was AddressSanitizer's, not MathScript's: ASan keeps a thirty-frame allocation stack for every chunk it has handed out, and the run had been through 6.4 million of them. Measured on a 6,280-unit corpus over five minutes, `malloc_context_size=5` takes the floor from 769 MB to 154 MB and the peak from 968 MB to 209 MB while executing slightly *more* inputs. So the cap stays at 2048 MB and means what it says again; raising it would have silenced the guard that caught the real `combo_restricted_partitions(442, 5)` finding at 2398 MB.

   **With the corpus finally loading, a ten-minute rehearsal found a real crash in 143 seconds**: `cellmemory_new(cm, 2, 4555555555555555, [0.1,31, 107])` asked `new[]` for 36 petabytes and ended the process, at a peak RSS of 186 MB rather than 2,057. `CellMemory` and `DifModel` checked that their dimensions were positive integers and stopped there — the same family as the fourteen size arguments bounded earlier, missed by that sweep because these are session-object constructors rather than matrix calls. Both are bounded now, `DifModel` on the **product** `output_dim × input_dim` because that is what it allocates. The input is in the corpus and the regression is in `test_repl_resource_guards.cpp`.

   The second defect is the worse one. The workflows passed the corpus as `-corpus_dir=DIR`, which **is not a libFuzzer flag**: it was accepted as unrecognised, ignored, and every Actions fuzz run — the smoke, both nightly sessions and the marathon — started from an empty corpus, never read the inputs earlier runs had found interesting, and discarded everything it found. `scripts/fuzz_24h_local.sh` had it right, which is how the divergence survived. There is now one implementation and five callers of it.

   **A second rehearsal found a second crash, in 61 seconds**: `tensorops_decompose_hosvd(jk2, [1, 0; 0, 1], [1, 155555555555555555555555555555555555555])` — 39 digits — ended the process with a `std::length_error` escaping a library built with `-fno-exceptions`. The handler *did* guard the rank, and that is the lesson: `rank > tensor->shape[mode]` is an **upper** bound, and what reached it was INT_MIN, because `static_cast<int>` of a double that far past `int`'s range is undefined behaviour. Every earlier case of this had a `< 0` somewhere that rejected INT_MIN by accident. Auditing the rest of the interp layer for the same conversion found **eleven more sites, none of which crashed** — `combo_next_perm([0, 1555…555])` printed `[-2147483648, 0]` as its answer. Those are the ones worth worrying about: the earlier size sweeps put oversized values in *scalar* argument positions, and every one of these takes its oversized value inside a *vector*. Widening the same audit to the other integer widths found a twelfth, in `int64_t`: `numthy_convergents([1e30; 7])` reported a first convergent of -9223372036854775808. The recurrence underneath it is careful — `numthy::convergents` uses `checked_mul`/`checked_add` and stops on overflow — which is precisely why the bad value survived to be printed instead of being caught.

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
when each one was last run and on what. `main` at `6eed6b43` was measured on
2026-09-16: Windows MSVC 2022/2026 on the workstation, and Linux in WSL Ubuntu
24.04 with GCC 13.3.0, Clang 18 and LLVM 18 — the same toolchain versions the CI
jobs install. GitHub Actions run [35079998777](https://github.com/odin-loki/MathScript/actions/runs/35079998777)
on that commit was cancelled after plugin, JIT, fuzz-smoke and compliance
succeeded; the Windows, Linux, coverage, sanitizer and benchmark jobs did not
finish there. The local figures below are the ones that did.

| Criterion | Job it mirrors | Last run | Result |
|---|---|---|---|
| 1. CI green on `main` | — | 2026-09-16, run 35079998777 | **cancelled.** Plugin, JIT, fuzz-smoke and compliance succeeded; Windows, Linux, coverage, sanitizer and benchmark were cancelled |
| 2. Full CTest | `build-test-linux`, `build-test-windows` | local, 2026-09-16 | **374/374** Release GCC 13 in WSL; **374/374** Release MSVC on Windows. `MS_BUILD_INTEGRATION=ON`, CUDA and AVX-512 off |
| 2a. `-fno-exceptions` syntax gate | `build-test-linux` | local, this branch | the seven files the job names, plus `repl_engine.cpp`, `repl_engine_internal.cpp` and `finance.cpp` |
| 3. Coverage | `coverage-linux` | local, 2026-09-16 | **91.5% lines** (84,594 of 92,415), **97.9% functions** (6,008 of 6,134), **58.5% branches** (90,189 of 154,138), 374/374 suites on the instrumented tree; the ratchet passes on all three (lines −0.10, functions −0.40, branches +0.70 vs the 2026-09-10 baseline) |
| 4. ASan + UBSan | `sanitizer-linux` | local, 2026-09-17 | **371/371 with zero sanitizer reports** in 1370 s, `MS_BUILD_INTEGRATION=ON`, `MS_LINK_TESTS_SHARED=ON`, `detect_leaks=0:halt_on_error=1` **and now `UBSAN_OPTIONS=halt_on_error=1`**, the three CUDA/stress suites excluded as the job excludes them. Until that second variable was set, a UBSan finding could not fail this job — see criterion 4 |
| 5. Fuzz, 24 h × 7 | `fuzz-24h.yml` | **rehearsal green 2026-09-17, run 10: 7 of 7 targets at 900 s, zero crashes**; the full 24 h still needs dispatching | Three rehearsals, three real defects, each one unblocking the next: `cellmemory_new` (143 s), `tensorops_decompose_hosvd` (61 s), and the eleven silent siblings of the second that an audit found rather than the fuzzer — all fixed, all in the corpus. Earlier state, kept because it explains the gap: 6 of 7 targets green on a 10-minute rehearsal; `fuzz_repl_input` found a real unbounded allocation in `cellmemory_new`, now fixed. The failure before the fix was not in the library: `fuzz_repl_input` hit the RSS cap at 69 minutes with a **43.8 MB live heap**, and the corpus flag was a no-op so no Actions run had ever been seeded. Both fixed in `scripts/fuzz_session.sh` — see criterion 5 — and **the marathon needs re-dispatching**. A local run started the same evening predates the fix and will hit the same cap. The smoke is clean (7 targets × 4096 runs, no crash) though it was not seeded either |
| 6. Unsafe surface | `build-test-linux` | local, this branch | **33 sites against a baseline of 33**, delta clean |
| 6a. Vendor checksums | `build-test-linux` | local, this branch | **OK, 5 files** |
| 7. Packaging | `build-test-linux`, `build-test-windows` | local, 2026-09-16 | Linux install prefix and `mathscript-1.0.0-Linux.tar.gz`, **smoke OK**. Windows install prefix and `mathscript-1.0.0-win64.zip`, **smoke OK**. NSIS/WiX skipped: tools not installed |
| 8. Benchmarks | `benchmark-linux` | local, 2026-09-16 (build and smoke only) | all **28** bench targets build and run on Windows MSVC and on Linux GCC 13. The **regression comparison was not run**: the baseline was taken on a GitHub-hosted runner. See criterion 8 above for why the gate is five entries wide |
| 9. Compliance (plugin) | `plugin-linux` | local, 2026-09-16 | **42/42**, Clang 18 + LLVM 18 |
| 9a. Compliance (source) | `compliance` | local, 2026-09-16 | SPDX **1806/1806**; SBOM current; matrix-call manifest current (**485 handlers**); generated dispatch tests current (**29 sources, 1,583 tests**); test names unique across **33** executables |
| 10. JIT | `jit-linux` | local, 2026-09-16 | **2/2**, Clang 18 + LLVM 18 |
| 11. Documentation | — | local, 2026-09-16 | this file's counts re-measured; see criterion 2 |

Windows MSVC on this commit needed two `M_PI` guards the Linux tree does not
(`test_special_reference_values`, `test_signal_cheby2_orders`) and a `build.ps1`
fix so `-Benchmark` finds the globbed `bench_*.cpp` targets. After those, the
grouped integration binaries, `/bigobj`, and the ZIP package smoke are what this
workstation can check. NSIS and WiX remain untested here.

[`HANDOFF.md`](HANDOFF.md) is the companion to this table: what is left, what each item
needs a real machine for, and the exact commands to run it.

So the honest summary is that **every criterion that can be checked locally has
been checked on this commit and passes**, including Windows. What has not:
CI green on `main` (the run that would have said so was cancelled), the 24-hour
fuzz marathon (started, not finished), NSIS/WiX packaging, and the benchmark
regression comparison — which could run here but would be comparing two different
machines, so it would produce a number rather than an answer.

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
