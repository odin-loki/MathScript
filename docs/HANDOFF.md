# What to run on your machine

Everything in this file is work that cannot be done from a cloud session: it needs a
24-hour budget, a machine whose timings mean something, a Windows box, or a decision
that is yours. It is written to be picked up and executed without further context.

## How to use this file

Work top to bottom. Each task states **why** it exists, the exact commands to **run**,
and what **done** looks like. Tasks 1 and 2 are the release. Task 3 is optional before
the tag. Task 4 is everything after it, ranked.

If you are an agent: do one task per branch, run the full gate block in the appendix
before you push, and do not mark a task done on a partial result — every number in this
repository is expected to be one somebody measured. `docs/RELEASE.md` is the contract;
`docs/PLAN_STATUS.md` is the running record; add to both when you close something.

## Where things stand

`main` is at the release candidate. CMake reports **1.0.0** and the tag `v1.0.0` has not
been cut. Everything in `docs/RELEASE.md`'s criteria list has been measured except the
24-hour fuzz marathon, and that is the last hard blocker.

Before starting anything below, confirm the CI run for the current `main` is green:

```bash
gh run list --branch main --limit 1
gh run view --log-failed   # only if it is not
```

A summary of what has been verified, on what, and when is in the "What has been
verified, and where" section of `docs/RELEASE.md`. Read it before quoting any number
from this repository at anyone.

## Environment setup

### Linux or WSL (everything except the Windows job)

```bash
sudo apt-get update
sudo apt-get install -y \
    g++-13 gcc-13 clang-18 lld llvm-18-dev libclang-18-dev \
    ninja-build cmake lcov
```

CMake ≥ 3.28. The tree vendors GoogleTest and xsimd; Google Benchmark is fetched by
`FetchContent` at configure time and needs network access, or a local checkout passed as
`-DFETCHCONTENT_SOURCE_DIR_BENCHMARK=/path/to/benchmark` (tag `v1.8.5`).

### Windows

Visual Studio 2022 or 2026 with the C++ workload, CMake, Ninja, and — for the packaging
smoke — NSIS and the WiX Toolset. Then:

```powershell
.\build.ps1 -Test
pwsh -NoProfile -File scripts/package_smoke.ps1 build-msvc install-smoke
.\scripts\tag_1.0.0_checklist.ps1
```

CI runs the Windows job on every push to `main`, so this is confirmation rather than
coverage. Run it if you are about to tag, or if you have changed anything that touches
path lengths, `/bigobj`, or the installer.

---

## 1. The 24-hour fuzz marathon — the last hard blocker

**Why.** Tag criterion 5: seven libFuzzer targets, 86,400 seconds each, zero crashes.
The smoke run in CI is 4,096 runs per target and finds nothing that the corpora do not
already contain. The marathon earns its place — the last one found an out-of-memory in
`combo_restricted_partitions(442, 5)` at 2,398 MB RSS that 353 million local executions
had not.

**Run it one of two ways.**

On GitHub Actions, one 2-core runner per target:

```bash
bash scripts/fuzz_24h_dispatch.sh      # or: gh workflow run fuzz-24h.yml
gh run list --workflow=fuzz-24h.yml
```

On your own machine, all seven at once with every spare core behind them — faster, and
the way to get the evidence if you would rather not spend the Actions minutes:

```bash
# Rehearse for an hour first; it uses the same code path as the real thing.
MS_FUZZ_SECONDS=3600 bash scripts/fuzz_24h_local.sh

# The real run. Findings land in tests/fuzz/corpus/<target>/ and artifacts/<target>/.
bash scripts/fuzz_24h_local.sh
```

It holds CI's 2,048 MB RSS limit so an out-of-memory reproduces identically, seeds from
the checked-in corpora, writes new coverage back into them, and fails if any target
leaves a `crash-`/`oom-`/`leak-`/`timeout-` artifact — including when libFuzzer's own
exit code is 0, which happens when a worker rather than the parent finds the input.

**Done when** all seven targets complete their full budget with no artifact. Commit any
new corpus entries the run produces; they are the record of what was explored — and there
will now be some, which there were not before. The 2026-09-16 dispatch failed on two
defects in the job rather than in the library: the workflows passed the corpus as
`-corpus_dir=`, which is not a libFuzzer flag, so every Actions run started from an empty
corpus and discarded what it found; and the RSS cap was being filled by AddressSanitizer's
per-allocation stacks rather than by anything MathScript allocated. Both are fixed in
`scripts/fuzz_session.sh`, which every fuzz job now calls — read its header before
changing how any of them is invoked.

**If it finds something:** the crashing input is in `artifacts/<target>/`. Reproduce with
`./build-fuzz-24h/tests/fuzz/<target> artifacts/<target>/<file>`, fix it, add the input
to `tests/fuzz/corpus/<target>/` so the `replay_fuzz_*` CTest suites keep it, and say in
the commit message what the input was and why the code accepted it.

## 2. Cut the tag

**Why.** Everything else is done. Do not start this until task 1 is clean and CI is green
on the commit you intend to tag.

```bash
# 1. Move the changelog's [Unreleased] heading to [1.0.0] - <today>.
$EDITOR CHANGELOG.md

# 2. The local gate. Warn-only, so read the output rather than the exit code.
bash scripts/pre_release.sh

# 3. Read-only checklist, one last time.
bash scripts/tag_1.0.0_checklist.sh

# 4. Tag and push.
git tag -a v1.0.0 -m "MathScript 1.0.0"
git push origin main
git push origin v1.0.0
```

**Done when** the tag exists, the release artifacts from the CI run attach to it, and
`docs/RELEASE.md` records the run ID that was green when you cut it.

## 3. Worth doing before the tag, but not blocking

### 3.1 A benchmark baseline that means something

**Why.** `benchmark-linux` gates on `tests/performance/baselines/linux-gcc13.json`, and
only **5 of its 445 entries** carry a measured `median_time_ns`. The other 440 are nulls
the comparison skips. That is deliberate and the evidence is in `docs/RELEASE.md`
criterion 8: two dispatches of the same code to two GitHub-hosted runners disagreed by
more than 10% on **94%** of benchmarks, the worst by 214%. A fuller baseline on shared
hardware would fail on almost every run.

Your machine changes that, if it is one you can pin.

```bash
cmake -S . -B build-bench -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=OFF -DMS_BUILD_BENCHMARKS=ON \
  -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build-bench --target $(bash scripts/bench_cmake_targets.sh | xargs)

# Run the same set twice, hours apart, and compare the two before trusting either.
bash scripts/bench_smoke.sh build-bench
MS_BENCH_REGRESSION=on MS_BENCH_TOLERANCE=10 bash scripts/bench_regression.sh build-bench
```

**Done when** either the baseline is broadened with entries whose run-to-run spread you
have measured on the machine that will run them, or `docs/RELEASE.md` records that it
was tried and what the spread was. Do not widen the gate on a single run.

### 3.2 §8.6 — differential tests against reference BLAS and LAPACK

**Why.** `docs/ENGINEERING_PLAN.md` §8.6 says it plainly: for this audience, "agrees with
OpenBLAS and LAPACK to 1e-12 across 10,000 random inputs" carries more weight than a
coverage figure. The dgemm kernels have such tests; the wider LAPACK surface does not.
This is the largest remaining claim the project cannot make, and it needs a machine with
reference BLAS installed — which the cloud session did not have.

```bash
sudo apt-get install -y libopenblas-dev liblapack-dev liblapacke-dev
```

Then extend `tests/numerical/` with differential suites for the routines
`src/runtime/cpu/` implements, comparing against LAPACKE on random well-conditioned
inputs across a shape grid. `tests/unit/linalg/test_lapack_svd_properties.cpp` is the
model for what a property suite over a shape grid looks like; it found four real defects
on its first run.

**Done when** the routines `src/runtime/cpu/lapack_*.cpp` implements are each compared
against a reference over a documented shape and conditioning range, and §8.6 in
`docs/PLAN_STATUS.md` moves off "Partial".

---

## 4. After the tag, ranked

These are ordered by what the evidence says is weakest, not by how interesting they are.

### 4.1 Branch coverage — 58.6% against 91.5% lines

The single most informative number in the project. `docs/RELEASE.md` criterion 3a has the
argument: this tree's largest files are dispatch chains, and a dispatch chain reaches
high line coverage with one branch of each test taken. **91.5% line coverage on this
codebase means considerably less than it sounds like.**

Start from the ranked report, which names the worst files:

```bash
cmake -S . -B build-cov -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-O0 -g1 -fno-omit-frame-pointer" \
  -DMS_ENABLE_COVERAGE=ON -DMS_BUILD_INTEGRATION=ON -DMS_LINK_TESTS_SHARED=ON \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build-cov
ctest --test-dir build-cov --output-on-failure
MS_COVERAGE_MIN=80 bash scripts/coverage_report.sh build-cov
less build-cov/coverage-ranked.txt
python3 scripts/coverage_ratchet.py build-cov
```

**Watch function coverage while you are in there.** It is at 97.9% against a 98.3%
baseline — inside the 0.5-point tolerance, and using most of it. It is the one metric
moving the wrong way.

### 4.2 Split `repl_engine.cpp`

25,002 lines in one translation unit, with `repl_engine_internal.cpp` at 20,112 beside
it: **27% of `src/` in two files**, and every mutant of either costs a full recompile of
the largest TU in the tree. Three specific things inside it, each found by §8.4 and each
recorded rather than fixed:

- **A 354-line `||` chain** in `is_scalar_expression_rhs` naming every matrix-valued
  command. It is about 4% of the file's mutable sites, so a uniform mutation sample keeps
  landing in it, and every mutant there is equivalent — measured: dropping `ml_lasso_fit`,
  `ml_lasso_predict` and `ml_decision_tree_fit` out of the list changes nothing, because
  the matrix-call dispatch upstream claims them first. It also repeats four of its own
  lines verbatim. Establish how much of it is reachable before deleting any of it.
- **Twelve argument-splitting regexes**, `unary` through `nonary`, whose groups are
  `[^,]+`. `execute` currently retries through the assignment path when one of them cuts
  a matrix literal at its first comma; that is a workaround, and the fix is for the
  bare-call path to use `split_call_args` as the assignment path does.
- **`split_call_args` tracks brackets and quotes but not parentheses**, so `f(g(1, 2), 3)`
  would split into three arguments. No command reaches it that way today — nested calls
  go through the scalar expression reader, which does track parentheses — so it is a
  latent edge with no reproduction, and it should get one before it gets a fix.

Also: guards duplicated across the assignment and bare-call paths with **different
messages**, so `numthy_prime_pi(-1)` and `a = numthy_prime_pi(-1)` report differently for
the same input. A test that asked for the shorter of two such messages passed with the
outer guard deleted. Unify the messages, or delete the duplicates, but pin whichever
survives.

### 4.3 §9.5 — the SIMD tiers

`docs/PLAN_STATUS.md` rows 2324–2328. Tier 1 AVX2/FMA `dgemm` is done; NEON `dgemm` and
`sgemm` are open, then xsimd across `vector_ops`/`linalg`/`fft`/`special` (Tier 2), the
dispatch table and threading (Tier 3), and masked epilogues, VNNI/BF16, SVE and AES-NI
(Tier 4). **§13.1 bit-reproducibility depends on Tier 3** and cannot start before it.

### 4.4 §13 — the remaining features

Open: 13.1 bit-reproducibility (blocked on 9.5 Tier 3), 13.3 structured audit log,
13.4 Python bindings, 13.5 language server, 13.6 sparse direct solvers, 13.8 checkpoint
and restart. 13.2 and 13.7 are done.

### 4.5 §12 — the GUI

Open, and `MS_BUILD_GUI` defaults to **OFF**, so nothing untested ships today. It stays
out of the coverage denominator until §12.4's offscreen harness exists. If the GUI is
ever to be part of a release, that harness comes first.

### 4.6 §10 — finish the `ms::sym2` port

The core exists and is verified against the old engine by
`tests/unit/sym2/test_sym2_differential.cpp` (4,000 random expressions at three points
each, plus a round trip and a fixed-point check). What is left: the transforms, series,
limits, linear solve and ODE solvers are still `ms::symbolic`'s, and **nothing in the
REPL calls `sym2` yet**. §10.5's discipline is to port one function at a time with the
differential test as the gate; keep to it.

---

## Appendix — every gate, in one block

Run this before any push. It is the local equivalent of the CI matrix, minus the Windows
job and the benchmark comparison. On the container this was last run on, the whole block
takes about two hours, most of it the sanitizer and coverage builds.

```bash
set -euo pipefail

# --- source-only compliance (seconds) ---
python3 scripts/add_spdx.py --check
python3 scripts/gen_sbom.py --check
python3 scripts/extract_manifest.py --check --strict
python3 scripts/gen_matrix_call_tests.py --check
python3 scripts/check_test_names.py
bash scripts/verify_vendor.sh

# --- build and test, Release (the catalogue is 374 suites) ---
cmake -S . -B build-linux -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build-linux
ctest --test-dir build-linux --output-on-failure

# --- the -fno-exceptions syntax gate, plus anything you changed ---
for f in src/core/tensor.cpp src/poly/poly.cpp src/bignum/bignum.cpp \
         src/control/control.cpp src/fem/fem.cpp src/cpu/blas.cpp \
         src/runtime/cpu/blas_dgemm.cpp; do
  g++-13 -std=c++23 -fno-exceptions -fsyntax-only -Iinclude -Ibuild-linux/include -c "$f" -o /dev/null
done

# --- unsafe surface and packaging ---
bash scripts/unsafe_report.sh build-linux/unsafe_report.txt
bash scripts/unsafe_delta.sh build-linux/unsafe_report.txt
bash scripts/package_smoke.sh build-linux install-smoke

# --- ASan + UBSan (371 suites; the three the CI job excludes stay excluded) ---
cmake -S . -B build-sanitizer -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-O0 -g1 -fno-omit-frame-pointer" \
  -DMS_ENABLE_ASAN=ON -DMS_BUILD_INTEGRATION=ON -DMS_LINK_TESTS_SHARED=ON \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build-sanitizer
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
  ctest --test-dir build-sanitizer --output-on-failure \
  -E "test_fuzz_stress|test_cuda_matmul|test_cuda_stub"

# --- Clang plugin compliance (42 tests) ---
cmake -S . -B build-plugin -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=ON -DMS_BUILD_PLUGIN=ON \
  -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang
cmake --build build-plugin --target ms_plugin test_plugin_smoke
ctest --test-dir build-plugin -R 'test_plugin_smoke|compliance_' --output-on-failure

# --- LLVM ORC JIT smoke ---
cmake -S . -B build-jit -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release -DMS_BUILD_TESTS=ON -DMS_BUILD_JIT=ON \
  -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm
cmake --build build-jit --target test_jit_backend test_plot_console
ctest --test-dir build-jit -R 'test_jit_backend|test_plot_console' --output-on-failure

# --- libFuzzer smoke (the marathon is task 1) ---
cmake -S . -B build-fuzz -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMS_BUILD_FUZZ=ON \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build-fuzz --target fuzz_special_fns fuzz_matrix_ops fuzz_repl_input \
  fuzz_sym_parser fuzz_poly_ops fuzz_bignum fuzz_mpi_message
for t in fuzz_special_fns fuzz_matrix_ops fuzz_repl_input fuzz_sym_parser \
         fuzz_poly_ops fuzz_bignum fuzz_mpi_message; do
  ./build-fuzz/tests/fuzz/$t -runs=4096 -max_total_time=30 \
    -corpus_dir="tests/fuzz/corpus/$t"
done
```

The coverage build is in task 4.1 and is the slowest of the lot; run it when you have
changed tests, not on every push.

### Disk

The sanitizer tree is about 16 GB and the coverage tree about 9 GB. Build them one at a
time and delete each before the next, or keep 40 GB free.

### Mutation testing

§8.4 is done — twenty-five files measured, against a plan that asked for ten — but the
harness is the tool for any new subsystem, and it has one rule worth repeating:

```bash
# Measure a file.
python3 scripts/mutation_test.py --source src/<area>/<file>.cpp \
    --target test_<area> --build-dir build-linux --limit 24 --seed 1

# After writing tests for the survivors, RE-TEST THOSE SURVIVORS -- do not re-sample.
python3 scripts/mutation_test.py --source src/<area>/<file>.cpp \
    --target test_<area> --build-dir build-linux --replay <the previous output>
```

A run is reproducible from its seed only while the source is unchanged: the sample is
drawn from character offsets, so a two-line fix renumbers every site after it and the
same seed then draws a different set. `--replay` re-tests the survivors a report names,
located by the text of their line rather than its number. **A test that does not kill its
mutant is the same silence with more lines.**
