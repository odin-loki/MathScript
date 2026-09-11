# MathScript — Engineering Plan

> **Preserved verbatim.** This is the audit document as written, against the tree
> at `4acd942`. It is deliberately not edited as work lands: its value is as a
> dated record of what was true when it was written, including the three places
> where it turned out to be wrong.
>
> For what has actually been done, what remains, and where the plan was mistaken,
> see [`PLAN_STATUS.md`](PLAN_STATUS.md).

**Byline:** Odin Loch
**Date:** 2026-08-31
**Tree audited:** `github.com/odin-loki/MathScript` @ `4acd942`
**Scope:** whole repository. No module excluded, nothing deferred for convenience.

This document consolidates three passes over the tree — a coverage audit, a
whole-repository defect and claims audit, and a feature design study for the
symbolic engine, GUI and SIMD layer — into one plan with a single ordering.

---

## Contents

1. [The short version](#1-the-short-version)
2. [What the tree actually contains](#2-what-the-tree-actually-contains)
3. [Work already completed](#3-work-already-completed)
4. [Blocking: legal and commercial](#4-blocking-legal-and-commercial)
5. [False and stale claims](#5-false-and-stale-claims)
6. [Correctness defects](#6-correctness-defects)
7. [Stubs and half-implementations](#7-stubs-and-half-implementations)
8. [Coverage and testing programme](#8-coverage-and-testing-programme)
9. [Performance and intrinsics](#9-performance-and-intrinsics)
10. [Symbolic engine](#10-symbolic-engine)
11. [LaTeX and notation interchange](#11-latex-and-notation-interchange)
12. [GUI](#12-gui)
13. [Other features worth building](#13-other-features-worth-building)
14. [Sequencing](#14-sequencing)
15. [Appendix](#15-appendix)

---

## 1. The short version

**Do not cut the `v1.0.0` tag yet.** Five things must close first: the AGPL-3.0
licence is chosen but not applied and needs a §7 exception before a CUDA build can be
distributed at all; there is no export-control position on the crypto; 94% of commits
are attributed to a bot rather than to you; `simd/isa.cpp` can SIGILL inside `dgemm`
depending on the host; and `crypto::random_bytes` is not a CSPRNG. A 1.0 with no
licence and a crash in the hottest path is worse than no 1.0.

**The test volume is not the problem.** ~816 CTest targets, ~24,768 test cases.
The problems are that a quarter of the tree was outside the coverage denominator,
branch coverage was switched off, and nothing verifies that the tests assert rather
than merely execute.

**The single largest performance gap** is that there is no AVX2/FMA `dgemm`. One
hand-written intrinsics file exists in 139k lines. On every machine without AVX-512 —
all of Zen 1–3, every Intel client part since Alder Lake, every ARM machine — matrix
multiply falls to scalar.

**The single largest feature blocker** is the symbolic core representation. It is a
binary tree over `double`, so exact arithmetic is impossible and simplification can
never be correct. Both the LaTeX work and the GUI math rendering depend on replacing
it.

**What is genuinely good, and should not be disturbed:** zero `throw` statements
across 139k lines under `-fno-exceptions`, a `Result<T>` discipline applied almost
everywhere, constant-time GCM tag verification, crypto tested against published
FIPS-197 / FIPS-180 / RFC 4231 vectors, numerical tests against NIST DLMF reference
values, and a matrix-call registry generated from the file glob so it cannot drift
from the source tree. That is disciplined work and it is the reason the rest of this
document is worth writing.

---

## 2. What the tree actually contains

| | |
|---|---|
| Source | 139,382 LOC across 722 files |
| Tests | 398,002 LOC across 925 files |
| CTest targets | ~816 (207 unit + 573 integration + numerical + compliance) |
| `TEST(...)` macros | 24,768 |
| `src/interp` | 70,170 LOC — **50% of the codebase** |
| `repl_engine.cpp` + `repl_engine_internal.cpp` | 39,806 LOC in two files |
| `matrix_calls/` | 29,478 LOC across 478 generated files (18,484 after §3.2) |
| `src/gui` | 4,569 LOC, one test file |
| `src/plugin` | 1,189 LOC, **zero** test files |
| `src/cuda` | 1,164 LOC + one `.cu` file |
| `src/simd` + AVX-512 kernel | 550 + 23 LOC of intrinsics |
| `src/symbolic` | 2,835 LOC |
| Licence files | **none** |

**A correction worth recording.** My first read took the CHANGELOG's "115 CTest
suites, 1,339 test cases" at face value and reported the project as under-tested by
an order of magnitude. That number is stale. `tests/CMakeLists.txt` globs
`unit/*.cpp` and `tests/integration/CMakeLists.txt` globs the remaining 573 files
into their own targets, so the README's figure of 816 is the correct one. The
CHANGELOG's wave-by-wave counts contradict both the README and each other, which is
itself a finding — see §5.

---

## 3. Work already completed

Delivered as `mathscript-coverage-phase0-2.patch` — **516 files changed,
+8,465 / −11,902**.

### 3.1 Coverage tooling made honest

`scripts/coverage_report.sh` previously removed `matrix_calls`, `gui`, `plugin`,
`cuda` and `ms_bundle.cpp` from the denominator — roughly a quarter of the tree. The
published percentage was therefore not the project's coverage.

- Exclusions moved into `scripts/coverage_exclusions.txt` as declared data, one glob
  per line, each with a written reason. `matrix_calls`, `plugin` and `ms_bundle` are
  back in. Only `cuda` (no GPU runner) and `gui` (no display server) remain out.
- The rule is now stated in the file: an exclusion asserts the code *cannot* be
  reached on a CI runner, not that nobody has got to it yet.
- Branch **and** function coverage enabled (`lcov_branch_coverage=1`); the previous
  configuration measured lines only, which on a codebase whose largest files are
  dispatch chains measures very little.
- Three independent gates: `MS_COVERAGE_MIN`, `MS_COVERAGE_BRANCH_MIN`,
  `MS_COVERAGE_FUNC_MIN`.
- Each run prints how many lines the exclusions hid, so the list stays visible rather
  than silently growing.
- Per-file ranked report emitted via the existing `cov_rank.py`.

### 3.2 11,472 lines deleted before a single test was written

The generator that produced `src/interp/matrix_calls/` stamped an identical
four-lambda prologue into all 478 handler TUs, regardless of use:

| helper | defined in | actually used in |
|---|---|---|
| `resolve_operand` | 478 | 405 |
| `parse_scalar_arg` | 478 | 112 |
| `parse_positive_size_arg` | 478 | **9** |
| `parse_uint64_arg` | 478 | **1** |

38.9% of that directory was copy-paste, most of it unreachable. Writing tests for it
would have been writing tests for a code-generation error.

`scripts/hoist_prologue.py` moves the helpers onto `MatrixCallCtx` and
`matrix_call.hpp` and rewrites all 478 call sites to `ctx.<helper>(...)`. Bodies and
`DomainError` payloads are unchanged. **29,478 → 18,484 lines: −11,472, or 8.2% of
the entire codebase.**

### 3.3 1,348 generated test cases

`scripts/extract_manifest.py` parses all 478 handlers into
`matrix_calls_manifest.json` — callee, accepted arities, open-ended arity
comparisons, helpers used, and every reachable `DomainError`. It parses 478/478 with
one flagged anomaly (§6.6).

`scripts/gen_matrix_call_tests.py` emits `tests/unit/matrix_calls/` — 29 TUs wired as
the `test_matrix_calls` CTest target:

- 478 registration tests — every handler reachable through `dispatch_matrix_call`
- 469 wrong-arity tests — fall-through to `DomainError{"assign","unsupported matrix call"}`
- 401 undefined-operand tests — error propagates rather than crashing

Happy paths are deliberately **not** generated. Valid inputs are per-callee semantics
and belong in the hand-written REPL suites and the golden corpus of §8.3.

CI should re-run both scripts and fail on a dirty tree, so a new handler cannot land
without its dispatch tests.

---

## 4. Blocking: legal and commercial

These stop a sale before any technical review begins. All four are cheap relative to
the rest of this document.

### 4.1 Licensing — AGPL-3.0 (decided)

139,382 lines of source currently ship with **no top-level licence, no copyright
headers, no SPDX tags, no NOTICE**. Without a licence file the default is exclusive
copyright: nobody may legally use, copy or redistribute the code, which makes
evaluation by an agency impossible. No government or prime-contractor procurement
clears this.

**Decision: AGPL-3.0-or-later**, with the author retaining copyright so that
commercial licences can be sold separately.

#### 4.1.1 Work required

- `LICENSE` — the verbatim AGPL-3.0 text. Do not edit it; modifications invalidate
  compatibility.
- `COPYRIGHT` — `Copyright (C) 2026 Odin Loch`, and the statement that the author
  retains full copyright.
- `SPDX-License-Identifier: AGPL-3.0-or-later` in every source header, enforced in CI
  so a file cannot land without one.
- `NOTICE` / `THIRD_PARTY.md` — every vendored and fetched dependency with its terms.
- The standard AGPL notice in `--version` output and at REPL startup, including the
  network-use clause pointer. Section 13 obligations are only discharged if users
  interacting over a network are told where to get the source.

#### 4.1.2 Dependency compatibility — one real problem

| Dependency | Licence | Compatible with AGPL-3.0? |
|---|---|---|
| googletest | BSD-3 | Yes (permissive, one-way) |
| xsimd | BSD-3 | Yes |
| curve25519-donna | Public domain | Yes |
| LLVM / ORC JIT | Apache-2.0 with LLVM exception | Yes (Apache-2.0 → GPLv3 is one-way compatible) |
| Qt 6 | LGPLv3 or commercial | Yes under LGPLv3. If you ever link Qt **statically**, LGPL relinking obligations attach — keep it dynamic. |
| MPI | OpenMPI BSD, MPICH permissive | Yes |
| **CUDA — cuBLAS, cuSolver, NCCL** | **Proprietary NVIDIA EULA** | **No, not without an added exception** |

The CUDA case is the one to handle deliberately. Linking AGPL-licensed code against
proprietary NVIDIA libraries is the classic GPL system-library problem. The
"system library" exception in GPLv3 §1 is arguable for the driver but weak for
cuBLAS and cuSolver, which are not part of the operating system.

The standard remedy is an **additional permission under GPLv3 §7**, in the style of
the widely-used OpenSSL exception, granting permission to link against the NVIDIA
CUDA runtime and libraries and to distribute the combination. You are the sole
copyright holder, so you can grant this. Write it into `LICENSE.exceptions` and
reference it from every affected file. Without it, distributing a CUDA-enabled binary
is a licence violation of your own licence.

#### 4.1.3 What AGPL means for your sales strategy

AGPL is a defensible choice, but go in with the trade-offs stated:

- **The dual-licence model works, and it is the right one for you.** Sole copyright
  means you can sell proprietary licences to agencies that cannot accept AGPL terms,
  while the public tree stays AGPL. This is the Qt/MongoDB/GitLab model.
- **The corollary is that outside contributions break it.** The moment someone else's
  patch lands without a signed CLA or copyright assignment, you can no longer
  relicense that code, and the dual-licence business model quietly stops working. Put
  a CLA in place *before* accepting the first external pull request, not after.
- **AGPL is on several procurement blocklists.** Google bans it internally, and a
  number of primes and agencies reject it by policy — usually because the §13 network
  clause is hard to reason about in classified or air-gapped deployments. Expect it
  to come up. The answer is the commercial licence, so have terms and a price ready
  before you submit anywhere.
- **§13 is the whole point of choosing AGPL over GPL**, and it is live here: the tree
  has a `server_cli` target, so MathScript can be operated over a network. That is
  exactly the case AGPL covers and GPL does not.

### 4.2 Trademark collision

"MathScript" is National Instruments' registered mark for LabVIEW MathScript. Rename
before the tag, not after. A rename touches the namespace, the binaries
(`mathscript`, `mathscriptc`, `mathscript-repl`), the `.ms` file extension, CMake
target names, package names and the repository URL — cheap now, expensive once
shipped and referenced in a customer's documentation.

### 4.3 No export-control position

The tree implements AES-128/256, AES-GCM, SHA-256/512, HMAC, X25519 and Ed25519. For
an Australian entity selling to Five Eyes agencies that engages the Defence Trade
Controls Act and DSGL Part 2 Category 5. There is no written position anywhere in the
repository. This needs a documented determination before distribution, and the
outcome affects whether the crypto module ships at all.

### 4.4 No SECURITY.md, no SBOM

No vulnerability contact, no disclosure policy, no CycloneDX or SPDX SBOM. All three
are routine procurement asks now. `vendor/CHECKSUMS.sha256` is good practice and is
already most of the way to an SBOM — finish it. The SBOM and the `NOTICE` file of
§4.1.1 are the same inventory; generate both from one source.

### 4.5 The git history does not establish your authorship

Commit authorship across all 1,440 commits:

| Author | Commits | Share |
|---|---|---|
| `MathScript Bot <bot@mathscript.local>` | 1,351 | 93.8% |
| `odin-loki <odin-loki@users.noreply.github.com>` | 72 | 5.0% |
| `MathScript Agent <agent@mathscript.local>` | 13 | 0.9% |
| `MathScript Agent <agent@local>` | 2 | 0.1% |
| `odinl <odinl@local>` | 1 | — |
| `Cursor Agent <agent@cursor.com>` | 1 | — |

**94% of the work is attributed to an identity that is not you.** Only 72 commits
carry your GitHub identity. The apparent "second contributor" on the GitHub page is
the single `agent@cursor.com` commit of 2026-07-11 — a Cursor session that committed
under its own default identity rather than your git config. GitHub maps commits to
accounts by author email; the `*.local` addresses resolve to nothing, so those show
as unlinked rather than as contributors.

This matters more than any technical defect in this document. The strategy is cold
submission to agencies pointing at this repository. The first thing an IP or
procurement review does is read the history, and what it currently shows is a project
where the named human wrote one commit in twenty and a bot wrote the rest.

It also directly undermines §4.1. The dual-licence model rests on your being the sole
copyright holder. A history that attributes 94% of the code to `bot@mathscript.local`
is poor evidence of that, and it is the kind of thing a counterparty's lawyer raises
precisely when you are trying to sell a commercial licence.

#### 4.5.1 Fix

- **Rewrite the history.** `git filter-repo --mailmap`, collapsing every identity onto
  `Odin Loch <odin.loch@outlook.com.au>`. Note that a `.mailmap` file alone is not
  sufficient: GitHub's contributor graph ignores mailmap, which only affects local
  `git shortlog` and `git log --use-mailmap`.
- This rewrites every SHA and requires a force push. Do it **now**, while nothing
  external depends on the hashes, and do it in the same pass as the §4.2 rename and
  the §4.1 header insertion so the history is disturbed exactly once.
- Set `user.name` and `user.email` in the repository's local git config so agent tools
  inherit your identity instead of inventing one.
- Add a CI check rejecting any commit whose author email is not on an allowlist.
- Keep a signed tag at the rewrite point as a provenance anchor.

---

## 5. False and stale claims

Each is checkable by a reviewer in five minutes, and each costs more credibility than
the underlying gap it conceals.

| Claim | Location | Reality |
|---|---|---|
| "CI enforces **90%** line coverage" | README §Status | `ci.yml` sets `MS_COVERAGE_MIN: "80"` |
| That 80% is the project's coverage | implied | ~25% of the tree was excluded from the denominator (§3.1) |
| "115 CTest suites (1,339 test cases)" | CHANGELOG Wave 22 | ~816 targets, ~24,768 cases |
| "95 CTest suites (932 test cases)" | CHANGELOG, same file | contradicts the line above it |
| "`v1.0.0` tag is not cut" | README | CHANGELOG carries a `[1.0.0]` link to a tag that does not exist |
| "28 benchmarks passed locally with `--benchmark_min_time=0.001s`" | README | the regression gate runs 0.1s / 5 reps — validated at a setting the gate never uses |
| `dist_cg`, `dist_gmres`, … are distributed solvers | `src/distributed/iterative.cpp` | nine `stub_gather_*` functions gather to rank 0 and solve serially |
| The coverage report is trustworthy | `coverage_report.sh` | `--ignore-errors mismatch,gcov,empty,source,unused` suppressed exactly the errors that indicate stale gcov data |

**Structural fix.** Generate the status block from the build rather than maintaining
it by hand. A CI step that emits `docs/STATUS.md` from `ctest -N`, the coverage
summary and the benchmark JSON eliminates this entire class of problem permanently.
Nothing on this list would have survived a generated status page.

---

## 6. Correctness defects

### 6.1 CPUID without OSXSAVE / XGETBV — `src/simd/isa.cpp` (crash)

```cpp
f.avx     = (info[2] & (1 << 28)) != 0;
f.avx512f = (info[1] & (1 << 16)) != 0;
```

CPUID reports what the *silicon* supports. It does not report whether the *OS* has
enabled the extended register state. The correct sequence is `CPUID.1:ECX.OSXSAVE[27]`,
then `XGETBV(0)`, requiring `XCR0 & 0x6 == 0x6` for AVX/AVX2/FMA and
`XCR0 & 0xE6 == 0xE6` for AVX-512 (opmask + ZMM_Hi256 + Hi16_ZMM).

Without it, `avx512::available()` can return true on a host where executing `_mm512_*`
faults — certain hypervisors, sandboxes, older kernels, and any configuration where
AVX-512 is masked off. `avx512_dgemm.cpp` is compiled with `-mavx512f` and dispatched
on this flag, so the failure mode is **SIGILL inside `dgemm`**, not a fallback to a
slower path.

This is the highest-priority technical item in the repository: a crash in the hottest
path, triggered by the deployment environment rather than by input, on a code path
with no way for the caller to defend itself. It is also a hard prerequisite for §9 —
adding ISA paths on top of broken detection multiplies the bug rather than adding
capability.

### 6.2 Miller-Rabin is broken for large inputs — `src/bignum/bignum.cpp:406`

```cpp
std::mt19937_64 rng(42);
long long nll = nm1.to_ll();
if (nll <= 2) nll = 3;
long long ull = 2 + (long long)(rng() % (std::abs(nll) - 2));
```

Four defects in five lines:

- `to_ll()` on an arbitrary-precision value that exceeds `long long` — the witness
  range is garbage for exactly the inputs a bignum primality test exists to handle.
- `std::abs(nll)` on that garbage; if it is `LLONG_MIN`, `abs` is undefined behaviour.
- If `std::abs(nll) == 2`, the expression is `% 0` — division by zero.
- Seed 42 is fixed, so the witness set is deterministic and publicly known.
  Composites that pass are constructible by anyone who reads the source.

This is a crypto-adjacent primitive. It needs a rewrite tested against known
Carmichael numbers and the standard deterministic witness sets for 64-bit inputs.

### 6.3 `crypto::random_bytes` is not a CSPRNG

```cpp
std::random_device rd;
for (...) out[i] = static_cast<std::uint8_t>(rd());
```

One `std::random_device` call per **byte**, discarding 24+ bits each time, and
`std::random_device` carries no cryptographic guarantee in the standard — on some
toolchains it has historically been a fixed-sequence PRNG. This is the key-generation
path.

Use `getrandom(2)` on Linux, `BCryptGenRandom` on Windows, `arc4random_buf` on BSD,
and fail loudly if none is available. Add a startup self-test.

### 6.4 AES is table-driven — cache-timing side channel

The S-box, `xtime`/`aes_mul` and `MixColumns` use data-dependent table lookups. This
is not constant-time and leaks key material to a co-resident attacker. Acceptable in
a computer algebra system; not acceptable in anything described as crypto for a
defence customer. Either document it as non-hardened and explicitly out of scope, or
move to AES-NI with a bitsliced fallback (see §9.5, Tier 4).

### 6.5 `--allow-multiple-definition` masks ODR violations

`src/CMakeLists.txt:102` passes this to the `ms_bundle` link. It silences duplicate
symbols rather than fixing them; whatever is duplicated is still duplicated, and the
linker is now choosing arbitrarily between definitions. Remove the flag, read the
errors, fix them. The `MS_LINK_TESTS_SHARED` path is also untested.

### 6.6 Sole handler without an arity guard

`src/interp/matrix_calls/frameworks/izaac_vrf_keygen.cpp` uses `assign.args.empty()`
where the other 477 use an explicit `size()` check. Found by the manifest parser in
§3.3. Either normalise it or document why zero-argument handlers differ.

### 6.7 Undocumented fixed RNG seeds

`std::mt19937 rng(seed)` appears throughout `optim`, `stats`, `finance` and `linalg`.
Determinism is the right default for reproducibility, but a user calling a Monte
Carlo routine twice and receiving identical output is surprised unless it is
documented. Needs an explicit seeding contract in `docs/API.md`. See also the
reproducibility work in §13.

### 6.8 Sentinel returns in the symbolic API

Nine functions return a valid-looking `SymExpr` to signal failure. Covered in §10.2,
since the fix is part of the core rewrite rather than a standalone patch.

---

## 7. Stubs and half-implementations

Each needs a ship-or-cut decision, and anything cut needs to say so in the
documentation rather than in a function body.

| Item | State | Recommendation |
|---|---|---|
| `distributed/iterative.cpp` | Nine `stub_gather_*`: gather to rank 0, solve serially | **Rename the API.** A `dist_cg` that is not distributed is a false claim in the symbol table. Call them `gather_cg` and document the topology, or implement real domain decomposition. |
| `cuda/nccl.cpp` | Documented stubs | Keep, but return errors rather than success, and say so in `docs/API.md`. |
| `cuda/solver.cpp` | cuSolver `getrf` real; `cuda lu` returns `DeviceError` | Honest. Leave as is. |
| `src/cuda/kernels.cu` | The only `.cu` file in the tree | The CUDA surface is thin — mostly cuBLAS/cuSolver wrappers. Acceptable, but the README should not imply a GPU compute stack. |
| `frameworks/axiom/axiom.cpp` | Placeholders | Cut from 1.0 or implement. |
| `src/gui/MainWindow.cpp` | Placeholders; 4,569 LOC module, one test file | Do not cut — expand it (§12), but decompose and test it first. |
| `ms::inv` | Missing; Wave 22 worked around it with `solve` | Implement it or remove it from the documentation. Users will look for it. |
| `interp/jit_orc_stub.cpp` | The non-LLVM backend | Correct by design. Leave. |

---

## 8. Coverage and testing programme

Volume is not the problem — 24,768 cases is real work. Distribution and verification
are the problems.

### 8.1 Baseline first (must run on real hardware)

Nothing below is worth attempting before `coverage-ranked.txt` exists. Guessing at
what is already covered is how `src/plugin` ended up with 1,189 lines and zero tests.

Commands and outputs are in §15.3. Expect the headline number to fall sharply —
that is the exclusions coming off, not a regression.

### 8.2 `src/plugin` — 1,189 LOC, zero tests

It was excluded from coverage, so nobody saw it. It is the Clang enforcement plugin:
the component that guarantees the rest of the tree's safety properties. It is
entirely untested. This is the highest-value gap on the list because everything else
in the repository leans on it.

### 8.3 `repl_engine.cpp` + `repl_engine_internal.cpp` — 39,806 LOC in two files

Golden-transcript corpus: `tests/repl_corpus/*.ms` with committed expected stdout and
stderr, one CTest per file, driven through `mathscriptc`. Adding a command then means
adding a transcript, not editing a 21,533-line test file.

A second pass, driven by the ranked report, targets every error return in the
dispatch chain. The existing fuzz corpus under `tests/fuzz/corpus/` should be
replayed into the coverage build — it already contains inputs no hand-written test
would think of.

### 8.4 Mutation testing

24,768 cases prove lines execute. They do not prove anything asserts. Run `mull`
against the ten highest-ranked files. A surviving mutant in `crypto` or `linalg` is
the finding that matters — a line that is covered and untested is precisely the
failure mode a coverage percentage conceals.

### 8.5 Property-based testing

The numerical core is full of invariants nobody checks: `A·A⁻¹ ≈ I`, `Q·Qᵀ ≈ I`,
`L·U ≈ P·A`, `ifft(fft(x)) ≈ x`, `det(AB) = det(A)·det(B)`. RapidCheck over random
well-conditioned matrices finds what 24,768 fixed cases do not. The same technique
applies to the printer round-trips in §11.

### 8.6 Differential testing against reference BLAS/LAPACK

`src/linalg` is 2,438 LOC of hand-written kernels. For a product aimed at defence,
"agrees with OpenBLAS and LAPACK to 1e-12 across 10,000 random inputs" is the claim
that carries weight — considerably more than a coverage figure.

### 8.7 Remaining gaps

- `exe/` (493 LOC) and the `--jit`, `--debug`, `--jit-stats`, `--eval-file` flag
  matrix. The CLI is what a customer touches first.
- Fuzzing beyond smoke: seven libFuzzer targets and a 24-hour workflow exist, but the
  REPL parser is the untrusted-input boundary and should be targeted directly.
- The `ms_bundle` / `MS_LINK_TESTS_SHARED` path, which also carries the ODR
  suppression from §6.5.
- `src/gui` needs the offscreen harness of §12.4 before it can leave the exclusion
  list.

### 8.8 Build cost — why coverage work is currently painful

573 integration tests are 573 executables, each linking the whole library. On an
instrumented build that dominates the wall clock. Group them into per-domain binaries
(29 targets rather than 573): the same tests, an order of magnitude less link time,
and coverage builds become routine rather than a nightly event. Do this early — it
makes every subsequent item in this section cheaper.

### 8.9 Lock it in

- **Ratchet gate.** CI reads the previous run's number and fails on any decrease. No
  fixed threshold to argue about, no silent slippage.
- CI re-runs `extract_manifest.py` and `gen_matrix_call_tests.py`, failing on a dirty
  tree.
- Generated `docs/STATUS.md` (§5) as the single source of truth for every published
  number.

---

## 9. Performance and intrinsics

### 9.1 Prerequisite

**§6.1 must be fixed first.** Adding ISA paths on top of detection that can SIGILL
multiplies the bug rather than adding capability. Alongside the fix, add a
`MS_FORCE_ISA` environment override so every path can be exercised on one machine,
and a CI matrix that runs scalar, SSE2, AVX2 and AVX-512 paths under Intel SDE.

### 9.2 The gap

One hand-written intrinsics file in 139,382 lines (`avx512_dgemm.cpp`, 23 lines of
actual intrinsics usage), and `xsimd` appears in 12 places despite being vendored at
13.2.0.

There is **no AVX2/FMA `dgemm`**. On every machine without AVX-512 — all of Zen 1
through 3, every Intel client part since Alder Lake, every ARM machine — matrix
multiply falls to scalar. That is most of the market, and it is the single largest
performance gap in the project.

### 9.3 Strategy: xsimd first, hand-written only where measured

`xsimd` gives SSE2 through AVX-512, plus NEON and SVE, from one source. Write kernels
once against `xsimd::batch`, measure, and drop to raw intrinsics only where the
generic version leaves real performance on the table. The current position — one
hand-written AVX-512 kernel and nothing else — is the worst of both: maximum
maintenance burden for minimum coverage.

### 9.4 Fix the existing kernel before writing more

Two problems in `avx512_dgemm.cpp` that will be copied into every kernel modelled on
it:

**`load_b_panel` is a gather in disguise.**
```cpp
return _mm512_set_pd(B[(j0+7)*ldb + p], B[(j0+6)*ldb + p], ... );
```
Eight strided scalar loads per vector, on every iteration of the inner loop. Pack B
into a contiguous panel once, outside the loop, then load with `_mm512_load_pd`.
Typically 2–4x on its own.

**No cache blocking.** There is a 4×8 micro-kernel and nothing above it. Without
L2/L3 tiling it will not scale past matrices that fit in cache. The Goto/BLIS
structure — outer loops over `nc`/`kc`/`mc` blocks, packed A and B panels,
micro-kernel at the bottom — is well documented and worth following rather than
reinventing.

### 9.5 Roadmap

**Tier 1 — the gap that costs sales**
- AVX2/FMA `dgemm`, with B-panel packing and BLIS-style cache blocking.
- NEON `dgemm`. Graviton and Apple silicon are default numerical platforms now.
- Apply the same packing fix to the existing AVX-512 kernel.
- `sgemm` — single precision doubles the lane count, and ML workloads want it.

**Tier 2 — broad, cheap wins through xsimd**
- `src/simd/vector_ops.cpp` (414 LOC): elementwise add/mul/scale, dot, axpy, norms.
- Reductions with pairwise or Kahan summation. Vectorised reduction changes the
  summation order, which changes results — decide the contract and document it.
- `src/linalg` triangular solves and LU/QR/Cholesky panel updates.
- FFT butterflies in `src/fft`; radix-4 and split-radix vectorise well.
- Elementwise transcendentals in `src/special`. Vectorised `exp`/`log`/`sin` is where
  most numerical code actually spends its time.

**Tier 3 — architecture**
- Runtime dispatch table selected once at startup rather than a branch per call, with
  per-ISA translation units and no `-march` on common code.
- Alignment and aliasing discipline: aligned allocators for `Matrix`, `__restrict` on
  kernel pointers.
- Threading. The kernels are single-threaded; a blocked parallel `dgemm` over an
  existing thread pool is usually a larger win than any single-core SIMD work.
- **Bit-reproducibility mode** — see §13.1.

**Tier 4**
- AVX-512 masked epilogues instead of scalar remainder loops.
- VNNI/BF16, ARM SVE/SME behind runtime detection.
- AES-NI with a bitsliced fallback, which also closes the side channel in §6.4.

### 9.6 Correctness discipline for every kernel

A scalar reference implementation kept in the tree; a differential test against it
over random inputs at every size, including the ragged edges around block boundaries;
a check that every ISA path agrees within tolerance; and a documented tolerance.
Vectorised reductions reorder floating-point arithmetic — that is legitimate, but it
must be stated rather than discovered by a user.

---

## 10. Symbolic engine

**This section gates §11 and most of §12.** Both the LaTeX work and the GUI math
rendering depend on replacing the core representation. If only one track can be
pushed on, push on this one.

### 10.1 The core representation is the blocker

```cpp
struct SymExpr {
    SymOp op = SymOp::Const;         // 15 ops
    double value = 0.0;              // <-- exact arithmetic impossible
    std::string name;
    std::unique_ptr<SymExpr> left;   // <-- strictly binary
    std::unique_ptr<SymExpr> right;
};
```

Every limitation encountered when adding features traces back to these five fields.

**No exact arithmetic.** `value` is a `double`. `1/3` is `0.333…`, so
`sym_simplify(x/3*3)` cannot return `x`. Every simplification is approximate, and
approximate simplification is not simplification — it is rounding with extra steps. A
CAS that cannot represent ⅓ is not a CAS.

**The fix is already in the tree and unused.** `include/ms/bignum/bignum.hpp` defines
`BigInt` and a `Rational` class with `BigInt` numerator and denominator, full
comparison operators, and string parsing (`"3/4"`, `"1.5"`). `src/symbolic` does not
include it. Exact rational arithmetic is available for the cost of a `#include`.

**Strictly binary.** `a+b+c` parses as `(a+b)+c`. Term collection, canonical ordering
and structural equality all fight the tree shape. `sym_collect` and the
`flatten_linear_sum` helper at `symbolic.cpp:1943` exist specifically to work around
this — flattening a binary tree back into an n-ary sum at every call site is the
symptom. `Add` and `Mul` must be n-ary with sorted children, or simplification stays
O(n²) and brittle.

**Single-argument functions only.** `parse_function_call` reads exactly one
expression between the parentheses and dispatches on six known names. That rules out
`atan2`, `besselj(n,x)`, `gcd`, `binomial`, `mod`, `log(x, base)` and `hypergeom` —
and it walls the whole of `src/special`, hundreds of multi-argument functions, off
from the symbolic layer.

**No shared subexpressions.** `unique_ptr` means a pure tree. `sym_expand` on nested
products is exponential in memory, not merely in time. Hash-consing to a DAG with
`shared_ptr<const Node>` fixes this and yields O(1) structural equality as a bonus.

**Missing heads.** No `Integer` distinct from `Const`, no `Rational`, no `Complex`,
no symbolic π or e, no ∞, no `Matrix`, no `Sum`/`Product`, no unevaluated `Integral`
or `Limit`, no relations (`=`, `<`, `≤`), no `Piecewise`, no `Undefined`. Several of
these are required just to represent the *answers* the existing functions produce —
an integral that cannot be evaluated should return an unevaluated `Integral` node,
not a sentinel.

### 10.2 The sentinel-return contract is a bug generator

```cpp
// Unsupported forms return sym_deriv(expr, var) as an explicit sentinel
SymExpr sym_integrate(const SymExpr&, const std::string&);
SymExpr sym_laplace(...);  SymExpr sym_dsolve(...);   // same convention
```

Nine functions return a *valid-looking* `SymExpr` on failure. The caller must know
the convention and check for it, and nothing in the type system helps.

This is also inconsistent with the rest of the codebase, which uses
`std::expected`-based `Result<T>` almost everywhere and contains zero `throw`
statements across 139k lines. `sym_parse` and `sym_solve_linear` already return
`std::expected`; the transforms should too.

The same problem in a different shape:

```cpp
double sym_eval(const SymExpr&, const std::map<std::string, double>&);
double sym_limit(const SymExpr&, const std::string&, double);
```

Bare `double`. An unbound variable, a division by zero and a divergent limit are all
indistinguishable from a legitimate result. Both should return `Result<double>`.

### 10.3 The printer

`sym_to_string` fully parenthesises every node and formats numbers with
`std::to_string`, so `2*x + 1` renders as `((2.000000 * x) + 1.000000)`. Unreadable —
and the LaTeX printer would inherit both faults. Precedence-aware printing is a
prerequisite for every output format in §11, not a separate task.

### 10.4 Proposed core

```cpp
enum class Head {
    Integer, Rational, Real, Complex,      // exact where possible
    Symbol, Constant,                      // pi, e, i, inf, undefined
    Add, Mul, Pow,                         // n-ary Add/Mul, binary Pow
    Function,                              // n-ary, named, table-driven
    Relation, Piecewise, Matrix,
    Integral, Derivative, Sum, Product, Limit,   // unevaluated heads
};

struct Node;
using ExprRef = std::shared_ptr<const Node>;     // immutable, hash-consed

struct Node {
    Head head;
    std::variant<BigInt, Rational, double, std::complex<double>,
                 std::string, std::monostate> atom;
    std::vector<ExprRef> args;             // n-ary
    std::size_t hash;                      // cached, for hash-consing
};
```

Immutable plus hash-consed gives structural sharing, O(1) equality, and a natural
memoisation key for `simplify`. `Result<ExprRef>` throughout.

**Automatic simplification on construction** is the design decision that matters
most: `Add` sorts its arguments into a canonical order, folds numeric terms exactly,
and collects like terms at build time. `simplify` then handles only the hard cases
rather than everything. This is how every serious CAS is built, and retrofitting it
later is considerably harder than doing it now.

### 10.5 Migration

Do not rewrite in place. The current engine has real capability behind it — Laplace,
Mellin, Hankel and Fourier transforms, Z-transform, series, limits, linear solve,
separable ODEs. That is worth keeping.

1. Build the new core beside the old one (`ms::sym2`), with the algebra, canonical
   ordering and exact rationals.
2. Write `ExprRef → SymExpr` and `SymExpr → ExprRef` bridges.
3. Port function by function, differentially testing old against new on random
   expressions — the same discipline as PBSD. Each ported function must agree with
   the old one numerically at 1,000 random points before the old one is deleted.
4. Flip the public header; delete the bridge.

Expect the rewrite to *find bugs* in the current engine. Differential testing usually
does.

### 10.6 Capability roadmap once the core lands

| Capability | Depends on |
|---|---|
| Exact rational and integer arithmetic | new core |
| `gcd`, `factor`, `expand`, `together`, `apart` over polynomials | exact arithmetic |
| Multivariate polynomial arithmetic, resultants, Gröbner bases | polynomial layer |
| Risch-lite integration (rational functions fully decided) | polynomial factoring |
| Pattern matching and user-defined rewrite rules | canonical ordering |
| Assumptions system (`x > 0`, `n ∈ ℤ`) | required for correct `sqrt(x²) = x` |
| `solve` beyond linear — polynomial, then systems | resultants |
| Symbolic linear algebra: det, inverse, eigenvalues over ℚ | matrix head + exact arithmetic |
| Symbolic → compiled numeric codegen | new core |

That last entry is worth flagging separately. An ORC JIT backend already exists in
`src/interp`. Compiling a simplified symbolic expression down to a native function is
a genuinely strong feature, and very little in the open-source CAS space does it
well.

---

## 11. LaTeX and notation interchange

Two problems with very different difficulty. Split them and treat them separately.

### 11.1 Output — `to_latex(ExprRef)`

Straightforward once precedence-aware printing exists (§10.3). Roughly a week.

- A precedence and associativity table drives minimal parenthesisation.
- `Div` → `\frac{}{}`; `Pow` with a fractional exponent → `\sqrt[n]{}`.
- Symbol name mapping: `alpha` → `\alpha`, `x_1` → `x_{1}`, `hbar` → `\hbar`.
- Unevaluated heads render properly: `\int f(x)\,dx`, `\lim_{x \to 0}`,
  `\sum_{i=1}^{n}`.
- Matrices → `pmatrix` / `bmatrix`.
- Options: display versus inline, `\left(...\right)` sizing, `\cdot` versus
  juxtaposition, `\times` versus `\cdot`, decimal separator.

Ship the following alongside it, because they are the same tree walk with a different
symbol table: **Presentation MathML**, **Content MathML**, **Unicode plain text**
(`x²·√y`), **SymPy-compatible** and **Mathematica-compatible** strings for pasting
into what people already use, and **C/C++/Python source** emission to feed the
codegen path in §10.6. Build one visitor interface and five tables — not five
printers.

### 11.2 Input — parsing LaTeX

Substantially harder, and the difficulty is intrinsic rather than a matter of effort.

**LaTeX is presentation markup, not semantics.** There is no correct general parser,
because the source does not carry the meaning:

- `f(x+1)` — function application, or `f` multiplied by `(x+1)`?
- `2x` is implicit multiplication. Is `xy` one symbol named `xy`, or `x·y`?
- `dx` in `\int f\,dx` is a delimiter, not a product with a variable named `d`.
- `\frac{d}{dx}` is an operator; `\frac{da}{db}` probably is not.
- `e` — Euler's number, or a variable? `i` — the imaginary unit, or an index?
- `'` is a derivative prime, or a transpose, or part of an identifier.
- Anything defined by `\newcommand` is unresolvable without macro expansion.

**Recommended approach.** Define a documented subset, parse it strictly, and reject
everything outside it with a precise source position. Do not guess. Publish the
subset as a grammar in `docs/LATEX_SUBSET.md` so the contract is explicit and
testable.

Layered:

1. **Tokenizer** for TeX: control sequences, groups, sub/superscripts, delimiters.
2. **Presentation tree**, faithful to the source, with no semantics attached.
3. **Semantic lift** with an explicit, configurable disambiguation policy — implicit
   multiplication on or off, single-letter-symbols-only on or off, a known-function
   name table, integral and derivative recognition. Ambiguous input returns
   `Result<ExprRef>` carrying an error that names *which* ambiguity and offers the
   candidate readings, rather than silently choosing one.
4. **Content MathML importer** as the escape hatch. It is genuinely semantic, so
   round-trip is exact, and it is what should be recommended for machine
   interchange.

**Round-trip property tests** are the right correctness discipline:
`parse(to_latex(e))` is structurally equal to `e` for randomly generated
expressions. That finds printer and parser bugs by the hundred, and the same harness
covers all the output formats in §11.1. This is the §8.5 technique applied to a new
surface.

**Do not** attempt full TeX macro expansion. TeX is Turing-complete, and that is not
the feature being built.

---

## 12. GUI

### 12.1 Current state

4,569 LOC of Qt6 Widgets; `MainWindow.cpp` alone is 3,037 lines. One test file
(`test_gui_text_transforms.cpp`), covering `TextTransforms` only. `MS_BUILD_GUI`
defaults to `OFF` and the module was excluded from coverage entirely, so nothing has
been looking at it.

Reading the action list, the balance is off: `snake_case_selection`,
`kebab_case_selection`, `camel_case_selection`, `screaming_snake_case_selection`,
`title_case`, `invert_case`, `sort_lines`, `unique_lines`, `reverse_lines`,
`join_lines`. That is a general-purpose text editor's feature set. A mathematician
does not need kebab-case conversion; they need rendered math, a variable inspector
and good plots.

`PlotWidget` is 2D `QPainter`. `PlotSurfWidget` is legacy fixed-function OpenGL.

### 12.2 Decompose before expanding

`MainWindow.cpp` at 3,037 lines cannot absorb a notebook, a math renderer and an
inspector. Split it first — `EditorPane`, `ConsolePane`, `PlotPane`,
`InspectorPane`, `Actions`, `Theme` — reducing `MainWindow` to composition and
layout. Then add.

This also makes the module testable, which is the precondition for it not being cut.

### 12.3 The rendering decision

Displaying typeset math in Qt has three routes, and this is a genuine architectural
fork:

| Option | Cost | Problem |
|---|---|---|
| `QWebEngineView` + KaTeX/MathJax | Fast to build | Pulls **Chromium** into the dependency tree. For a customer whose procurement reviews every dependency, and for your own SBOM and CVE exposure, that is a serious cost. Also roughly 150 MB and a separate process per view. |
| Shell out to a TeX installation | Trivial | Requires TeX on the target machine; slow, fragile, no interactivity. Non-starter. |
| **Native Qt math layout** | ~2–3k LOC | Real work, but bounded work. |

**Recommendation: native.** A math layout engine over `QPainter` needs a box model
(TeX's horizontal and vertical list model is well documented and worth following), a
font with the right glyph coverage — STIX Two Math or Latin Modern Math, both
OFL-licensed and embeddable — and layout rules for fractions, radicals, scripts, big
operators, matrices and stretchy delimiters.

It gives you no Chromium, a clean SBOM (§4.4), output that is selectable and
hit-testable so subexpressions can be clicked and acted on, and reuse for SVG and PDF
export of the same tree. That last point matters: the layout engine that draws to
`QPainter` draws to `QSvgGenerator` and `QPdfWriter` for free, so
publication-quality export falls out as a side effect.

Build it against the printer of §11.1, consuming `ExprRef` directly rather than a
LaTeX string, so there is no parse round-trip in the render path.

### 12.4 Features, in order

1. **Notebook / cell interface.** Input cells with rendered math output, re-executable
   in place, saved as a document. This is the single change that makes the tool feel
   like a product rather than a REPL with a text editor attached.
2. **Rendered math output** in the console and notebook (§12.3).
3. **Variable inspector** — a dock listing session variables with type, dimensions and
   a preview; matrices open in a spreadsheet-style viewer with sortable columns and a
   heatmap mode. Cheap to build against the existing `SessionState`.
4. **Equation editor** — a palette for entering math structurally, with live LaTeX in
   one pane and the rendered form in the other. This is where §11 pays off twice.
5. **Plot overhaul.** Move `PlotSurfWidget` from fixed-function OpenGL to
   `QOpenGLShaderProgram`, or to `QRhi`, which brings Metal, Vulkan and D3D for free.
   Add log axes, legends, multiple series, error bars, contour and vector fields,
   colour maps, and export to SVG/PDF/PNG at publication resolution.
6. **Interactive plot manipulation** — pan, zoom, crosshair readout, data-point
   tooltips, and range selection that feeds back into the session as a variable.
7. **Symbolic manipulation by direct interaction.** Click a subexpression in rendered
   output and get a context menu of applicable transformations — factor, expand,
   substitute, differentiate with respect to. This is a genuinely differentiating
   feature, and it falls out almost for free once §12.3 provides hit-testable layout
   boxes.
8. **Debugger** — breakpoints, stepping and watch expressions against the interpreter.
9. **Documentation pane** — the `help` output already exists in `repl_engine`; render
   it with math and make it searchable.
10. **Offscreen test harness** — `QTest::qWait`, `QOpenGLFramebufferObject` and
    image-comparison baselines. This is what removes `src/gui` from the coverage
    exclusion list (§8.7), and it should be built alongside the features rather than
    after them.

---

## 13. Other features worth building

Ordered by what a defence or agency buyer actually asks for.

### 13.1 Bit-reproducibility mode

A flag that pins the ISA path, thread count and reduction order so results are
bit-identical across machines. Nothing in the open numerical stack does this well.
For a customer who has to reproduce an analysis years later, in a context where the
result may have to be defended, this is a real differentiator rather than a
convenience. It also resolves the undocumented seeding contract of §6.7 and depends
on the dispatch work in §9.5 Tier 3.

### 13.2 Reproducibility manifest

Every run emits version, ISA path taken, thread count and seed. Near-free given
`isa_summary()` already exists, and it is the minimum viable version of §13.1.

### 13.3 Structured audit log

A record of every operation performed in a session. A compliance requirement in most
agency environments, and a CAS is unusually well placed to provide it.

### 13.4 Python bindings

pybind11 over the `Result<T>` API. This is the realistic path to anyone evaluating
the project without committing to a C++ build, and it is how most numerical software
is actually adopted.

### 13.5 Language server

The IDE is listed as a separate product, but an LSP server is a weekend of work
against the existing parser and makes the tool usable from editors people already
have.

### 13.6 Sparse direct solvers

`sparse` currently offers `to_dense` and `spmv`. There is no sparse LU or Cholesky,
which is what real PDE and FEM work needs — and `src/fem` and `src/pde` are already
in the tree waiting for it.

### 13.7 Arbitrary-precision transcendentals

`RELEASE_DECISIONS.md` lists APFloat as rewrite-scale work. It is also something a
CAS is simply expected to have.

### 13.8 Checkpoint and restart

For long-running solvers. Straightforward, and the kind of thing that gets noticed
only when it is missing.

---

## 14. Sequencing

### Now — cheap, and each is a blocker on its own

These four touch the history or the identity of the project, so do them in **one
pass** and disturb the tree only once:

| Item | Section |
|---|---|
| Rewrite commit authorship onto your identity (`git filter-repo`) | §4.5 |
| AGPL-3.0 `LICENSE`, `COPYRIGHT`, SPDX headers, CUDA §7 exception | §4.1 |
| The rename off "MathScript" | §4.2 |
| `NOTICE` / `THIRD_PARTY.md` + SBOM | §4.1.1, §4.4 |

Then, independently:

| Item | Section |
|---|---|
| OSXSAVE / XGETBV fix in `simd/isa.cpp` | §6.1 |
| CSPRNG for `crypto::random_bytes` | §6.3 |
| Export-control determination | §4.3 |
| Reconcile README / CHANGELOG / `ci.yml`; generate `docs/STATUS.md` | §5 |
| CLA in place before any external pull request | §4.1.3 |

### Then — the coverage baseline on real hardware

Run the §8.1 commands. `build-cov/coverage-ranked.txt` drives everything after this
point. Expect the number to fall; that is the exclusions coming off.

### Then — three tracks that run in parallel

| Track | First milestone | Blocks |
|---|---|---|
| **Symbolic** | New core: exact `Rational`, n-ary `Add`/`Mul`, hash-consing, `Result<T>` API, precedence-aware printer (§10.4–10.5) | §11 and most of §12 |
| **Intrinsics** | AVX2/FMA `dgemm` with packed panels and cache blocking, then NEON (§9.4–9.5) | nothing |
| **GUI** | Decompose `MainWindow.cpp`; build the offscreen harness (§12.2, §12.4.10) | nothing; removes `src/gui` from the exclusion list |

Two of the three tracks want the symbolic core finished first. **If only one thing
can be pushed on, push on that.**

### Then — hardening, in this order

1. Miller-Rabin rewrite (§6.2) and the ODR flag removal (§6.5).
2. Integration test grouping, 573 targets → 29 (§8.8) — do this early, it makes
   everything after it cheaper.
3. `src/plugin` tests (§8.2).
4. REPL golden corpus (§8.3).
5. Mutation and property testing (§8.4–8.5).
6. Reference BLAS/LAPACK differential tests (§8.6).
7. CI ratchet and generated status page (§8.9).
8. Ship-or-cut decisions on §7.

### Then — features

LaTeX and MathML output (§11.1) → native math layout engine (§12.3) → notebook
interface (§12.4.1) → LaTeX input (§11.2) → interactive symbolic manipulation
(§12.4.7) → the §13 list.

### The gate on `v1.0.0`

Do not cut the tag until §4.1 (licence, including the CUDA §7 exception), §4.3
(export control), §4.5 (authorship), §6.1 (SIGILL in `dgemm`) and §6.3 (CSPRNG) are
closed. Everything else in this document can ship in a point release. Those five
cannot — three are legal and two are defects a customer would hit on first contact.

---

## 15. Appendix

### 15.1 Artifacts produced

| Path | Purpose |
|---|---|
| `scripts/extract_manifest.py` | Parses all 478 matrix-call handlers into `matrix_calls_manifest.json` |
| `scripts/gen_matrix_call_tests.py` | Emits `tests/unit/matrix_calls/` from the manifest |
| `scripts/hoist_prologue.py` | One-shot transform removing the duplicated lambda prologue |
| `scripts/coverage_report.sh` | Rewritten: branch and function coverage, declared exclusions, ranked report, three gates |
| `scripts/coverage_exclusions.txt` | Exclusions as declared data, each with a written reason |
| `tests/unit/matrix_calls/` | 29 generated TUs, 1,348 cases |
| `mathscript-coverage-phase0-2.patch` | All of the above: 516 files, +8,465 / −11,902 |

### 15.2 Regenerating the matrix-call suite

```bash
python3 scripts/extract_manifest.py .
python3 scripts/gen_matrix_call_tests.py matrix_calls_manifest.json tests/unit/matrix_calls
```

CI should run both and fail on a dirty tree, so a new handler cannot land without its
dispatch tests.

### 15.3 Coverage baseline

```bash
cmake -S . -B build-cov -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DMS_BUILD_TESTS=ON -DMS_BUILD_INTEGRATION=ON \
  -DMS_ENABLE_COVERAGE=ON -DMS_ENABLE_CUDA=OFF
cmake --build build-cov -j
ctest --test-dir build-cov --output-on-failure
bash scripts/coverage_report.sh build-cov

# outputs:
#   build-cov/coverage.info            lcov data, branch coverage included
#   build-cov/coverage-summary.txt     line / function / branch percentages
#   build-cov/coverage-ranked.txt      per-file ranking by uncovered lines
```

Gates are set by environment variable: `MS_COVERAGE_MIN`,
`MS_COVERAGE_BRANCH_MIN`, `MS_COVERAGE_FUNC_MIN`.

### 15.4 Defects by severity

| # | Defect | Section | Severity |
|---|---|---|---|
| 1 | CPUID without OSXSAVE/XGETBV — SIGILL in `dgemm` | §6.1 | Crash |
| 2 | No LICENSE file — AGPL-3.0 chosen, not yet applied | §4.1 | Blocks sale |
| 3 | `random_bytes` is not a CSPRNG | §6.3 | Key material |
| 4 | No export-control position | §4.3 | Blocks distribution |
| 5 | Miller-Rabin: overflow, UB, `% 0`, fixed seed | §6.2 | Wrong answers |
| 6 | AES table-driven — cache timing | §6.4 | Side channel |
| 7 | `--allow-multiple-definition` masks ODR | §6.5 | Latent |
| 8 | `src/plugin` — 1,189 LOC, zero tests | §8.2 | Unverified |
| 9 | Seven false or stale published claims | §5 | Credibility |
| 10 | `izaac_vrf_keygen` missing arity guard | §6.6 | Minor |
| 11 | 94% of commits attributed to `bot@mathscript.local`, not you | §4.5 | Provenance / IP |
| 12 | AGPL code links proprietary cuBLAS/cuSolver with no §7 exception | §4.1.2 | Licence violation |

### 15.5 Note on measurement

The audit was performed by static inspection. No build was run: the analysis
environment was a single vCPU with 3 GB of RAM and no `cmake`, `ninja` or `lcov`, and
a coverage build of 139k source plus 398k test LOC will not complete there. Every
figure quoted in this document comes from parsing the tree directly, and every claim
about runtime behaviour — §6.1 in particular — should be confirmed on real hardware
before it is acted on. The §8.1 baseline is the first step in doing so.
