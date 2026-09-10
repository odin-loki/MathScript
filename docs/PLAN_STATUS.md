# Engineering plan — execution status

Status of every item in [`ENGINEERING_PLAN.md`](ENGINEERING_PLAN.md), which is kept
verbatim as the dated audit it was.

Three conventions here, because the distinction is the whole point of the document:

- **Done** — implemented, and a test or a check in CI fails if it regresses.
- **Open** — not done. No partial credit, no "mostly".
- **Held** — deliberately not done, with the reason stated. Two items are one-way
  doors that are the repository owner's to open, not a contributor's.

---

## Where the plan was wrong

Worth stating first, because these were found by measuring rather than by reading,
and each one changed what the work had to be.

### §3 "Work already completed" had not been applied to this tree

The plan opens by describing coverage tooling, a prologue hoist and a generated
test suite as finished. None of it was present: no `extract_manifest.py`, no
`gen_matrix_call_tests.py`, no `hoist_prologue.py`, no `coverage_exclusions.txt`,
no `tests/unit/matrix_calls/`. The plan was written against a tree where that work
existed; the published repository is not that tree.

The consequence was not cosmetic. Without `coverage_exclusions.txt`, 37,738 lines —
25% of `src/` — sat outside the coverage denominator. **The 92.0% figure that had
been reported repeatedly was measured over 75% of the tree.** That is the single
most misleading number the project has published, and it was produced honestly by a
tool asked the wrong question.

### §6.2's defects 2 and 3 are unreachable

The plan lists three defects in the Miller-Rabin implementation. The first is real
and is fixed. The second (`std::abs` overflow on `INT64_MIN`) and third (a modulo by
zero) cannot occur: `if (nll <= 2) nll = 3;` executes before either, which
constrains the value past the point where those paths exist.

This is not an argument from reading the code. Both ESBMC and CBMC were run against
harnesses that assert each condition, and both report the properties unreachable.
The harnesses are in [`../verification/`](../verification) and run in CI.

### §6.5's diagnosis was wrong

The plan attributes `--allow-multiple-definition` to rival implementations of the
same symbol. It was not that. Every archive reached the link line twice, and all 20
duplicate symbols were byte-identical. The fix was to build the bundle from
`$<TARGET_OBJECTS:>` rather than to reconcile competing definitions — a different
change with a different risk profile from the one the plan describes.

### Counts have moved since the audit

The plan counts 478 matrix-call handlers and 29 integration domains. The tree now
has 485 and 31. Every generated figure in this document comes from the tree, not
from the plan.

---

## §3 — Work the plan recorded as complete

| Item | Status | Where |
|---|---|---|
| 3.1 Coverage tooling made honest | Done | `scripts/coverage_report.sh`, `scripts/coverage_exclusions.txt` |
| 3.2 Duplicated handler prologue hoisted | Done | `scripts/hoist_prologue.py`, `MatrixCallCtx` in `src/interp/matrix_call.hpp` |
| 3.3 Generated dispatch tests | Done | `scripts/extract_manifest.py`, `scripts/gen_matrix_call_tests.py` |

Only paths that *cannot* execute on a CI runner are excluded from coverage, and each
run prints how many lines the exclusions hid. A number that hides its own denominator
is how the 92.0% happened.

### What the generated suite found on its first run

1,373 of 1,377 passed immediately. The four failures were all real, and none was a
bug.

`mat_row`, `mat_col`, `mat_reshape` and `mat_submatrix` guard their arity with an
early return that names the signature they expected —
`DomainError{"mat_row", "expected mat_row(A, i)"}` — where the other 481 handlers
leave the pre-initialised `DomainError{"assign", "unsupported matrix call"}` in
place. The four give the *better* diagnostic. What failed was the generator's
assumption that one message covered every handler.

The fix was to stop assuming. `extract_manifest.py` now reads each handler's
rejection error out of its own guard, and the generated test asserts that specific
contract. A test that asserted the convention would have passed on all 485 and
noticed nothing if a handler changed its rejection message; this one notices.

---

## §4 — Legal and commercial

| Item | Status | Where |
|---|---|---|
| 4.1 AGPL-3.0, `COPYRIGHT`, SPDX headers, CUDA §7 exception | Done | `LICENSE`, `COPYRIGHT`, `LICENSE.exceptions`, `scripts/add_spdx.py` |
| 4.2 Rename off "MathScript" | **Declined** | see below |
| 4.3 Export-control position | Done | `docs/EXPORT_CONTROL.md` |
| 4.4 `SECURITY.md` and SBOM | Done | `SECURITY.md`, `sbom.cdx.json`, `scripts/gen_sbom.py` |
| 4.5 Commit authorship | **Declined** | see below |

SPDX headers are on all 1,735 source files; the 18 translation units that link the
NVIDIA libraries additionally name `LICENSE.exceptions`, because a file that
participates in that link should say so rather than leaving the grant discoverable
only from the repository root. `vendor/` is untouched: it is not ours to mark.
`scripts/add_spdx.py --check` fails CI if a file lands without one.

### The two decisions

Both were one-way doors, both belonged to the repository owner rather than to a
contributor, and both have now been decided. Neither was ever blocked on work.

**§4.2, the rename — declined. The project keeps the name MathScript.**

The plan's finding stands on its facts: National Instruments ships a MathScript
alongside LabVIEW, so the name is not distinctive in this field. Keeping it is an
accepted risk rather than a refutation of the finding, and the risk is not constant —
it is small for a personal repository and grows if the project is ever sold, packaged
commercially under that name, or put forward as a mark. Recorded here so a later
reader does not mistake the absence of a rename for the absence of the question.

**§4.5, the authorship rewrite — declined.**

`git filter-repo --mailmap` and a force push would have rewritten all 1,440 commit
SHAs, breaking every existing clone and every link to a commit. The owner's judgement
is that the benefit does not justify that, which is a reasonable reading: the
repository is not disputed and the history is not load-bearing evidence of anything.

One thing worth separating out, because it is not what was declined: a checked-in
`.mailmap` maps author identities for `git log`, `git shortlog` and GitHub's own
display **without rewriting a single commit**. It is additive and reversible. It
addresses the presentation half of §4.5's concern and none of the provenance half. It
has not been added — this is a note, not a plan.

## §5 — False and stale claims

**Done.** The structural fix was to stop maintaining the numbers by hand.
`scripts/gen_status.py` reads them from build artefacts — `ctest -N` for the suite
count, `coverage-summary.txt` for the percentages, the benchmark baseline for its
entry count, `ci.yml` for the thresholds actually enforced — and writes
[`STATUS.md`](STATUS.md). Anything it cannot read is written as "not measured"
rather than carried forward, because a number that is absent is honest and a stale
one is not.

`STATUS.md` shows the measured coverage and the CI gate in separate columns on
purpose. The README once claimed CI enforced 90% while `ci.yml` set 80%, and nothing
in the repository reconciled them.

## §6 — Correctness defects

| Item | Status | Note |
|---|---|---|
| 6.1 CPUID without OSXSAVE/XGETBV (SIGILL) | Done | `src/simd/isa.cpp` |
| 6.2 Miller-Rabin for large inputs | Done | defect 1 fixed; 2 and 3 proved unreachable |
| 6.3 `crypto::random_bytes` not a CSPRNG | Done | OS CSPRNG on every platform |
| 6.4 AES is table-driven (cache timing) | Done | masked full-table scan; 26x slower, `docs/PERFORMANCE.md` |
| 6.5 `--allow-multiple-definition` | Done | bundle built from `$<TARGET_OBJECTS:>` |
| 6.6 Sole handler without an arity guard | Done | `izaac_vrf_keygen.cpp` |
| 6.7 Undocumented fixed RNG seeds | Done | `docs/API.md`, "Randomness and the seeding contract" |
| 6.8 Sentinel returns in the symbolic API | Partly | leak fixed and detection made recursive; the `Result<T>` API is still §10 |

**6.1** is the item the plan ranks highest, and it is worth being precise about what
it was. CPUID reports what the silicon implements; it does not report whether the OS
has enabled the extended register state. A hypervisor, a sandbox, or a kernel booted
with `noxsave` leaves AVX-512 masked off on a CPU that advertises it, and dispatching
on the CPUID bit alone then selects a kernel whose first instruction faults. The
failure is SIGILL inside `dgemm`, decided by the deployment environment rather than by
anything the caller passed in. The fix reads `CPUID.1:ECX.OSXSAVE` and then
`XGETBV(0)`, requiring `XCR0 & 0x6` for YMM state and `XCR0 & 0xE6` for ZMM. FMA is
gated with AVX, not separately: it operates on YMM and needs the same agreement even
though its CPUID bit sits apart.

A related hazard was found while doing §9 and fixed: `src/simd/isa.cpp` was being
compiled with `-mavx2 -mfma`, which permits the compiler to place an AVX instruction
inside the routine whose entire job is to determine whether AVX instructions will
fault. Today's object contains none, so the defect was latent rather than active —
but it depended on a compiler's choice. `isa.cpp` now builds at baseline ISA and only
`vector_ops.cpp` gets the wide flags.

**6.6** looks like a style nit and is not. `izaac_vrf_keygen.cpp` guarded its arity
with `assign.args.empty()` where the other 484 handlers use an explicit
`assign.args.size() == N`. The two are identical to the compiler. They are not
identical to `extract_manifest.py`, which reads every guard as a predicate over the
argument count — so that one handler parsed as "depends on something other than the
count", dropped out of the manifest, and would have dropped out of the generated
tests, with nothing failing. The parser found it independently, which is the
strongest evidence available that the parser is reading the guards correctly.

## §7 — Stubs and half-implementations

**Open.** Ship-or-cut decisions, tracked in
[`RELEASE_DECISIONS.md`](RELEASE_DECISIONS.md).

## §8 — Coverage and testing programme

| Item | Status | Note |
|---|---|---|
| 8.1 Baseline on real hardware | Done | 91.2% lines, 98.3% functions, 57.3% raw branches, 71.8% over decision lines |
| 8.2 `src/plugin` tests | Partial | `unsafe_registry` is tested (273 lines of test against 282 of audit bookkeeping that had never run); the Clang AST rules themselves are covered only by the plugin smoke job |
| 8.3 REPL golden corpus | Done | `tests/repl_corpus/*.ms` with committed stdout and stderr, run through the real `mathscriptc` |
| 8.4 Mutation testing | Open | |
| 8.5 Property-based testing | Done | seeded invariants over the linalg/FFT core, and the §11 printer round-trips |
| 8.6 Differential tests vs reference BLAS/LAPACK | Partial | the dgemm kernels have them; the wider LAPACK surface does not |
| 8.7 Remaining gaps | Open | |
| 8.8 Group 573 integration targets | Done | 573 executables → 31 |
| 8.9 Lock it in | Done | four source-only gates plus the coverage ratchet, all gating in CI |

### What the corpus found on its first run

§8.3's first eight transcripts turned up seven things on the run that generated them,
which is the argument for the shape of the test rather than for the tests in it. None
were regressions; all were already true and none had a test.

- **`^` was not an operator in the REPL's scalar evaluator** -- `2^3` did not parse,
  and `pow(2, 3)` was the only spelling, while `^` *was* an operator in every symbolic
  command and in the matrix literal syntax. **Fixed**, in all three evaluators at once
  (the interpreter's two and the ORC JIT's own copy), and the two backends were checked
  against each other on the same seventeen expressions rather than assumed to agree.

  The first implementation of it was wrong in a way worth recording, because it is the
  same shape as the audit findings: both backends agreed, and they agreed on **4** for
  `-2^2`. Unary minus binds looser than exponentiation -- in mathematics and in every
  language that has a power operator -- so the answer is **-4**. Both readings evaluate
  and nothing but an assertion distinguishes them. It also had to stay compatible with
  an earlier fix in the same function: the additive level must be searched *before* a
  leading sign is taken as unary, or `-4 + 1` becomes `-(4 + 1)`, while the power level
  must be searched *after* it.
- **`transpose(A)` has no no-target form** although `matmul(A, A)` does. The CHANGELOG
  says the registry gives every matrix-returning callee a bare form; it does not reach
  this one.
- **The no-target matrix form printed a result it did not store.** Recorded first as a
  labelling problem -- it said `C` where the CHANGELOG says `_` -- which understated
  it: the name was not merely wrong, nothing was stored under it or any other name, so
  the value could be read and not used. **Fixed.** `matmul`, `tensorops_matmul`,
  `tensorops_einsum`, `signal_conv2`, `ml_mat_mul` and `dist_matmul` had hand-written
  branches predating the registry fallback and shadowing it; they now print under `_`
  and store there, so `B = matmul(_, A)` works on the next line as it already did after
  `rand(2, 2)`.

  Two things nearby are **not** defects, and are recorded so they are not "fixed"
  later by someone reading the first sentence. A bare `lu(A)` printing `L =`, `U =`,
  `P =` is right -- those are the factors' names, not invented ones -- and the
  assignment form printing the name the user chose is right for the same reason. What
  is arguably still open is that the multi-output bare forms store nothing either;
  that needs a decision about what `_` should mean when a command yields three
  matrices, which is why it was not answered here.
- **`stats_one_way_anova` and `rle_encode_vec` are unknown in the bare form** and
  reachable only through an assignment.
- **`1 / 0` reported "could not parse"** rather than anything about division. **Fixed**:
  the bare-expression fallback discarded the evaluator's error and replaced it with a
  parse failure, so a real diagnosis was thrown away and the reader was sent looking for
  a typo that was not there. A line carrying a top-level operator or a call now reports
  what actually went wrong. A bare word still reports the parse error, deliberately:
  `load` is an incomplete command and `no_such_variable` is a missing name, they are the
  same line shape, and calling either an unknown *scalar* asserts a category this code
  cannot know.
- **`A + B` said "unknown scalar: A"** once errors were propagated, with `A` sitting in
  `vars` as a matrix. **Fixed** in passing: the resolver now distinguishes a name that
  does not exist from one that exists as a matrix, and says that matrices have no
  operator arithmetic.
- **`not_a_function(1)` reports `unknown matrix: 1`** -- the diagnostic names the
  argument rather than the function it could not find.
- **`sym_simplify("x + x")` returns `(x + x)`.** Like terms are collected during
  `sym_expand` and not during `sym_simplify`.

The transcripts record all seven as they are, with the two most misleading marked in
the script files, so each becomes a readable diff the day it is fixed rather than an
assertion someone has to reconstruct.

One further note the corpus settled: the one-dimensional root finders bind **`x0`**,
not `x`. `c88f0ef`'s message says `x` -- the guard it describes is correct and the
name in the prose is not. `docs/API.md` now states the binding on the row a user
reads.

### 8.1, the number the plan asked for

Measured on an instrumented Debug build with `MS_BUILD_INTEGRATION=ON`, 336/336 CTest
suites passing, over the denominator declared in `coverage_exclusions.txt`:

| | Measured |
|---|---|
| Lines | **91.2%** (76,687 of 84,077) |
| Functions | **98.3%** (5,492 of 5,588) |
| Branches (raw gcov) | **57.3%** (82,082 of 143,192) |
| Branches (decision lines only) | **71.8%** |

Two branch rows, and **neither replaces the other**. The raw gcov figure counts every
edge gcov emits, which on this codebase includes edges inside library code inlined
into our lines -- a `std::vector` growth path, an allocation-failure branch -- that no
test of ours can reach and that we would not write a test for if we could. The
decision-line figure, which `scripts/decision_coverage.py` computes, counts only the
branches on lines that carry a decision we wrote. The gap between 57.3% and 71.8% is
that inlined machinery, and quoting either number alone overstates something: the raw
one understates what our own decisions cover, the decision one hides how much
uncovered generated code the binary contains. `docs/STATUS.md` is regenerated each
coverage run and carries the current pair.

The exclusions hide 7,698 of 91,775 instrumented lines — 8% — and the run prints that
figure, because a list nobody sees is a list that grows. What remains excluded is
`/usr/*`, `vendor/`, GoogleTest, the tests themselves, `src/cuda` and `src/gui`; in
this configuration three of those patterns matched nothing at all, which the run
reports rather than passing over in silence.

The plan said to expect the line figure to fall as the exclusions came off. It barely
moved — 92.0% to 91.2% — despite a substantially larger denominator. The likely
reason is that the 1,377 generated dispatch tests of §3.3 landed on the same branch
and exercise precisely the `matrix_calls` code the old exclusions removed. That is an
explanation rather than a measurement: the two changes were not isolated from each
other. What is measured is that `matrix_calls` handlers dominate the lowest-coverage
entries in `coverage-ranked.txt`, so the newly-included code is still the weakest in
the tree.

**The branch figure is the one that matters, and it had never been measured.**
`coverage_report.sh` asked lcov for branch data under `lcov_branch_coverage`; lcov 2.x
renamed that to `branch_coverage`, still accepts the old name, still warns that it is
deprecated, and then collects nothing. Every run reported `branches...: no data found`
while the script's own header claimed branch coverage was being measured. Nothing was
silently green — a requested `MS_COVERAGE_BRANCH_MIN` fails as "not measured" rather
than passing vacuously — but the measurement never happened and the branch gate was
unusable by anyone who set it.

That header comment was right about why it mattered: this tree's largest files are
dispatch chains, and a dispatch chain reaches high line coverage with one branch of
each test taken. 91.2% line coverage against 57.3% branch coverage is that prediction
confirmed. **The line figure means considerably less on this codebase than it sounds
like.**

### 8.8, and the collision it exposed

573 integration tests were 573 executables, each linking the whole library. Linking
is what dominates a test build, and on an instrumented build it dominated it badly
enough that measuring coverage was a nightly event rather than something anyone did
before pushing. They are now 31 per-domain executables — the same tests, an order of
magnitude less link time. The build graph went from 2,431 steps to 1,460.

The prerequisite was not obvious. `TEST(Suite, Name)` expands to a class
`Suite_Name_Test` whose member functions are implicitly inline, so two files in one
executable declaring the same pair link with no diagnostic and one body silently
replaces the other. **71 pairs collided, and 29 of them had different bodies.**
Grouping without fixing that would have stopped running real tests while the test
count — the thing anyone would have checked — stayed exactly where it was. The names
were disambiguated first, and `scripts/check_test_names.py` runs in CI so it cannot
recur.

### 8.9, what is and is not gated

In CI, as a job that needs no build and fails in under a minute:

- `add_spdx.py --check` — every source file carries a licence identifier
- `gen_sbom.py --check` — the SBOM matches the tree
- `extract_manifest.py --check --strict` — every handler parses, no anomalies
- `gen_matrix_call_tests.py --check` — the generated suites are current
- `check_test_names.py` — no two tests in one executable share a name

`--strict` on the manifest is deliberate: a handler the parser cannot read is a
handler whose dispatch nothing tests, and that has to be a failure rather than a line
of console output nobody reads.

`STATUS.md` is regenerated in the coverage job and published as an artefact, with the
drift printed — but it is **not** a gate. The coverage figure moves between runners
and between runs, so failing on inequality would be failing on noise, and a gate that
cries wolf is how the stale numbers got published in the first place.

The **coverage ratchet** of §8.9 is built and gating: `scripts/coverage_ratchet.py`
runs in the coverage job and fails on a drop below `tests/coverage_baseline.json`.

It is a ratchet rather than a threshold because a threshold is a number to argue
about, where a ratchet only asks that the tree not go backwards. It allows a
tolerance of 0.5 points, and that number is set from measurement rather than taste:
the same tree measured locally and on the runner agreed on all three metrics to
within lcov's own 0.1 resolution, so 0.5 is about ten times any spread observed and
far below what removing a test suite would cost. The evidence is recorded in the
baseline file next to the number, because a tolerance with no stated basis is a
tolerance that grows.

`--update` raises the baseline and refuses to lower it; lowering needs `--force` and
therefore shows up in the diff. That is the difference between a run that
legitimately drops coverage and one that quietly moves the goalposts.

## §9 — Performance and intrinsics

| Item | Status |
|---|---|
| 9.1 §6.1 fixed first, `MS_FORCE_ISA` override | Done |
| 9.4 Fix `load_b_panel` and add cache blocking | Done |
| 9.5 Tier 1 — AVX2/FMA `dgemm` | Done |
| 9.5 Tier 1 — NEON `dgemm` | Open |
| 9.5 Tier 1 — `sgemm` | Open |
| 9.5 Tier 2 — xsimd across `vector_ops`, `linalg`, `fft`, `special` | Open |
| 9.5 Tier 3 — dispatch table, alignment, threading | Open |
| 9.5 Tier 4 — masked epilogues, VNNI/BF16, SVE, AES-NI | Open |
| 9.6 Correctness discipline | Done for the kernels that exist |
| CI matrix under Intel SDE | Open |

### What the AVX-512 kernel was doing

It read B through a gather written as a set:

```cpp
_mm512_set_pd(B[(j0+7)*ldb + p], B[(j0+6)*ldb + p], ...)
```

Eight strided scalar loads per vector, executed on every iteration of the innermost
loop. Storing C had the mirror-image problem: the accumulator held eight *columns*
for one row, so writing it back meant spilling to a stack buffer and scattering eight
scalars. And there was a 4×8 micro-kernel with nothing above it, which is fine while
the operands fit in cache and stops scaling the moment they do not.

Both are structural rather than local, and both are fixed by the same change:
orienting the vectors along rows — contiguous in a column-major matrix, for A and C
alike — and putting a Goto/BLIS loop nest underneath, in
`src/runtime/cpu/gemm_blocking.hpp`. B still has to be gathered, because it is
column-major and the kernel needs a row of it, but that now happens once per panel
during packing rather than k times per tile.

### The gap that mattered more

There was no AVX2/FMA `dgemm` at all. Every machine without AVX-512 — Zen 1 through
3, every Intel client part since Alder Lake, and the CI configuration itself, which
passes `-DMS_ENABLE_AVX512=OFF` — fell all the way to a rank-1 update loop for matrix
multiply. `src/runtime/cpu/avx2_dgemm.cpp` is an 8×6 micro-kernel over the same
blocking: 8 rows is two 4-wide vectors, and 6 columns keeps 12 accumulators, 2 A
vectors and 1 broadcast inside the 16 ymm registers AVX2 provides.

### Measured

n = 512 on one Xeon, single-threaded, same inputs through every path:

| Path | GFLOP/s |
|---|---|
| no vector kernel compiled in (the old CI configuration) | 5.03 |
| AVX2/FMA, packed and blocked | 36.06 |
| AVX-512, gather and no blocking (before) | 12.16 |
| AVX-512, packed and blocked (after) | 66.62 |

The plan predicted the packing fix alone would be "typically 2-4x"; with the
blocking above it, the AVX-512 kernel is 5.5× its previous self, and the two agree
to 6.0e-14 on the same inputs. One machine, one size, one thread — enough to
establish direction and magnitude, not enough to quote as a general figure.

### The tolerance contract

The kernels reassociate the sum over k and use fused multiply-add. They do not
reproduce a naive triple loop bit for bit and should not be expected to; both
behaviours are legitimate and both change the result. `tests/unit/linalg/
test_dgemm_kernels.cpp` compares each kernel against a reference accumulated in long
double, at every size that straddles a boundary in the decomposition, with a
tolerance of 16 × k × ε × max|A| × max|B|. That is loose enough to cover any
summation order the compiler chooses and far tighter than an indexing mistake could
hide behind: dropping one term of a k-term sum of unit-scale values is an error of
order 1, and the tolerance at k = 256 is about 1e-12.

The block sizes are defaults, not measurements. They are a `struct` rather than
constants baked into the loop because the right values are a property of the machine,
and claiming otherwise would be inventing a number.

## §10 — Symbolic engine

**Started.** The core the section specifies now exists as `ms::sym2`, built beside
`ms::symbolic` as §10.5 directs rather than replacing it: exact `BigInt`/`Rational`
atoms, n-ary sorted `Add`/`Mul` that collect like terms at construction, an interned
DAG so equality is a pointer comparison, `Derivative`/`Integral`/`Limit` heads and an
`undefined` value in place of the nine sentinel returns of §6.8, a precedence-aware
printer, and `Result<T>` on `evaluate`. `x/3*3` is `x`; `2*x + 1` prints as `2*x + 1`
rather than `((2.000000 * x) + 1.000000)`.

The bridge in `ms/sym2/bridge.hpp` converts both ways so functions can be ported one at
a time under the differential discipline §10.5 asks for, and
`tests/unit/sym2/test_sym2_differential.cpp` runs it: 4,000 random expressions against
the old engine at three points each, plus a round trip, a print-and-reparse check, and
a fixed-point check on the canonical form.

**What is left:** the transforms, series, limits, linear solve and ODE solvers are
still the old engine's, and nothing in the REPL calls `sym2` yet. §10.6's capability
roadmap — polynomial `gcd`/`factor`/`together`/`apart`, assumptions, `solve` beyond
linear — all sits on top of what now exists.

### What was fixed without it

An audit drove the REPL over the standard tables of all twelve symbolic families:
10,153 commands, 80 claimed gaps, 35 double-confirmed by an independent empirical and
mathematical check. Acting on it turned up nine results that were not declines but
**wrong answers with no error attached**, which are worse than the missing table rows
that prompted the audit:

| Defect | Symptom |
|---|---|
| Unary minus bound tighter than `^` | `-t^2` evaluated to `+9` at `t = 3`; `-2^2` to `4`; `-t^0.5` to NaN |
| `pi` and `e` parsed as free variables | `sym_eval("pi")` returned `0.000000`, as does any unbound variable |
| `sym_ztransform` folded a coefficient into the pole | `Z{3*2^n}` returned `z/(z-6)` instead of `3z/(z-2)` |
| `sym_solve_linear` dropped terms it could not read | `x + sin(y) - 1` solved to `x = 1`; `x^2 + x - 1` was answered as if linear |
| The unsupported sentinel leaked through linearity | one unsupported term in a sum returned part transform, part `d/dt(...)`, reported as success |
| `sym_hankel` on `r^n exp(-a r)`, `n >= 1` | `H0[r^2 e^{-2r}]` at `k = 1` returned exactly twice the true value |
| `sym_limit` on a one-sided domain | returned its `0.0` initialiser: `sqrt(x) + 5` at 0 gave `0.000000` |
| `sym_limit` under cancellation | drove `h` to 1e-15 and returned the resulting 0 as converged |
| `sym_to_string` on small constants | printed `1e-9` as `0.000000`, losing the term entirely |

and one that was neither a decline nor a wrong answer: `sym_expand("((x+1)^8)^8")` did
not terminate.

The table gaps behind the audit are closed in `sym_integrate`, `sym_laplace`,
`sym_ilaplace`, `sym_fourier`/`sym_ifourier`, `sym_mellin`/`sym_imellin`,
`sym_hankel`/`sym_ihankel`, `sym_ztransform`/`sym_iztransform` and `sym_dsolve`, mostly
by stating a general rule — the shifting theorems, frequency differentiation, a
first-degree numerator over three denominator families — rather than adding rows.

`tests/unit/symbolic/test_symbolic_tables.cpp` checks entries against the definitions
they come from rather than against the implementation: antiderivatives are
differentiated and compared with the integrand, forward transforms against their
defining integral evaluated numerically, inverse transforms by forward-transforming the
result, ODE solutions by substitution into the equation, and Mellin entries on a mesh
substituted to remove the singularity at each end exactly. That is what caught the
Hankel factor of two, which no amount of asserting the expected closed form would have.

Both of the items left open here are now closed. `sym_expand` collects like terms on a
canonical polynomial form, so `((x+1)^8)^8` is the degree-64 binomial in 8 ms rather
than a hang, and the REPL's scalar output round-trips.

Chasing that second one down through the REPL turned up five more wrong answers, none
of them symbolic:

| Defect | Symptom |
|---|---|
| A leading unary sign applied to the whole expression | `-4 + 1` evaluated to **-5**, `-4 - 1` to **-3**; with `x = 4`, `-x + y` to **-6** |
| The top-level operator scan did not know an exponent sign | `1e-09 * 2` was split at the minus and reported "could not parse" |
| Every scalar printed with `printf("%f")` | `x = 0.000000001` echoed as **0.000000**; above 1e16 the same format grew a spurious `.000000` tail |
| `save_session` wrote six significant digits | `x = 1.23456789` saved as `1.23457` and reloaded 2.1e-06 wrong, silently, for every scalar, matrix entry and plot sample |
| `combo::binomial` overflowed its intermediate product | `C(67,33)` returned **8829174638479413** for 14226520737620288370 — a value well inside `uint64_t` |

The last of those came out of asking a narrower question: the REPL was printing
`combo`'s `UINT64_MAX` overflow sentinel as an answer, so `combo_factorial(25)` said
**18446744073709551615**. Guarding the twenty-one call sites was the fix for that;
checking the counts against exact arithmetic while writing the test is what showed the
counting functions themselves were wrapping. `permutations`, `multinomial`,
`combinations_with_rep` and the four rank/unrank functions had the same problem in
different forms.

The pattern across all fourteen: the code was wrong in a way that looked right. The
tests that catch this class compare against an independent definition — quadrature for
a transform, exact integer arithmetic for a count, a bit pattern rather than a printed
form for a round trip — because a test written from the implementation's own output
agrees with the bug.

## The second audit

A read-only sweep of the whole tree, run along eight dimensions in parallel, with every
finding then handed to an independent verifier told to refute it: **40 claims, 36
confirmed, 4 refuted.** All 36 are fixed.

The question it asked was narrower than the first audit's and turned out to be more
productive: not "what is missing" but "what produces a value a user would read as an
answer and that is not one". The categories it found:

| Category | Examples |
|---|---|
| A different function entirely | `jordan_totient` computed the Euler-totient shape; header, implementation and test all agreed with each other |
| Overflow with no report | `binomial`'s intermediate product, `catalan_num` via the central binomial, `crt`'s modulus, `sum_divisors`, `convergents`, `lucas_sequence`, `BigInt::to_ll` |
| A marker printed as a value | `combo` and `numthy`'s `UINT64_MAX`, `primitive_root`'s -1, `quantum_fidelity`'s 0.0, `graph_diameter` on a disconnected graph |
| A rule that is not an identity | `sym_mellin`'s exponential rows dropped Gamma(s); `sym_limit` averaged a two-sided divergence to zero |
| Success reported for a run that failed | the adaptive ODE step budget; `converged = 1` from five optimisers |
| A value silently changed on the way through | `matrix_to_bytes` rescaling by 255; the compress round trips; `BigInt` turning a bad literal into 0 |
| State lost or shadowed | `load_session` clearing the session before failing; `save_session` writing files it cannot read; `A(1,2) = 5`; scalars and matrices shadowing each other |
| A display that erased its value | 60-odd sites at six significant digits; `saveplot` writing a rounded preview |

Two of them are worth separating out, because they say something about how the rest were
found rather than only what they were.

**`jordan_totient` was wrong in three places at once.** The header stated
`J_k(n) = n^k prod (1 - 1/p)`, the implementation computed that, and the test asserted
`J_2(6) = 12` with a comment deriving it from the same formula. Nothing in the tree
disagreed with anything else in the tree. The test that catches it counts the k-tuples
J_k is defined as, which is not a formula and so cannot carry the same error.

**The ORC JIT repeated the interpreter's unary-minus defect exactly.** The same
`if (expr.front() == '-')` before the binary-operator scan, in the backend whose job is
to agree with the interpreter. Fixing one and not the other would have left the two
disagreeing about `-2 + 1` -- which is a worse state than both being wrong.

## §11 — LaTeX and notation interchange

| Item | Status |
|---|---|
| 11.1 Output — `to_latex(ExprRef)` and the sibling formats | Done |
| 11.2 Input — parsing LaTeX | Open |

**§11.1 is done, as one walk and five tables rather than as ten printers.** The plan
insisted on that shape and the reason held up: almost everything a printer does is
structural, and structure is the same in every notation. `src/sym2/notation.cpp` makes
every structural decision once -- which factors are a denominator, which sum terms are
subtractions, which powers are roots, display order, where a grouping is needed, how a
symbol name splits -- and a notation is a `Syntax` table that only spells what has
already been decided.

Ten notations come out of five tables: LaTeX, Presentation MathML, Content MathML,
Unicode, ASCII, SymPy, Mathematica, and C / C++ / Python source. In the REPL:
`sym_latex("expr")` and `sym_export("expr", "notation")`.

What made this worth doing as one walk rather than ten printers is visible in what each
table got wrong on its own terms and had to be told: `1e+20` is not a LaTeX numeral,
`1/3` in a Python session is a float, `1/3` in Wolfram Language is not, `1/3` in C is
zero, `Sin(x)` in Wolfram Language is a product rather than a call, `√` has no vinculum
in text so it does not group its argument, and a LaTeX value written as a product may
not enter a superscript without a grouping or the document does not compile. Every one
of those is a case where the *obvious* string parses to a different expression than the
one printed -- and none of them are visible by reading the output.

Also settled, and recorded because it is the same defect class as the audits: a
derivative, integral or limit has no source form. The C, C++ and Python tables emit an
identifier that does not exist, so the code fails to compile and names the problem,
rather than emitting a plausible call.

**§11.2 stays open**, and the plan's assessment of it stands: LaTeX is presentation
markup and there is no correct general parser, so the work is to define a subset, parse
it strictly, and reject everything outside it with a source position.

## §12 — GUI

**Open.**

## §13 — Other features

| Item | Status |
|---|---|
| 13.1 Bit-reproducibility mode | Open — depends on §9.5 Tier 3 |
| 13.2 Reproducibility manifest | Done — `ms/runtime/repro.hpp` |
| 13.3 Structured audit log | Open |
| 13.4 Python bindings | Open |
| 13.5 Language server | Open |
| 13.6 Sparse direct solvers | Open |
| 13.7 Arbitrary-precision transcendentals | Done previously |
| 13.8 Checkpoint and restart | Open |

`ms::runtime::capture()` reports version and commit, the ISA path actually taken
after the OS register-state check, whether `MS_FORCE_ISA` applied a ceiling, the
worker count, and the seed — as text or JSON. It reports conditions rather than
pinning them, which is the minimum viable form of §13.1 and the part that was nearly
free.

## §14 — The gate on `v1.0.0`

The plan names five items that must close before the tag:

| Gate | Status |
|---|---|
| §4.1 Licence, including the CUDA §7 exception | Done |
| §4.3 Export-control determination | Done |
| §4.5 Authorship | **Waived** — declined by the repository owner |
| §6.1 SIGILL in `dgemm` | Done |
| §6.3 CSPRNG | Done |

All five are now resolved: four closed by work, the fifth by an explicit decision not
to do it. **On the plan's own criteria the tag is no longer blocked.**

What still gates `v1.0.0` is `RELEASE.md`'s own eleven criteria rather than this list
— in practice criterion 1 (CI green) and criterion 5 (the 24 h fuzz marathon, which
the author is running on their own hardware).

---

## Formal verification

Not in the plan, and worth recording. `verification/` holds bounded-model-checking
harnesses run under both ESBMC and CBMC in CI:

- `isa_gating.c` — the ISA hierarchy cannot report a wider path than the OS enabled,
  and `MS_FORCE_ISA` can only narrow.
- `miller_rabin_witness.c` — the §6.2 defects 2 and 3 are unreachable.

One thing that went wrong here is worth stating, because it is the failure mode this
kind of tooling is most prone to. The harnesses were first written with
`__CPROVER_assume`, which ESBMC accepts and silently ignores. They therefore verified
considerably less than they claimed, and reported success while doing it. They now
use `__VERIFIER_assume` through a portability shim, and both provers agree.
