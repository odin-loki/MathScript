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

### A size argument sized an allocation, and the guard in front of it was not one

One idiom, in 178 REPL commands:

    const int n_i = static_cast<int>(n_d);
    if (n_i < 0 || n_d != n_i) { /* reject */ }
    ... eval_something(static_cast<std::size_t>(n_i)) ...

Three things go wrong with a size argument before any work starts. It catches none of
them.

**The cast IS the check.** `static_cast<int>` of a double outside `int`'s range is
undefined behaviour, not a wrap, so by the time the guard reads `n_i` there is no value
there to test. On x86-64 the conversion happens to produce `INT_MIN`, so `n_i < 0`
rejected `fem_poisson1d(1e18)` — by accident, in a way that reads exactly like a guard
and is not one. The range has to be decided on the double.

**A count that fits is not a count that is affordable.** `fem_poisson1d(100000000)` is a
perfectly ordinary `int` and asks for 800 MB.

**A cap on each extent alone is not a cap on the allocation**, because what is allocated
is the *product*. `imresize(A, 100000, 100000)` names two extents that each look like a
resolution and together are ten billion elements — which is why a per-dimension bound,
the obvious fix, would not have been one.

With `-fno-exceptions` none of this produces a diagnostic. The `std::bad_alloc` out of
`std::vector` reaches `std::terminate` and the process is gone, with nothing written to
either stream. Every case below was reproduced under a 2 GB address-space cap and came
back `rc=134`:

| Command | The argument | What it asked for |
|---|---|---|
| `fem_poisson1d(100000000)` | `n` | 1e8 elements |
| `fem_poisson2d(100000, 100000)` | `nx`, `ny` | the product, 1e10 |
| `fem_poisson3d(5000, 5000, 5000)` | `nx`, `ny`, `nz` | the product, 1.25e11 |
| `cfd_advection1d(100000000, ...)` | `nx` | 1e8 elements |
| `cfd_advection2d(100000, 100000, ...)` | `nx`, `ny` | the product |
| `cfd_advection3d(2000, 2000, 2000, ...)` | `nx`, `ny`, `nz` | the product, 8e9 |
| `numthy_farey(1000000)` | `n` | about 3e11 rows -- the length of F_n is *quadratic* in n |
| `impad(A, 1000000)` | `pad` | 4e12, because the padding grows all four sides |
| `imresize(A, 100000, 100000)` | `rows`, `cols` | the product |
| `hough_lines(A, 0.5, 1e8, 1e8, 1)` | `n_theta`, `n_rho` | the accumulator, one cell per pair |
| `hough_circles(A, 1, 100000000)` | `r_max` | one plane of the image per radius |
| `ml_pca_fit(A, 100000000)` | `n_components` | a component matrix, for a 2x2 input |
| `ml_pca_fit_transform(A, 100000000)` | `n_components` | the same, plus the transform |
| `ml_kmeans_fit(A, 100000000)` | `k` | a centroid per cluster, for two rows |

**The budget is not a new number.** `kMaxReplMatrixElems` is 262144, and the REPL
already refused to *store* a larger matrix — in `assign_matrix_call`, after the
allocation. All the guard moves is when: from a diagnostic about a matrix that has
already been built, to one about the argument that asked for it. `ExtentBudget` in
`src/interp/matrix_call.hpp` reads extents one at a time and divides the budget down as
it goes, so the bound lands on the product without any handler having to multiply.

**Two of them are shape relationships rather than sizes, and capping them would have
been wrong.** A principal component is a direction in feature space and there are only
`min(samples, features)` of them; k clusters need k points to put in them.
`ml_pca_fit(A, 100000000)` on a 2x2 is not an expensive request, it is a request with no
answer, and it now says so. `numthy_farey` is a third kind: its length is quadratic in
its argument, so the guard computes `|F_n|` exactly with a totient sieve rather than
estimating it — an estimate would have to be conservative, and a conservative estimate
refuses an order whose sequence actually fits.

**And one was in `src/compress`, found by AddressSanitizer in CI rather than by any of
this.** `lz77_decode` reads a back-reference as a distance BACK from the end of what it
has decoded so far:

    size_t start = out.size() - t.offset;
    for (uint16_t i = 0; i < t.length; ++i) out.push_back(out[start + i]);

`t.offset` comes straight from the caller's data and the subtraction is unsigned, so an
offset larger than the output wrapped to an index near 2^64 and read roughly four billion
bytes past the buffer. It is reached from `lz77_decode_vec(M3)` -- a 3x3 matrix of small
numbers, which is to say from the first malformed token stream anyone hands the REPL --
and `test_repl_malformed_sweep` has been handing it one all along. It has failed on both
ASan runs that finished (`7bc6e38` and `75c870f`); the runs before those were cancelled
by the next push, so how far back it goes is not established here.

A stream that back-references a byte it never emitted is not one this can decode, and no
prefix of it is meaningful either, so the answer is nothing rather than a guess.
`eval_lz77_decode_vec` names the offending token before it gets that far, and also stops
truncating: `static_cast<uint16_t>(70000)` is 4464, so an offset past the type's range
used to become a DIFFERENT, valid offset and decode silently to the wrong bytes.

**Two of them were in `src/image` rather than at the REPL boundary, and a guard at the
boundary does not reach them.** `image::impad` computes `img.rows + 2*pad` in `int`. At
`pad >= (INT_MAX - 2) / 2` that overflows to a negative, `Image`'s constructor clamps a
non-positive extent to an EMPTY image, and the copy loop -- bounded by the *source's*
extents rather than the destination's -- ran anyway and wrote an index near 2^30 into a
zero-length vector, about 4.29 GB past a null base. A negative `pad` reached the same
write from the other end. `image::imresize` indexed its destination with
`(r * nc + c) * channels` in `int`, which wraps negative once the output passes INT_MAX
elements -- and 46341 x 46341 single-channel is an image this type can legitimately
hold. Both are settled inside the library now: a caller can be asked to keep a request
affordable, and cannot be asked to keep a function inside its own allocation.

**The cast is gone from all 256 of them.** The list above is what was measured to end
the process; the same conversion was in front of every other integer argument the REPL
takes, and every one is now decided on the double. The second family needs a different
bound, because it is not an extent:

    legendre_p(1750000000, 0.5)

*returns.* Nothing is allocated per unit of an order; what it does is drive a recurrence,
one step per unit, in a REPL with no way to interrupt one -- so it takes longer than the
twenty seconds a probe will wait for it. `checked_int_argument` bounds these at 1e7,
which is the number `repl_engine_internal.cpp` has used for a matrix index and a matrix
count since the accessors were added. It is a WORK bound and not an accuracy one: these
recurrences still carry several correct digits well past it; what they do not do is
finish. No order anyone writes down is within four orders of magnitude of it.

The same helper refuses truncation, which is the half of this that has nothing to do
with undefined behaviour. `static_cast<int>(2.5)` is perfectly well defined, and
`bessel_j(1.5, 1)` answered as though 1 had been written, with nothing to say so.

Removing the cast made 109 conditions unreachable -- `if (n < 0 || n_d != n)` cannot
reach its second half once the double has been checked -- and those are removed with it,
because a condition that reads as a guard and cannot fire is the thing this whole entry
is about. 42 tests that asserted a combined message ("expected integer l and m") now
assert the per-argument one, which names which argument.

**What a linear bound does not cover.** The number above is 1e7 and the comment beside
it justifies it by a duration -- "a three-term recurrence is about a tenth of a second".
Which means it was never a bound on the argument's MAGNITUDE. It was a bound on the WORK
a linear command does per unit of it, read out in the argument's own units, because for a
linear command the two coincide. For a command whose cost is not linear they do not, and
the linear reading is not conservative, it is catastrophic:
`finance_binomial_call(S,K,T,r,sigma,10000000)` is a perfectly ordinary integer that asks
for 5e13 node visits, about twelve days.

That gap is now closed, and closing it turned up **nine more commands that end the
process**, in the same class as the fourteen above and missed by the same sweep that
found those. The reason they were missed is worth stating plainly, because it is a fact
about the test rather than about the code: `test_repl_malformed_sweep` probes
`3000000000` and `1e18`, which the linear cap REJECTS. **A sweep made of values the guard
turns away cannot find a command that dies on a value the guard lets through.**

Each was reproduced under a 4 GB address-space cap and came back `rc=134`, a
`std::bad_alloc` reaching `std::terminate` with nothing on either stream:

| Command | At | Why |
|---|---|---|
| `pde_heat_1d(ones(200,1),0.1,0.1,0.001,10000000)` | 10.9 s | 16 GB of history for 200 numbers |
| `pde_heat_1d_cn(...)` | 26.2 s | the same |
| `pde_advection_1d(...)` | 8.6 s | the same |
| `pde_advection_1d_lax_wendroff(...)` | 12.3 s | the same |
| `pde_reaction_diffusion_1d(...)` | 12.2 s | the same |
| `pde_burgers_1d(...)` | 11.9 s | the same |
| `pde_wave_1d(...)` | 8.5 s | the same |
| `pde_heat_2d(ones(60,60),...)` | 22.7 s | the same, per 3600-cell grid |
| `pde_wave_2d(...)` | 26.4 s | the same |
| `gria_alpha_ca(30,10000000,10000000)` | 0.02 s | `bits.reserve(steps*width)` -- 800 TB |

`pde_heat_2d_cn_adi` is the tenth and was still running at 35 s rather than aborting
inside it.

The shape is the same in all nine solvers: they accumulate the whole trajectory,
`result.u.push_back(u)` once per step, and the REPL reads only `.back()`. So the request
is for a 16 GB history in order to return 200 numbers. (The waste is not fixed here --
the library's return type is a trajectory and other callers read it -- but it no longer
reaches a size that matters, because the bound on `steps` scales with the grid.)

**Two policy numbers and a measurement at every call site.** A bound on a super-linear
argument is really a bound on TIME, since a REPL command runs to completion with no
interrupt. That splits cleanly into a judgement and a fact, and the two should not be
confused:

  - `kMaxReplCommandWorkNanos` (0.25 s) and `kMaxReplSimulationWorkNanos` (4 s) are the
    POLICY. Two rather than one, because the distinction is in what the argument MEANS: a
    binomial tree converges like 1/steps and is finished by about a thousand, so nobody
    types `steps = 1000000` on purpose and bounding it costs no one anything; a Monte
    Carlo converges like 1/sqrt(n_paths), so a hundred thousand paths is not a slip, it is
    the command doing its job. Holding the second to a quarter of a second would take the
    command away rather than protect it.
  - `nanos_per_unit`, passed by each call site, is the MEASUREMENT. These differ by a
    factor of a hundred and twenty, which is exactly why a single shared "quadratic
    arguments" cap would have been wrong in both directions at once:

| Command | ns per unit | Measured from |
|---|---|---|
| `finance_binomial_call` / `_put` | 11 | 2.69 s at steps=16000 |
| `finance_american_option` | 28 | 2.65 s at steps=10000 |
| `finance_trinomial_option` | 28 | 2.84 s at n_steps=10000 |
| `diffgeo_sphere_gauss_bonnet` (+`_residual`) | 1970 | 1.97 s at n=1000 |
| `pde_advection_1d` | 14 | 0.56 s at 200x2e5 |
| `pde_heat_1d` | 16 | 0.62 s |
| `pde_wave_1d` | 18 | 0.71 s |
| `pde_reaction_diffusion_1d` | 20 | 0.78 s |
| `pde_advection_1d_lax_wendroff` | 21 | 0.82 s |
| `pde_burgers_1d` | 25 | 0.97 s |
| `pde_heat_2d` | 38 | 1.50 s at 3600x1.1e4 |
| `pde_heat_2d_cn_adi` | 101 | 4.03 s |
| `pde_wave_2d` | 45 | 1.78 s |
| `quantum_schrodinger` (+`_final`) | 25 | 15.7 s at 8x8, n=1e7 |
| `gria_alpha_ca` | 177 | 17.7 s at 1e5x1e3 |
| the Monte Carlo family | 190 | 1.68 s at 1e4 paths x 1e3 steps |

A binomial tree at 3000 steps takes 0.21 s and is allowed; Gauss-Bonnet at 3000 takes
17.4 s and is not. One cap could not have said both.

**`WorkBudget` is `ExtentBudget`'s shape applied to work.** Where the cost is a PRODUCT
-- `steps` sweeps of a grid, `n_paths` walks of `n_steps` -- no single factor looks wrong
and `checked_int_argument` bounds each at 1e7 independently, so the product it admits is
1e14. The budget multiplies the factors as they are read, and charges the operand matrix
first so the bound on `steps` shrinks as the grid grows: a hundred steps of a large grid
costs what ten thousand steps of a small one does, and that is the relationship that
actually holds.

**A shape ceiling is not a cost ceiling.** `tensorops_decompose_cp` and `_nmf` already
refused a rank past the tensor's element count -- which is a statement about whether the
decomposition is informative, and says nothing about what it costs. Underneath it,
`tensorops_decompose_nmf(h, ones(80,80), 200)` is rank 200 of 6400, an entirely ordinary
request, and had not finished after 45 s. The two differ in shape and are bounded
differently: NMF runs every one of its `max_iter` sweeps, so the iteration count is
charged before the rank is read; CP's ALS converges out of `max_iter` long before
reaching it -- 1e7 iterations of a 40x40 still returns in 0.02 s -- so only the rank
drives it. `tensorops_decompose_tucker` measured the same way as CP and is left alone:
the mode-dimension ceiling it already has is the right one.

**The `uint64_t` family: eighty-one sites, and the answers were fabricated rather than
absent.** Every one guarded the bottom of the range and none the top --

    if (arg < 0.0 || std::floor(arg) != arg) { /* reject */ }
    ... static_cast<uint64_t>(arg) ...

-- which rejects negatives and fractions and then converts anything else, 1e300 included.
What that looked like from the prompt was an answer:

| Typed | Printed |
|---|---|
| `numthy_gcd(1e300, 18)` | 18 |
| `numthy_lcm(1e300, 3)` | 0 |
| `numthy_num_divisors(1e300)` | 1 |
| `numthy_euler_phi(1e300)` | 0 |
| `numthy_sum_divisors(18446744073709551615)` | 0 |

The last is the sharpest. That literal is 2^64-1, which no double represents; it rounds
UP to exactly 2^64, one past the last value the destination holds, so the conversion had
nothing to return and sigma was reported as zero. `checked_u64_argument` decides the range
on the double, and the clamp is written against 2^64 - 2048 -- the largest double that is
also a `uint64_t` -- because `kTwoPow64 - 1.0` rounds straight back to `kTwoPow64` and
would have admitted the one value that cannot be converted.

The argument names in those diagnostics are not invented. They are read out of the
signatures the REPL's own help prints, so `numthy_mod_pow(1e300, 2, 7)` says `base` and
`gria_gf2n_inv` says `poly`.

**Fifteen seeds, and two of them were the same seed.** `static_cast<unsigned>` has the
same problem one type down, and here it does not merely admit nonsense:

    finance_mc_european_call(100,100,1,0.05,0.2,1000,42)           10.799620
    finance_mc_european_call(100,100,1,0.05,0.2,1000,4294967296)   10.757478
    finance_mc_european_call(100,100,1,0.05,0.2,1000,1e300)        10.757478

The last two agree because neither conversion had a value to produce. Somebody varying
the seed to see the Monte Carlo spread would have been reading one sample twice and
calling it two.

**The exclusion list is gone, and neither entry needed a cap.** `test_repl_malformed_sweep`
skipped `numthy_prime_nth` and `numthy_sum_divisors` through an `is_proportional_cost`
predicate -- a record of an unfixed defect rather than of a test that does not apply. The
earlier note here said both ran indefinitely on `3000000000`; measured, only `prime_nth`
did. `sum_divisors(3000000000)` answers in 0.01 s and it is `sum_divisors(1e18)` that took
3.55 s. In both cases the right answer turned out not to be a bound at all:

  - `sum_divisors` built the divisor list by trial division to sqrt(n) -- 4.3e9 iterations
    at the top of the range. Sigma is multiplicative, so the exponents are enough, which
    is what `num_divisors` and `euler_phi` on either side of it already did: they answer
    the same n in 0.01 s out of the same factorisation.
  - `prime_nth` ran one Miller-Rabin test per prime up to n: 4.3 s at n=1e6 and still
    going at half a minute for 3e9. It sieves once instead, to the Rosser-Schoenfeld
    bound p_n < n(ln n + ln ln n), with the first five primes listed because that bound is
    not valid below n=6.

**A cap on the result is not a cap on the working set.** `fem_poisson1d(262144)` still
aborted the process AFTER §54's extent guard was in place, and at exactly the number that
guard enforces. The guard charged `n` against `kMaxReplMatrixElems`, which is right for
the RESULT -- a vector of `n` node values. It is not what gets allocated:
`assemble_stiffness_1d` builds a **dense** `n_nodes` by `n_nodes` `ColMatrix`, so
`n = 262144` asks for 6.9e10 doubles, 550 GB. `fem_poisson2d` and `fem_poisson3d` assemble
the same way from `(nx+1)(ny+1)` and `(nx+1)(ny+1)(nz+1)` nodes.

It is the same sentence as §54's own "a cap on each extent is not a cap on the
allocation", one level further out, and it is worth separating because §54 read as closed.
`ExtentBudget::charge_dense_order` charges the mesh order a second time, so what the
budget bounds is the stiffness matrix rather than the answer, at all seven dispatch sites.

The `fem_mesh` family had the plain version of the same defect and no budget at all:
`parse_positive_size_arg` bounds each extent at 1e7 on its own, so
`fem_mesh2d(0, 0, 1, 1, 10000000, 10000000)` asks for 1e14 nodes. Measured aborting;
`fem_mesh2d`, `fem_mesh2d_rectangular`, `fem_mesh3d` and `fem_mesh3d_box` now charge the
product.

Both were found the same way -- by running the probes the audit proposed, against the
guard that was supposed to have closed them. A guard is not a fix until the input that
motivated it has been re-run against it.

**Eight more that ended the session, from running the audit's own probes.** A read-only
sweep of all ten library domains proposed 86 candidates with a probe line each; running
them turned up eight further aborts, and their shapes are the four the guards already
knew, arriving where the earlier sweeps had not looked:

| Shape | Commands |
|---|---|
| an output that is a MULTIPLE of the input | `signal_upsample`, `signal_interpolate`, `signal_resample` |
| an output that is the SQUARE of an extent | `quantum_identity_n`, `topo_pairwise_distances` |
| a PRODUCT of two arguments | `topo_persistence_landscape` |
| a parameter whose MAGNITUDE sizes a matrix | `mathieu_a`'s `q`, and `lbfgs`'s history `m` |

The last row is the one worth naming. `mathieu_a(n, q)` looks like it takes two ordinary
numbers, and `q` is a size argument wearing a parameter's clothes: the characteristic
matrix is sized `max(24, index + 16 + ceil(sqrt(|q|)))`, so `q = 1e18` asks for a
1e9-entry tridiagonal, and at `q = 1e300` the `static_cast<int>` of that square root is
undefined before it gets there. `checked_matrix_sized_parameter` bounds the DIMENSION
rather than `q`, because what has to fit is the matrix and `q` is only how the command
spells it; the same guard covers `mathieu_b`, `_ce`, `_se` and the three spheroidal
commands.

`lbfgs`'s `m` came through `parse_optional_positive_int`, which bounded the bottom of the
range and not the top across **nineteen** call sites. `max_value` is the caller's there,
because what those nineteen bound is not one kind of thing -- an iteration count is
bounded by work, and a stored history by memory, and `m` is the second.

**Two of the fixes had to move to the funnel.** `signal_resample` and
`topo_pairwise_distances` are each reached by more than one dispatch path -- the
assignment form goes through the matrix-call registry and the bare form does not -- and
guarding the handler left the other route intact, which the probe caught by still
aborting. The guard belongs in `eval_signal_resample` and `eval_topo_pairwise_distances`,
where every route converges.

**And one was wrong in a way only the probe showed.** `charge_dense_order` charges the
order ONCE, as the SECOND factor: at the FEM sites an earlier `take` had already charged
the first. `topo_pairwise_distances` had no earlier take, so a 131072-point set passed a
check it should have failed by a factor of 131072. It aborted again, which is the only
reason it was caught.

**The image filters: twelve commands, and the no-assignment form is a second path.**
Every one of them visits each pixel once per kernel cell, so the cost is the image times
the kernel -- times its SQUARE for the morphology and median filters. Measured on a
256x256: `medfilt2` at ksize 21 takes 1.73 s, so the `medfilt2(ones(512,512), 999)` the
audit proposed is 2.6e11 pixel-cells, about four hours.

| Command | ns per pixel-cell | Kernel enters as |
|---|---|---|
| `medfilt2` | 60 | its square |
| `imdilate`, `imerode`, `imopen`, `imclose`, `imtophat`, `imbothat`, `imgradient_morph` | 20 | its square |
| `bilateral` | 45 | its square, from `half = 2*sigma_s` |
| `boxfilter` | 45 | its width (separable) |
| `imgaussfilt`, `laplacian_of_gaussian` | 35 | its width, from `half = 3*sigma` |

`sigma` is not a tuning knob on the cost: the kernel half-width is a multiple of it, so
what sigma names IS the kernel, and the bound is stated on the kernel because the kernel
is the thing that has to fit. There is also a shape argument available -- a kernel wider
than the image is meaningless, every window being the whole image -- but it would not
have been enough on its own: 512 x 512 x 512^2 is still four hours.

And the trap that had already caught `signal_resample` and `topo_pairwise_distances`
caught this family too, in its own way. Guarding the twelve handlers left
`medfilt2(A, 999)` -- the same call with no `B =` in front of it -- running for four
hours, because the no-assignment form does not go through the matrix-call registry. It
was the TEST that found it: the suite went from 130 s to 1570 s and timed out, which is
the same signal as an abort and nearly as loud.

**A step count that is not an argument at all.** The six CFD advection commands take
`t_end` and `dt` and no step count: the number of sweeps is `ceil(t_end/dt)`. Neither
number looks like a size and their QUOTIENT is one, so no per-argument guard can see it --
1.0 and 1e-9 are both unremarkable, and together they are a billion sweeps of the grid
with one whole grid retained per step. Measured at 70 ns per cell-step in 1-D, 200 in 2-D
and 370 in 3-D. The bound is on the quotient, which is the only place the size actually
appears, and it sits in the six `eval_cfd_*` functions because that is where every
dispatch path converges -- applying the lesson from `signal_resample` rather than
relearning it.

**Work for an answer that could never be shown.** The REPL's scalar is a double and
`bigint_to_scalar` requires the exact BigInt to round-trip through one, so 21! already
fails and so does fib(79). What the bignum commands did with a large argument was compute
the exact answer FIRST and refuse it afterwards: `bigint_fib(200000)` spent 15.3 s
building a number it then declined to print, and `bigint_factorial(20000)` 1.4 s. Both are
quadratic in n, because each of the n steps operates on a number that is itself growing.
The bounds added sit far above where the round trip stops succeeding, so they refuse
nothing that could have worked; all they do is stop the computing.

**And the Schmidt family is cubic in the subsystem dimension.** The Gram matrix is
`dim_a` by `dim_a` and the Jacobi sweep over it is cubic: 0.65 s at dim_a = 1024, so
dim_a = 4096 is 6.9e10 units and 82 s. Ten qubits is an ordinary subsystem to decompose
and stays inside the bound; twelve does not. Five commands share it --
`quantum_schmidt_rank`, `_number`, `_decomposition`, `_bases` and
`quantum_entanglement_entropy`.

**One reported finding did not survive a probe.** The allocation audit recorded
`graph_bipartite_match` aborting at its second argument. It does not:
`graph_bipartite_match(M3, 3000000000)` is refused by the argument guard and
`graph_bipartite_match(M3, 10000000)` returns "not bipartite" promptly, which is the
right answer for that matrix. Recorded here rather than dropped, because an unreproduced
report left in a list reads later like an unfixed defect.

Both now report the same sieve-span sentinel `prime_pi` does, and that sentinel got an
honest message on the way past: it used to say "result does not fit in 64 bits" about
pi(3000000000), a number near 1.4e8 that fits in a double with room to spare. The limit
is the sieve, not the width, and it now says so.

## §7 — Stubs and half-implementations

**Open.** Ship-or-cut decisions, tracked in
[`RELEASE_DECISIONS.md`](RELEASE_DECISIONS.md).

## §8 — Coverage and testing programme

| Item | Status | Note |
|---|---|---|
| 8.1 Baseline on real hardware | Done | 91.2% lines, 98.3% functions, 57.3% raw branches, 71.8% over decision lines |
| 8.2 `src/plugin` tests | Partial | `unsafe_registry` is tested (273 lines of test against 282 of audit bookkeeping that had never run); the Clang AST rules themselves are covered only by the plugin smoke job |
| 8.3 REPL golden corpus | Done | `tests/repl_corpus/*.ms` with committed stdout and stderr, run through the real `mathscriptc` |
| 8.4 Mutation testing | Started | `scripts/mutation_test.py`; run over `src/compress/compress.cpp`, 66.7% -> 80.0% over viable mutants, three remaining classified rather than counted as gaps |
| 8.5 Property-based testing | Done | seeded invariants over the linalg/FFT core, and the §11 printer round-trips |
| 8.6 Differential tests vs reference BLAS/LAPACK | Partial | the dgemm kernels have them; the wider LAPACK surface does not |
| 8.7 Remaining gaps | Open | |
| 8.8 Group 573 integration targets | Done | 573 executables → 31 |
| 8.9 Lock it in | Done | four source-only gates plus the coverage ratchet, all gating in CI |

### §8.4, and the four kinds of survivor

`scripts/mutation_test.py` changes one character-range of a source file, rebuilds the
target that covers it, runs it, and reports what happened. Coverage says a line ran; a
surviving mutant says nothing asserted it.

**Mutants that do not compile are reported separately and are not counted as killed.**
Folding them in is the standard way a mutation score is inflated: a harness that
generates mostly uncompilable mutants and calls them killed reports 95% while testing
nothing. The score is over viable mutants only, and the raw counts are printed.

First file: `src/compress/compress.cpp`, 16 mutants at seed 3. **Five of fifteen viable
mutants survived — 66.7% — on a file with 105 tests and full line coverage.** What made
the run worth more than the number is that the five were four different things:

| Survivor | What it was |
|---|---|
| `:791` wavelet header byte order | A missing test. Every wavelet test was short enough that the length fits in the last header byte, so reading the wrong one gave the same answer. **Killed.** |
| `:329` ANS frequency normalisation | A missing test *that no round trip can supply*. The excess comes off the largest frequency; the encoder writes its choice into `freq_table`, which the decoder rebuilds from, so the decoder absorbs it wherever the encoder did. **Killed by pinning the bytes.** |
| `:270` the decoder's count clamp | **Equivalent.** `symbol_for_count` clamps to the last index by construction, so `total - 1` and `total + 1` select the same symbol. Not a gap. |
| `:359` `index_of`'s `-1` | **Unreachable.** All three callers look up a symbol the model was built from. Not a gap — but all three then index `freq` and `cum` with the result unchecked. |
| `:185` the range coder's carry | **Dead, or as good as.** Instrumented and counted: zero hits across 800,000 bytes in four distributions, on both the encode and decode paths. The condition tests bit 56 of a 64-bit `low` while `kTop` is `1u << 24`, a 32-bit coder's constant, and the expression truncates through `static_cast<uint32_t>`. Recorded rather than changed: altering a working entropy coder's carry logic with no reproducing input would be reckless. **Open.** |

The general lesson is the second row. **A property that says "decode undoes encode" is
blind to any change applied symmetrically**, and for a codec that is most of the
implementation. `CompressFormat.TheEncodedBytesAreWhatTheyHaveAlwaysBeen` pins the exact
output of both entropy coders, which makes the compressed format a contract —
deliberately, since `bzip2_compress_vec` and its siblings hand a user a matrix they can
save and read back in a later build.

Score after: **80.0%**, with the three remaining classified above rather than counted as
gaps.

Second file: `src/combo/combo.cpp`, 14 mutants at seed 5 against all three targets that
cover it. **13 of 14 killed, 92.9%**, one survivor: `combo.cpp:144`, `if (i < 0) return
false;` -- a line that ran and that nothing asserted.

Third file: `src/numthy/numthy.cpp`, 16 mutants at seed 7 against `test_numthy` and
`test_numthy_overflow`. **12 of 15 viable killed, 80.0%** -- and all three survivors are
classified, none of them a gap. Each was settled by MEASUREMENT and not only by the
argument for it, because an equivalence that is merely argued is how an untested line
gets written off:

| Survivor | What it was |
|---|---|
| `:312` `if (e > 1)` in `pow_u64` | **Not compiled.** It sits inside the `#else` of `#if defined(__SIZEOF_INT128__)`, and `__int128` is available on every platform CI builds, so the mutant produced a byte-identical program. Traced by hand, the mutant *would* be wrong if that branch were ever taken -- `pow_u64(3, 2)` would return 3 -- so the fallback is correct and simply has no coverage anywhere. |
| `:479` `M > UINT64_MAX / t` in `crt` | **Equivalent.** `t` is a residue mod `m[i]`, so `t < m[i]`, so `UINT64_MAX / t > UINT64_MAX / m[i]`: the boundary `M == UINT64_MAX / t` that `>=` would newly reject always trips the modulus guard four lines below, which returns the identical message. Confirmed over 400,000 random systems weighted towards moduli large enough to reach the overflow guards -- byte-identical output. |
| `:180` `(c % (n - 1)) + 1` in `pollard_rho` | **Equivalent through the only caller.** `pollard_rho` is `static` and `factor_recursive` is its sole caller; that caller trial-divides by 2, 3, 5, 7, 11 and 13 first and passes `c` from 1 to 20, and `c % (n-1) == c == c % (n+1)` whenever `c < n - 1`. The two differ only for `n <= 21`, where the result is unchanged anyway. Confirmed by running `factor(n)` for every `n` from 2 to 300,000 under both -- identical. |

The first row is the one worth keeping. A mutation score cannot see a branch the
preprocessor removed, so a fallback implementation behind a `#if` is invisible to this
technique *and* to the test suite at the same time -- and the two silences look exactly
alike from the outside.

Fourth file: `src/sym2/expr.cpp`, 16 mutants at seed 11 against all eight targets that
cover it. **10 of 16 killed, 62.5%** -- the lowest of the four, and the §10 core. Six
survivors, and the spread is the finding rather than the number:

  - **One real gap, and it is in `evaluate`.** `*base == 0.0 && *exponent < 0.0` widened
    to `<=` -- which turns `0^0` into a division-by-zero error -- survived all eight
    suites. `test_sym2_core` *does* assert `0^0`, but on the BUILDER, which folds the
    literal to `undefined` before any Pow node exists; the evaluator's own guard is a
    different path reached only when base and exponent both come out zero from the
    environment, and nothing was asserting it. Measured: `evaluate(b^e, {b:0, e:0})` is
    `1.0`, following `std::pow`. **Tested now**, and the new assertion was checked
    against the mutant: it fails.
  - **Two are unreachable, and for the same reason.** `:308`, the `hash !=` fast path in
    `structurally_equal`'s structural fallback, and `:347`, the equal-value branch of
    `compare` on two Reals. Both functions short-circuit on `a == b` first, and
    interning means the public API cannot produce two structurally equal nodes that are
    not the same pointer -- measured over integers, reals, symbols, sums and a nested
    power, every pair came back pointer-identical. `structurally_equal`'s comment
    justifies the fallback by "a node built before an interner reset, or handed in from
    a bridge that constructed one directly"; there is no interner-reset API, so it is
    defensive code for a caller that does not exist yet.
  - **Three are equivalent.** `:371` and `:363` sit after an explicit equality test
    (`if (x != y)`, `if (xb == yb) return 0;`), so widening `<` to `<=` cannot change
    the branch taken; `:363`'s other mutant returns 2 where the code returns 1, and a
    three-way comparator is consumed as a sign. `:958` sorts by `display_key`, and `mul`
    collects same-base factors at construction, so two factors with an equal key do not
    arise.

The number to take from this is not 62.5%. It is that the newest code in the tree, whose
tests were written alongside it, had its one real gap exactly where a test asserted the
*builder* and the reader would reasonably believe the behaviour was covered.

Fifth file: `src/sym2/latex_parse.cpp`, 18 mutants at seed 13 against
`test_sym2_latex_parse`, `test_sym2_latex_roundtrip` and `test_sym2_notation_roundtrip`:
**12 of 17 viable killed, 70.6%**. Five survivors, and unlike `expr.cpp` above, three of
them are real gaps rather than one:

  - **`:202`, the multi-character escapes.** `i += sizeof("\\textasciitilde{}") - 1`
    widened to `- 2` leaves the closing brace unconsumed, so `\operatorname{a\textasciitilde{}b}`
    reads as the name `a~}b` -- and all eight sym2 suites still passed, because §1.2's
    three named escapes had **no test anywhere in the tree**. All three are asserted now,
    not just the one a mutant happened to land on.
  - **`:974`, the adjacency clause of the scientific numeral.** §2.4 spells the base as
    the single terminal `"10"`, so `1 0` -- two tokens with a space between them -- is
    not it. Rewriting one `||` of the five-way chain to `&&` regroups it so a
    NON-adjacent `1 0^{3}` satisfies it. Measured on the mutant: `2 \times 1 0^{3}` reads
    as 2000.
  - **`:1984`, the derivative denominator's opening brace.** §2.6 requires the second
    `{`; the mutant returns before reading the variable, calling `\frac{d}x` a derivative
    of the empty name. This one took three attempts to kill, and the failures are the
    lesson: asserting that a MALFORMED string is rejected separates nothing, because the
    mutant rejects it too. What separates them is a **positive** case --  `\frac{d}x` is
    legal (§2.9 lets one token stand for a group) and means the quotient `d/x`.

The other two are not gaps, and each was settled by reading every path rather than by
eye:

  - `:1547`, `std::size_t bad_at = 0`. **Equivalent.** `unescape_name` has exactly two
    `return false` statements and both are immediately preceded by `bad_at = i`, so the
    initialiser is never the value anyone reads.
  - `:1931`, `at_leibniz_fraction`'s own `if (!at_fraction()) return false`.
    **Unreachable.** Its single caller sits inside `if (at_fraction())`, and the only
    thing between them is a `while (try_derivative_operator(var))` loop that restores
    `pos_` on failure -- and the branch is only reached when that loop matched nothing.

Every one of the three new tests was checked against its mutant: apply, rebuild, and
confirm the test fails. A test added for a survivor that does not actually kill it is
the same silence with more lines in it.

Three crashes turned up while reading for those, all in code a frequency table reaches
from `ans_decode_vec`, and all verified before and after:

- every count zero: `raw_total` was zero and `from_counts` divided by it. **SIGFPE.**
- a count above `INT_MAX`: `static_cast<int>` made it negative, skewing every scaled
  frequency computed from the total.
- and one that appeared only *after* the first two were fixed: a table normalising to no
  usable symbols left the decoders indexing an empty model. **SIGSEGV.** A guard that
  returns an empty model turns a division by zero into an out-of-bounds read unless the
  caller is guarded too.

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
- **`transpose(A)` had no no-target form** although `matmul(A, A)` does. The CHANGELOG
  says the registry gives every matrix-returning callee a bare form. **Fixed**, and
  the count is the finding: it was not one name but **98**.

  A hand-written chain of 92 `else if (fn == ...)` branches, continued in a second
  function of 53 more because MSVC would not compile it as one, sits in front of the
  registry fallback. Its terminal `else` did not decline a name it had never heard of;
  it *claimed* the line and reported `unknown function`, six lines above the registry
  that knows every matrix-returning callee there is. So the rule was: a unary call
  whose argument resolves to a matrix reaches the registry only if somebody remembered
  to add the name to the list by hand. `prewitt`, `scharr` and `roberts` were on it;
  `sobel`, `sobel_x` and `sobel_y` were not. `graph_laplacian` was; `laplacian` was
  not. `matmul(A, A)` worked only because `A, A` fails to resolve as one matrix name
  and takes a different path entirely.

  The tail returns `std::nullopt` for a name it does not know now, which is a
  different answer from rejecting the line, and the caller asks the registry. The
  fallback prints **and stores** under `_`, so `B = transpose(_)` works on the next
  line. 13 more callees that take a scalar (`zeros(A)`, `eye(A)`, `fftfreq(A)`, ...)
  stop answering `unknown function: zeros` and give the handler's own diagnosis.

  Four names -- `boxfilter`, `imgaussfilt`, `laplacian_of_gaussian`, `medfilt2` --
  turned out to be listed at arity 1 in the arity table with no arity-1 form in the
  handler. **Fixed**, and not uniformly, because the four are not the same case.
  `medfilt2` and `boxfilter` now take the form: a default exists to be used rather than
  invented -- `image::medfilt2` declares `int ksize = 3` in its own signature, and 3 is
  what the eight neighbours in the same arity group use. `imgaussfilt` and
  `laplacian_of_gaussian` answer arity 1 with a diagnostic instead, because sigma is not
  a setting on a Gaussian blur, it IS the blur: there is no width that a caller who did
  not name one meant. Answering the arity rather than removing it from the table is
  deliberate -- the table decides whether the line is a matrix call at all, so dropping
  the row would stop `B = imgaussfilt(A)` being recognised as a call and report
  something further still from the truth.

  The fix adds no nesting to the 92-deep chain, deliberately: everything new sits
  after it, at the depth of the `return` it replaces. A change that deepened that
  chain would be paid for on the Windows runner an hour later, which is what the
  chain was split in two to avoid.
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
- **`stats_one_way_anova` and `rle_encode_vec` were unknown in the bare form** and
  reachable only through an assignment. **Fixed** by the same change: they are two of
  the 98. Both now answer with their handler's own diagnosis, which is what the
  transcript line was written to see -- `the test is not defined for these groups` and
  `byte values must be whole numbers in [0, 255]` rather than `unknown function`.
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
- **`not_a_function(1)` reported `unknown matrix: 1`** -- the diagnostic named the
  argument rather than the function it could not find, sending the reader to look for a
  matrix called 1. **Fixed**, and what the fix is *not* is the point.

  Saying "unknown function" instead would be a different false claim. `f(x)` reaches
  that return from every dispatch form in the REPL, not just the one whose name list is
  nearby: sweeping all 1,278 `fn == "..."` names through the binary as `N(1)` shows 502
  reaching it, of which 278 are real, working callees -- `mat_at`, `finance_npv`,
  `stats_percentile`, `finance_bs_call`, `stats_ttest` among them -- dispatched from
  blocks no predicate at that point enumerates. Nothing there knows whether the callee
  exists.

  So the message says only what was established: that no reading of the line worked. It
  keeps naming the argument when the argument is a *name*, because `det(no_such)` should
  still say `no_such`.

  The durable fix this finding really wants is one shared "is this a known REPL callee"
  predicate generated from every dispatch form. `scripts/extract_manifest.py` covers the
  485-entry matrix-call registry, which is one form of several. **Open.**

- **`bigint("495.0")` was not diagnosed by name.** Noticed while fixing the line above.
  The literal was rejected -- it no longer answers 0, which was the recorded defect --
  but no reading of the line claimed it, so it reported the generic "could not read"
  where it used to report a phantom matrix. **Fixed**, and the cause was not the
  message: `bigint` existed only as an ASSIGNMENT form, so the bare line was not a
  command at all. `bigint("495")` did not work either. Both go through the same
  reporting parse now, and `bigint("495.0")` says `invalid decimal literal: 495.0`.
- **`sym_simplify("x + x")` returned `(x + x)`.** Like terms were collected during
  `sym_expand` and not during `sym_simplify`, which folded a constant into a constant,
  dropped a zero, and stopped: `sym_simplify("2*x + 3*x")` came back as
  `((2.000000 * x) + (3.000000 * x))`, the input with the spaces moved. **Fixed**, and
  the shape of the fix is the point.

  The obvious route is to reuse expansion's polynomial normal form. It would also
  multiply products out -- `x*x` becoming `(x ^ 2.000000)` -- in all ~180 of simplify's
  callers, including the ODE solvers that dispatch on the *op* of what it returns. So
  the collector flattens the sum, adds the coefficients of terms that are the same
  term, and changes nothing else; when nothing merges it returns the expression it was
  given rather than a rebuilt copy, so a sum with no like terms in it is untouched.
  `x*x + y + y` is `((x * x) + (2.000000 * y))`: the product survives a sum that
  collected around it.

  The first version keyed addends by `sym_to_string`, and an existing test caught it
  within the hour. The printer renders a constant with six decimals, so
  `sin(1.0000001*x)` and `sin(1.0000002*x)` print identically and their difference
  collapsed to exactly zero. `SymbolicTables.ExpansionKeepsDistinctAtomsApart` exists
  because expansion made the same mistake earlier, and its comment names the trap in
  advance. The text is a bucket key now and `sym_equal` decides; the simplify half of
  the property is asserted too, so the next person to reach for a printed form as an
  identity has two tests telling them not to.

The transcripts record all seven as they are, with the two most misleading marked in
the script files, so each becomes a readable diff the day it is fixed rather than an
assertion someone has to reconstruct.

One further note the corpus settled: the one-dimensional root finders bind **`x0`**,
not `x`. `c88f0ef`'s message says `x` -- the guard it describes is correct and the
name in the prose is not. `docs/API.md` now states the binding on the row a user
reads.

### What the corpus found on Windows, and what it did not

**The Windows job never ran a single transcript, and had not since the corpus was
added.** Four CI cycles were spent on hypotheses about what the output meant when
there was no output, and the correction is worth more than any of them.

The test invokes `mathscriptc` through `std::system` with a command holding four
quoted paths -- the program, the script, and two redirect targets:

    "...\mathscriptc.exe" "...\arithmetic.ms" > "...\arithmetic.actual.out" 2> "...\arithmetic.actual.err"

On Windows `system()` runs `cmd.exe /c <command>`, and cmd's documented rule is that
unless the command holds *exactly two* quote characters it strips the first quote and
the last one. This one holds eight. Stripping takes the closing quote off the final
redirect target, cmd finds an unterminated quote where a filename should be, and says

    The filename, directory name, or volume label syntax is incorrect.

and exits **1** without running anything. That is the whole of it: empty stdout, empty
stderr, exit 1, for every transcript, on every Windows run since the corpus existed.
An extra outer pair of quotes is what the stripping is there to consume, and is the
fix.

Two things were in plain view and were read as noise:

- **The exit code was 1.** A Windows access violation is 3221225477 and a stack
  overflow is 3221225725. `mathscriptc` cannot return 1 without first writing to
  standard error. So the status could not have come from the program at all, which
  named the shell as the suspect three cycles before it was one.
- **cmd printed the reason every time.** It went to the test's own standard error,
  which is 5,800 lines deep in a CTest log whose API reaches only the end. Re-running
  the failed tests after a failing run -- added for an unrelated reason -- is what put
  it at the tail where it could be read, and it was legible on the first run after
  that.

#### What that means for the two "Windows failures" recorded here before

Both were inferences from an empty transcript, not observations, and neither can have
happened: with no output there is no line to differ. The record said otherwise and was
wrong.

- `-nan` versus MSVC's `-nan(ind)` in `dispatch_errors.out`: never observed. The fix
  it prompted stands on its own and is not withdrawn -- **a NaN reaching the display
  is a marker printed as a value**, `ms::sym2::evaluate` already declined both
  `log(-1)` and `sqrt(-1)`, and the REPL and the symbolic core disagreeing about the
  same expression is a defect whatever Windows does. `check_scalar_domain` now reports
  for `sqrt`, `log`, `log2`, `log10`, `log1p`, `asin`, `acos`, `acosh` and `atanh` in
  both of the REPL's scalar evaluators. Reaching those reports also required
  distinguishing a diagnosis from a decline among the three readings of `f(x)`, which
  is a real improvement to the resolver.
- `gamma(20)` at full precision pinning the host libm's last bit: never observed
  either. The rule it produced is still right and still followed -- **an expected file
  may hold exact integers, parsed literals printed back, and diagnostics, and may not
  hold the result of a transcendental at full precision** -- because a golden file
  that depends on which libm ran is a golden file that will fail eventually. It is
  `combo_factorial(19)` now, the same number by a path with no libm in it.

So two good changes were made for a stated reason that was not true. Keeping the
changes and correcting the reason is the only honest way to hold both, and the reason
mattered: each was offered as evidence that the transcripts were running on Windows,
which is exactly what needed testing and was never tested.

#### The lesson that generalises

A transcript that produced *nothing* is not a transcript that produced the wrong
thing, and reporting it as a content difference sends the reader to the content. The
test now separates them: empty stdout, empty stderr and a non-zero status is reported
as its own case, with whether the redirect files exist at all -- a missing file means
the shell could not create the redirect, so the status is the shell's -- and a re-run
with nothing redirected, whose output lands on the test's own streams instead of in a
file the shell may not have opened.

`mathscriptc` also flushes standard output after every line now. Redirected to a
file, `std::cout` is fully buffered, so a script runner that flushes only at exit
loses everything it printed if it dies partway; one flush per line means the file that
survives says how far it got. It also puts the two streams in true order, since
`std::cerr` is unit-buffered and `std::cout` was not.

#### The three readings of `f(x)`

Recorded here because it came out of the same thread and is not withdrawn: making
`check_scalar_domain`'s reports reachable took a second fix, and it is the more
interesting one.

`sqrt(-1)` reported **"unknown matrix: -1"**. The line has the shape `f(x)`, which is
also the shape of a call on a matrix and of a matrix constructor, and three readings
compete for it. Whichever fails last was reporting, so the useful diagnosis lost to
one about a variable the user never mentioned. They are now ordered by how much each
reading actually established:

1. the scalar reading when it *diagnosed* the line rather than declining it,
2. then a matrix call that got as far as dispatching and rejected its own arguments,
3. then the outer failure to resolve an argument as a matrix name.

Distinguishing a diagnosis from a decline is the whole of it, and it took three
attempts to get right — each wrong version traded one misleading message for another.
`invalid scalar expression`, `unknown scalar function: X` and `unknown scalar: X` all
mean "not my kind of line"; `'C' is a matrix, not a scalar` means that too when
another reading is available, and is the whole answer when none is. That last
distinction is why the rule is two predicates rather than one.

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
| 11.2 Input — parsing LaTeX | Done — `docs/LATEX_SUBSET.md` defines the subset, `parse_latex` reads it, `sym_from_latex("tex")` in the REPL |

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

**§11.2 is done, and the plan's assessment of it is what shaped it**: LaTeX is
presentation markup and there is no correct general parser, so the work was to define a
subset, parse it strictly, and reject everything outside it with a source position.

The subset is defined by the printer rather than by taste. `docs/LATEX_SUBSET.md` fixes
the accepted language as **everything `notation_latex.cpp` can emit, under every
`NotationOptions` combination**, plus twelve human spellings listed by name. That is
what turns

    parse_latex(to_latex(e, options)) == e

from an aspiration into an assertion -- and the document lists, exhaustively, the
twenty-eight shapes where it does not hold, each one a case where the printed form
carries less than the node did: a whole-valued `Real` prints as an integer, a total and
a partial derivative are spelled the same way, `\sqrt{x}` is a half power rather than a
call. Every one of the twenty-eight has its own test pinning what *does* come back. An
exception list nobody tests is an exception list that grows.

Every rejection names the ambiguity rather than the rule: `\sin^{2}(x)` is the square
at 2 and the inverse at -1; `\int_{a}^{b}` would have its bounds silently discarded
because `Head::Integral` records none; `\hat{x}` and `x` are different symbols to a
reader and the same name to a parser.

**The parser and its 107 tests were written in parallel by two authors, neither seeing
the other's work, both writing from the document.** They disagreed thirteen times. Ten
were the parser's. The other three were not bugs on either side -- they were places the
document was wrong or silent, and each is recorded in it now:

- three of them had **one** cause, and it was a markdown table cell. A literal `|` in a
  table has to be escaped as `\|`, which is also LaTeX's control symbol for the norm
  delimiter, so A34, A36, H2 and H3 all wrote the same two characters and meant
  different things by them. §2.4's grammar, which is in a code block where the character
  survives, settles it. Every table writes `&#124;` for a literal bar now.
- **N24** claimed a Constant and a Symbol of the same name print byte-identically. They
  do not: a Constant never goes through `split_subscript`, so `constant("gamma_E")`
  prints `\mathrm{gamma\_E}` while `symbol("gamma_E")` splits and prints
  `\gamma_{E}`. The exception is real; the reason given for it was not.
- **`\frac{a}{b \cdot c}`** was a question the document had not asked.
  `mul({a, b^-1, c^-1})` and `mul({a, (b c)^-1})` print the same string, so one of them
  cannot read back as itself. The parser distributes, keeping the shape a canonical node
  actually has; the other is N28, whose stated consequence is that `\frac{d}{d \cdot x}`
  reads as `1/x`.

That is what writing the tests from the document rather than from the implementation
buys. A test read off a parser agrees with that parser's reading of an ambiguous
sentence, and the sentence stays ambiguous.

**An adversarial review of the committed parser found five more, and no crash.** 220,000
fuzzed inputs under ASan and UBSan produced no report, and the depth guard holds to
20,000 nestings on a 256 KB stack. What it did find was two wrong results and three
diagnostics that were worse than useless, all now fixed:

- **`\frac{dy}{dx}` came back `y/x`.** `match_derivative_operator` requires the numerator
  to be exactly `d`, so every Leibniz spelling except `\frac{d}{dx} f` fell through to
  `parse_fraction` -- which distributes the denominator (N28), leaving `mul` to collect
  `d^1 * d^-1` and cancel. `\frac{d^{2}y}{dx^{2}}` came back `d*y/x^2`, carrying a factor
  of `d` the author never wrote, standing exactly where the order of the derivative had
  been. The partial form was already rejected, so the asymmetry was in the parser rather
  than in the subset. Now **A53 / E-LATEX-0045**.
- **`\int x \, dx + 1` came back `integral(d*x^2 + 1)`.** The integrand runs to the end
  of the enclosing group, so text after the differential breaks the backwards scan, no
  differentials are found, and the empty-variable-list exemption -- written for `\int f`,
  which has no differential at all -- re-read `\, dx` as the factors `d` and `x`. Two
  integral signs rejected the same input correctly, so the hole was exactly the one-sign
  case. Now **A54 / E-LATEX-0046**.
- **`(x\right)` was diagnosed as "expected ')' ... found ')'".** `mismatch_site` skipped
  over `\right` unconditionally and pointed at the token after it. That is right only
  when the opener licenses `\right` as its closer's prefix: a bare `(` does not, so the
  one thing wrong with the input was never named and the reader was sent to a perfectly
  good `)` six columns further on. The rule is now "skip the prefix this opener
  licenses", which fixes the mirror case (`\left(x\big)`) at the same time.
- **E-LATEX-0018 handed out advice that parsed to something else.** It told the author of
  `f'(x)` to write `\frac{d}{dx} f(x)` -- but an unmarked juxtaposition before a
  parenthesis is a product (A1), so following it gave the derivative of `f` times `x`,
  with no call in it, and no second diagnostic to say so. It names the `\operatorname`
  form now, and the test takes the advice *out of the message* and parses it, so the two
  cannot drift.
- **`\mathrm{ }` was accepted as `symbol(" ")`**, which §4.1 promises will round-trip and
  which did not: a one-character name printed bare, so the whole printed form was a
  single space, and a single space is empty input. The fence went into the printer rather
  than into a §4.2 row, because the guarantee as §4.1 words it is the one worth having.

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
