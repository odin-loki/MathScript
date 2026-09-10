# Status

**Generated** by `scripts/gen_status.py` at 2026-09-10 04:40 UTC from commit `b1951d9`.
Do not edit by hand; run the script.

Every figure here is read from a build artefact. Anything the script could not
read says "not measured" rather than carrying a value forward, because a number that
is absent is honest and a stale one is not. This file exists because seven
published claims were checkable in five minutes and wrong.

## Tests

| | |
|---|---|
| CTest suites | 336 |
| Source lines (`src` + `include`, excluding vendor) | 153,290 |

## Coverage

Measured over the denominator declared in `scripts/coverage_exclusions.txt`.
Only paths that *cannot* execute on a CI runner are excluded, and each run
prints how many lines they hid.

| | Measured | CI gate | Ratchet floor |
|---|---|---|---|
| Lines | 91.2% | 80% | 91.2% |
| Functions | 98.3% | not measured | 98.3% |
| Branches (raw gcov) | 57.3% | not measured | 57.3% |
| Branches (decision lines only) | 71.4% | — | — |

Three columns, three different things. The measurement is what this build reported.
The CI gate is the fixed minimum `ci.yml` sets. The ratchet floor is the previous
committed measurement, which `scripts/coverage_ratchet.py` fails on a drop below by
more than 0.5 points.

They are separate on purpose. The README once claimed CI enforced 90% while
`ci.yml` set 80%, and nothing reconciled them.

Read the branch rows before quoting the line row. This tree's largest files are
dispatch chains, and a dispatch chain reaches high line coverage with one branch of
each test taken.

The two branch rows measure different denominators and **neither replaces the
other**. The raw gcov figure counts every edge gcov emits, including those inside
library code inlined into our lines -- `std::vector` growth and allocation-failure
arms, `std::string` short/long checks -- which is 43% of the denominator and largely
unreachable from a test. The decision-line figure counts only slots on a line
containing `if`, `while`, `for`, `switch`, `&&`, `||` or `?`. The first understates
how well this project's logic is tested; the second ignores real edges the compiler
generated. Quoting only the flattering one is how a 92.0% line figure measured over
75% of the repository came to be published.

## Benchmarks

| | |
|---|---|
| Baseline entries | 445 |
| Of which carry a measured median | 5 |
| Regression tolerance | 10% |

A large gap between those first two rows means the comparison is skipping most
of the suite. See `docs/RELEASE.md` criterion 8 for why that is deliberate.

## Compliance

| | |
|---|---|
| Reviewed unsafe sites | 33 |
| SPDX headers | enforced by `scripts/add_spdx.py --check` |
| SBOM | `sbom.cdx.json`, checked by `scripts/gen_sbom.py --check` |
| Formal verification | `verification/run.sh` under ESBMC and CBMC |

## Release

| | |
|---|---|
| `v1.0.0` tag | not cut |

Tag criteria are in [`RELEASE.md`](RELEASE.md).
