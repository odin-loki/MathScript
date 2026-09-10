# Status

**Generated** by `scripts/gen_status.py` at 2026-09-10 02:40 UTC from commit `345e025`.
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
| Branches | 57.3% | not measured | 57.3% |

Three columns, three different things. The measurement is what this build reported.
The CI gate is the fixed minimum `ci.yml` sets. The ratchet floor is the previous
committed measurement, which `scripts/coverage_ratchet.py` fails on a drop below by
more than 0.5 points.

They are separate on purpose. The README once claimed CI enforced 90% while
`ci.yml` set 80%, and nothing reconciled them.

Read the branch row before quoting the line row. This tree's largest files are
dispatch chains, and a dispatch chain reaches high line coverage with one branch of
each test taken.

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
