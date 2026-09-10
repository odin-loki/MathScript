# Status

**Generated** by `scripts/gen_status.py` at 2026-09-10 00:05 UTC from commit `60a9674`.
Do not edit by hand; run the script.

Every figure here is read from a build artefact. Anything the script could not
read says "not measured" rather than carrying a value forward, because a number that
is absent is honest and a stale one is not. This file exists because seven
published claims were checkable in five minutes and wrong.

## Tests

| | |
|---|---|
| CTest suites | 875 |
| Source lines (`src` + `include`, excluding vendor) | 151,124 |

## Coverage

Measured over the denominator declared in `scripts/coverage_exclusions.txt`.
Only paths that *cannot* execute on a CI runner are excluded, and each run
prints how many lines they hid.

| | Measured | CI gate |
|---|---|---|
| Lines | not measured | 80% |
| Functions | not measured | not measured |
| Branches | not measured | not measured |

The gate and the measurement are separate columns on purpose. The README once
claimed CI enforced 90% while `ci.yml` set 80%, and nothing reconciled them.

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
