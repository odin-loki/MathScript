# MathScript 1.0.0 Release Checklist

Version **1.0.0** is tagged when Phase 10 (Hardening) is complete. The authoritative
item list lives in the README under **[Phase 10 checklist](../README.md#phase-10-checklist)**.

## Tag criteria (summary)

All of the following must be true before `git tag v1.0.0`:

1. **CI green** — every job in `.github/workflows/ci.yml` passes on `main` without `continue-on-error` overrides.
2. **Tests** — full CTest suite passing (currently **57** suites; CUDA optional/off in CI).
3. **Coverage** — line coverage ≥ **90%** enforced in CI (target **~91%**; see `coverage-linux` job).
4. **Memory** — Valgrind memcheck clean on the test suite (`valgrind-linux` job).
5. **Fuzz** — all libFuzzer targets run ≥ **24 hours** total with **no crashes** (see `fuzz-24h.yml` and nightly workflows).
6. **Unsafe surface** — every site in `UNSAFE_REVIEW.md` reviewed; `scripts/unsafe_report.sh` and `scripts/unsafe_delta.sh` report no new unreviewed sites.
7. **Packaging** — install smoke + TGZ/ZIP/DEB/RPM (Linux) and NSIS/WiX (Windows) build on clean CI runners.
8. **Performance** — `benchmark-linux` regression gate within **10%** of `linux-gcc13.json` baseline.
9. **Compliance** — `plugin-linux` job green with `MS_BUILD_PLUGIN=ON` and **twenty** enforced compile-fail rules (all `DiagnosticID`s in `diagnostics.hpp` except partial `UnsafeAudit` via `MS_UNSAFE` + audit scripts) plus compliance tests passing.
10. **JIT** — `jit-linux` job green with `-DMS_BUILD_JIT=ON` (ORC LLJIT smoke; scalar literal/expression/libm-call JIT; native dispatch for matrix/scalar/multi-target REPL call assignments).
11. **Documentation** — `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/CONTRIBUTING.md`, and `CHANGELOG.md` current for the release.

## Pre-tag commands (Linux CI parity)

```bash
cmake -S . -B build-linux -G Ninja \
  -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
cmake --build build-linux
ctest --test-dir build-linux --output-on-failure
bash scripts/unsafe_report.sh build-linux/unsafe_report.txt
bash scripts/unsafe_delta.sh build-linux/unsafe_report.txt
bash scripts/package_smoke.sh build-linux install-smoke
```

## Fuzz marathon (required before v1.0.0)

The nightly workflow runs **15 min × 7** targets for smoke. Phase 10 exit requires the manual **24 h × 7** campaign with zero crashes:

1. Open GitHub → Actions → **Fuzz 24h** → **Run workflow**, or:
   ```bash
   gh workflow run fuzz-24h.yml
   gh run list --workflow=fuzz-24h.yml
   ```
2. Wait for all **7** matrix jobs to complete (`-max_total_time=86400` per target; job timeout **1500 min**).
3. Confirm no crashes in any job log before tagging.

Helper: `bash scripts/fuzz_24h_dispatch.sh` (wraps `gh workflow run` + prints monitor command).

Pre-tag read-only gate: `bash scripts/tag_1.0.0_checklist.sh` (does not bump version).

## Remaining before 1.0.0

See README **Remaining** under the Phase 10 checklist: extended 24 h fuzz marathon,
full ORC JIT v2 matrix LLVM IR lowering (native dispatch for all REPL call forms today),
and version bump to **1.0.0** after fuzz marathon completes.
