# MathScript 1.0.0 Release Checklist

Version **1.0.0** is tagged when Phase 10 (Hardening) is complete. `README.md` has a short
**Phase progress** summary table; the authoritative, detailed tag criteria are the numbered
list below. For the original phase-by-phase plan and current known gaps, see
`mathscript-master-plan.md` §15.12.1.

## Tag criteria (summary)

All of the following must be true before `git tag v1.0.0`:

1. **CI green** — every job in `.github/workflows/ci.yml` passes on `main` without `continue-on-error` overrides.
2. **Tests** — full CTest suite passing (currently **368** suites; CUDA optional/off in CI).
3. **Coverage** — line coverage ≥ **90%** enforced in CI (target **~91%**; see `coverage-linux` job).
4. **Memory** — Valgrind memcheck clean on the test suite (`valgrind-linux` job).
5. **Fuzz** — all libFuzzer targets run ≥ **24 hours** total with **no crashes** (see `fuzz-24h.yml` and nightly workflows).
6. **Unsafe surface** — every site in `UNSAFE_REVIEW.md` reviewed; `scripts/unsafe_report.sh` and `scripts/unsafe_delta.sh` report no new unreviewed sites.
7. **Packaging** — install smoke + TGZ/ZIP/DEB/RPM (Linux) and ZIP (+ optional NSIS/WiX) on Windows; verified via `scripts/package_smoke.sh` / `scripts/package_smoke.ps1`.
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

## Pre-tag commands (Windows packaging)

After a Release configure/build of `build-msvc`:

```powershell
pwsh -NoProfile -File scripts/package_smoke.ps1 build-msvc install-smoke
```

This runs `cmake --install`, checks the install tree, then `cpack -G ZIP` and verifies `mathscript-*.zip` in the build directory. On Linux, `scripts/package_smoke.sh` also runs `cpack -G TGZ` and verifies `mathscript-*.tar.gz`. Equivalent manual step: `cmake --build build-msvc --target package` (generates all configured CPack artifacts).

## Fuzz marathon (required before v1.0.0)

The nightly workflow runs **15 min × 7** targets on schedule. Manual **`fuzz-marathon`** runs **60 min × 7**. Phase 10 exit requires the separate **24 h × 7** campaign with zero crashes:

1. Open GitHub → Actions → **Fuzz 24h** → **Run workflow**, or:
   ```bash
   gh workflow run fuzz-24h.yml
   gh run list --workflow=fuzz-24h.yml
   ```
2. Wait for all **7** matrix jobs to complete (`-max_total_time=86400` per target; job timeout **1500 min**).
3. Confirm no crashes in any job log before tagging.

Helper: `bash scripts/fuzz_24h_dispatch.sh` (wraps `gh workflow run` + prints monitor command).

Pre-tag read-only gate: `bash scripts/tag_1.0.0_checklist.sh` (does not bump version).

## Final release procedure

Run these steps **in order** after tag criteria are met:

1. **Fuzz marathon** — `gh workflow run fuzz-24h.yml`; confirm **zero crashes** in all **7** jobs (`gh run list --workflow=fuzz-24h.yml`).
2. **Version bump** — set `project(MathScript VERSION …)` in `CMakeLists.txt` to **1.0.0**, reconfigure, then:
   - Windows: `.\build.ps1 -Test`
   - Linux: `ctest --test-dir build-linux --output-on-failure`
3. **Changelog** — edit `CHANGELOG.md`: change `## [Unreleased]` to `## [1.0.0] - <YYYY-MM-DD>` (tag date) and remove the `### Remaining (Phase 10)` section if still present.
4. **Pre-release gate** — `bash scripts/pre_release.sh` (Linux); review warnings.
5. **Push** — push to `main` and confirm CI is green on the push.
6. **Tag** — `git tag v1.0.0 && git push origin v1.0.0`

## Post-tag

After `v1.0.0` is on `main`:

1. **GitHub release** — create a release from the `v1.0.0` tag (GitHub → Releases → Draft a new release → choose tag `v1.0.0`).
2. **Artifacts** — download `mathscript-linux-tgz` / `mathscript-windows-zip` from the green `build-test-linux` / `build-test-windows` CI jobs (uploaded after `package_smoke`), or build locally with `cpack` and upload.
3. **Announce** — publish release notes summarizing Phase 10 hardening and link to `CHANGELOG.md` for the full **1.0.0** change list.

## Post-1.0.0 items

- Extended fuzz (**24 h × 7**) — manual campaign via `gh workflow run fuzz-24h.yml`
- Full ORC JIT v2 matrix LLVM IR lowering (enhancement)
- Windows installer / Linux packages (packaging)
