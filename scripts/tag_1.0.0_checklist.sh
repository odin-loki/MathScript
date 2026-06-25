#!/usr/bin/env bash
# Pre-tag gate for MathScript v1.0.0 (read-only checklist; does not modify version).
#
# Usage:
#   bash scripts/tag_1.0.0_checklist.sh
#
# After fuzz-24h.yml completes with zero crashes:
#   1. Update project(MathScript VERSION ...) in CMakeLists.txt to 1.0.0
#   2. Re-run cmake configure (regenerates include/ms/version.hpp)
#   3. Update CHANGELOG.md [Unreleased] -> [1.0.0] with date
#   4. bash scripts/pre_release.sh
#   5. Confirm CI green on main, then: git tag v1.0.0 && git push origin v1.0.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_LINE="$(grep -E '^project\(MathScript VERSION ' "${ROOT}/CMakeLists.txt" || true)"

echo "=== MathScript v1.0.0 tag checklist ==="
echo "Current: ${VERSION_LINE:-unknown}"
echo
echo "Required before tagging (see docs/RELEASE.md):"
echo "  [ ] CI green on main (.github/workflows/ci.yml)"
echo "  [ ] 351 CTest suites passing"
echo "  [ ] Coverage >= 90% (coverage-linux)"
echo "  [ ] Valgrind clean (valgrind-linux)"
echo "  [ ] fuzz-24h.yml: 86400 s x 7 targets, zero crashes"
echo "  [ ] Verify gh run list --workflow=fuzz-24h.yml shows 7 completed with no crashes"
echo "  [ ] plugin-linux + jit-linux green"
echo "  [ ] benchmark-linux within tolerance"
echo "  [ ] unsafe_delta clean"
echo "  [ ] CHANGELOG + docs current"
echo "  [ ] Packaging smoke (scripts/package_smoke.sh or scripts/package_smoke.ps1)"
echo
echo "Fuzz marathon: bash scripts/fuzz_24h_dispatch.sh"
echo "Local gate:    bash scripts/pre_release.sh"
echo "Packaging:     bash scripts/package_smoke.sh build-linux install-smoke"
echo "Windows:       .\\build.ps1 -Test ; .\\scripts\\tag_1.0.0_checklist.ps1"
echo "               pwsh -NoProfile -File scripts/package_smoke.ps1 build-msvc install-smoke"
echo
if [[ "${VERSION_LINE}" == *"1.0.0"* ]]; then
    echo "NOTE: CMakeLists.txt already references 1.0.0 — verify fuzz marathon completed."
else
    echo "NOTE: Version not yet 1.0.0 in CMakeLists.txt (expected until fuzz marathon passes)."
fi
