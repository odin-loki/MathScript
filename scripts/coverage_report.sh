#!/usr/bin/env bash
# Generate an lcov report from a coverage-instrumented MathScript build tree.
set -euo pipefail

BUILD_DIR="${1:-build-cov}"
MIN_PCT="${MS_COVERAGE_MIN:-0}"

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "Build directory not found: ${BUILD_DIR}" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFO="${BUILD_DIR}/coverage.info"

lcov --quiet --capture --directory "${BUILD_DIR}" --output-file "${INFO}" \
    --rc lcov_branch_coverage=0 --ignore-errors source,gcov,empty

lcov --quiet --remove "${INFO}" '/usr/*' '*/tests/*' '*/googletest/*' \
    --output-file "${INFO}"

echo "=== Line coverage summary ==="
lcov --summary "${INFO}" 2>&1 | tee "${BUILD_DIR}/coverage-summary.txt"

LINE_SUMMARY="$(lcov --summary "${INFO}" 2>&1 | grep -E '^  lines\.\.\.\.\.\.' || true)"
if [[ -z "${LINE_SUMMARY}" ]]; then
    echo "Could not parse line coverage summary." >&2
    exit 1
fi

# Example: "  lines......: 72.3% (1234 of 1706 lines)"
PCT="$(echo "${LINE_SUMMARY}" | sed -n 's/.*: \([0-9.]*\)%.*/\1/p')"
if [[ -z "${PCT}" ]]; then
    echo "Could not parse coverage percentage from: ${LINE_SUMMARY}" >&2
    exit 1
fi

echo "Line coverage: ${PCT}% (minimum requested: ${MIN_PCT}%)"

awk -v pct="${PCT}" -v min="${MIN_PCT}" 'BEGIN {
    if (pct + 0 < min + 0) {
        printf("Coverage %.2f%% is below minimum %.2f%%\n", pct, min) > "/dev/stderr";
        exit 1;
    }
}'
