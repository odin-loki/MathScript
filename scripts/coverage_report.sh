#!/usr/bin/env bash
# Generate an lcov report from a coverage-instrumented MathScript build tree.
#
# Three things this script is careful about, because each was wrong before and
# each moved the published number without moving the work:
#
#   1. Exclusions are declared data in coverage_exclusions.txt, one glob per line
#      with the reason it cannot be reached on a runner. They used to be inline
#      here, and they removed matrix_calls, plugin and ms_bundle -- about a
#      quarter of src/ -- so the percentage was not the project's coverage.
#   2. Branch and function coverage are measured, not only lines. On a codebase
#      whose largest files are dispatch chains, line coverage alone says very
#      little: a chain can be fully "covered" with one branch of each test taken.
#   3. The run reports how many lines the exclusions hid. A list nobody sees is a
#      list that grows.
#
# Gates, each independent and off unless set:
#   MS_COVERAGE_MIN         line percentage
#   MS_COVERAGE_BRANCH_MIN  branch percentage
#   MS_COVERAGE_FUNC_MIN    function percentage
set -euo pipefail

BUILD_DIR="${1:-build-cov}"
MIN_PCT="${MS_COVERAGE_MIN:-0}"
MIN_BRANCH="${MS_COVERAGE_BRANCH_MIN:-0}"
MIN_FUNC="${MS_COVERAGE_FUNC_MIN:-0}"

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "Build directory not found: ${BUILD_DIR}" >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFO="${BUILD_DIR}/coverage.info"
INFO_RAW="${BUILD_DIR}/coverage-raw.info"
EXCLUSIONS="${ROOT}/scripts/coverage_exclusions.txt"

if [[ ! -f "${EXCLUSIONS}" ]]; then
    echo "Exclusion list not found: ${EXCLUSIONS}" >&2
    exit 1
fi

# Strip comments (whole-line and trailing) and blanks.
mapfile -t EXCLUDE_GLOBS < <(
    sed -E 's/[[:space:]]+#.*$//; s/^[[:space:]]*#.*$//' "${EXCLUSIONS}" \
        | sed -E 's/[[:space:]]+$//' \
        | grep -vE '^[[:space:]]*$'
)

if [[ ${#EXCLUDE_GLOBS[@]} -eq 0 ]]; then
    echo "No exclusion globs parsed from ${EXCLUSIONS}" >&2
    exit 1
fi

# --ignore-errors is deliberately narrow. The previous form passed
# "source,gcov,empty,mismatch,unused", which suppressed exactly the errors that
# indicate stale gcov data -- the report would come out clean while describing a
# previous build. "source", "gcov" and "empty" stay off for that reason.
#
# Two are kept, for opposite reasons:
#
#   unused    an exclusion glob that matches nothing is a maintenance signal, not
#             a failure, and it is reported below instead of aborting the run.
#
#   mismatch  not a staleness signal at all, and removing it was a mistake that
#             cost a CI cycle. This tree builds the library with -fno-exceptions
#             and gives the test executables -fexceptions under coverage, because
#             GoogleTest's EXPECT_NO_THROW expands to try/catch. The same inline
#             function in a libstdc++ header therefore carries different exception
#             tags in different objects, and lcov 2.x treats that as a hard error:
#
#               geninfo: ERROR: bits/stl_construct.h:97: mismatched exception tag
#
#             The line counts either side of the mismatch are correct; what
#             differs is metadata about a deliberate configuration. Suppressing it
#             hides nothing about whether the data is current.
LCOV_IGNORE="unused,mismatch"

lcov --quiet --capture --directory "${BUILD_DIR}" --output-file "${INFO_RAW}" \
    --rc lcov_branch_coverage=1 --ignore-errors "${LCOV_IGNORE}"

lines_in() {
    lcov --summary "$1" 2>/dev/null \
        | sed -n 's/.*lines\.*: *[0-9.]*% *(\([0-9]*\) of \([0-9]*\) lines).*/\2/p'
}

RAW_LINES="$(lines_in "${INFO_RAW}")"

lcov --quiet --remove "${INFO_RAW}" "${EXCLUDE_GLOBS[@]}" \
    --ignore-errors "${LCOV_IGNORE}" \
    --output-file "${INFO}"

KEPT_LINES="$(lines_in "${INFO}")"
HIDDEN=$(( ${RAW_LINES:-0} - ${KEPT_LINES:-0} ))

echo "=== Coverage exclusions ==="
printf '  %s\n' "${EXCLUDE_GLOBS[@]}"
if [[ "${RAW_LINES:-0}" -gt 0 ]]; then
    printf '  hidden: %s of %s instrumented lines (%s%%)\n' \
        "${HIDDEN}" "${RAW_LINES}" "$(( HIDDEN * 100 / RAW_LINES ))"
fi
echo

echo "=== Coverage summary ==="
lcov --rc lcov_branch_coverage=1 --summary "${INFO}" 2>&1 \
    | tee "${BUILD_DIR}/coverage-summary.txt"

SUMMARY="$(lcov --rc lcov_branch_coverage=1 --summary "${INFO}" 2>&1)"

pct_of() {
    echo "${SUMMARY}" | sed -n "s/.*$1\.*: *\([0-9.]*\)%.*/\1/p" | head -1
}

PCT="$(pct_of lines)"
PCT_FUNC="$(pct_of functions)"
PCT_BRANCH="$(pct_of branches)"

if [[ -z "${PCT}" ]]; then
    echo "Could not parse line coverage from the summary." >&2
    exit 1
fi

# Per-file ranking by uncovered lines: the list that says where to work next.
if [[ -x "${ROOT}/scripts/cov_rank.py" ]] || [[ -f "${ROOT}/scripts/cov_rank.py" ]]; then
    python3 "${ROOT}/scripts/cov_rank.py" "${INFO}" \
        > "${BUILD_DIR}/coverage-ranked.txt" 2>/dev/null \
        && echo "Ranked per-file report: ${BUILD_DIR}/coverage-ranked.txt" \
        || true
fi

echo
printf 'Line coverage:     %s%% (minimum requested: %s%%)\n' "${PCT}" "${MIN_PCT}"
printf 'Function coverage: %s%% (minimum requested: %s%%)\n' "${PCT_FUNC:-n/a}" "${MIN_FUNC}"
printf 'Branch coverage:   %s%% (minimum requested: %s%%)\n' "${PCT_BRANCH:-n/a}" "${MIN_BRANCH}"

FAILED=0
check_gate() {
    local name="$1" value="$2" minimum="$3"
    if [[ -z "${value}" || "${value}" == "n/a" ]]; then
        if awk -v m="${minimum}" 'BEGIN { exit !(m + 0 > 0) }'; then
            echo "${name} coverage was requested at ${minimum}% but not measured." >&2
            FAILED=1
        fi
        return
    fi
    if awk -v v="${value}" -v m="${minimum}" 'BEGIN { exit !(v + 0 < m + 0) }'; then
        printf '%s coverage %.2f%% is below minimum %.2f%%\n' \
            "${name}" "${value}" "${minimum}" >&2
        FAILED=1
    fi
}

check_gate "Line" "${PCT}" "${MIN_PCT}"
check_gate "Function" "${PCT_FUNC}" "${MIN_FUNC}"
check_gate "Branch" "${PCT_BRANCH}" "${MIN_BRANCH}"

exit "${FAILED}"
