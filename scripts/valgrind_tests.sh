#!/usr/bin/env bash
# Run the MathScript test suite under Valgrind (Linux only).
set -euo pipefail

BUILD_DIR="${1:-build-linux}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "${ROOT}/${BUILD_DIR}" && ! -d "${BUILD_DIR}" ]]; then
    echo "Build directory not found: ${BUILD_DIR}" >&2
    exit 1
fi

if [[ -d "${BUILD_DIR}" ]]; then
    TEST_DIR="${BUILD_DIR}"
else
    TEST_DIR="${ROOT}/${BUILD_DIR}"
fi

if ! command -v valgrind >/dev/null 2>&1; then
    echo "valgrind not found in PATH" >&2
    exit 1
fi

SUPPRESSIONS="${ROOT}/cmake/valgrind.supp"
MEMCHECK_OPTS=(
    --leak-check=full
    --errors-for-leak-kinds=definite,indirect
    --error-exitcode=1
    --show-leak-kinds=definite,indirect
)
if [[ -f "${SUPPRESSIONS}" ]]; then
    MEMCHECK_OPTS+=(--suppressions="${SUPPRESSIONS}")
fi

export CTEST_MEMORYCHECK_COMMAND=valgrind
export CTEST_MEMORYCHECK_COMMAND_OPTIONS="${MEMCHECK_OPTS[*]}"

echo "Valgrind memcheck in ${TEST_DIR} (excluding test_fuzz_stress)"
ctest --test-dir "${TEST_DIR}" -T memcheck -E test_fuzz_stress --output-on-failure -j1
