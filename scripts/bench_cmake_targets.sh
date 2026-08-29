#!/usr/bin/env bash
# Collect bench_* executable names from tests/performance/bench_*.cpp.
# Sourceable: sets MS_BENCH_TARGETS array.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${ROOT}/tests/performance"

mapfile -t MS_BENCH_TARGETS < <(
    find "${BENCH_DIR}" -name 'bench_*.cpp' \
        | sed 's|.*/||; s|\.cpp$||' | sort
)

if [[ ${#MS_BENCH_TARGETS[@]} -eq 0 ]]; then
    echo "No bench_*.cpp targets found in ${BENCH_DIR}" >&2
    exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    printf '%s\n' "${MS_BENCH_TARGETS[@]}"
fi
