#!/usr/bin/env bash
# Print add_ms_bench target names from tests/performance/CMakeLists.txt.
# Sourceable: sets MS_BENCH_TARGETS array.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE_LISTS="${ROOT}/tests/performance/CMakeLists.txt"

mapfile -t MS_BENCH_TARGETS < <(
    grep -E '^\s*add_ms_bench\s*\(' "${CMAKE_LISTS}" \
        | sed -E 's/^\s*add_ms_bench\s*\(\s*([[:alnum:]_]+).*/\1/'
)

if [[ ${#MS_BENCH_TARGETS[@]} -eq 0 ]]; then
    echo "No add_ms_bench targets found in ${CMAKE_LISTS}" >&2
    exit 1
fi

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    printf '%s\n' "${MS_BENCH_TARGETS[@]}"
fi
