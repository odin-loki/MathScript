#!/usr/bin/env bash
# Run performance benchmarks with a short min time for CI smoke.
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
    BUILD_DIR="${1:-build-bench}"

    if [[ -d "${BUILD_DIR}" ]]; then
        BUILD_PATH="${BUILD_DIR}"
    else
        BUILD_PATH="${ROOT}/${BUILD_DIR}"
    fi

    if [[ ! -d "${BUILD_PATH}" ]]; then
        echo "Build directory not found: ${BUILD_DIR}" >&2
        exit 1
    fi

    for bench in "${MS_BENCH_TARGETS[@]}"; do
        path="${BUILD_PATH}/tests/performance/${bench}"
        if [[ ! -x "${path}" ]]; then
            echo "Missing benchmark executable: ${path}" >&2
            exit 1
        fi
        "${path}" --benchmark_min_time=0.001s
    done

    echo "Benchmark smoke OK (${#MS_BENCH_TARGETS[@]} executables)"
fi
