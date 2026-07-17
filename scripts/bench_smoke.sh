#!/usr/bin/env bash
# Run performance benchmarks with a short min time for CI smoke.
set -euo pipefail

# All targets from tests/performance/CMakeLists.txt (add_ms_bench).
MS_BENCH_TARGETS=(
    bench_matmul
    bench_fft
    bench_linalg
    bench_repl
    bench_special
    bench_stats
    bench_rng_dispatch
    bench_simd
    bench_signal_linalg
    bench_ode_pde
    bench_special_memory
    bench_optim_symbolic
    bench_frameworks
    bench_distributed_cellai
    bench_poly_domain
    bench_prob
    bench_optim_ml
)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    BUILD_DIR="${1:-build-bench}"
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
        "${path}" --benchmark_min_time=0.01s
    done

    echo "Benchmark smoke OK"
fi
