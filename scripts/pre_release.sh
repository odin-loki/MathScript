#!/usr/bin/env bash
# Local pre-release checklist (see docs/RELEASE.md). Warn-only — always exits 0.
#
# Linux parity build (run once before this script):
#   cmake -S . -B build-linux -G Ninja \
#     -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
#   cmake --build build-linux
#
# Optional coverage tree (90% gate):
#   cmake -S . -B build-cov -G Ninja \
#     -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
#     -DCMAKE_BUILD_TYPE=Debug -DMS_ENABLE_COVERAGE=ON \
#     -DMS_BUILD_TESTS=ON -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
#   cmake --build build-cov && ctest --test-dir build-cov --output-on-failure
#
# Optional plugin compliance (mirrors plugin-linux CI job):
#   cmake -S . -B build-plugin -G Ninja \
#     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DMS_BUILD_TESTS=ON -DMS_BUILD_PLUGIN=ON \
#     -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF \
#     -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
#     -DClang_DIR=/usr/lib/llvm-18/lib/cmake/clang
#   cmake --build build-plugin --target ms_plugin test_plugin_smoke
#   ctest --test-dir build-plugin -R 'test_plugin_smoke|compliance_' --output-on-failure
#
# Optional benchmark regression (mirrors benchmark-linux CI job):
#   cmake -S . -B build-bench -G Ninja \
#     -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DMS_BUILD_TESTS=OFF -DMS_BUILD_BENCHMARKS=ON \
#     -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
#   cmake --build build-bench --target $(bash scripts/bench_cmake_targets.sh | xargs)
#   (28 add_ms_bench targets: bench_matmul bench_fft bench_linalg bench_repl bench_special
#    bench_stats bench_rng_dispatch bench_simd bench_signal_linalg bench_signal_filters
#    bench_ode_pde bench_fem bench_special_memory bench_optim_symbolic bench_frameworks
#    bench_tensorops bench_distributed_cellai bench_poly_domain bench_prob bench_optim_ml
#    bench_crypto bench_graph bench_topo bench_image bench_geo bench_quantum bench_compress
#    bench_finance)
#   MS_BENCH_REGRESSION=on MS_BENCH_TOLERANCE=10 bash scripts/bench_regression.sh build-bench
#
# Phase 10 fuzz marathon (not run by this script — trigger in GitHub Actions):
#   workflow: .github/workflows/fuzz-24h.yml (workflow_dispatch)
#   bash scripts/fuzz_24h_dispatch.sh   # or: gh workflow run fuzz-24h.yml
#   7 parallel jobs × 86400 s (24 h) per libFuzzer target; required before v1.0.0 tag.
#
# Env overrides: MS_PRE_RELEASE_BUILD, MS_PRE_RELEASE_COV_BUILD, MS_PRE_RELEASE_PLUGIN_BUILD,
#   MS_PRE_RELEASE_BENCH_BUILD, MS_COVERAGE_MIN (default 90), MS_BENCH_TOLERANCE (default 10)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${MS_PRE_RELEASE_BUILD:-build-linux}"
COV_DIR="${MS_PRE_RELEASE_COV_BUILD:-build-cov}"
PLUGIN_DIR="${MS_PRE_RELEASE_PLUGIN_BUILD:-build-plugin}"
BENCH_DIR="${MS_PRE_RELEASE_BENCH_BUILD:-build-bench}"
MIN_PCT="${MS_COVERAGE_MIN:-90}"
BENCH_TOLERANCE="${MS_BENCH_TOLERANCE:-10}"
WARNINGS=0

warn() {
    echo "WARN: $*" >&2
    WARNINGS=$((WARNINGS + 1))
}

note() {
    echo "NOTE: $*"
}

resolve_dir() {
    local dir="$1"
    if [[ "${dir}" = /* ]]; then
        echo "${dir}"
    elif [[ -d "${dir}" ]]; then
        echo "${dir}"
    elif [[ -d "${ROOT}/${dir}" ]]; then
        echo "${ROOT}/${dir}"
    else
        echo "${dir}"
    fi
}

BUILD_PATH="$(resolve_dir "${BUILD_DIR}")"
COV_PATH="$(resolve_dir "${COV_DIR}")"
PLUGIN_PATH="$(resolve_dir "${PLUGIN_DIR}")"
BENCH_PATH="$(resolve_dir "${BENCH_DIR}")"

echo "=== MathScript pre-release checklist ==="
echo "Release build: ${BUILD_PATH}"
echo "Coverage build: ${COV_PATH} (min ${MIN_PCT}%)"
echo "Plugin build: ${PLUGIN_PATH} (optional; mirrors plugin-linux CI)"
echo "Benchmark build: ${BENCH_PATH} (optional; mirrors benchmark-linux CI)"
echo "Fuzz marathon: trigger fuzz-24h.yml in GitHub Actions before tagging (see docs/RELEASE.md)"
echo

echo "--- 1/5: CTest (57 suites, CUDA off) ---"
if [[ ! -d "${BUILD_PATH}" ]]; then
    warn "Build directory missing: ${BUILD_PATH} — configure per header comments or docs/RELEASE.md"
else
    if ! ctest --test-dir "${BUILD_PATH}" --output-on-failure; then
        warn "ctest failed in ${BUILD_PATH}"
    else
        echo "ctest OK"
    fi
fi
echo

echo "--- 2/5: Coverage (Linux, min ${MIN_PCT}%) ---"
if [[ ! -d "${COV_PATH}" ]]; then
    note "Coverage build not found (${COV_PATH}); skipping — see script header for configure steps"
elif ! command -v lcov >/dev/null 2>&1; then
    note "lcov not installed; skipping coverage report"
else
    if ! MS_COVERAGE_MIN="${MIN_PCT}" bash "${ROOT}/scripts/coverage_report.sh" "${COV_PATH}"; then
        warn "Coverage below ${MIN_PCT}% or report failed (${COV_PATH})"
    else
        echo "Coverage OK (>= ${MIN_PCT}%)"
    fi
fi
echo

echo "--- 3/5: Plugin compliance (plugin-linux parity) ---"
if [[ ! -d "${PLUGIN_PATH}" ]]; then
    note "Plugin build not found (${PLUGIN_PATH}); skipping — see script header for configure steps (LLVM 18 + Clang)"
else
    if ! ctest --test-dir "${PLUGIN_PATH}" -R 'test_plugin_smoke|compliance_' --output-on-failure; then
        warn "Plugin/compliance ctest failed in ${PLUGIN_PATH} (plugin-linux CI job would fail)"
    else
        echo "Plugin compliance OK"
    fi
fi
echo

echo "--- 4/5: Benchmark regression (benchmark-linux parity) ---"
if [[ ! -d "${BENCH_PATH}" ]]; then
    note "Benchmark build not found (${BENCH_PATH}); skipping — see script header for configure steps"
else
    if ! MS_BENCH_REGRESSION=on MS_BENCH_TOLERANCE="${BENCH_TOLERANCE}" bash "${ROOT}/scripts/bench_regression.sh" "${BENCH_PATH}"; then
        warn "Benchmark regression failed in ${BENCH_PATH} (benchmark-linux CI job would fail)"
    else
        echo "Benchmark regression OK"
    fi
fi
echo

echo "--- 5/5: Unsafe surface delta ---"
UNSAFE_REPORT="${BUILD_PATH}/unsafe_report.txt"
if [[ ! -d "${BUILD_PATH}" ]]; then
    warn "Cannot run unsafe delta — build directory missing"
elif [[ ! -f "${UNSAFE_REPORT}" ]]; then
    note "Generating unsafe report: ${UNSAFE_REPORT}"
    if ! bash "${ROOT}/scripts/unsafe_report.sh" "${UNSAFE_REPORT}"; then
        warn "unsafe_report.sh failed"
    elif ! bash "${ROOT}/scripts/unsafe_delta.sh" "${UNSAFE_REPORT}"; then
        warn "unsafe_delta.sh reported new unreviewed sites"
    else
        echo "Unsafe delta OK"
    fi
else
    if ! bash "${ROOT}/scripts/unsafe_delta.sh" "${UNSAFE_REPORT}"; then
        warn "unsafe_delta.sh reported new unreviewed sites"
    else
        echo "Unsafe delta OK"
    fi
fi
echo

if [[ "${WARNINGS}" -gt 0 ]]; then
    echo "Pre-release checklist finished with ${WARNINGS} warning(s). See docs/RELEASE.md before tagging v1.0.0."
else
    echo "Pre-release checklist passed locally (CI still required: coverage-linux, valgrind-linux, fuzz-linux, plugin-linux, benchmark-linux; plus manual fuzz-24h.yml before v1.0.0)."
fi

exit 0
