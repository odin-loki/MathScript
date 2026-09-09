#!/usr/bin/env bash
# Run the Phase 10 fuzz marathon (24 h x 7 libFuzzer targets) on this machine.
#
# The companion script fuzz_24h_dispatch.sh sends the same marathon to GitHub
# Actions, where each target gets its own 2-core runner. On a workstation the
# whole matrix fits at once and every spare core can be put behind it, which is
# the faster way to get the criterion-5 evidence.
#
#   scripts/fuzz_24h_local.sh                  # 24 h, cores split across targets
#   MS_FUZZ_SECONDS=3600 scripts/fuzz_24h_local.sh    # a 1 h rehearsal
#   MS_FUZZ_WORKERS=4 scripts/fuzz_24h_local.sh       # 4 workers per target
#   MS_FUZZ_BUILD_DIR=build-fuzz scripts/fuzz_24h_local.sh
#
# Findings land in tests/fuzz/corpus/<target>/ (new coverage) and in
# artifacts/<target>/ (the crashing input for anything that fails).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${MS_FUZZ_BUILD_DIR:-${ROOT}/build-fuzz-24h}"
SECONDS_PER_TARGET="${MS_FUZZ_SECONDS:-86400}"
# Match the CI rss limit so an out-of-memory finding reproduces identically.
RSS_LIMIT_MB="${MS_FUZZ_RSS_LIMIT_MB:-2048}"
OUT_DIR="${MS_FUZZ_OUT_DIR:-${ROOT}/fuzz-runs/$(date -u +%Y%m%dT%H%M%SZ)}"

TARGETS=(
    fuzz_special_fns
    fuzz_matrix_ops
    fuzz_repl_input
    fuzz_sym_parser
    fuzz_poly_ops
    fuzz_bignum
    fuzz_mpi_message
)

NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
# One worker per core, split evenly, but never fewer than one.
DEFAULT_WORKERS=$(( NPROC / ${#TARGETS[@]} ))
[[ "${DEFAULT_WORKERS}" -lt 1 ]] && DEFAULT_WORKERS=1
WORKERS="${MS_FUZZ_WORKERS:-${DEFAULT_WORKERS}}"

CC_BIN="${MS_FUZZ_CC:-clang}"
CXX_BIN="${MS_FUZZ_CXX:-clang++}"

if ! command -v "${CXX_BIN}" >/dev/null 2>&1; then
    echo "error: ${CXX_BIN} not found. libFuzzer needs Clang; set MS_FUZZ_CXX." >&2
    exit 1
fi

echo "MathScript fuzz marathon"
echo "  targets      : ${#TARGETS[@]}"
echo "  duration     : ${SECONDS_PER_TARGET}s per target"
echo "  workers      : ${WORKERS} per target (${NPROC} cores detected)"
echo "  rss limit    : ${RSS_LIMIT_MB} MB"
echo "  build dir    : ${BUILD_DIR}"
echo "  run output   : ${OUT_DIR}"
echo

if [[ ! -x "${BUILD_DIR}/tests/fuzz/${TARGETS[0]}" ]]; then
    echo "Configuring ${BUILD_DIR} ..."
    cmake -S "${ROOT}" -B "${BUILD_DIR}" -G Ninja \
        -DCMAKE_C_COMPILER="${CC_BIN}" -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DMS_BUILD_FUZZ=ON -DMS_BUILD_TESTS=ON \
        -DMS_ENABLE_CUDA=OFF -DMS_ENABLE_AVX512=OFF
    echo "Building ${#TARGETS[@]} fuzz targets ..."
    cmake --build "${BUILD_DIR}" -j"${NPROC}" --target "${TARGETS[@]}"
fi

mkdir -p "${OUT_DIR}"

start_epoch="$(date -u +%s)"
pids=()
for target in "${TARGETS[@]}"; do
    bin="${BUILD_DIR}/tests/fuzz/${target}"
    if [[ ! -x "${bin}" ]]; then
        echo "error: ${bin} missing. Build it or unset MS_FUZZ_BUILD_DIR." >&2
        exit 1
    fi
    corpus="${ROOT}/tests/fuzz/corpus/${target}"
    mkdir -p "${corpus}" "${OUT_DIR}/${target}"

    # -artifact_prefix keeps each target's crashing inputs in its own directory.
    # The corpus directory is passed positionally so libFuzzer both seeds from it
    # and writes newly interesting inputs back into it.
    (
        cd "${OUT_DIR}/${target}"
        "${bin}" \
            -max_total_time="${SECONDS_PER_TARGET}" \
            -rss_limit_mb="${RSS_LIMIT_MB}" \
            -workers="${WORKERS}" -jobs="${WORKERS}" \
            -print_final_stats=1 \
            -artifact_prefix="${OUT_DIR}/${target}/" \
            "${corpus}" > "${OUT_DIR}/${target}/run.log" 2>&1
        echo "$?" > "${OUT_DIR}/${target}/exit_code"
    ) &
    pids+=("$!")
    echo "  started ${target} (pid ${pids[-1]})"
done

echo
echo "Waiting. Tail a target with: tail -f ${OUT_DIR}/<target>/run.log"
for pid in "${pids[@]}"; do
    wait "${pid}" || true
done

elapsed=$(( $(date -u +%s) - start_epoch ))
echo
echo "=== Fuzz marathon complete (${elapsed}s) ==="

failed=0
total_execs=0
for target in "${TARGETS[@]}"; do
    code="$(cat "${OUT_DIR}/${target}/exit_code" 2>/dev/null || echo "?")"
    execs="$(grep -hoE 'stat::number_of_executed_units: *[0-9]+' \
        "${OUT_DIR}/${target}"/*.log 2>/dev/null | grep -oE '[0-9]+' \
        | awk '{s+=$1} END {print s+0}')"
    total_execs=$(( total_execs + execs ))
    artifacts="$(find "${OUT_DIR}/${target}" -maxdepth 1 \
        \( -name 'crash-*' -o -name 'oom-*' -o -name 'leak-*' -o -name 'timeout-*' \) \
        2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${code}" != "0" || "${artifacts}" != "0" ]]; then
        failed=$(( failed + 1 ))
        printf '  %-18s FAIL (exit %s, %s artifact(s), %s execs)\n' \
            "${target}" "${code}" "${artifacts}" "${execs}"
        find "${OUT_DIR}/${target}" -maxdepth 1 \
            \( -name 'crash-*' -o -name 'oom-*' -o -name 'leak-*' -o -name 'timeout-*' \) \
            -printf '      %p\n' 2>/dev/null || true
        # A sanitizer abort can kill the process between libFuzzer announcing the
        # artifact and the file reaching disk: an observed ASan heap-buffer-overflow
        # logged "Test unit written to ./crash-98c0..." and left nothing behind.
        # That is why a target fails on its exit code and not only on an artifact,
        # and why the input has to be recoverable from the log.
        if [[ "${artifacts}" == "0" ]]; then
            echo "      no artifact file was written; recover the input from the log:"
            echo "        grep 'Base64:' ${OUT_DIR}/${target}/run.log | tail -1 \\"
            echo "          | sed 's/.*Base64: //' | base64 -d > input.bin"
        fi
    else
        printf '  %-18s ok   (%s execs)\n' "${target}" "${execs}"
    fi
done

echo
echo "  total executions: ${total_execs}"
if [[ "${failed}" -ne 0 ]]; then
    echo
    echo "${failed} target(s) failed. Reproduce one with:"
    echo "  ${BUILD_DIR}/tests/fuzz/<target> -rss_limit_mb=${RSS_LIMIT_MB} <artifact>"
    echo "Then add the artifact to tests/fuzz/corpus/<target>/ so the replay_fuzz_*"
    echo "CTest suites cover it without a libFuzzer runtime."
    exit 1
fi

echo "All ${#TARGETS[@]} targets clean."
