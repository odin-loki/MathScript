#!/usr/bin/env bash
# Run one libFuzzer target for a total budget, in bounded chunks.
#
#   scripts/fuzz_session.sh BUILD_DIR TARGET TOTAL_SECONDS [CHUNK_SECONDS] [extra args...]
#
# This exists because the same fifteen lines were written out four times -- the
# CI smoke, the two nightly sessions and the 24-hour marathon -- and three of the
# four had the same two defects. One implementation is one place to be wrong.
#
# THE CORPUS DIRECTORY IS POSITIONAL. libFuzzer seeds from the directories given
# as positional arguments and writes newly interesting inputs back into the
# first. `-corpus_dir=DIR` is NOT a libFuzzer flag: it was accepted as an
# unrecognised flag and ignored, so every Actions run of every fuzz job started
# from an empty corpus, never read the inputs previous runs had found
# interesting, and threw away everything it found. `scripts/fuzz_24h_local.sh`
# had it right; the workflows did not.
#
# THE MEMORY THE 2026-09-16 MARATHON DIED OF WAS THE SANITIZER'S, NOT THE
# TARGET'S. That job hit `out-of-memory (used: 2057Mb; limit: 2048Mb)` after 69
# minutes with a LIVE HEAP OF 43.8Mb and a 602Kb corpus. AddressSanitizer stores
# an allocation stack for every chunk it has ever handed out, thirty frames deep
# by default, and the run had been through 6.4 million chunks. Measured here, on
# the same target and a 6280-unit corpus, over five minutes:
#
#                          floor at INITED    peak RSS    executions
#   default ASAN_OPTIONS         769Mb          968Mb       449,810
#   malloc_context_size=5        154Mb          209Mb       501,245
#
# A 4.6x cut in peak RSS and slightly MORE throughput. So the cap stays at 2048Mb
# and gets its meaning back: with a 154Mb floor, what fills 2GB is the target.
# Raising the limit instead would have silenced the guard that caught the real
# `combo_restricted_partitions(442, 5)` out-of-memory at 2398Mb peak RSS.
#
# What that costs: a use-after-free or double-free report shows five frames of
# the allocation stack rather than thirty. The CRASH stack is unaffected, and the
# artifact is written out -- re-run it with the default ASAN_OPTIONS to get the
# full allocation context back.
#
# THE BUDGET IS ALSO SPENT IN CHUNKS, each one a fresh process, because the
# remaining growth is slow but was never bounded: it climbed 402Mb to 597Mb over
# eight minutes locally before the options above. Restarting bounds what cannot
# be predicted, and costs only the corpus reload.
#
# Env: MS_FUZZ_RSS_LIMIT_MB (2048), MS_FUZZ_ARTIFACTS (./artifacts),
#      MS_FUZZ_ASAN_OPTIONS (the options above; set it to override them)
set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 BUILD_DIR TARGET TOTAL_SECONDS [CHUNK_SECONDS] [extra args...]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$1"
TARGET="$2"
TOTAL="$3"
CHUNK="${4:-900}"
shift 3
[[ $# -gt 0 ]] && shift || true
EXTRA=("$@")

BIN="${BUILD_DIR}/tests/fuzz/${TARGET}"
if [[ ! -x "${BIN}" ]]; then
    BIN="${ROOT}/${BUILD_DIR}/tests/fuzz/${TARGET}"
fi
if [[ ! -x "${BIN}" ]]; then
    echo "error: no fuzz binary for ${TARGET} under ${BUILD_DIR}" >&2
    exit 1
fi

CORPUS="${ROOT}/tests/fuzz/corpus/${TARGET}"
ARTIFACTS="${MS_FUZZ_ARTIFACTS:-${PWD}/artifacts}"
mkdir -p "${CORPUS}" "${ARTIFACTS}"

ARGS=(-rss_limit_mb="${MS_FUZZ_RSS_LIMIT_MB:-2048}"
      -print_final_stats=1
      -artifact_prefix="${ARTIFACTS}/")

# Match each harness's own cap. `fuzz_repl_input` returns immediately above 512
# bytes and `fuzz_sym_parser` above 256, and libFuzzer defaults to generating up
# to 4096 -- so most of what it produced was discarded before it reached a line
# of code. The other five read a fixed-size prefix and have no upper bound.
case "${TARGET}" in
    fuzz_repl_input) ARGS+=(-max_len=512) ;;
    fuzz_sym_parser) ARGS+=(-max_len=256) ;;
esac

elapsed=0
chunk_index=0
while [[ "${elapsed}" -lt "${TOTAL}" ]]; do
    slice=$(( TOTAL - elapsed ))
    if [[ "${slice}" -gt "${CHUNK}" ]]; then
        slice="${CHUNK}"
    fi
    chunk_index=$(( chunk_index + 1 ))
    echo "=== ${TARGET}: chunk ${chunk_index}, ${slice}s (${elapsed}s of ${TOTAL}s done)"
    ASAN_OPTIONS="${MS_FUZZ_ASAN_OPTIONS:-malloc_context_size=5:quarantine_size_mb=16}" \
        "${BIN}" "${ARGS[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" \
            -max_total_time="${slice}" "${CORPUS}"
    code=$?
    if [[ "${code}" -ne 0 ]]; then
        echo "error: ${TARGET} exited ${code} in chunk ${chunk_index}" >&2
        exit "${code}"
    fi
    elapsed=$(( elapsed + slice ))
done

# libFuzzer can leave an artifact behind and still exit 0, which happens when a
# worker rather than the parent found the input. Exit code alone is not the
# answer to "did this find anything".
found=0
for kind in crash oom leak timeout; do
    for artifact in "${ARTIFACTS}/${kind}-"*; do
        [[ -e "${artifact}" ]] || continue
        echo "error: ${TARGET} left ${artifact}" >&2
        found=1
    done
done
exit "${found}"
