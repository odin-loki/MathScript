#!/usr/bin/env bash
# Static unsafe-surface audit for Phase 10 hardening.
# Full [[ms::unsafe]] enforcement requires MS_BUILD_PLUGIN (Clang plugin).
# MS_UNSAFE(reason) macro: include/ms/unsafe/unsafe.hpp (included from error_types.hpp).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVIEW="${ROOT}/UNSAFE_REVIEW.md"
OUT="${1:-${ROOT}/build/unsafe_report.txt}"

mkdir -p "$(dirname "${OUT}")"

scan_dir() {
    local label="$1"
    local dir="$2"
    if [[ ! -d "${dir}" ]]; then
        return
    fi
    echo "=== ${label} ==="
    rg -n --glob '*.cpp' --glob '*.hpp' \
        'reinterpret_cast|const_cast|\[\[ms::unsafe|UNSAFE_SITE\(' "${dir}" || true
    echo
}

{
    echo "MathScript unsafe surface report"
    echo "Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo
    scan_dir "src" "${ROOT}/src"
    scan_dir "include" "${ROOT}/include"
} | tee "${OUT}"

TOTAL="$(rg -c 'reinterpret_cast|const_cast|\[\[ms::unsafe|UNSAFE_SITE\(' \
    --glob '*.cpp' --glob '*.hpp' "${ROOT}/src" "${ROOT}/include" 2>/dev/null \
    | awk -F: '{sum += $2} END {print sum + 0}')"

echo "Total matched lines: ${TOTAL}"

if [[ -f "${REVIEW}" ]]; then
    APPROVED="$(grep -E '^approved_sites:' "${REVIEW}" | tail -1 | sed 's/.*: *//')"
    if [[ -n "${APPROVED}" ]] && [[ "${TOTAL}" -gt "${APPROVED}" ]]; then
        echo "ERROR: ${TOTAL} unsafe-site matches exceed approved baseline ${APPROVED} in UNSAFE_REVIEW.md" >&2
        exit 1
    fi
    echo "Approved baseline: ${APPROVED:-not set}"
else
    echo "WARNING: ${REVIEW} missing; set approved_sites after review." >&2
fi
