#!/usr/bin/env bash
# Compare unsafe_report.txt against a stored baseline of reviewed sites.
# Exit 0 when no new unreviewed sites; warn and exit 1 on delta.
# MS_UNSAFE_DELTA=off       — skip compare (always exit 0).
# MS_UNSAFE_DELTA_STRICT=on — same exit behaviour; for future blocking CI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${ROOT}/tests/compliance/unsafe_baseline.txt"
REVIEW="${ROOT}/UNSAFE_REVIEW.md"
CURRENT="${ROOT}/build/unsafe_report.txt"
WRITE_BASELINE=0
DELTA="${MS_UNSAFE_DELTA:-on}"

usage() {
    echo "Usage: $0 [--write-baseline] [REPORT_PATH]" >&2
    echo "  --write-baseline  Normalize REPORT_PATH and overwrite unsafe_baseline.txt" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --write-baseline)
            WRITE_BASELINE=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            CURRENT="$1"
            shift
            ;;
    esac
done

if [[ "${CURRENT}" != /* ]]; then
    CURRENT="${ROOT}/${CURRENT#./}"
fi

normalize_sites() {
    local report="$1"
    local line rel
    if [[ ! -f "${report}" ]]; then
        echo "Report not found: ${report}" >&2
        return 1
    fi
    while IFS= read -r line; do
        if [[ "${line}" =~ .*((src|include)/[^:]+\.(cpp|hpp):[0-9]+:.*) ]]; then
            rel="${BASH_REMATCH[1]}"
            echo "${rel}"
        fi
    done < <(grep -E '(src|include)/[^:]+\.(cpp|hpp):[0-9]+:' "${report}" || true) \
        | sort -u
}

# The line number in a baseline row is there so a reviewer can find the site; it
# is NOT part of the site's identity. Editing anything above a reviewed site
# shifts it, and comparing on the raw row then reports a site that has not
# changed as new -- which it did three times in one afternoon, once for a
# comment and twice for an added include. Identity is the file plus the matched
# text, and multiplicity still counts, so a genuinely added cast is still caught.
strip_line_numbers() {
    sed -E 's#^([^:]+):[0-9]+:#\1:#'
}

count_sites() {
    normalize_sites "$1" | wc -l | tr -d ' '
}

approved_count_from_review() {
    if [[ ! -f "${REVIEW}" ]]; then
        return
    fi
    grep -E '^approved_sites:' "${REVIEW}" | tail -1 | sed 's/.*: *//'
}

if [[ "${WRITE_BASELINE}" -eq 1 ]]; then
    if [[ ! -f "${CURRENT}" ]]; then
        echo "Report not found for --write-baseline: ${CURRENT}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${BASELINE}")"
    {
        echo "# MathScript unsafe baseline — one reviewed site per line (file:line:match)."
        echo "# The line number locates the site for a reviewer; identity is file+match,"
        echo "# so an edit that only shifts a site does not read as a new one."
        echo "# Regenerate: bash scripts/unsafe_delta.sh --write-baseline [REPORT_PATH]"
        echo "# Source report: ${CURRENT}"
        normalize_sites "${CURRENT}"
    } > "${BASELINE}"
    echo "Wrote baseline ($(count_sites "${BASELINE}") sites): ${BASELINE}"
    exit 0
fi

if [[ "${DELTA}" == "off" ]]; then
    echo "Unsafe delta check skipped (MS_UNSAFE_DELTA=off)"
    exit 0
fi

if [[ ! -f "${CURRENT}" ]]; then
    echo "ERROR: unsafe report missing: ${CURRENT}" >&2
    echo "Run: bash scripts/unsafe_report.sh" >&2
    exit 1
fi

if [[ -f "${BASELINE}" ]]; then
    _ud_base="$(mktemp)"
    _ud_cur="$(mktemp)"
    trap 'rm -f "${_ud_base}" "${_ud_cur}"' EXIT

    grep -Ev '^\s*#' "${BASELINE}" | grep -E '^(src|include)/' \
        | strip_line_numbers | sort > "${_ud_base}"
    normalize_sites "${CURRENT}" | strip_line_numbers | sort > "${_ud_cur}"

    mapfile -t new_sites < <(comm -13 "${_ud_base}" "${_ud_cur}")
    mapfile -t removed < <(comm -23 "${_ud_base}" "${_ud_cur}")

    if [[ ${#new_sites[@]} -eq 0 ]]; then
        echo "Unsafe delta check OK ($(wc -l < "${_ud_cur}" | tr -d " ") sites, baseline $(wc -l < "${_ud_base}" | tr -d " "))"
        if [[ ${#removed[@]} -gt 0 ]]; then
            echo "NOTE: ${#removed[@]} baseline site(s) no longer present (review table may need cleanup)"
            printf '  %s\n' "${removed[@]}"
        fi
        exit 0
    fi

    echo "WARN: ${#new_sites[@]} new unreviewed unsafe site(s) vs baseline ${BASELINE##*/}:" >&2
    printf '  %s\n' "${new_sites[@]}" >&2
    echo "Add rows to UNSAFE_REVIEW.md, bump approved_sites, then:" >&2
    echo "  bash scripts/unsafe_delta.sh --write-baseline ${CURRENT#${ROOT}/}" >&2
    exit 1
fi

# Fallback: count-only check against UNSAFE_REVIEW.md approved_sites.
APPROVED="$(approved_count_from_review)"
CURRENT_COUNT="$(count_sites "${CURRENT}")"

if [[ -z "${APPROVED}" ]]; then
    echo "WARNING: ${BASELINE##*/} missing and approved_sites unset in UNSAFE_REVIEW.md" >&2
    echo "Current unsafe sites: ${CURRENT_COUNT}"
    exit 0
fi

if [[ "${CURRENT_COUNT}" -gt "${APPROVED}" ]]; then
    echo "WARN: ${CURRENT_COUNT} unsafe sites exceed approved baseline ${APPROVED} in UNSAFE_REVIEW.md" >&2
    echo "Create ${BASELINE##*/} after review: bash scripts/unsafe_delta.sh --write-baseline" >&2
    exit 1
fi

echo "Unsafe delta check OK (${CURRENT_COUNT} sites, approved_sites=${APPROVED})"
echo "NOTE: line-level baseline ${BASELINE##*/} not found; using count-only fallback"
