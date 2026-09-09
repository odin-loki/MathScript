#!/usr/bin/env bash
# Run every verification harness under whichever bounded model checker is present.
# See README.md for what each harness checks and what the results mean.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED=0

run_one() {
    local tool="$1" file="$2"; shift 2
    printf '  %-24s %-28s ' "$(basename "${file}")" "${tool}"
    local out
    if [[ "${tool}" == esbmc ]]; then
        out="$(esbmc "${file}" --unwind 4 --overflow-check --memory-leak-check "$@" 2>&1)"
        if grep -q "VERIFICATION SUCCESSFUL" <<<"${out}"; then echo "SUCCESSFUL"; return 0; fi
    else
        out="$(cbmc "${file}" --unwind 4 --bounds-check --pointer-check \
               --conversion-check --signed-overflow-check --div-by-zero-check "$@" 2>&1)"
        if grep -q "VERIFICATION SUCCESSFUL" <<<"${out}"; then echo "SUCCESSFUL"; return 0; fi
    fi
    echo "FAILED"
    sed -n '/^\*\* Results:/,$p' <<<"${out}" | head -30
    FAILED=$((FAILED + 1))
}

echo "Verification harnesses in ${ROOT}"
TOOLS=()
command -v cbmc  >/dev/null 2>&1 && TOOLS+=(cbmc)
command -v esbmc >/dev/null 2>&1 && TOOLS+=(esbmc)
if [[ ${#TOOLS[@]} -eq 0 ]]; then
    echo "error: neither cbmc nor esbmc found." >&2
    echo "  apt-get install cbmc      # Ubuntu/Debian" >&2
    echo "  ESBMC: https://github.com/esbmc/esbmc/releases" >&2
    exit 1
fi
echo "  tools: ${TOOLS[*]}"
echo

for tool in "${TOOLS[@]}"; do
    for f in "${ROOT}"/*.c; do
        run_one "${tool}" "${f}"
    done
done

echo
if [[ "${FAILED}" -ne 0 ]]; then
    echo "${FAILED} harness run(s) failed."
    exit 1
fi
echo "All harnesses verified."
