#!/usr/bin/env bash
# Verify SHA-256 checksums for vendored sources (master plan §13.8).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKSUMS="${ROOT}/vendor/CHECKSUMS.sha256"

if [[ ! -f "${CHECKSUMS}" ]]; then
    echo "Missing vendor checksum file: ${CHECKSUMS}" >&2
    exit 1
fi

mapfile -t ENTRIES < <(grep -vE '^\s*(#|$)' "${CHECKSUMS}" || true)

if [[ ${#ENTRIES[@]} -eq 0 ]]; then
    echo "no vendored sources"
    exit 0
fi

sha256_of() {
    local filepath="$1"

    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${filepath}" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${filepath}" | awk '{print $1}'
    elif command -v certutil >/dev/null 2>&1; then
        certutil -hashfile "${filepath}" SHA256 | awk 'NR==2 {print tolower($0)}'
    elif command -v powershell >/dev/null 2>&1; then
        MS_VERIFY_FILE="${filepath}" powershell -NoProfile -Command \
            '$env:MS_VERIFY_FILE | ForEach-Object { (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLower() }'
    else
        echo "No SHA-256 tool found (need sha256sum, shasum, certutil, or powershell)" >&2
        return 1
    fi
}

failures=0

for entry in "${ENTRIES[@]}"; do
    entry="${entry//$'\r'/}"
    if [[ "${entry}" =~ ^([0-9a-fA-F]{64})[[:space:]]+(.*)$ ]]; then
        expected="${BASH_REMATCH[1],,}"
        relpath="${BASH_REMATCH[2]}"
    else
        echo "Invalid checksum line: ${entry}" >&2
        failures=$((failures + 1))
        continue
    fi

    relpath="${relpath#./}"
    relpath="${relpath#\\}"
    filepath="${ROOT}/${relpath}"

    if [[ ! -f "${filepath}" ]]; then
        echo "Missing vendored file: ${relpath}" >&2
        failures=$((failures + 1))
        continue
    fi

    actual="$(sha256_of "${filepath}")"
    actual="${actual,,}"

    if [[ "${actual}" != "${expected}" ]]; then
        echo "Hash mismatch: ${relpath}" >&2
        echo "  expected: ${expected}" >&2
        echo "  actual:   ${actual}" >&2
        failures=$((failures + 1))
    fi
done

if [[ ${failures} -gt 0 ]]; then
    echo "Vendor integrity check failed (${failures} error(s))" >&2
    exit 1
fi

echo "Vendor integrity check OK (${#ENTRIES[@]} file(s))"
