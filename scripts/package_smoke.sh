#!/usr/bin/env bash
# Verify MathScript install tree after cmake --install.
set -euo pipefail

BUILD_DIR="${1:-build-linux}"
PREFIX="${2:-install-smoke}"
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

if [[ "${PREFIX}" = /* ]]; then
    PREFIX_PATH="${PREFIX}"
else
    PREFIX_PATH="${ROOT}/${PREFIX}"
fi

cmake --install "${BUILD_PATH}" --prefix "${PREFIX_PATH}"

for bin in mathscript-repl mathscriptc mathscript-server; do
    path="${PREFIX_PATH}/bin/${bin}"
    if [[ ! -x "${path}" ]]; then
        echo "Missing or non-executable: ${path}" >&2
        exit 1
    fi
done

if [[ ! -f "${PREFIX_PATH}/lib/libms_core.a" ]]; then
    echo "Missing: ${PREFIX_PATH}/lib/libms_core.a" >&2
    exit 1
fi

if [[ ! -f "${PREFIX_PATH}/include/ms/version.hpp" ]]; then
    echo "Missing: ${PREFIX_PATH}/include/ms/version.hpp" >&2
    exit 1
fi

echo "Install smoke OK: ${PREFIX_PATH}"

pushd "${BUILD_PATH}" >/dev/null
cpack -G TGZ
shopt -s nullglob
tgz=(mathscript-*.tar.gz)
if (( ${#tgz[@]} == 0 )); then
    echo "CPack TGZ artifact not found in ${BUILD_PATH}" >&2
    exit 1
fi
echo "Package smoke OK: ${BUILD_PATH}/${tgz[0]}"
popd >/dev/null
