#!/usr/bin/env python3
"""Generate a CycloneDX SBOM from the same inventory as THIRD_PARTY.md.

Procurement asks for an SBOM and a NOTICE, and they are the same list of
components seen twice. Keeping two hand-maintained copies is how they drift, so
both come from the table below.

    python3 scripts/gen_sbom.py                 # writes sbom.cdx.json
    python3 scripts/gen_sbom.py --check         # fails if the file is stale

The vendored entries carry a SHA-256 over the vendored tree, computed from
vendor/CHECKSUMS.sha256 so the SBOM hash and the checked-in checksums cannot
disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The single inventory. THIRD_PARTY.md is the prose view of this table; keep the
# two in step when adding a component.
COMPONENTS = [
    {
        "name": "googletest",
        "version": "1.14.0",
        "purl": "pkg:github/google/googletest@v1.14.0",
        "licence": "BSD-3-Clause",
        "scope": "excluded",  # test-only, not in any shipped binary
        "path": "vendor/googletest",
        "description": "Test framework. Not linked into any shipped binary.",
    },
    {
        "name": "xsimd",
        "version": "13.2.0",
        "purl": "pkg:github/xtensor-stack/xsimd@13.2.0",
        "licence": "BSD-3-Clause",
        "scope": "required",
        "path": "vendor/xsimd",
        "description": "Header-only SIMD abstraction.",
    },
]

# Found on the host rather than vendored, and only when the matching option is on.
OPTIONAL = [
    ("llvm", "Apache-2.0 WITH LLVM-exception", "MS_BUILD_JIT / MS_BUILD_PLUGIN"),
    ("qt6", "LGPL-3.0-or-later", "MS_BUILD_GUI (dynamic linking only)"),
    ("openmpi", "BSD-3-Clause", "MS_ENABLE_MPI"),
    ("cuda", "LicenseRef-NVIDIA-EULA", "MS_ENABLE_CUDA (see LICENSE.exceptions)"),
]


def project_version() -> str:
    """Read the version from CMakeLists rather than restating it here."""
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"project\s*\([^)]*VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", text, re.S | re.I)
    return m.group(1) if m else "0.0.0"


def vendored_digest(rel_path: str) -> str | None:
    """SHA-256 over the recorded checksums for one vendored subtree.

    Derived from vendor/CHECKSUMS.sha256 rather than re-hashing the files, so the
    SBOM cannot claim a different content than the checked-in manifest that CI
    verifies.
    """
    manifest = ROOT / "vendor" / "CHECKSUMS.sha256"
    if not manifest.is_file():
        return None
    lines = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#") and rel_path in line
    ]
    if not lines:
        return None
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def build_sbom() -> dict:
    version = project_version()
    components = []

    for c in COMPONENTS:
        entry = {
            "type": "library",
            "name": c["name"],
            "version": c["version"],
            "purl": c["purl"],
            "scope": c["scope"],
            "description": c["description"],
            "licenses": [{"license": {"id": c["licence"]}}],
        }
        digest = vendored_digest(c["path"])
        if digest:
            entry["hashes"] = [{"alg": "SHA-256", "content": digest}]
        components.append(entry)

    for name, licence, gate in OPTIONAL:
        lic: dict = (
            {"license": {"name": licence}}
            if licence.startswith("LicenseRef")
            else {"license": {"id": licence}}
        )
        components.append(
            {
                "type": "library",
                "name": name,
                "version": "provided-by-host",
                "scope": "optional",
                "description": f"Used only when built with {gate}.",
                "licenses": [lic],
            }
        )

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        # A stable serial number: a regenerated SBOM for unchanged input should be
        # byte-identical, so --check can be a plain comparison.
        "serialNumber": "urn:uuid:"
        + str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://github.com/odin-loki/MathScript@{version}")),
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "MathScript",
                "version": version,
                "licenses": [{"license": {"id": "AGPL-3.0-or-later"}}],
                "description": (
                    "Computer algebra and numerical computing library. "
                    "See LICENSE.exceptions for the additional permission covering "
                    "the NVIDIA CUDA libraries."
                ),
            },
            "authors": [{"name": "Odin Loch"}],
        },
        "components": components,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if sbom.cdx.json is stale")
    ap.add_argument("-o", "--output", default=str(ROOT / "sbom.cdx.json"))
    args = ap.parse_args()

    rendered = json.dumps(build_sbom(), indent=2, sort_keys=True) + "\n"
    out = pathlib.Path(args.output)

    if args.check:
        if not out.is_file():
            print(f"{out} is missing; run scripts/gen_sbom.py", file=sys.stderr)
            return 1
        if out.read_text(encoding="utf-8") != rendered:
            print(f"{out} is stale; run scripts/gen_sbom.py", file=sys.stderr)
            return 1
        print(f"{out} is up to date")
        return 0

    out.write_text(rendered, encoding="utf-8")
    print(f"wrote {out} ({len(build_sbom()['components'])} components)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
