#!/usr/bin/env python3
"""Put an SPDX licence identifier at the top of every source file.

139k lines shipped with no per-file licence marking. A reviewer opening any one
file could not tell what terms it was under, and automated licence scanners --
which is how most procurement actually checks -- report the tree as unlicensed.

    python3 scripts/add_spdx.py            # insert where missing
    python3 scripts/add_spdx.py --check    # fail if any file lacks one (CI)

Files under src/cuda/ and include/ms/cuda/ additionally name LICENSE.exceptions.
Those are the translation units that link the proprietary NVIDIA libraries, and
the GPLv3 section 7 additional permission is what makes distributing that
combination lawful; a file that participates in the link should say so rather
than leaving the grant discoverable only from the repository root.

vendor/ is never touched: it is not ours to mark.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

SPDX = "SPDX-License-Identifier: AGPL-3.0-or-later"
CUDA_NOTE = "SPDX-FileComment: links the NVIDIA CUDA libraries; see LICENSE.exceptions"
COPYRIGHT = "SPDX-FileCopyrightText: 2026 Odin Loch"

SCAN_DIRS = ["src", "include", "tests"]
SUFFIXES = {".cpp", ".hpp", ".h", ".cc", ".cu"}

# Not ours, or generated.
SKIP_PARTS = {"vendor", "build", "build-test", "build-cov", "build-odr", "build-fuzz"}


def is_cuda_linked(path: pathlib.Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    return rel.startswith("src/cuda/") or rel.startswith("include/ms/cuda/")


def header_for(path: pathlib.Path) -> str:
    lines = [f"// {SPDX}", f"// {COPYRIGHT}"]
    if is_cuda_linked(path):
        lines.append(f"// {CUDA_NOTE}")
    return "\n".join(lines) + "\n"


def has_spdx(text: str) -> bool:
    # Only the top of the file counts: an SPDX string inside a plugin diagnostic
    # or a test fixture is not a licence marking.
    head = text.split("\n")[:10]
    return any("SPDX-License-Identifier" in line for line in head)


def insert(text: str, header: str) -> str:
    """Insert after a shebang or a UTF-8 BOM, otherwise at the very top."""
    bom = ""
    if text.startswith("﻿"):
        bom, text = "﻿", text[1:]
    if text.startswith("#!"):
        first, _, rest = text.partition("\n")
        return f"{bom}{first}\n{header}{rest}"
    return f"{bom}{header}{text}"


def candidates() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.suffix not in SUFFIXES or not p.is_file():
                continue
            if SKIP_PARTS & set(p.relative_to(ROOT).parts):
                continue
            out.append(p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report and fail; change nothing")
    args = ap.parse_args()

    files = candidates()
    if not files:
        print("no source files found", file=sys.stderr)
        return 1

    missing: list[pathlib.Path] = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="surrogateescape")
        if has_spdx(text):
            continue
        missing.append(p)
        if not args.check:
            p.write_text(insert(text, header_for(p)), encoding="utf-8", errors="surrogateescape")

    marked = len(files) - len(missing)
    if args.check:
        print(f"{marked} of {len(files)} files carry an SPDX identifier")
        if missing:
            print(f"\n{len(missing)} file(s) missing one:", file=sys.stderr)
            for p in missing[:20]:
                print(f"  {p.relative_to(ROOT)}", file=sys.stderr)
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
            print("\nrun: python3 scripts/add_spdx.py", file=sys.stderr)
            return 1
        return 0

    cuda = sum(1 for p in missing if is_cuda_linked(p))
    print(f"scanned {len(files)} files; added a header to {len(missing)}")
    print(f"  of which {cuda} also reference LICENSE.exceptions (CUDA-linked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
