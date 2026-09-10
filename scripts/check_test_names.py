#!/usr/bin/env python3
"""Fail if two tests in one binary share a name.

`TEST(Suite, Name)` expands to a class `Suite_Name_Test` whose member functions
are implicitly inline. Two translation units in the same executable declaring the
same pair therefore link cleanly, and one body silently replaces the other: the
binary reports the same number of passes while running one of the two tests
twice. Nothing in the toolchain says a word.

This mattered the moment the integration tests were grouped from 573 executables
into 29 per-domain ones. 71 pairs collided at that point and 29 of them had
different bodies, so the naive grouping would have quietly stopped running real
tests -- and the test count, the thing anyone would have checked, would not have
moved.

    python3 scripts/check_test_names.py

Each directory that becomes one executable is checked independently, because that
is the scope in which the collision does damage.
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_RE = re.compile(r'\bTEST(?:_F|_P)?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)')

# Each entry is a directory whose sources are linked into a single executable.
GROUPED = [
    (ROOT / "tests" / "integration", True),   # one target per immediate subdir
    (ROOT / "tests" / "unit" / "matrix_calls", False),  # one target for the whole dir
    (ROOT / "tests" / "unit" / "repl", False),
]


def pairs_in(path: pathlib.Path) -> list[tuple[str, str]]:
    return TEST_RE.findall(path.read_text(encoding="utf-8", errors="replace"))


def check_group(name: str, files: list[pathlib.Path]) -> list[str]:
    seen: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for f in sorted(files):
        for pair in pairs_in(f):
            seen[pair].append(f.name)
    problems = []
    for pair, where in sorted(seen.items()):
        if len(set(where)) > 1:
            problems.append(
                f"{name}: TEST({pair[0]}, {pair[1]}) declared in {sorted(set(where))}")
    return problems


def main() -> int:
    problems: list[str] = []
    checked = 0
    for base, per_subdir in GROUPED:
        if not base.is_dir():
            continue
        if per_subdir:
            for sub in sorted(p for p in base.iterdir() if p.is_dir()):
                files = sorted(sub.rglob("*.cpp"))
                if files:
                    problems += check_group(f"{sub.relative_to(ROOT)}", files)
                    checked += 1
        else:
            files = sorted(base.glob("*.cpp"))
            if files:
                problems += check_group(f"{base.relative_to(ROOT)}", files)
                checked += 1

    if problems:
        print(f"{len(problems)} duplicate test name(s) within a single executable:",
              file=sys.stderr)
        for p in problems[:40]:
            print(f"  {p}", file=sys.stderr)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
        print("\nOne of each pair would be silently discarded at link time. "
              "Rename one.", file=sys.stderr)
        return 1

    print(f"{checked} test executables checked; no duplicate test names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
