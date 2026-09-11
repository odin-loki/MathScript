#!/usr/bin/env python3
"""Branch coverage restricted to lines that actually contain a decision.

gcov's branch total is not a count of decisions in the source. At -O0 it also
counts edges inside library code that got inlined into your line, and attributes
them to your line number. In this tree that is 43% of the denominator:

    matrix.hpp:34: : m_rows_(rows), m_cols_(cols), m_data_(rows * cols) {}
    matrix.hpp:53: m_data_.resize(m_rows_ * m_cols_);

Neither line has a decision in it. The branches are std::vector's growth and
allocation-failure arms and std::string's short/long representation check. Most
cannot be reached from a test at all -- the allocation-failure arm needs the
allocator to fail, and this library is built with -fno-exceptions.

So the raw figure understates how well the project's own logic is tested, and no
amount of test writing moves it much. This computes the other number: of the branch
slots that sit on a line containing `if`, `while`, `for`, `switch`, `&&`, `||` or
`?`, how many were taken.

    python3 scripts/decision_coverage.py build-cov/coverage.info

BOTH NUMBERS MATTER, AND THIS ONE DOES NOT REPLACE THE OTHER. Reporting only this
figure would be the same mistake that produced a published 92.0% line coverage
measured over 75% of the repository: a denominator quietly chosen to flatter the
result. It is reported beside the raw gcov figure, never instead of it.

The classification is a heuristic over source text. It cannot tell a decision that
the compiler duplicated from one a human wrote, and a line holding both a real
condition and an inlined call is counted as a decision line. It is a better
denominator than the raw count, not a perfect one.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

DECISION = re.compile(r'\b(if|while|for|switch|case|catch)\b|&&|\|\||\?')


def analyse(info_path: pathlib.Path) -> dict:
    per_file: dict[str, list[tuple[int, bool]]] = collections.defaultdict(list)
    cur = None
    for line in info_path.open(encoding="utf-8", errors="replace"):
        if line.startswith("SF:"):
            cur = line[3:].strip()
        elif line.startswith("BRDA:") and cur:
            parts = line[5:].strip().split(",")
            try:
                ln = int(parts[0])
            except ValueError:
                continue
            per_file[cur].append((ln, parts[-1] not in ("-", "0")))

    stats = {"decision_total": 0, "decision_hit": 0,
             "inlined_total": 0, "inlined_hit": 0,
             "missing_source": 0}
    worst: dict[str, tuple[int, int]] = {}
    for path, branches in per_file.items():
        try:
            src = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            stats["missing_source"] += len(branches)
            continue
        d_tot = d_hit = 0
        for ln, taken in branches:
            text = src[ln - 1] if 0 < ln <= len(src) else ""
            if DECISION.search(text):
                stats["decision_total"] += 1
                d_tot += 1
                stats["decision_hit"] += taken
                d_hit += taken
            else:
                stats["inlined_total"] += 1
                stats["inlined_hit"] += taken
        if d_tot:
            worst[path] = (d_tot - d_hit, d_tot)
    stats["worst"] = worst
    return stats


def pct(hit: int, total: int) -> float:
    return 100.0 * hit / total if total else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("info", nargs="?", default="build-cov/coverage.info")
    ap.add_argument("--top", type=int, default=15, help="worst N files to list")
    ap.add_argument("--quiet", action="store_true", help="print only the percentage")
    args = ap.parse_args()

    info = pathlib.Path(args.info)
    if not info.is_file():
        print(f"{info} is missing; run scripts/coverage_report.sh first", file=sys.stderr)
        return 1

    s = analyse(info)
    d_pct = pct(s["decision_hit"], s["decision_total"])
    if args.quiet:
        print(f"{d_pct:.1f}")
        return 0

    raw_total = s["decision_total"] + s["inlined_total"]
    raw_hit = s["decision_hit"] + s["inlined_hit"]
    print(f"raw gcov branches        {raw_hit}/{raw_total} = {pct(raw_hit, raw_total):.1f}%")
    print(f"  on decision lines      {s['decision_hit']}/{s['decision_total']} "
          f"= {d_pct:.1f}%")
    print(f"  on non-decision lines  {s['inlined_hit']}/{s['inlined_total']} "
          f"= {pct(s['inlined_hit'], s['inlined_total']):.1f}%   "
          f"({pct(s['inlined_total'], raw_total):.0f}% of the denominator, "
          f"mostly inlined library code)")
    if s["missing_source"]:
        print(f"  source unreadable      {s['missing_source']} slots skipped")

    worst = sorted(s["worst"].items(), key=lambda kv: -kv[1][0])[:args.top]
    if worst:
        root = str(pathlib.Path(__file__).resolve().parent.parent) + "/"
        print(f"\nWorst {len(worst)} files by uncovered decisions:")
        for path, (miss, tot) in worst:
            print(f"  {miss:6d} of {tot:6d} ({pct(tot - miss, tot):5.1f}%)  "
                  f"{path.replace(root, '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
