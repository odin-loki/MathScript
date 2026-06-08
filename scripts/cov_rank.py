#!/usr/bin/env python3
import re
import sys
from pathlib import Path

info = Path(sys.argv[1] if len(sys.argv) > 1 else "build-cov/coverage.info")
data = info.read_text()
rows = []
for rec in data.split("end_of_record"):
    sf = re.search(r"^SF:(.+)$", rec, re.M)
    if not sf:
        continue
    path = sf.group(1)
    if "/src/" not in path:
        continue
    if "tests/" in path or "googletest" in path:
        continue
    das = [line for line in rec.split("\n") if line.startswith("DA:")]
    if not das:
        continue
    hit = sum(1 for line in das if int(line.split(",")[1]) > 0)
    total = len(das)
    short = path.split("MathsScript/")[-1]
    rows.append((100 * hit / total, hit, total, short))

rows.sort()
print("Lowest 30 files by line coverage:")
for pct, hit, total, path in rows[:30]:
    print(f"{pct:5.1f}% ({hit:4}/{total:4}) {path}")
print("---")
print("Top 20 by uncovered lines:")
by_miss = sorted(((total - hit, pct, path) for pct, hit, total, path in rows), reverse=True)
for miss, pct, path in by_miss[:20]:
    print(f"{miss:4} miss ({pct:5.1f}%) {path}")
print("---")
total_hit = sum(r[1] for r in rows)
total_lines = sum(r[2] for r in rows)
print(f"Overall src: {100 * total_hit / total_lines:.1f}% ({total_hit}/{total_lines})")
