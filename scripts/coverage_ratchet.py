#!/usr/bin/env python3
"""Fail when coverage goes down.

The plan's §8.9 asks for a ratchet rather than a fixed threshold: CI reads the
previous number and fails on any decrease. The appeal is that there is no figure to
argue about and no room for silent slippage -- a change that removes tests stops
being invisible.

    python3 scripts/coverage_ratchet.py build-cov            # check
    python3 scripts/coverage_ratchet.py build-cov --update   # raise the baseline

The baseline lives in tests/coverage_baseline.json and is committed, because a
ratchet with no memory between runs is just a threshold.

TOLERANCE, and why this is not "any decrease at all".

A gate that fires on noise is a gate people learn to re-run until it passes, and
this repository already has one cautionary example: the benchmark job was left with
five measured entries out of 445 precisely because 94% of them varied by more than
the threshold between two runs of identical code. Coverage is far more stable than
timing, but it is not exactly reproducible either -- code behind a timeout, a thread
count, or a filesystem probe can execute on one runner and not another.

So the ratchet allows a small slack, and the number is set from measurement rather
than taste: see `tolerance` in the baseline file for the evidence behind the value
in use. Slippage larger than the slack fails; slippage smaller than it is invisible
either way, which is the honest description of what a percentage can tell you.

--update raises the baseline and refuses to lower it. Lowering is a deliberate act
and needs --force, so that a run which legitimately drops coverage leaves a trace in
the diff rather than quietly moving the goalposts.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "coverage_baseline.json"
METRICS = ("lines", "functions", "branches")


def parse_summary(build_dir: pathlib.Path) -> dict[str, float]:
    """Read the percentages out of lcov's own summary output."""
    path = build_dir / "coverage-summary.txt"
    if not path.is_file():
        print(f"{path} is missing; run scripts/coverage_report.sh first", file=sys.stderr)
        raise SystemExit(1)
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, float] = {}
    for metric in METRICS:
        m = re.search(rf"{metric}\.*:\s*([0-9.]+)%", text)
        if m:
            out[metric] = float(m.group(1))
    if "lines" not in out:
        print(f"could not parse a line-coverage percentage from {path}", file=sys.stderr)
        raise SystemExit(1)
    return out


def load_baseline() -> dict:
    if not BASELINE.is_file():
        return {}
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("build_dir", nargs="?", default="build-cov")
    ap.add_argument("--update", action="store_true", help="raise the baseline to match")
    ap.add_argument("--force", action="store_true",
                    help="with --update, allow the baseline to be lowered")
    args = ap.parse_args()

    build_dir = pathlib.Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir

    measured = parse_summary(build_dir)
    baseline = load_baseline()
    recorded = baseline.get("metrics", {})
    tolerance = float(baseline.get("tolerance", 0.5))

    if args.update:
        merged = dict(recorded)
        lowered: list[str] = []
        declined: list[str] = []
        for metric, value in measured.items():
            previous = recorded.get(metric)
            if previous is None or value > previous:
                merged[metric] = value
            elif value < previous:
                change = f"{metric} {previous:.1f}% -> {value:.1f}%"
                if args.force:
                    merged[metric] = value
                    lowered.append(change)
                else:
                    merged[metric] = previous
                    declined.append(change)
        # Saying nothing here would let "--update" look like it had recorded the
        # run when it had in fact kept the old, higher numbers.
        if declined:
            for change in declined:
                print(f"declined to lower: {change}", file=sys.stderr)
            print("the baseline keeps the higher figure; use --force to lower it "
                  "deliberately", file=sys.stderr)
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps({
            "comment": baseline.get("comment", ""),
            "tolerance": tolerance,
            "tolerance_evidence": baseline.get("tolerance_evidence", ""),
            "measured_at": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%d %H:%M UTC"),
            "commit": commit(),
            "metrics": {k: round(merged[k], 1) for k in sorted(merged)},
        }, indent=2) + "\n", encoding="utf-8")
        for line in lowered:
            print(f"lowered: {line}")
        print(f"wrote {BASELINE.relative_to(ROOT)}")
        return 0

    if not recorded:
        print(f"{BASELINE.relative_to(ROOT)} has no metrics; "
              "seed it with --update", file=sys.stderr)
        return 1

    failed = False
    for metric in METRICS:
        value = measured.get(metric)
        previous = recorded.get(metric)
        if previous is None:
            continue
        if value is None:
            print(f"{metric} coverage is in the baseline at {previous:.1f}% but was "
                  f"not measured in this run", file=sys.stderr)
            failed = True
            continue
        delta = value - previous
        if delta < -tolerance:
            print(f"{metric} coverage fell {abs(delta):.2f} points: "
                  f"{previous:.1f}% -> {value:.1f}% (tolerance {tolerance})",
                  file=sys.stderr)
            failed = True
        else:
            arrow = "+" if delta >= 0 else ""
            print(f"{metric:<10} {value:5.1f}%  baseline {previous:5.1f}%  "
                  f"({arrow}{delta:.2f})")

    if failed:
        print("\nCoverage went down. Either add tests, or raise the tolerance with a "
              "reason,\nor lower the baseline deliberately with --update --force.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
