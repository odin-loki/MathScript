#!/usr/bin/env python3
"""Generate docs/STATUS.md from the build rather than from memory.

Engineering plan §5 listed seven published claims that were false or stale --
a coverage figure the CI gate did not enforce, two CTest counts in one file that
contradicted each other and both contradicted the README, a benchmark validated
at a setting the gate never uses, and a link to a tag that did not exist. Every
one was checkable in five minutes, and each cost more credibility than the gap it
concealed.

The structural answer is to stop maintaining those numbers by hand. This reads
them from the artefacts the build already produces:

    ctest -N                       suite count
    coverage-summary.txt           line / function / branch percentages
    benchmark JSON                 benchmark count and tolerance
    ci.yml                         the gate values actually enforced
    git                            the current commit and whether a tag exists

    python3 scripts/gen_status.py --build-dir build-test
    python3 scripts/gen_status.py --check      # fail if docs/STATUS.md is stale

Anything this script cannot read is written as "not measured" rather than
guessed. A number that is absent is honest; a number that is stale is not.
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
UNKNOWN = "not measured"


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd or ROOT, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def ctest_suites(build_dir: pathlib.Path) -> str:
    if not build_dir.is_dir():
        return UNKNOWN
    out = run(["ctest", "--test-dir", str(build_dir), "-N"])
    if not out:
        return UNKNOWN
    m = re.search(r"Total Tests:\s*(\d+)", out)
    return m.group(1) if m else UNKNOWN


def coverage(build_dir: pathlib.Path) -> dict[str, str]:
    summary = build_dir / "coverage-summary.txt"
    result = {"lines": UNKNOWN, "functions": UNKNOWN, "branches": UNKNOWN}
    if not summary.is_file():
        return result
    text = summary.read_text(encoding="utf-8", errors="replace")
    for key in result:
        m = re.search(rf"{key}\.*:\s*([0-9.]+)%", text)
        if m:
            result[key] = f"{m.group(1)}%"
    return result


def ci_gates() -> dict[str, str]:
    """The thresholds CI actually enforces, read from the workflow."""
    wf = ROOT / ".github" / "workflows" / "ci.yml"
    gates = {"MS_COVERAGE_MIN": UNKNOWN, "MS_COVERAGE_BRANCH_MIN": UNKNOWN,
             "MS_COVERAGE_FUNC_MIN": UNKNOWN, "MS_BENCH_TOLERANCE": UNKNOWN}
    if not wf.is_file():
        return gates
    text = wf.read_text(encoding="utf-8", errors="replace")
    for key in gates:
        m = re.search(rf'{key}:\s*"?([0-9]+)"?', text)
        if m:
            gates[key] = m.group(1)
    return gates


def benchmarks() -> dict[str, str]:
    baseline = ROOT / "tests" / "performance" / "baselines" / "linux-gcc13.json"
    info = {"entries": UNKNOWN, "measured": UNKNOWN}
    if not baseline.is_file():
        return info
    try:
        data = json.loads(baseline.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return info
    bench = data.get("benchmarks", data)
    if isinstance(bench, dict):
        info["entries"] = str(len(bench))
        info["measured"] = str(
            sum(1 for v in bench.values()
                if isinstance(v, dict) and v.get("median_time_ns") is not None)
        )
    return info


def unsafe_sites() -> str:
    baseline = ROOT / "tests" / "compliance" / "unsafe_baseline.txt"
    if not baseline.is_file():
        return UNKNOWN
    n = sum(
        1 for line in baseline.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return str(n)


def git_facts() -> dict[str, str]:
    commit = (run(["git", "rev-parse", "--short", "HEAD"]) or "").strip() or UNKNOWN
    tags = run(["git", "tag", "--list", "v1.0.0"]) or ""
    return {"commit": commit, "v100_tag": "present" if tags.strip() else "not cut"}


def source_lines() -> str:
    total = 0
    for d in ("src", "include"):
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.suffix in {".cpp", ".hpp", ".h", ".cu"} and p.is_file():
                if "vendor" in p.parts:
                    continue
                try:
                    total += p.read_text(encoding="utf-8", errors="replace").count("\n")
                except OSError:
                    pass
    return f"{total:,}" if total else UNKNOWN


def render(build_dir: pathlib.Path) -> str:
    cov = coverage(build_dir)
    gates = ci_gates()
    bench = benchmarks()
    git = git_facts()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def pct(v: str) -> str:
        """A gate value with its unit, or the unknown marker without one."""
        return UNKNOWN if v == UNKNOWN else f"{v}%"

    return f"""# Status

**Generated** by `scripts/gen_status.py` at {stamp} from commit `{git['commit']}`.
Do not edit by hand; run the script.

Every figure here is read from a build artefact. Anything the script could not
read says "{UNKNOWN}" rather than carrying a value forward, because a number that
is absent is honest and a stale one is not. This file exists because seven
published claims were checkable in five minutes and wrong.

## Tests

| | |
|---|---|
| CTest suites | {ctest_suites(build_dir)} |
| Source lines (`src` + `include`, excluding vendor) | {source_lines()} |

## Coverage

Measured over the denominator declared in `scripts/coverage_exclusions.txt`.
Only paths that *cannot* execute on a CI runner are excluded, and each run
prints how many lines they hid.

| | Measured | CI gate |
|---|---|---|
| Lines | {cov['lines']} | {pct(gates['MS_COVERAGE_MIN'])} |
| Functions | {cov['functions']} | {pct(gates['MS_COVERAGE_FUNC_MIN'])} |
| Branches | {cov['branches']} | {pct(gates['MS_COVERAGE_BRANCH_MIN'])} |

The gate and the measurement are separate columns on purpose. The README once
claimed CI enforced 90% while `ci.yml` set 80%, and nothing reconciled them.

## Benchmarks

| | |
|---|---|
| Baseline entries | {bench['entries']} |
| Of which carry a measured median | {bench['measured']} |
| Regression tolerance | {pct(gates['MS_BENCH_TOLERANCE'])} |

A large gap between those first two rows means the comparison is skipping most
of the suite. See `docs/RELEASE.md` criterion 8 for why that is deliberate.

## Compliance

| | |
|---|---|
| Reviewed unsafe sites | {unsafe_sites()} |
| SPDX headers | enforced by `scripts/add_spdx.py --check` |
| SBOM | `sbom.cdx.json`, checked by `scripts/gen_sbom.py --check` |
| Formal verification | `verification/run.sh` under ESBMC and CBMC |

## Release

| | |
|---|---|
| `v1.0.0` tag | {git['v100_tag']} |

Tag criteria are in [`RELEASE.md`](RELEASE.md).
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir", default="build-test")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("-o", "--output", default=str(ROOT / "docs" / "STATUS.md"))
    args = ap.parse_args()

    build_dir = pathlib.Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir

    rendered = render(build_dir)
    out = pathlib.Path(args.output)

    if args.check:
        if not out.is_file():
            print(f"{out} is missing; run scripts/gen_status.py", file=sys.stderr)
            return 1
        # The timestamp and commit line moves every run, so compare the body.
        def body(t: str) -> str:
            return "\n".join(l for l in t.splitlines() if not l.startswith("**Generated**"))
        if body(out.read_text(encoding="utf-8")) != body(rendered):
            print(f"{out} is stale; run scripts/gen_status.py", file=sys.stderr)
            return 1
        print(f"{out} is up to date")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
