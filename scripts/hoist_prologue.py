#!/usr/bin/env python3
"""Remove the duplicated argument-helper prologue from the generated handlers.

The generator that produced src/interp/matrix_calls/ stamped the same four
lambdas into every handler regardless of use:

    resolve_operand           defined in 481 files, called in 485
    parse_scalar_arg          defined in 485 files, called in 117
    parse_positive_size_arg   defined in 480 files, called in   9
    parse_uint64_arg          defined in 480 files, called in   1

The bodies are identical in every copy, so they now live on MatrixCallCtx
(src/interp/matrix_call.hpp) and this script deletes the local definitions and
points the call sites at the context.

    python3 scripts/hoist_prologue.py --check    # report, change nothing
    python3 scripts/hoist_prologue.py            # rewrite in place

Only the prologue is touched. Handler bodies and every DomainError payload are
left exactly as they were, so behaviour does not move -- the point is to delete
code that was never reachable, not to rewrite what runs.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANDLERS = ROOT / "src" / "interp" / "matrix_calls"

# Each lambda as the generator emitted it. Matched structurally rather than by
# exact text so a stray difference in whitespace does not silently skip a file.
LAMBDA_PATTERNS = {
    "resolve_operand": re.compile(
        r"[ \t]*auto resolve_operand = \[&ctx\]\(const std::string& text\) \{[^\n]*\n?",
    ),
    "parse_scalar_arg": re.compile(
        r"[ \t]*auto parse_scalar_arg = \[&ctx\]\(.*?\n(?:.*?\n)*?[ \t]*\};[ \t]*\n",
    ),
    "parse_positive_size_arg": re.compile(
        r"[ \t]*auto parse_positive_size_arg = \[\]\(.*?\n(?:.*?\n)*?[ \t]*\};[ \t]*\n",
    ),
    "parse_uint64_arg": re.compile(
        r"[ \t]*auto parse_uint64_arg = \[\]\(.*?\n(?:.*?\n)*?[ \t]*\};[ \t]*\n",
    ),
}

HELPERS = list(LAMBDA_PATTERNS)


def strip_definition(text: str, helper: str) -> tuple[str, bool]:
    """Remove one lambda definition. Returns (text, removed)."""
    pattern = LAMBDA_PATTERNS[helper]
    m = pattern.search(text)
    if not m:
        return text, False
    return text[: m.start()] + text[m.end() :], True


def redirect_calls(text: str, helper: str) -> tuple[str, int]:
    """Point bare calls at the context. Skips anything already qualified."""
    # (?<![\w.>]) keeps ctx.parse_scalar_arg( and obj->parse_scalar_arg( intact.
    pattern = re.compile(r"(?<![\w.])(?<!->)\b" + re.escape(helper) + r"\(")
    new_text, n = pattern.subn(f"ctx.{helper}(", text)
    return new_text, n


def process(path: pathlib.Path, apply: bool) -> dict:
    original = path.read_text(encoding="utf-8")
    text = original
    removed: list[str] = []

    for helper in HELPERS:
        text, was_removed = strip_definition(text, helper)
        if was_removed:
            removed.append(helper)

    redirected = 0
    for helper in HELPERS:
        text, n = redirect_calls(text, helper)
        redirected += n

    # A handler that used none of them still declares ctx; leaving an unused
    # variable would trip -Wunused-variable, so keep it only if referenced.
    if "ctx." not in text and "ctx)" not in text:
        text = re.sub(r"[ \t]*MatrixCallCtx ctx\(interp\);[ \t]*\n", "", text, count=1)
        text = re.sub(
            r"(Result<Matrix<double>> handle_\w+\()Interpreter& interp,",
            r"\1Interpreter& /*interp*/,",
            text,
            count=1,
        )

    changed = text != original
    if changed and apply:
        path.write_text(text, encoding="utf-8")

    return {
        "path": path,
        "changed": changed,
        "removed": removed,
        "redirected": redirected,
        "lines_before": original.count("\n"),
        "lines_after": text.count("\n"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report without rewriting")
    args = ap.parse_args()

    if not HANDLERS.is_dir():
        print(f"handler directory not found: {HANDLERS}", file=sys.stderr)
        return 1

    files = sorted(HANDLERS.rglob("*.cpp"))
    if not files:
        print(f"no handlers under {HANDLERS}", file=sys.stderr)
        return 1

    results = [process(f, apply=not args.check) for f in files]

    removed_counts = {h: 0 for h in HELPERS}
    for r in results:
        for h in r["removed"]:
            removed_counts[h] += 1

    before = sum(r["lines_before"] for r in results)
    after = sum(r["lines_after"] for r in results)
    changed = sum(1 for r in results if r["changed"])
    redirected = sum(r["redirected"] for r in results)

    print(f"handlers:            {len(files)}")
    print(f"files changed:       {changed}")
    print(f"call sites redirect: {redirected}")
    for h in HELPERS:
        print(f"  removed {h:<26} from {removed_counts[h]} files")
    print(f"lines: {before} -> {after}  ({after - before:+d})")

    if args.check and changed:
        print("\n--check: files would change; run without --check to apply.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
