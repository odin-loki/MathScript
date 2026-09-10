#!/usr/bin/env python3
"""Parse every matrix-call handler into a machine-readable manifest.

478 handlers dispatch on `assign.callee` and then on `assign.args.size()`. What
each one accepts is currently knowable only by opening it, which is why the one
handler that guards its arity differently from the other 477 went unnoticed until
something read all of them at once.

    python3 scripts/extract_manifest.py                 # write the manifest
    python3 scripts/extract_manifest.py --check         # fail if it is stale (CI)

The manifest is not a summary. It is the input to `gen_matrix_call_tests.py`, so
a handler this cannot parse is a handler whose dispatch behaviour nothing tests --
hence the refusal to guess. Every anomaly is reported and, with --strict, fails.

What is extracted, per handler:

    callee          the string dispatched on
    arities         the argument counts that reach the body, solved rather than
                    pattern-matched: the guard is translated to a predicate over
                    n and evaluated for every n in 0..MAX_ARITY
    open_ended      the guard admits arities above MAX_ARITY (a bare `>= k`)
    resolves_first  arg[0] goes through ctx.resolve_operand, so an undefined name
                    must surface as an error rather than a crash
    helpers         which MatrixCallCtx helpers the body uses
    domain_errors   every DomainError reachable in the file

The guard is read, not assumed. A handler whose top-level condition mentions
anything other than the callee and the argument count is an anomaly by
construction: such a guard cannot be reasoned about from arity alone, and
pretending otherwise would generate tests that assert the wrong thing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HANDLER_DIR = ROOT / "src" / "interp" / "matrix_calls"
MANIFEST = ROOT / "tests" / "unit" / "matrix_calls" / "matrix_calls_manifest.json"

# The largest arity any handler accepts, plus headroom. Guards are evaluated over
# 0..MAX_ARITY; one that still accepts at the top of the range is recorded as
# open-ended rather than silently truncated.
MAX_ARITY = 14

HELPERS = [
    "resolve_operand",
    "parse_scalar_arg",
    "parse_positive_size_arg",
    "parse_uint64_arg",
    "session_objects",
    "parse_number",
    "eval_scalar_expr",
]


class ParseError(Exception):
    """The handler does not match the shape the manifest can describe."""


def strip_comments(text: str) -> str:
    """Remove comments without disturbing string literals.

    A naive regex would eat the `//` inside a URL in a message string and take the
    rest of the line with it, quietly changing a DomainError the manifest reports.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j == -1 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def balanced(text: str, open_at: int) -> tuple[str, int]:
    """Return the contents of the parenthesised group starting at `open_at`."""
    assert text[open_at] == "("
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i], i + 1
    raise ParseError("unbalanced parentheses in guard condition")


def find_guard(body: str) -> tuple[str, bool, tuple[str, str] | None]:
    """The condition that decides whether the call is this handler's, and whether
    it is written as a rejection.

    Two idioms are in the tree: a positive guard whose body does the work, and an
    early return whose body rejects. They mean opposite things, so which one this
    is has to be read rather than assumed.
    """
    m = re.search(r"\bif\s*\(", body)
    while m:
        cond, after = balanced(body, m.end() - 1)
        if "assign.callee" in cond:
            tail = body[after:].lstrip()
            if tail.startswith("{"):
                tail = tail[1:].lstrip()
            # An early-return guard states its own rejection error, and four
            # handlers use that to say what signature they expected rather than
            # the generic sentinel. Capture it: a test that assumed one message
            # for all 485 would be asserting a convention rather than a contract.
            m_err = re.match(
                r'return\s+std::unexpected\s*\(\s*DomainError\s*\{\s*'
                r'"([^"]*)"\s*,\s*"((?:[^"\\]|\\.)*)"', tail)
            rejects = m_err is not None
            rejection = (m_err.group(1), m_err.group(2)) if m_err else None
            return cond, rejects, rejection
        m = re.search(r"\bif\s*\(", body[after:])
        if m:
            m = re.search(r"\bif\s*\(", body, after)
    raise ParseError("no dispatch guard mentioning assign.callee")


# What a guard is permitted to contain once the callee and the argument count have
# been substituted out. Anything else means the guard depends on argument content,
# and its accepted arities are not a function of n alone.
RESIDUE = re.compile(r"^[\s()n0-9<>=!]*$")


def solve_arities(cond: str, callee: str, rejects: bool) -> tuple[list[int], bool]:
    # The guard is one expression however it is wrapped in the source; eval is not
    # so forgiving about the line breaks clang-format put in it.
    expr = " ".join(cond.split())
    expr = re.sub(r'assign\.callee\s*==\s*"([^"]*)"',
                  lambda m: "True" if m.group(1) == callee else "False", expr)
    expr = re.sub(r'assign\.callee\s*!=\s*"([^"]*)"',
                  lambda m: "False" if m.group(1) == callee else "True", expr)
    expr = expr.replace("assign.args.size()", "n")

    # Validate before translating the operators, so the check runs against C++
    # tokens rather than against Python ones that could have come from anywhere.
    residue = expr.replace("True", "").replace("False", "")
    residue = residue.replace("&&", "").replace("||", "")
    if not RESIDUE.match(residue):
        raise ParseError(
            f"guard is not a function of the argument count alone: {cond.strip()!r}")

    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"!(?!=)", " not ", expr)

    accepted: list[int] = []
    for n in range(MAX_ARITY + 1):
        try:
            taken = bool(eval(expr, {"__builtins__": {}}, {"n": n}))  # noqa: S307
        except Exception as exc:  # pragma: no cover - shape already validated
            raise ParseError(f"could not evaluate guard {cond.strip()!r}: {exc}") from exc
        if taken != rejects:
            accepted.append(n)
    return accepted, MAX_ARITY in accepted


def parse_handler(path: pathlib.Path) -> dict:
    text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))

    fn = re.search(r"Result<Matrix<double>>\s+handle_(\w+)\s*\(", text)
    if not fn:
        raise ParseError("no handle_* function")
    brace = text.find("{", fn.end())
    if brace == -1:
        raise ParseError("handler has no body")
    body, _ = balanced(text.replace("{", "(", 1) if False else text, brace) if False else (None, 0)

    # Take the body by brace matching from the opening brace of the function.
    depth, end = 0, None
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise ParseError("handler body is not brace-balanced")
    body = text[brace + 1 : end]

    reg = re.search(r'register_matrix_call\s*\(\s*"([^"]+)"', text)
    if not reg:
        raise ParseError("no register_matrix_call call")
    callee = reg.group(1)

    cond, rejects, rejection = find_guard(body)
    guard_callees = sorted(set(re.findall(r'assign\.callee\s*[!=]=\s*"([^"]*)"', cond)))
    arities, open_ended = solve_arities(cond, callee, rejects)
    if not arities:
        raise ParseError(f"guard accepts no arity at all: {cond.strip()!r}")

    errors = sorted({
        (m.group(1), m.group(2))
        for m in re.finditer(r'DomainError\s*\{\s*"([^"]*)"\s*,\s*"((?:[^"\\]|\\.)*)"', body)
    })

    return {
        "file": path.relative_to(ROOT).as_posix(),
        "callee": callee,
        "handler": f"handle_{fn.group(1)}",
        "guard": " ".join(cond.split()),
        "guard_rejects": rejects,
        # What a call this handler declines actually reports. Handlers with a
        # positive guard leave the pre-initialised sentinel in place; the
        # early-return ones name their own signature.
        "rejection": ({"fn": rejection[0], "message": rejection[1]} if rejection
                      else {"fn": "assign", "message": "unsupported matrix call"}),
        # Guards naming callees other than the registered one are dead branches:
        # dispatch_matrix_call has already selected this handler by name.
        "dead_guard_callees": [c for c in guard_callees if c != callee],
        "arities": arities,
        "open_ended": open_ended,
        "resolves_first": "resolve_operand(assign.args[0])" in body.replace(" ", ""),
        "helpers": sorted(h for h in HELPERS if h in body),
        "domain_errors": [{"fn": f, "message": msg} for f, msg in errors],
    }


def build() -> tuple[dict, list[str]]:
    entries: list[dict] = []
    anomalies: list[str] = []
    for path in sorted(HANDLER_DIR.rglob("*.cpp")):
        try:
            entries.append(parse_handler(path))
        except ParseError as exc:
            anomalies.append(f"{path.relative_to(ROOT).as_posix()}: {exc}")

    seen: dict[str, str] = {}
    for e in entries:
        if e["callee"] in seen:
            anomalies.append(
                f"{e['file']}: callee {e['callee']!r} also registered by {seen[e['callee']]}")
        seen[e["callee"]] = e["file"]

    entries.sort(key=lambda e: e["callee"])
    manifest = {
        "generated_by": "scripts/extract_manifest.py",
        "max_arity_probed": MAX_ARITY,
        "handler_count": len(entries),
        "handlers": entries,
    }
    return manifest, anomalies


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if the manifest is stale")
    ap.add_argument("--strict", action="store_true", help="fail on any anomaly")
    ap.add_argument("-o", "--output", default=str(MANIFEST))
    args = ap.parse_args()

    manifest, anomalies = build()
    if not manifest["handlers"]:
        print(f"no handlers found under {HANDLER_DIR}", file=sys.stderr)
        return 1

    rendered = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    out = pathlib.Path(args.output)

    for a in anomalies:
        print(f"anomaly: {a}", file=sys.stderr)

    if args.check:
        if not out.is_file():
            print(f"{out} is missing; run scripts/extract_manifest.py", file=sys.stderr)
            return 1
        if out.read_text(encoding="utf-8") != rendered:
            print(f"{out} is stale; run scripts/extract_manifest.py", file=sys.stderr)
            return 1
        print(f"{out} is up to date ({manifest['handler_count']} handlers)")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
        print(f"wrote {out}: {manifest['handler_count']} handlers, {len(anomalies)} anomalies")

    return 1 if (anomalies and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
