#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Odin Loch
"""§8.4: change the code on purpose and see whether a test notices.

Coverage says a line ran. It does not say anything asserted. A line that is covered
and untested is exactly the failure a coverage percentage conceals, and the only way
to find one is to break the line and watch what happens: if every test still passes,
nothing was checking it.

    python3 scripts/mutation_test.py --source src/compress/compress.cpp \\
                                     --target test_compress

For each mutation the harness edits one character-range of the source, rebuilds just
the target that covers it, runs that target, and puts the result in one of four boxes:

    killed        a test failed -- something was asserting the behaviour
    survived      every test passed -- THE FINDING
    not viable    the mutant did not compile
    timed out     the mutant hangs; counted as killed, since a hang is a failure

**"Not viable" is reported separately and is not counted as killed.** Folding it in is
the standard way a mutation score is inflated: a harness that generates mostly
uncompilable mutants and calls them killed reports 95% while testing nothing. The score
here is over viable mutants only, and the raw counts are printed so the reader can see
how many there were.

The mutations are the classic set -- a relational operator moved by one, an arithmetic
operator swapped, a boolean connective flipped, a small constant changed, a `return
true` turned into `return false`. They are applied one at a time, and the file is
restored after each, including on Ctrl-C: a harness that leaves a mutated tree behind
is worse than no harness.

Sites are found after blanking comments and string literals, so a `<` inside a message
is not a site. Some sites are still nonsense in context (`<` inside a template argument
list, `+` in a fold expression); those come back as not viable, which costs a compile
and no correctness. Deciding it statically would mean parsing C++, and the compiler is
already here.
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Mutation:
    offset: int
    length: int
    replacement: str
    kind: str

    def line_of(self, text: str) -> int:
        return text.count("\n", 0, self.offset) + 1

    def context(self, text: str) -> str:
        start = text.rfind("\n", 0, self.offset) + 1
        end = text.find("\n", self.offset)
        return text[start:end if end != -1 else len(text)].strip()


def blank_strings_and_comments(text: str) -> str:
    """`text` with every comment and string literal replaced by spaces of equal length.

    Offsets are preserved so a site found in the blanked text is the same site in the
    original. Newlines are kept so line numbers still work.
    """
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "//":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif two == "/*":
            while i < n and text[i:i + 2] != "*/":
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            for j in range(i, min(i + 2, n)):
                out[j] = " "
            i += 2
        elif text[i] in "\"'":
            quote = text[i]
            out[i] = " "
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    out[i] = " "
                    i += 1
                if i < n and text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


# (pattern, replacement, kind). Order matters: the two-character operators are matched
# first so that `<=` is never seen as a `<`.
OPERATORS: list[tuple[str, str, str]] = [
    (r"<=", ">=", "relational"),
    (r">=", "<=", "relational"),
    (r"==", "!=", "equality"),
    (r"!=", "==", "equality"),
    (r"&&", "||", "boolean"),
    (r"\|\|", "&&", "boolean"),
]

SINGLE: list[tuple[str, str, str]] = [
    ("<", "<=", "relational"),
    (">", ">=", "relational"),
    ("+", "-", "arithmetic"),
    ("-", "+", "arithmetic"),
    ("*", "/", "arithmetic"),
]


def find_sites(text: str, rng: random.Random) -> list[Mutation]:
    code = blank_strings_and_comments(text)
    sites: list[Mutation] = []
    taken: set[int] = set()

    for pattern, replacement, kind in OPERATORS:
        for match in re.finditer(pattern, code):
            if any(offset in taken for offset in range(match.start(), match.end())):
                continue
            taken.update(range(match.start(), match.end()))
            sites.append(Mutation(match.start(), match.end() - match.start(), replacement, kind))

    for index, character in enumerate(code):
        if index in taken:
            continue
        for symbol, replacement, kind in SINGLE:
            if character != symbol:
                continue
            neighbours = code[max(0, index - 1):index + 2]
            # Skip the spellings where the character is part of something else: shifts
            # and streams, arrows, compound assignment, increment, and a template's
            # angle brackets after an identifier.
            if any(token in neighbours for token in ("<<", ">>", "->", "+=", "-=", "*=",
                                                     "++", "--", "/*", "*/", "//")):
                continue
            if symbol in "<>" and re.match(r"[A-Za-z_]", code[max(0, index - 1)]):
                continue
            if symbol == "*" and re.match(r"[\s(,=]", code[max(0, index - 1)] or " "):
                continue  # a dereference or a declarator, not a multiply
            sites.append(Mutation(index, 1, replacement, kind))
            break

    # `return true;` -> `return false;` and back. These are the cheapest way to find a
    # predicate nobody checks.
    for match in re.finditer(r"return\s+(true|false)\s*;", code):
        word = code[match.start(1):match.end(1)]
        sites.append(Mutation(match.start(1), len(word),
                              "false" if word == "true" else "true", "return"))

    # A small integer literal moved by one. Only 0..4: past that the constant is usually
    # a size or a limit whose exact value no test could reasonably pin.
    for match in re.finditer(r"(?<![\w.])([0-4])(?![\w.])", code):
        value = int(match.group(1))
        sites.append(Mutation(match.start(1), 1, str(value + 1), "constant"))

    rng.shuffle(sites)
    return sites


def run(command: list[str], timeout: int) -> tuple[int, str]:
    try:
        finished = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return finished.returncode, finished.stdout + finished.stderr
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="the file to mutate")
    parser.add_argument("--target", required=True,
                        help="the CTest/ninja target that covers it, e.g. test_compress")
    parser.add_argument("--build-dir", default="build-test")
    parser.add_argument("--limit", type=int, default=30, help="mutants to try")
    parser.add_argument("--seed", type=int, default=1,
                        help="which mutants get tried; a run is reproducible from it")
    parser.add_argument("--build-timeout", type=int, default=1800,
                        help="per-mutant build; the baseline gets four times this, "
                             "because it may be a cold build of the whole tree")
    parser.add_argument("--test-timeout", type=int, default=600)
    args = parser.parse_args()

    source = Path(args.source)
    original = source.read_text()
    sites = find_sites(original, random.Random(args.seed))
    if not sites:
        print(f"no mutation sites in {source}", file=sys.stderr)
        return 1

    build = ["ninja", "-C", args.build_dir, args.target]
    test = [str(Path(args.build_dir) / "tests" / args.target)]

    print(f"{source}: {len(sites)} sites, trying {min(args.limit, len(sites))} "
          f"(seed {args.seed})")
    # The baseline may be a cold build of everything the target links, which is a
    # different order of cost from a mutant's rebuild of one translation unit. Giving it
    # the same budget is how the first run of this harness reported "the unmutated tree
    # does not build" about a tree that builds perfectly well and was still compiling.
    print("establishing the baseline...", flush=True)
    code, output = run(build, args.build_timeout * 4)
    if code == 124:
        print(f"the baseline build did not finish within {args.build_timeout * 4}s. That "
              f"is a timeout, not a failure: build the target once by hand and run this "
              f"again, or raise --build-timeout.", file=sys.stderr)
        return 1
    if code != 0:
        print("the unmutated tree does not build; fix that first\n" + output[-2000:],
              file=sys.stderr)
        return 1
    code, output = run(test, args.test_timeout)
    if code != 0:
        print("the unmutated tree does not pass; a survivor would mean nothing\n"
              + output[-2000:], file=sys.stderr)
        return 1
    print("baseline green\n", flush=True)

    killed = survived = not_viable = timed_out = 0
    survivors: list[Mutation] = []
    try:
        for index, mutation in enumerate(sites[:args.limit], start=1):
            mutated = (original[:mutation.offset] + mutation.replacement
                       + original[mutation.offset + mutation.length:])
            source.write_text(mutated)
            label = (f"[{index}/{min(args.limit, len(sites))}] line "
                     f"{mutation.line_of(original)} {mutation.kind}: "
                     f"{original[mutation.offset:mutation.offset + mutation.length]!r}"
                     f" -> {mutation.replacement!r}")
            build_code, _ = run(build, args.build_timeout)
            if build_code != 0:
                not_viable += 1
                print(f"{label}: not viable", flush=True)
                continue
            test_code, _ = run(test, args.test_timeout)
            if test_code == 124:
                timed_out += 1
                print(f"{label}: TIMED OUT (counted as killed)", flush=True)
            elif test_code != 0:
                killed += 1
                print(f"{label}: killed", flush=True)
            else:
                survived += 1
                survivors.append(mutation)
                print(f"{label}: SURVIVED", flush=True)
    finally:
        source.write_text(original)
        run(build, args.build_timeout)

    viable = killed + survived + timed_out
    print()
    print(f"viable {viable}, killed {killed + timed_out}, survived {survived}, "
          f"not viable {not_viable}")
    if viable:
        print(f"mutation score over viable mutants: "
              f"{100.0 * (killed + timed_out) / viable:.1f}%")
    if survivors:
        print("\nSURVIVORS -- each is a line that ran and that nothing asserted:")
        for mutation in survivors:
            print(f"  {source}:{mutation.line_of(original)}  {mutation.kind}  "
                  f"{original[mutation.offset:mutation.offset + mutation.length]!r}"
                  f" -> {mutation.replacement!r}")
            print(f"      {mutation.context(original)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
