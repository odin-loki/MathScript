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

`--target` takes every target that covers the file, comma-separated. A mutant is killed
if any of them fails, and naming only some of them reports survivors that are not.

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

Each survivor is reported as `file:line:column`, with the line printed twice -- as it
is written and as the mutant has it. A line and an operator are not an address: `a && b`
beside `c && d`, or the two `n - 1` in one constructor call, are different mutants that
print identically, and hand-reproducing the wrong one credits a kill to a mutant that
was never run. The column says which, and the `now:` line says exactly what to write.

A run is reproducible from its seed ONLY while the source is unchanged. The sample is
drawn from the list of character offsets in the file, so inserting a line renumbers every
site after it and the same seed then draws a different set: a "re-run" after a fix is a
second sample, not a before-and-after. `--replay` closes that gap. Point it at a previous
run's output and it re-tests exactly the survivors that run reported -- located by the
text of the line they sit on and the column within it, so they are still found after the
file moves underneath them:

    python3 scripts/mutation_test.py --source src/interp/repl_engine_internal.cpp \
                                     --target test_repl_commands --replay previous.txt

That is what turns "I wrote a test for this survivor" into "this survivor is dead", which
is the only claim worth making: a test that does not kill its mutant is the same silence
with more lines. A survivor whose line has since been edited cannot be located and is
reported as unresolved rather than quietly dropped.

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


# `fn == "sqrt"`, `callee == "det"`, `name == "plot"` -- the shape a dispatch
# branch takes throughout this interpreter.
DISPATCH_GUARD = re.compile(r'\b(?:fn|callee|name)\s*==\s*"')


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

    def column_of(self, text: str) -> int:
        """0-based column, so a line carrying several sites can be told apart.

        A line and an operator are not an address. `a && b` beside `c && d`, or
        the two `n - 1` in one constructor call, are different mutants that print
        identically, and reproducing the wrong one by hand is how a kill gets
        credited to a mutant that was never run.
        """
        return self.offset - (text.rfind("\n", 0, self.offset) + 1)

    def enclosing(self, text: str) -> str:
        """The nearest anchor above this site, as a second half of its address.

        Line text alone does not identify a line: `if (m < 1) {` occurs twice in
        `repl_engine_internal.cpp` and `if (model.rows() < 2 || model.cols() < 5) {`
        occurs four times. What differs is the context, and there are two useful
        stand-ins for it, whichever is nearer:

        - the dispatch guard the site sits under, `if (fn == "...")` and its
          spellings. `Interpreter::execute` is thirteen thousand lines of them, and
          its own signature tells none of its branches apart: twenty-four lines in
          it read `!parse_number(trim(match[4].str()), T) || ...` identically, one
          per option pricer, and the guard is the only thing that separates them.
        - failing that, the last line above starting in column 0 that is not a
          closing brace, a label or a preprocessor directive -- which, in this
          style, is the enclosing function's signature.
        """
        line_start = text.rfind("\n", 0, self.offset) + 1
        for candidate in reversed(text[:line_start].split("\n")):
            stripped = candidate.strip()
            if DISPATCH_GUARD.search(stripped):
                return stripped
            if not candidate or candidate[0].isspace():
                continue
            if candidate[0] in "}#" or candidate.startswith("//"):
                continue
            return stripped
        return ""

    def mutated_line(self, text: str) -> str:
        """The line as the mutant has it -- the thing to reproduce, not infer."""
        mutated = (text[:self.offset] + self.replacement
                   + text[self.offset + self.length:])
        start = mutated.rfind("\n", 0, self.offset) + 1
        end = mutated.find("\n", self.offset)
        return mutated[start:end if end != -1 else len(mutated)].strip()


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


def parse_ranges(spec: str) -> list[tuple[int, int]]:
    """`"100:200,340:355"` as inclusive (first, last) line pairs."""
    ranges: list[tuple[int, int]] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise SystemExit(f"--lines wants START:END, got {piece!r}")
        first, _, last = piece.partition(":")
        try:
            lo, hi = int(first), int(last)
        except ValueError:
            raise SystemExit(f"--lines wants two integers, got {piece!r}") from None
        if lo > hi:
            raise SystemExit(f"--lines range runs backwards: {piece!r}")
        ranges.append((lo, hi))
    if not ranges:
        raise SystemExit("--lines named no range")
    return ranges


SURVIVOR_LINE = re.compile(
    r"^\s*(?P<path>\S+):(?P<line>\d+):(?P<column>\d+)\s+(?P<kind>\w+)\s+"
    r"'(?P<was>.*)' -> '(?P<now>.*)'\s*$")


def parse_replay(report: str) -> list[dict]:
    """The survivors a previous run printed, as dicts.

    Reads the report the harness itself writes, so there is nothing to transcribe
    by hand -- transcription is exactly where a kill gets credited to the wrong
    mutant. Anything that is not a survivor header followed by its `was:` line is
    ignored, which lets the whole run log be passed in.
    """
    entries: list[dict] = []
    lines = report.split("\n")
    for index, line in enumerate(lines):
        match = SURVIVOR_LINE.match(line)
        if not match:
            continue
        tail = [following.strip() for following in lines[index + 1:index + 3]]
        enclosing = next((entry[len("in:  "):] for entry in tail
                          if entry.startswith("in:  ")), None)
        context = next((entry[len("was: "):] for entry in tail
                        if entry.startswith("was: ")), None)
        if context is None:
            continue
        entries.append({"path": match["path"],
                        "line": int(match["line"]),
                        "column": int(match["column"]),
                        "kind": match["kind"],
                        "was": match["was"],
                        "now": match["now"],
                        "text": context,
                        # Reports written before the `in:` line existed do not carry
                        # one; then the line text is the whole key, and a tie is
                        # reported rather than guessed.
                        "enclosing": enclosing})
    return entries


def locate_replayed(
        text: str, entries: list[dict]) -> tuple[list[Mutation], list[str], list[str]]:
    """Each replayed entry as a site in `text`, with notes for the ones that move.

    The address a report prints is `line:column`, and the line number is the part
    that goes stale the moment anything above it changes. The line's TEXT does not,
    so that is the key here: find the lines that read exactly as the report says the
    line read, and inside them the site at the recorded column with the recorded
    mutation. A site that matches in more than one place is as unusable as one that
    matches nowhere, and both are returned as notes rather than guessed at.

    Returns (sites, unresolved, relaxed). `relaxed` names the entries that were only
    found once the enclosing declaration was ignored -- renaming a function moves
    every site inside it by that key, and dropping them all would report a file as
    unmeasurable for a rename. Nothing is relaxed silently.
    """
    sites = find_sites(text, random.Random(0))
    by_line: dict[int, list[Mutation]] = {}
    for site in sites:
        by_line.setdefault(site.line_of(text), []).append(site)

    lines = text.split("\n")

    def candidates(entry: dict, use_enclosing: bool) -> list[Mutation]:
        return [site
                for number, raw in enumerate(lines, start=1)
                if raw.strip() == entry["text"]
                for site in by_line.get(number, ())
                if site.kind == entry["kind"]
                and text[site.offset:site.offset + site.length] == entry["was"]
                and site.replacement == entry["now"]
                and site.column_of(text) == entry["column"]
                and (not use_enclosing or entry["enclosing"] is None
                     or site.enclosing(text) == entry["enclosing"])]

    found: list[Mutation] = []
    unresolved: list[str] = []
    relaxed: list[str] = []
    for entry in entries:
        matches = candidates(entry, use_enclosing=True)
        where = f"{entry['line']}:{entry['column']} {entry['kind']} " \
                f"{entry['was']!r} -> {entry['now']!r}"
        if not matches and entry["enclosing"] is not None:
            widened = candidates(entry, use_enclosing=False)
            if widened:
                matches = widened
                relaxed.append(f"{where}: no longer inside {entry['enclosing']!r}, "
                               f"matched on the line alone")
        if len(matches) > 1:
            # `is_scalar_expression_rhs` repeats four of its lines verbatim, so the
            # enclosing declaration does not always separate them either. A candidate
            # sitting at exactly the line number the report gave is the one the report
            # meant unless the file moved out from under it, and if it had moved there
            # would be nothing at that number to match.
            here = [site for site in matches if site.line_of(text) == entry["line"]]
            if len(here) == 1:
                matches = here
        if len(matches) == 1:
            found.append(matches[0])
        elif not matches:
            unresolved.append(f"{where}: nothing in {entry['enclosing'] or 'the file'} "
                              f"now reads {entry['text']!r} with that site -- it was "
                              f"edited, which is often the point")
        else:
            unresolved.append(f"{where}: {len(matches)} lines read {entry['text']!r} "
                              f"identically; the report cannot say which")
    found.sort(key=lambda site: site.offset)
    return found, unresolved, relaxed


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
                        help="the target(s) covering it, comma-separated, e.g. "
                             "test_combo,test_combo_exact_tables,test_combo_overflow. A "
                             "mutant is killed if ANY of them fails, so leaving one out "
                             "reports a survivor that is not one")
    parser.add_argument("--build-dir", default="build-test")
    parser.add_argument("--lines", default=None,
                        help="restrict the sample to these inclusive line ranges, "
                             "comma-separated, e.g. 2680:2820,3100:3190. A uniform sample "
                             "over a four-thousand-line file lands a couple of mutants in "
                             "any one function, which is not a measurement of that "
                             "function; this aims the sample where the question is")
    parser.add_argument("--replay", default=None,
                        help="a previous run's output; re-test exactly the survivors it "
                             "reported instead of drawing a fresh sample. Sites are "
                             "located by the text of their line, so they survive the file "
                             "moving underneath them -- which a seed does not. --seed, "
                             "--lines and --limit do not apply: the sites are named")
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
    if args.replay:
        report = Path(args.replay).read_text()
        entries = parse_replay(report)
        if not entries:
            print(f"{args.replay} carries no survivor block to replay", file=sys.stderr)
            return 1
        elsewhere = sorted({entry["path"] for entry in entries
                            if Path(entry["path"]).name != source.name})
        if elsewhere:
            print(f"{args.replay} reports survivors in {', '.join(elsewhere)}, and "
                  f"--source is {source}; replaying only what is named for this file "
                  f"would be a silent half-measurement", file=sys.stderr)
            return 1
        sites, unresolved, relaxed = locate_replayed(original, entries)
        print(f"{source}: replaying {len(sites)} of {len(entries)} survivors "
              f"from {args.replay}")
        for note in relaxed:
            print(f"  relaxed:    {note}")
        for note in unresolved:
            print(f"  unresolved: {note}")
        if not sites:
            print("none of them could be located in the current source", file=sys.stderr)
            return 1
        # The sample is the report; a limit drawn from a different run would silently
        # measure a prefix of it.
        args.limit = len(sites)
        if args.lines:
            print("--lines is ignored when replaying: the sites are already named",
                  file=sys.stderr)
    else:
        sites = find_sites(original, random.Random(args.seed))
    if args.lines and not args.replay:
        ranges = parse_ranges(args.lines)
        before = len(sites)
        sites = [site for site in sites
                 if any(lo <= site.line_of(original) <= hi for lo, hi in ranges)]
        print(f"{source}: {before} sites, {len(sites)} within {args.lines}")
    if not sites:
        where = f" within {args.lines}" if args.lines else ""
        print(f"no mutation sites in {source}{where}", file=sys.stderr)
        return 1

    targets = [name.strip() for name in args.target.split(",") if name.strip()]
    if not targets:
        print("--target named nothing", file=sys.stderr)
        return 1
    build = ["ninja", "-C", args.build_dir, *targets]

    # Test binaries are not all directly under <build>/tests: §8.8 grouped the
    # integration and numerical suites into per-domain executables that land in
    # subdirectories, so `numerical_linalg_svd_adv` is at tests/numerical/. Assuming
    # the flat layout made the harness die with a FileNotFoundError from deep inside
    # subprocess -- after the baseline build, which on a cold tree is several
    # minutes -- and said nothing about which target it could not find.
    def locate(name: str) -> str:
        flat = Path(args.build_dir) / "tests" / name
        if flat.is_file():
            return str(flat)
        found = sorted(Path(args.build_dir).glob(f"tests/**/{name}"))
        found = [path for path in found if path.is_file()]
        if len(found) == 1:
            return str(found[0])
        if not found:
            print(f"no test binary named {name} under {args.build_dir}/tests -- "
                  f"build it first, or check the name", file=sys.stderr)
            sys.exit(1)
        print(f"{name} is ambiguous: {', '.join(str(p) for p in found)}",
              file=sys.stderr)
        sys.exit(1)

    tests = [[locate(name)] for name in targets]

    def run_tests() -> tuple[str, int]:
        """The first target that fails, and its code. ("", 0) when they all pass.

        A mutant is killed by ANY of them. Running one target of several is how a
        harness reports a survivor that is not one -- the mutation WAS caught, by a test
        the run did not execute -- and every false survivor costs somebody the
        investigation it takes to find that out. `src/combo/combo.cpp` is covered by
        three targets; `src/compress/compress.cpp` happened to be covered by one.
        """
        for name, command in zip(targets, tests):
            code, _ = run(command, args.test_timeout)
            if code != 0:
                return name, code
        return "", 0

    how = (f"replaying {len(sites)} named survivors" if args.replay
           else f"{len(sites)} sites, trying {min(args.limit, len(sites))} "
                f"(seed {args.seed})")
    print(f"{source}: {how} against {', '.join(targets)}")
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
    failed, code = run_tests()
    if failed:
        print(f"the unmutated tree does not pass ({failed} exited {code}); a survivor "
              "would mean nothing", file=sys.stderr)
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
            failed_target, test_code = run_tests()
            if test_code == 124:
                timed_out += 1
                print(f"{label}: TIMED OUT in {failed_target} (counted as killed)",
                      flush=True)
            elif test_code != 0:
                killed += 1
                print(f"{label}: killed by {failed_target}", flush=True)
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
    if viable and args.replay:
        # NOT a mutation score. A replay re-tests a hand-picked set -- the
        # survivors of an earlier run -- so the fraction killed says how many of
        # those particular findings are now closed. Printing it as a mutation
        # score would put a number for a biased sample next to numbers for
        # uniform ones, which is how a percentage starts meaning nothing.
        print(f"{killed + timed_out} of {viable} replayed survivors are now killed; "
              f"this is not a mutation score -- the sample was chosen, not drawn")
    elif viable:
        print(f"mutation score over viable mutants: "
              f"{100.0 * (killed + timed_out) / viable:.1f}%")
    if survivors:
        print("\nSURVIVORS -- each is a line that ran and that nothing asserted:")
        for mutation in survivors:
            print(f"  {source}:{mutation.line_of(original)}"
                  f":{mutation.column_of(original)}  {mutation.kind}  "
                  f"{original[mutation.offset:mutation.offset + mutation.length]!r}"
                  f" -> {mutation.replacement!r}")
            print(f"      in:  {mutation.enclosing(original)}")
            print(f"      was: {mutation.context(original)}")
            print(f"      now: {mutation.mutated_line(original)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
