#!/usr/bin/env python3
"""Split tests/unit/test_repl_commands.cpp into helpers + wave-number TUs."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "tests" / "unit"
SRC = UNIT / "test_repl_commands.cpp"
CMAKE = ROOT / "tests" / "CMakeLists.txt"

TEST_RE = re.compile(r"TEST\s*\(\s*ReplCommandsTest\s*,\s*([A-Za-z_]\w*)\s*\)")
WAVE_RE = re.compile(r"^wave(\d+)_")
FUNC_START_RE = re.compile(
    r"^(void|double|std::pair<double, double>|std::vector<double>)\s+\w+\s*\("
)

HELPER_INCLUDES = """\
#pragma once

#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/interp/repl_engine.hpp"
"""

COMMON_INCLUDES = """\
#include <algorithm>
#include <cmath>
#include <set>
#include <fstream>
#include <gtest/gtest.h>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ms/cplx/cplx.hpp"
#include "ms/control/control.hpp"
#include "ms/error/error_types.hpp"
#include "ms/finance/finance.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/interp/repl_engine.hpp"
#include <ms/ml/ml.hpp>
#include "ms/ml/ml.hpp"
#include "ms/pde/pde.hpp"
#include "ms/prob/prob.hpp"
#include "ms/special/special.hpp"
#include "ms/frameworks/gria/gria.hpp"
#include "ms/quantum/quantum.hpp"
#include "ms/runtime/topology.hpp"
#include "ms/version.hpp"

#include "repl_test_helpers.hpp"

using namespace ms::interp;
"""


def find_matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    raise RuntimeError("unbalanced braces in anonymous namespace")


def qualify_interpreter(ns_body: str) -> str:
    ns_body = ns_body.replace("Interpreter&", "ms::interp::Interpreter&")
    ns_body = ns_body.replace("Interpreter::", "ms::interp::Interpreter::")
    return ns_body


def make_inline(ns_body: str) -> str:
    out: list[str] = []
    for line in ns_body.splitlines(keepends=True):
        if FUNC_START_RE.match(line):
            line = "inline " + line
        out.append(line)
    return "".join(out)


def extract_helpers(text: str) -> tuple[str, int]:
    ns_kw = text.find("\nnamespace {")
    if ns_kw < 0:
        raise RuntimeError("anonymous namespace not found")
    brace = text.find("{", ns_kw)
    close = find_matching_brace(text, brace)
    after = text[close + 1 :]
    comment_end = after.find("\n")
    ns_end = close + 1 + (comment_end + 1 if comment_end >= 0 else 0)
    inner = text[brace + 1 : close]
    inner = qualify_interpreter(inner)
    inner = make_inline(inner)
    header = (
        HELPER_INCLUDES
        + "\nnamespace {\n"
        + inner
        + "}  // namespace\n"
    )
    return header, ns_end


def collect_tests(text: str, search_from: int) -> list[tuple[str, str]]:
    matches = list(TEST_RE.finditer(text, search_from))
    if not matches:
        raise RuntimeError("no TEST macros found")
    tests: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        if not body.endswith("\n"):
            body += "\n"
        tests.append((m.group(1), body.rstrip() + "\n"))
    return tests


def wave_number(name: str) -> int | None:
    m = WAVE_RE.match(name)
    return int(m.group(1)) if m else None


def bucket_name(lo_thousand: int) -> str:
    lo = lo_thousand * 1000
    hi = lo + 999
    return f"test_repl_commands_w{lo:04d}_{hi:04d}.cpp"


def write_cpp(path: Path, bodies: list[str]) -> None:
    chunks = [COMMON_INCLUDES.rstrip(), ""]
    for body in bodies:
        chunks.append(body.rstrip())
        chunks.append("")
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8", newline="\n")


def cmake_snippet(wave_files: list[str]) -> str:
    sources = ["    unit/test_repl_commands.cpp"] + [
        f"    unit/{name}" for name in wave_files
    ]
    source_block = "\n".join(sources)
    return f"""add_executable(test_repl_commands
{source_block}
)
target_link_libraries(test_repl_commands PRIVATE mathscript GTest::gtest_main)
target_include_directories(test_repl_commands PRIVATE
    ${{CMAKE_SOURCE_DIR}}/include
    ${{CMAKE_BINARY_DIR}}/include
    ${{CMAKE_CURRENT_SOURCE_DIR}}/unit
)
if(MS_ENABLE_COVERAGE AND NOT MSVC)
    target_compile_options(test_repl_commands PRIVATE -fexceptions)
endif()
add_test(NAME test_repl_commands COMMAND test_repl_commands)
if(MSVC)
    target_compile_options(test_repl_commands PRIVATE /bigobj)
endif()
"""


def update_cmake(wave_files: list[str]) -> None:
    text = CMAKE.read_text(encoding="utf-8")
    old = (
        "add_ms_test(test_repl_commands unit/test_repl_commands.cpp)\n"
        "if(MSVC)\n"
        "    target_compile_options(test_repl_commands PRIVATE /bigobj)\n"
        "endif()\n"
    )
    new = cmake_snippet(wave_files)
    if old not in text:
        raise RuntimeError("expected add_ms_test(test_repl_commands ...) block not found")
    CMAKE.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def names_in_file(path: Path) -> list[str]:
    return [m.group(1) for m in TEST_RE.finditer(path.read_text(encoding="utf-8"))]


def main() -> int:
    original = SRC.read_text(encoding="utf-8")
    original_names = [m.group(1) for m in TEST_RE.finditer(original)]
    original_count = len(original_names)
    if len(set(original_names)) != original_count:
        dups = [n for n in original_names if original_names.count(n) > 1]
        raise RuntimeError(f"duplicate TEST names in original file: {sorted(set(dups))}")

    helpers, ns_end = extract_helpers(original)
    tests = collect_tests(original, ns_end)

    early: list[tuple[str, str]] = []
    waves: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for name, body in tests:
        wn = wave_number(name)
        if wn is None:
            early.append((name, body))
        else:
            waves[wn // 1000].append((name, body))

    helper_path = UNIT / "repl_test_helpers.hpp"
    helper_path.write_text(helpers, encoding="utf-8", newline="\n")

    write_cpp(SRC, [body for _, body in early])

    wave_filenames: list[str] = []
    per_file: dict[str, int] = {SRC.name: len(early)}
    for thousand in sorted(waves):
        fname = bucket_name(thousand)
        wave_filenames.append(fname)
        bodies = [body for _, body in waves[thousand]]
        write_cpp(UNIT / fname, bodies)
        per_file[fname] = len(bodies)

    update_cmake(wave_filenames)

    new_names: list[str] = []
    for fname in [SRC.name, *wave_filenames]:
        new_names.extend(names_in_file(UNIT / fname))

    orig_set = set(original_names)
    new_set = set(new_names)
    missing = sorted(orig_set - new_set)
    extra = sorted(new_set - orig_set)
    counts: dict[str, int] = defaultdict(int)
    for n in new_names:
        counts[n] += 1
    dup_after = sorted(n for n, c in counts.items() if c != 1)

    print(f"original TEST count: {original_count}")
    print(f"new TEST count:      {len(new_names)}")
    print(f"unique new names:    {len(new_set)}")
    print()
    print("file list / counts:")
    print(f"  {helper_path.relative_to(ROOT)}  (helpers, no TEST)")
    for fname in [SRC.name, *wave_filenames]:
        print(f"  tests/unit/{fname}  {per_file[fname]}")
    print()
    print("cmake snippet:")
    print(cmake_snippet(wave_filenames))

    errors: list[str] = []
    if missing:
        errors.append(f"missing names ({len(missing)}): {missing[:10]}")
    if extra:
        errors.append(f"extra names ({len(extra)}): {extra[:10]}")
    if dup_after:
        errors.append(f"names not appearing exactly once: {dup_after[:10]}")
    if len(new_names) != original_count:
        errors.append(f"count mismatch {original_count} -> {len(new_names)}")
    if errors:
        print("VERIFICATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    print("VERIFICATION OK: every original TEST name appears exactly once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
