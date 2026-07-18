"""Resolve wave262/combo-motzkin conflicts using index stages + union."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def show_stage(stage: int, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f":{stage}:{path}"],
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def extract(t: str, start: str, end: str) -> str:
    i = t.find(start)
    if i < 0:
        raise SystemExit(f"missing {start!r}")
    j = t.find(end, i + len(start))
    if j < 0:
        raise SystemExit(f"missing end after {start!r}")
    return t[i:j].rstrip() + "\n"


# --- repl_engine.cpp ---
ours = show_stage(2, "src/interp/repl_engine.cpp")
theirs = show_stage(3, "src/interp/repl_engine.cpp")
# Start from ours (has necklaces), inject motzkin pieces from theirs

text = ours

# 1) Insert eval helpers after de_bruijn helper (before next function)
# Find end of eval_combo_de_bruijn_sequence in ours
marker = "return int_vector_to_column(combo::de_bruijn_sequence(k, n));\n}"
pos = text.find(marker)
if pos < 0:
    # try alternate
    marker = "eval_combo_de_bruijn_sequence"
    pos = text.find(marker)
    if pos < 0:
        raise SystemExit("de_bruijn helper missing in ours")
    # find closing brace of function
    brace = text.find("\n}\n\n", pos)
    insert_at = brace + 3
else:
    insert_at = pos + len(marker)

motzkin_helpers = extract(
    theirs,
    "Result<Matrix<double>> eval_combo_motzkin_paths",
    "\nResult<",  # next Result function
)
# extract may cut at wrong place - get both helpers
h1 = extract(
    theirs,
    "Result<Matrix<double>> eval_combo_motzkin_paths",
    "Result<Matrix<double>> eval_combo_set_partitions",
)
h2_start = theirs.find("Result<Matrix<double>> eval_combo_set_partitions")
# end at next Result after set_partitions body
h2_end = theirs.find("\nResult<", h2_start + 10)
if h2_end < 0:
    h2_end = theirs.find("\nbool ", h2_start)
motzkin_helpers = theirs[h2_start:h2_end].rstrip() + "\n"
# Actually want both
motzkin_helpers = (
    theirs[
        theirs.find("Result<Matrix<double>> eval_combo_motzkin_paths") : h2_end
    ].rstrip()
    + "\n"
)
if "eval_combo_motzkin_paths" not in text:
    text = text[:insert_at] + "\n\n" + motzkin_helpers + text[insert_at:]
    print("inserted eval helpers")

# 2) Callee list: add motzkin near necklaces
if 'callee == "combo_motzkin_paths"' not in text:
    text = text.replace(
        'callee == "combo_necklaces" || callee == "combo_de_bruijn_sequence" ||',
        'callee == "combo_necklaces" || callee == "combo_de_bruijn_sequence" || '
        'callee == "combo_motzkin_paths" || callee == "combo_set_partitions" ||',
    )
    print("callee list")

# Also looks-like / is_matrix_call_callee other sites
for old, new in [
    (
        'fn == "combo_necklaces" || fn == "combo_de_bruijn_sequence" ||',
        'fn == "combo_necklaces" || fn == "combo_de_bruijn_sequence" || '
        'fn == "combo_motzkin_paths" || fn == "combo_set_partitions" ||',
    ),
]:
    if new.split("||")[-2].strip() not in text and old in text:
        text = text.replace(old, new)
        print("fn list")

# is_valid_matrix_call_arity — copy motzkin arity blocks from theirs if missing
if 'callee == "combo_motzkin_paths"' not in text[
    text.find("is_valid_matrix_call_arity") : text.find("is_valid_matrix_call_arity") + 8000
]:
    # find arity for necklaces in ours and append motzkin after
    pass

# Extract arity snippets from theirs
for needle in ['callee == "combo_motzkin_paths"', 'callee == "combo_set_partitions"']:
    if needle not in text:
        # find surrounding if-block in theirs
        i = theirs.find(needle)
        if i >= 0:
            # back up to "if ("
            start = theirs.rfind("\n    if (", 0, i)
            end = theirs.find("\n    }", i)
            end = theirs.find("\n", end + 1)
            block = theirs[start + 1 : end + 1]
            # insert after combo_necklaces arity if present
            ni = text.find('callee == "combo_necklaces"')
            if ni >= 0:
                # find end of that if block
                ne = text.find("\n    }", ni)
                ne = text.find("\n", ne + 1)
                text = text[: ne + 1] + block + text[ne + 1 :]
                print("arity", needle)

# 3) assign_matrix_call_tail2 arms: append motzkin arms before return result
neck_arm = extract(
    ours,
    '} else if (assign.callee == "combo_necklaces"',
    "\n    return result;\n}",
)
mot_arm = extract(
    theirs,
    '} else if (assign.callee == "combo_motzkin_paths"',
    "\n    return result;\n}",
)
# Rebuild: everything before combo_necklaces arm + neck + mot + return
start = text.find('} else if (assign.callee == "combo_necklaces"')
if start < 0:
    # maybe only de_bruijn in tail - find combo_de_bruijn
    start = text.find('} else if (assign.callee == "combo_de_bruijn_sequence"')
    if start < 0:
        # insert before return of tail2
        ret = text.find(
            "\n    return result;\n}\n\n\nResult<std::string> Interpreter::assign_matrix_call"
        )
        if ret < 0:
            ret = text.find(
                "\n    return result;\n}\n\nResult<std::string> Interpreter::assign_matrix_call"
            )
        body = neck_arm + mot_arm
        body = body.replace("}\n} else if", "} else if")
        text = text[:ret] + body + text[ret:]
        print("inserted arms at return")
    else:
        end = text.find(
            "\n    return result;\n}\n\n\nResult<std::string> Interpreter::assign_matrix_call",
            start,
        )
        if end < 0:
            end = text.find(
                "\n    return result;\n}\n\nResult<std::string> Interpreter::assign_matrix_call",
                start,
            )
        # Keep from start through de_bruijn from ours extract
        body = neck_arm + mot_arm
        body = body.replace("}\n} else if", "} else if")
        text = text[:start] + body + text[end:]
        print("replaced from de_bruijn/necklaces")
else:
    end = text.find(
        "\n    return result;\n}\n\n\nResult<std::string> Interpreter::assign_matrix_call",
        start,
    )
    if end < 0:
        end = text.find(
            "\n    return result;\n}\n\nResult<std::string> Interpreter::assign_matrix_call",
            start,
        )
    body = neck_arm.rstrip() + "\n" + mot_arm
    body = body.replace("}\n} else if", "} else if")
    if not body.endswith("\n"):
        body += "\n"
    text = text[:start] + body + text[end:]
    print("replaced necklaces..return with neck+mot")

# 4) Help name= lines
if "combo_motzkin_paths" not in text[text.find('name = combo_necklaces') : text.find('name = combo_necklaces') + 500]:
    text = text.replace(
        'name = combo_de_bruijn_sequence(k,n) De Bruijn sequence B(k,n) as k^n',
        'name = combo_de_bruijn_sequence(k,n) De Bruijn sequence B(k,n) as k^n',
    )
    # insert after de_bruijn help line
    db = 'name = combo_de_bruijn_sequence'
    di = text.find(db)
    if di >= 0:
        line_end = text.find("\n", di)
        insert = (
            '\n            "  name = combo_motzkin_paths(n) all Motzkin paths as Mxn matrix (+1 U, -1 D, 0 F steps)\\n"\n'
            '            "  name = combo_set_partitions(n) all set partitions of {0..n-1} as Bxn block-label matrix\\n"'
        )
        # careful with existing line
        if "combo_motzkin_paths" not in text[di : di + 400]:
            text = text[:line_end] + insert + text[line_end:]
            print("help name lines")

# 5) Catalog string — inject tokens
if "combo_motzkin_paths(n)" not in text:
    text = text.replace(
        "combo_de_bruijn_sequence(k,n), ",
        "combo_de_bruijn_sequence(k,n), combo_motzkin_paths(n), combo_set_partitions(n), ",
    )
    print("catalog")

# Also ensure is_matrix_call_callee has motzkin (may use different pattern)
if 'callee == "combo_motzkin_paths"' not in text:
    text = text.replace(
        'callee == "combo_de_bruijn_sequence" ||',
        'callee == "combo_de_bruijn_sequence" || callee == "combo_motzkin_paths" || '
        'callee == "combo_set_partitions" ||',
    )

Path("src/interp/repl_engine.cpp").write_text(text, encoding="utf-8", newline="")
print("wrote repl_engine.cpp")

# --- Completer ---
c2 = show_stage(2, "src/gui/ReplCompleter.cpp")
c3 = show_stage(3, "src/gui/ReplCompleter.cpp")
# take ours, add missing QStringLiterals from theirs
for lit in re.findall(r'QStringLiteral\("([^"]+)"\)', c3):
    if lit.startswith("combo_") and f'QStringLiteral("{lit}")' not in c2:
        # insert near combo_necklaces
        needle = 'QStringLiteral("combo_necklaces")'
        if needle in c2:
            c2 = c2.replace(
                needle,
                needle + f',\n        QStringLiteral("{lit}")',
            )
        else:
            needle = 'QStringLiteral("combo_de_bruijn_sequence")'
            if needle in c2:
                c2 = c2.replace(
                    needle,
                    needle + f',\n        QStringLiteral("{lit}")',
                )
Path("src/gui/ReplCompleter.cpp").write_text(c2, encoding="utf-8", newline="")
print("completer")

# --- tests ---
t2 = show_stage(2, "tests/unit/test_repl_commands.cpp")
t3 = show_stage(3, "tests/unit/test_repl_commands.cpp")
# append motzkin tests from t3 if missing
if "wave262_combo_motzkin" not in t2 and "wave262_combo_motzkin_paths" not in t2:
    # find TEST blocks in t3
    tests = []
    for name in ["wave262_combo_motzkin_paths", "wave262_combo_set_partitions"]:
        m = re.search(
            rf"TEST\(ReplCommandsTest, {name}\) \{{.*?\n\}}",
            t3,
            re.S,
        )
        if m:
            tests.append(m.group(0))
    if tests:
        t2 = t2.rstrip() + "\n\n" + "\n\n".join(tests) + "\n"
Path("tests/unit/test_repl_commands.cpp").write_text(t2, encoding="utf-8", newline="")
print("tests")

# --- API.md ---
a2 = show_stage(2, "docs/API.md")
a3 = show_stage(3, "docs/API.md")
for line in a3.splitlines():
    if "combo_motzkin_paths" in line or "combo_set_partitions" in line:
        if line not in a2:
            # insert after necklaces line
            if "combo_necklaces" in a2:
                for i, l in enumerate(a2.splitlines()):
                    if "combo_necklaces" in l or "combo_de_bruijn" in l:
                        last = i
                lines = a2.splitlines()
                lines.insert(last + 1, line)
                a2 = "\n".join(lines) + ("\n" if a2.endswith("\n") else "")
            else:
                a2 = a2.rstrip() + "\n" + line + "\n"
Path("docs/API.md").write_text(a2, encoding="utf-8", newline="")
print("api")

print("DONE")
