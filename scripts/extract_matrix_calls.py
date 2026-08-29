#!/usr/bin/env python3
"""One-shot migrate dest-tail matrix-call handlers into per-callee TUs."""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPL_CPP = ROOT / "src" / "interp" / "repl_engine.cpp"
REPL_HPP = ROOT / "include" / "ms" / "interp" / "repl_engine.hpp"
INTERNAL_CPP = ROOT / "src" / "interp" / "repl_engine_internal.cpp"
INTERNAL_HPP = ROOT / "src" / "interp" / "repl_engine_internal.hpp"
MATRIX_DIR = ROOT / "src" / "interp" / "matrix_calls"
CMAKE = ROOT / "src" / "interp" / "CMakeLists.txt"

CALLEE_RE = re.compile(r'assign\.callee\s*==\s*"([A-Za-z_][A-Za-z0-9_]*)"')
TAIL_SIG_RE = re.compile(
    r"^Result<Matrix<double>> Interpreter::assign_matrix_call_tail(\d*)"
    r"\(const MatrixCallAssign& assign\)",
    re.M,
)
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

HANDLER_PREAMBLE = """\
#include "matrix_call.hpp"
#include "repl_engine_internal.hpp"

namespace ms::interp {{

Result<Matrix<double>> handle_{name}(Interpreter& interp, const MatrixCallAssign& assign) {{
    using namespace detail;
    MatrixCallCtx ctx(interp);
    auto resolve_operand = [&ctx](const std::string& text) {{ return ctx.resolve_operand(text); }};
    auto parse_scalar_arg = [&ctx](const std::string& arg_text, const char* fn) -> Result<double> {{
        double value = 0.0;
        if (parse_number(arg_text, value)) return value;
        auto expr = eval_scalar_expr(ctx.state(), arg_text);
        if (!expr) {{
            return std::unexpected(DomainError{{fn, "expected numeric scalar argument"}});
        }}
        return *expr;
    }};
    auto parse_positive_size_arg = [](double value, const char* fn, const char* label) -> Result<std::size_t> {{
        const int i = static_cast<int>(value);
        if (i < 1 || value != static_cast<double>(i)) {{
            return std::unexpected(DomainError{{fn, label}});
        }}
        return static_cast<std::size_t>(i);
    }};
    auto parse_uint64_arg = [](double value, const char* fn, const char* label) -> Result<uint64_t> {{
        if (value < 0.0 || value != std::floor(value)) {{
            return std::unexpected(DomainError{{fn, label}});
        }}
        return static_cast<uint64_t>(value);
    }};

    Result<Matrix<double>> result =
        std::unexpected(DomainError{{"assign", "unsupported matrix call"}});
{chain}
    return result;
}}

void ms_register_matrix_call_{name}() {{
    register_matrix_call("{name}", &handle_{name});
}}

}} // namespace ms::interp
"""

ASSIGN_MATRIX_CALL_BODY = """\
Result<std::string> Interpreter::assign_matrix_call(const MatrixCallAssign& assign) {
    auto result = dispatch_matrix_call(*this, assign);
    if (!result) {
        return std::unexpected(result.error());
    }
    state_.matrices[assign.target] = *result;
    std::ostringstream out;
    out << assign.target << " =\\n";
    print_matrix(out, *result);
    return out.str();
}
"""

CMAKE_TEXT = """\
set(_MS_INTERP_SOURCES repl_engine.cpp plot_console.cpp jit_backend_factory.cpp jit_orc_stub.cpp)

file(GLOB _MS_MATRIX_CALL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/matrix_calls/*.cpp")

set(_MS_MATRIX_CALL_REGISTER "${CMAKE_CURRENT_BINARY_DIR}/matrix_call_register_all.cpp")
set(_MS_MATRIX_CALL_DECLS "")
set(_MS_MATRIX_CALL_CALLS "")
file(GLOB _MS_MATRIX_CALL_STEMS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/matrix_calls"
    "${CMAKE_CURRENT_SOURCE_DIR}/matrix_calls/*.cpp")
list(SORT _MS_MATRIX_CALL_STEMS)
foreach(_src ${_MS_MATRIX_CALL_STEMS})
    get_filename_component(_stem ${_src} NAME_WE)
    string(APPEND _MS_MATRIX_CALL_DECLS "void ms_register_matrix_call_${_stem}();\\n")
    string(APPEND _MS_MATRIX_CALL_CALLS "    ms_register_matrix_call_${_stem}();\\n")
endforeach()

file(WRITE "${_MS_MATRIX_CALL_REGISTER}"
"#include \\"matrix_call.hpp\\"

namespace ms::interp {

${_MS_MATRIX_CALL_DECLS}
void ensure_matrix_calls_registered() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
${_MS_MATRIX_CALL_CALLS}}

} // namespace ms::interp
")

list(APPEND _MS_INTERP_SOURCES
    repl_engine_internal.cpp
    matrix_call.cpp
    ${_MS_MATRIX_CALL_SOURCES}
    ${_MS_MATRIX_CALL_REGISTER}
)

if(MS_BUILD_JIT)
    list(APPEND _MS_INTERP_SOURCES jit_orc_llvm.cpp)
endif()

add_library(ms_interp STATIC ${_MS_INTERP_SOURCES})
target_include_directories(ms_interp PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)
target_include_directories(ms_interp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ms_interp PUBLIC
    ms_core ms_linalg ms_fft ms_runtime ms_simd ms_distributed ms_frameworks ms_special
    ms_image ms_compress ms_bignum ms_ml
    ms_graph ms_geo ms_combo ms_numthy ms_control ms_quantum
    ms_finance ms_info ms_cplx ms_tensorops ms_diffgeo ms_topo
    ms_crypto ms_fem ms_cfd
)
target_compile_features(ms_interp PUBLIC cxx_std_23)
if(MS_ENABLE_COVERAGE AND NOT MSVC)
    target_compile_options(ms_interp PRIVATE -fexceptions)
endif()

if(MS_BUILD_JIT)
    ms_apply_jit_llvm(ms_interp)
endif()
"""


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        tmp.replace(path)
    except PermissionError:
        try:
            path.write_text(text, encoding="utf-8", newline="\n")
            tmp.unlink(missing_ok=True)
        except PermissionError:
            tmp.unlink(missing_ok=True)
            fail(f"permission denied writing {path}")


def is_ident_start(c: str) -> bool:
    return c.isalpha() or c == "_"


def is_ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def skip_line_comment(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] != "\n":
        i += 1
    return i


def skip_block_comment(text: str, i: int) -> int:
    n = len(text)
    i += 2
    while i < n - 1:
        if text[i] == "*" and text[i + 1] == "/":
            return i + 2
        i += 1
    return n


def skip_quoted(text: str, i: int) -> int:
    """Skip a '...' or \"...\" literal starting at i."""
    n = len(text)
    quote = text[i]
    i += 1
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        i += 1
    return n


def skip_raw_string(text: str, i: int) -> int:
    """Skip a C++ raw string starting at R. i points at R."""
    n = len(text)
    j = i + 1
    if j < n and text[j] == '"':
        delim_start = j + 1
    elif j + 1 < n and text[j].isalpha() and text[j + 1] == '"':
        # Rprefix"
        while j < n and text[j] != '"':
            j += 1
        delim_start = j + 1
    else:
        return i + 1
    k = delim_start
    while k < n and text[k] != "(":
        k += 1
    if k >= n:
        return i + 1
    delim = text[delim_start:k]
    end = ")" + delim + '"'
    pos = text.find(end, k + 1)
    return n if pos < 0 else pos + len(end)


def skip_ws_and_comments(text: str, i: int) -> int:
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "/" and i + 1 < n:
            if text[i + 1] == "/":
                i = skip_line_comment(text, i + 2)
                continue
            if text[i + 1] == "*":
                i = skip_block_comment(text, i)
                continue
        break
    return i


def advance_code(text: str, i: int) -> int:
    """Advance one token/char, skipping strings and comments as units."""
    n = len(text)
    if i >= n:
        return n
    c = text[i]
    if c == "/" and i + 1 < n:
        if text[i + 1] == "/":
            return skip_line_comment(text, i + 2)
        if text[i + 1] == "*":
            return skip_block_comment(text, i)
    if c == "R" and i + 1 < n and (text[i + 1] == '"' or text[i + 1] == "8"):
        nxt = skip_raw_string(text, i)
        if nxt != i + 1:
            return nxt
    if c in ("'", '"'):
        return skip_quoted(text, i)
    return i + 1


def match_brace(text: str, open_pos: int) -> int:
    """Return index just past the matching '}' for '{' at open_pos."""
    if text[open_pos] != "{":
        fail(f"match_brace expected '{{' at {open_pos}")
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] in "/*":
            i = advance_code(text, i)
            continue
        if c == "R" and i + 1 < n:
            nxt = skip_raw_string(text, i)
            if nxt != i + 1:
                i = nxt
                continue
        if c in ("'", '"'):
            i = skip_quoted(text, i)
            continue
        if c == "{":
            depth += 1
            i += 1
            continue
        if c == "}":
            depth -= 1
            i += 1
            if depth == 0:
                return i
            continue
        i += 1
    fail(f"unbalanced '{{' at {open_pos}")
    return n


def match_paren(text: str, open_pos: int) -> int:
    if text[open_pos] != "(":
        fail(f"match_paren expected '(' at {open_pos}")
    depth = 0
    i = open_pos
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] in "/*":
            i = advance_code(text, i)
            continue
        if c == "R" and i + 1 < n:
            nxt = skip_raw_string(text, i)
            if nxt != i + 1:
                i = nxt
                continue
        if c in ("'", '"'):
            i = skip_quoted(text, i)
            continue
        if c == "(":
            depth += 1
            i += 1
            continue
        if c == ")":
            depth -= 1
            i += 1
            if depth == 0:
                return i
            continue
        i += 1
    fail(f"unbalanced '(' at {open_pos}")
    return n


def skip_template_head(text: str, i: int) -> int:
    """If text[i:] starts with template<...>, return index after it; else i."""
    j = skip_ws_and_comments(text, i)
    if not text.startswith("template", j):
        return i
    if j + 8 < len(text) and is_ident_char(text[j + 8]):
        return i
    j = skip_ws_and_comments(text, j + 8)
    if j >= len(text) or text[j] != "<":
        return i
    depth = 0
    n = len(text)
    while j < n:
        c = text[j]
        if c == "/" and j + 1 < n and text[j + 1] in "/*":
            j = advance_code(text, j)
            continue
        if c in ("'", '"'):
            j = skip_quoted(text, j)
            continue
        if c == "<":
            depth += 1
            j += 1
            continue
        if c == ">":
            depth -= 1
            j += 1
            if depth == 0:
                return j
            continue
        j += 1
    return i


def strip_leading_specifiers(sig: str) -> str:
    s = sig.strip()
    while True:
        t = skip_template_head(s, 0)
        if t != 0:
            s = s[t:].lstrip()
            continue
        changed = False
        for spec in ("static ", "inline ", "constexpr ", "consteval ", "extern "):
            if s.startswith(spec):
                s = s[len(spec) :].lstrip()
                changed = True
        if not changed:
            break
    return s


def classify_item(prefix: str, has_brace: bool) -> str:
    s = strip_leading_specifiers(prefix)
    if s.startswith("thread_local"):
        return "thread_local"
    if s.startswith("using ") or s.startswith("using\t"):
        return "using"
    if s.startswith(("struct ", "struct\t", "class ", "class\t", "enum ", "enum\t")):
        return "type" if has_brace else "fwd"
    if has_brace:
        if prefix.lstrip().startswith("template"):
            return "template_func"
        if strip_leading_specifiers(prefix).startswith("static") or prefix.lstrip().startswith(
            "static "
        ):
            return "static_func"
        # detect static more reliably
        lead = prefix.lstrip()
        lead = lead[len("template") :] if lead.startswith("template") else lead
        # already handled templates
        raw = prefix.strip()
        if re.match(r"(template[\s\S]*?)?static\b", raw):
            return "static_func"
        return "func"
    # semicolon-terminated
    if "(" in prefix:
        return "fwd"
    return "fwd"


def is_static_prefix(prefix: str) -> bool:
    s = prefix.strip()
    # strip template heads
    while s.startswith("template"):
        t = skip_template_head(s, 0)
        if t == 0:
            break
        s = s[t:].lstrip()
    return s.startswith("static") and (len(s) == 6 or not is_ident_char(s[6]))


def scan_namespace_items(body: str) -> list[tuple[str, int, int, str]]:
    """Return (kind, start, end, prefix) for depth-0 items in anonymous-namespace body."""
    items: list[tuple[str, int, int, str]] = []
    n = len(body)
    i = 0
    while i < n:
        i = skip_ws_and_comments(body, i)
        if i >= n:
            break
        if body[i] == "#":
            while i < n and body[i] != "\n":
                i += 1
            continue
        start = i
        paren = 0
        j = i
        while j < n:
            c = body[j]
            if c == "/" and j + 1 < n and body[j + 1] in "/*":
                j = advance_code(body, j)
                continue
            if c == "R" and j + 1 < n:
                nxt = skip_raw_string(body, j)
                if nxt != j + 1:
                    j = nxt
                    continue
            if c in ("'", '"'):
                j = skip_quoted(body, j)
                continue
            if c == "(":
                paren += 1
                j += 1
                continue
            if c == ")":
                paren -= 1
                j += 1
                continue
            if c == "{" and paren == 0:
                prefix = body[start:j]
                end = match_brace(body, j)
                k = skip_ws_and_comments(body, end)
                if k < n and body[k] == ";":
                    end = k + 1
                kind = classify_item(prefix, True)
                if is_static_prefix(prefix) and kind == "func":
                    kind = "static_func"
                items.append((kind, start, end, prefix))
                i = end
                break
            if c == ";" and paren == 0:
                prefix = body[start:j]
                kind = classify_item(prefix, False)
                items.append((kind, start, j + 1, prefix))
                i = j + 1
                break
            j += 1
        else:
            fail(f"unterminated namespace item starting at {start}")
    return items


def thread_local_extern(stmt: str) -> str:
    s = stmt.strip()
    if s.endswith(";"):
        s = s[:-1].rstrip()
    if "=" in s:
        s = s[: s.index("=")].rstrip()
    if not s.startswith("extern "):
        s = "extern " + s
    return s + ";"


def find_anon_namespace(text: str) -> tuple[int, int, int, int]:
    """Return (ns_kw_start, body_start, body_end, ns_end_after_close).

    ns_kw_start points at `namespace {`; body is inside; ns_end is after `} // namespace`.
    """
    marker = "ColMatrix<double> matrix_to_col_matrix(const Matrix<double>& matrix);"
    col = text.find(marker)
    if col < 0:
        fail("could not find ColMatrix forward declaration")
    search_from = col + len(marker)
    ns = text.find("namespace {", search_from)
    if ns < 0:
        fail("could not find anonymous namespace")
    brace = text.find("{", ns)
    body_start = brace + 1
    close = match_brace(text, brace)
    # include trailing ` // namespace` comment on the same line
    end = close
    rest = text[close:]
    m = re.match(r"[ \t]*//[^\n]*", rest)
    if m:
        end = close + m.end()
    return ns, body_start, close - 1, end


def generate_internal_header(includes: str, body: str, items: list) -> str:
    decls: list[str] = []
    for kind, start, end, prefix in items:
        chunk = body[start:end].rstrip()
        if kind == "type" or kind == "using":
            decls.append(chunk)
        elif kind == "thread_local":
            decls.append(thread_local_extern(chunk))
        elif kind == "template_func":
            decls.append(chunk)
        elif kind == "func":
            sig = prefix.rstrip()
            if sig.endswith("{"):
                sig = sig[:-1].rstrip()
            decls.append(sig + ";")
        elif kind in ("fwd", "static_func"):
            continue
        else:
            continue
    out = ["#pragma once", "", includes.rstrip(), "", "namespace ms::interp::detail {", ""]
    out.append("\n\n".join(decls))
    out.append("")
    out.append("} // namespace ms::interp::detail")
    out.append("")
    return "\n".join(out)


def generate_internal_cpp(includes: str, body: str) -> str:
    # includes already contain repl_engine.hpp (line 12); add it again per spec
    text = includes
    if '#include "ms/interp/repl_engine.hpp"' not in text:
        text += '#include "ms/interp/repl_engine.hpp"\n'
    text += "\nnamespace ms::interp::detail {\n"
    text += body
    if not body.endswith("\n"):
        text += "\n"
    text += "} // namespace ms::interp::detail\n"
    return text


def find_function(text: str, signature_re: re.Pattern[str]) -> tuple[int, int, int]:
    """Return (sig_start, body_open, body_close_after)."""
    m = signature_re.search(text)
    if not m:
        fail(f"function not found: {signature_re.pattern}")
    brace = text.find("{", m.end())
    if brace < 0:
        fail("missing '{' after function signature")
    end = match_brace(text, brace)
    return m.start(), brace, end


def find_named_function(text: str, name: str) -> tuple[int, int, int]:
    pat = re.compile(re.escape(name) + r"\s*\(", re.M)
    # prefer start-of-line definitions
    for m in pat.finditer(text):
        # walk back to line start; skip if this is a call
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix = text[line_start : m.start()].strip()
        if not prefix:
            continue
        if prefix.startswith("//"):
            continue
        brace = skip_ws_and_comments(text, m.end() - 1)
        # m.end()-1 is '(' 
        after_paren = match_paren(text, text.find("(", m.start()))
        after = skip_ws_and_comments(text, after_paren)
        if after < len(text) and text[after] == "{":
            end = match_brace(text, after)
            return line_start, after, end
    fail(f"definition not found: {name}")
    return 0, 0, 0


def _is_kw_at(text: str, i: int, kw: str) -> bool:
    n = len(kw)
    if not text.startswith(kw, i):
        return False
    if i > 0 and is_ident_char(text[i - 1]):
        return False
    if i + n < len(text) and is_ident_char(text[i + n]):
        return False
    return True


def extract_top_level_callee_ifs(fn_text: str, abs_base: int) -> list[tuple[list[str], int, int, str]]:
    """Collect function-scope if/else-if statements whose condition mentions assign.callee."""
    brace = fn_text.find("{")
    if brace < 0:
        fail("function body missing '{'")
    i = brace + 1
    n = len(fn_text)
    found: list[tuple[list[str], int, int, str]] = []
    while i < n:
        i = skip_ws_and_comments(fn_text, i)
        if i >= n:
            break
        if fn_text[i] == "}":
            break
        stmt_start = i
        is_else_if = False
        is_if = False
        if _is_kw_at(fn_text, i, "else"):
            j = skip_ws_and_comments(fn_text, i + 4)
            if _is_kw_at(fn_text, j, "if"):
                is_else_if = True
                i = j
        line_start = fn_text.rfind("\n", 0, i) + 1
        if fn_text[line_start:i].strip() == "":
            stmt_start = line_start
        if _is_kw_at(fn_text, i, "if"):
            is_if = True
            j = skip_ws_and_comments(fn_text, i + 2)
            if j < n and fn_text[j] == "(":
                after = match_paren(fn_text, j)
                cond = fn_text[j:after]
                k = skip_ws_and_comments(fn_text, after)
                if k < n and fn_text[k] == "{":
                    end = match_brace(fn_text, k)
                    names = CALLEE_RE.findall(cond)
                    if names:
                        chunk = fn_text[stmt_start:end]
                        found.append((names, abs_base + stmt_start, abs_base + end, chunk))
                    i = end
                    continue
                # if (...) stmt;
                i = after
                continue
        if is_else_if or is_if:
            # fall through to generic skip if we didn't parse a block
            i = stmt_start
        # skip one statement / declaration / lambda
        paren = 0
        while i < n:
            c = fn_text[i]
            if c == "/" and i + 1 < n and fn_text[i + 1] in "/*":
                i = advance_code(fn_text, i)
                continue
            if c == "R" and i + 1 < n:
                nxt = skip_raw_string(fn_text, i)
                if nxt != i + 1:
                    i = nxt
                    continue
            if c in ("'", '"'):
                i = skip_quoted(fn_text, i)
                continue
            if c == "(":
                paren += 1
                i += 1
                continue
            if c == ")":
                paren -= 1
                i += 1
                continue
            if c == "{" and paren == 0:
                i = match_brace(fn_text, i)
                break
            if c == ";" and paren == 0:
                i += 1
                break
            if c == "}" and paren == 0:
                break
            i += 1
    return found


def normalize_if_chunk(chunk: str, first: bool) -> str:
    indent_len = len(chunk) - len(chunk.lstrip(" \t"))
    indent = chunk[:indent_len]
    body = chunk[indent_len:]
    if first:
        if body.startswith("else if"):
            body = body[5:].lstrip()
        elif body.startswith("else"):
            rest = body[4:].lstrip()
            if rest.startswith("if"):
                body = rest
    else:
        if body.startswith("if ") or body.startswith("if(") or body.startswith("if\n"):
            body = "else " + body
    return indent + body


def collect_callees_from_tails(text: str) -> OrderedDict[str, list[str]]:
    grouped: OrderedDict[str, list[tuple[int, int, str]]] = OrderedDict()
    for n in range(249, 257):
        sig = (
            f"Result<Matrix<double>> Interpreter::assign_matrix_call_tail{n}"
            f"(const MatrixCallAssign& assign)"
        )
        pos = text.find(sig)
        if pos < 0:
            fail(f"missing {sig}")
        brace = text.find("{", pos + len(sig))
        end = match_brace(text, brace)
        fn_text = text[pos:end]
        for names, start, stmt_end, chunk in extract_top_level_callee_ifs(fn_text, pos):
            for name in names:
                grouped.setdefault(name, [])
                if not any(s == start and e == stmt_end for s, e, _ in grouped[name]):
                    grouped[name].append((start, stmt_end, chunk))
    result: OrderedDict[str, list[str]] = OrderedDict()
    for name, chunks in grouped.items():
        chunks.sort(key=lambda t: t[0])
        pieces = [normalize_if_chunk(c, i == 0) for i, (_, _, c) in enumerate(chunks)]
        result[name] = pieces
    return result


def sanitize_name(name: str) -> str:
    if not IDENT_RE.match(name):
        fail(f"callee name is not a C++ identifier: {name!r}")
    return name


def write_callee_files(grouped: OrderedDict[str, list[str]]) -> list[str]:
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for name, pieces in grouped.items():
        ident = sanitize_name(name)
        chain = "\n".join(p.replace("state_", "ctx.state()") for p in pieces)
        if not chain.endswith("\n"):
            chain += "\n"
        write_text(MATRIX_DIR / f"{ident}.cpp", HANDLER_PREAMBLE.format(name=ident, chain=chain))
        names.append(ident)
    return names


def patch_repl_engine(
    text: str,
    ns_start: int,
    ns_end: int,
) -> str:
    # Remove anonymous namespace (keep ColMatrix forward line before it)
    text = text[:ns_start] + text[ns_end:]

    # After #include "ms/interp/repl_engine.hpp" add the two includes
    needle = '#include "ms/interp/repl_engine.hpp"\n'
    idx = text.find(needle)
    if idx < 0:
        fail("missing #include \"ms/interp/repl_engine.hpp\"")
    insert = (
        needle
        + '#include "matrix_call.hpp"\n'
        + '#include "repl_engine_internal.hpp"\n'
    )
    text = text[:idx] + insert + text[idx + len(needle) :]

    # After namespace ms::interp { add using namespace detail;
    ns = "namespace ms::interp {\n"
    nsi = text.find(ns)
    if nsi < 0:
        fail("missing namespace ms::interp")
    text = text[: nsi + len(ns)] + "using namespace detail;\n" + text[nsi + len(ns) :]

    # Delete is_matrix_call_callee
    cal_re = re.compile(
        r"^bool is_matrix_call_callee\(const std::string& callee\)\s*\{",
        re.M,
    )
    m = cal_re.search(text)
    if not m:
        fail("could not find is_matrix_call_callee")
    cal_end = match_brace(text, text.find("{", m.start()))
    # drop following blank lines
    after = cal_end
    while after < len(text) and text[after] == "\n":
        after += 1
        if after < len(text) and text[after] == "\n":
            after += 1
            break
    text = text[: m.start()] + text[after:]

    # Delete ALL dest tails: first unnumbered tail through tail256 close
    first = re.search(
        r"^Result<Matrix<double>> Interpreter::assign_matrix_call_tail"
        r"\(const MatrixCallAssign& assign\)",
        text,
        re.M,
    )
    if not first:
        fail("could not find assign_matrix_call_tail(")
    last = re.search(
        r"^Result<Matrix<double>> Interpreter::assign_matrix_call_tail256"
        r"\(const MatrixCallAssign& assign\)",
        text,
        re.M,
    )
    if not last:
        fail("could not find assign_matrix_call_tail256")
    last_brace = text.find("{", last.end())
    last_end = match_brace(text, last_brace)
    while last_end < len(text) and text[last_end] in "\n":
        last_end += 1
        break
    # consume extra blank lines after tail256
    while last_end < len(text) and text[last_end] == "\n":
        last_end += 1
        if last_end < len(text) and text.startswith("Result<", last_end):
            break
        if last_end < len(text) and text[last_end] != "\n":
            break

    text = text[: first.start()] + text[last_end:]

    # Replace assign_matrix_call body
    am = re.search(
        r"^Result<std::string> Interpreter::assign_matrix_call"
        r"\(const MatrixCallAssign& assign\)\s*\{",
        text,
        re.M,
    )
    if not am:
        fail("could not find Interpreter::assign_matrix_call")
    am_end = match_brace(text, text.find("{", am.start()))
    text = text[: am.start()] + ASSIGN_MATRIX_CALL_BODY + text[am_end:]
    return text


def patch_repl_hpp(text: str) -> str:
    lines = text.split("\n")
    kept = [ln for ln in lines if not re.match(r"\s*Result<Matrix<double>> assign_matrix_call_tail", ln)]
    return "\n".join(kept)


def indent_callee_if_chains() -> None:
    """Restore 4-space indent on extracted if-chains that lost leading whitespace."""
    if not MATRIX_DIR.is_dir():
        return
    for path in MATRIX_DIR.glob("*.cpp"):
        text = read_text(path)
        marker = 'std::unexpected(DomainError{"assign", "unsupported matrix call"});'
        idx = text.find(marker)
        if idx < 0:
            continue
        head = text[: idx + len(marker)]
        tail = text[idx + len(marker) :]
        lines = tail.split("\n")
        out: list[str] = []
        for line in lines:
            if re.match(r"^(if |else if )", line):
                out.append("    " + line)
            else:
                out.append(line)
        new = head + "\n".join(out)
        if new != text:
            write_text(path, new)


def already_migrated(text: str) -> bool:
    return "using namespace detail;" in text and "dispatch_matrix_call" in text


def main() -> None:
    text = read_text(REPL_CPP)
    names: list[str] = []
    helper_lines = 0
    extra = ""

    if already_migrated(text) and INTERNAL_CPP.exists():
        body = read_text(INTERNAL_CPP)
        ns = body.find("namespace ms::interp::detail {")
        end = body.rfind("} // namespace ms::interp::detail")
        if ns >= 0 and end > ns:
            inner = body[body.find("\n", ns) + 1 : end]
            helper_lines = inner.count("\n")
        names = [
            p.stem
            for p in sorted(MATRIX_DIR.glob("*.cpp"), key=lambda p: p.stat().st_ctime)
        ]
        extra = " (resume: cpp already migrated)"
        indent_callee_if_chains()
        new_cpp = text
    else:
        includes = "\n".join(text.splitlines()[:76]) + "\n"
        ns_start, body_start, body_end, ns_end = find_anon_namespace(text)
        body = text[body_start:body_end]
        helper_lines = body.count("\n")

        items = scan_namespace_items(body)
        write_text(INTERNAL_CPP, generate_internal_cpp(includes, body))
        write_text(INTERNAL_HPP, generate_internal_header(includes, body, items))

        grouped = collect_callees_from_tails(text)
        names = write_callee_files(grouped)
        indent_callee_if_chains()

        new_cpp = patch_repl_engine(text, ns_start, ns_end)
        write_text(REPL_CPP, new_cpp)

    hpp = read_text(REPL_HPP)
    write_text(REPL_HPP, patch_repl_hpp(hpp))
    write_text(CMAKE, CMAKE_TEXT)

    leftover = re.findall(r"Interpreter::assign_matrix_call_tail\d*\(", new_cpp)
    leftover_hpp = [
        ln for ln in read_text(REPL_HPP).splitlines() if "assign_matrix_call_tail" in ln
    ]

    print(f"helper lines moved: {helper_lines}{extra}")
    print(f"callee files written: {len(names)}")
    print("first 10 names: " + ", ".join(names[:10]))
    if leftover:
        print(f"WARNING: leftover tail refs in cpp: {leftover[:8]}")
    else:
        print("tails deleted: yes (no assign_matrix_call_tail* left in repl_engine.cpp)")
    if leftover_hpp:
        print(f"WARNING: leftover tail decls in hpp: {leftover_hpp[:4]}")
    else:
        print("hpp tail decls deleted: yes")
    if "bool is_matrix_call_callee(" in new_cpp:
        print("WARNING: is_matrix_call_callee still in repl_engine.cpp")
    else:
        print("is_matrix_call_callee deleted from repl_engine.cpp: yes")
    print(f"created: {INTERNAL_CPP.relative_to(ROOT)}")
    print(f"created: {INTERNAL_HPP.relative_to(ROOT)}")
    print(f"created: {MATRIX_DIR.relative_to(ROOT)}/ ({len(names)} files)")
    print(f"updated: {CMAKE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
