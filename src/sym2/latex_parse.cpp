// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch

/// @file
/// @brief §11.2: the reader for the LaTeX subset `docs/LATEX_SUBSET.md` defines.
///
/// The subset is defined by the printer rather than by taste, so this file is written
/// against `notation_latex.cpp` and `notation.cpp` rather than against a LaTeX manual.
/// Where the two disagree the code is what is emitted and therefore what has to be
/// read back.
///
/// Three parts of the grammar are not ordinary recursive descent and are worth naming
/// before the code:
///
///   - **The integral's differentials are found by scanning backwards.** `\int f \, dx`
///     has no closing delimiter, so a descent that reads the integrand left to right
///     cannot know where it ends until it has already eaten `\, dx`. The scan of §2.6
///     runs to the end of the enclosing group, walks back over maximal `\, d<var>`
///     units, and hands what is left to the ordinary expression parser. The one-unit
///     lookahead a descent could afford gets `\int \mathrm{aa}\,d` wrong, and that is a
///     string the printer emits (N23).
///   - **A leading `-` before `\infty` is part of the constant, not a negation.** Ruling
///     A10: `constant("-inf")` and `neg(constant("inf"))` are both printed `-\infty`,
///     and the document picks the Constant. That decision has to be made before the
///     unary minus is applied, because `-\infty^{n}` is `pow(constant("-inf"), n)` --
///     the Constant is an atom and so reaches the superscript unfenced (N4).
///   - **Backtracking, in exactly one place.** `\frac{d}{dx} f` is a derivative and
///     `\frac{d}{d \cdot x}` is a quotient (A4), and they differ four tokens in. The
///     derivative shape is matched speculatively and the cursor restored on failure.
///     The alternative, a hand-written lookahead predicate, is the same walk written
///     twice and able to drift.
///
/// Everything else is precedence climbing over the table in §2.1, with the calculus
/// operators at statement position because `notation.cpp` places every sum term and
/// every product factor at `kPrecMul` or tighter and therefore always parenthesises
/// one there.
///
/// No construct recovers from an error. §5 rule 4: a strict parser that guesses its way
/// past a rejection is the failure mode the subset exists to prevent, and nine
/// diagnostics that are consequences of the first are worse than the first alone.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ms/sym2/latex_parse.hpp"

#include "latex_lex.hpp"

namespace ms::sym2 {

namespace {

namespace lex = detail::latex;

using lex::Token;
using lex::TokKind;

// --- The spellings the printer emits ------------------------------------------------

/// The thirty lower-case Greek names, in the order `notation_latex.cpp:93-104` lists
/// them. A control word matching one of these is the Symbol of that name, which is what
/// makes `symbol("alpha")` round-trip through `\alpha`.
constexpr const char* kGreekLower[] = {
    "alpha",      "beta",     "gamma", "delta",  "epsilon",  "zeta",   "eta",
    "theta",      "iota",     "kappa", "lambda", "mu",       "nu",     "xi",
    "omicron",    "pi",       "rho",   "sigma",  "tau",      "upsilon", "phi",
    "chi",        "psi",      "omega", "varepsilon", "vartheta", "varpi",
    "varrho",     "varsigma", "varphi",
};

/// The eleven Greek capitals LaTeX gives a control sequence. The other thirteen are
/// Latin letters and arrive as `\mathrm{<letter>}` instead; see `kUprightCapitals`.
constexpr const char* kGreekCapitals[] = {
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi",
    "Psi",   "Omega",
};

/// The inverse of the `\mathrm{<letter>}` column of `kGreek`. A one-character Symbol
/// prints bare (`symbol("A")` is `A`, notation_latex.cpp:152-153), so an upright single
/// capital can only have come from a Greek name -- ruling A9.
struct UprightCapital {
    char letter;
    const char* name;
};
constexpr UprightCapital kUprightCapitals[] = {
    {'A', "Alpha"}, {'B', "Beta"},  {'E', "Epsilon"}, {'Z', "Zeta"},    {'H', "Eta"},
    {'I', "Iota"},  {'K', "Kappa"}, {'M', "Mu"},      {'N', "Nu"},      {'O', "Omicron"},
    {'P', "Rho"},   {'T', "Tau"},   {'X', "Chi"},
};

constexpr const char* kGlyphs[] = {"hbar", "ell", "infty", "nabla", "partial", "aleph"};

/// `kOperators` from notation_latex.cpp:210-214. `\inf` is deliberately absent: `inf`
/// is the atom name of the infinity constant, and a reader that accepted `\inf` for it
/// would be reading the infimum operator as a number.
constexpr const char* kOperators[] = {
    "sin",    "cos",    "tan",    "log", "ln",  "exp", "sinh", "cosh", "tanh",
    "arcsin", "arccos", "arctan", "sec", "csc", "cot", "min",  "max",  "gcd",
    "det",    "arg",    "deg",    "dim", "ker", "lg",  "sup",
};

/// H2. These carry no meaning at all -- they are a request for a delimiter of a
/// particular height -- so they are accepted before `(`, `)` and `|` and dropped.
constexpr const char* kBigDelimiters[] = {
    "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
    "Biggl", "Biggr", "big", "Big", "bigg", "Bigg",
};

constexpr const char* kMatrixEnvironments[] = {"pmatrix", "bmatrix", "vmatrix",
                                               "Vmatrix", "matrix"};

const char* lookup(const char* const* table, std::size_t count, std::string_view name) {
    for (std::size_t i = 0; i < count; ++i) {
        if (name == table[i]) {
            return table[i];
        }
    }
    return nullptr;
}

template <std::size_t N>
const char* lookup(const char* const (&table)[N], std::string_view name) {
    return lookup(table, N, name);
}

const char* upright_capital_name(char letter) {
    for (const UprightCapital& entry : kUprightCapitals) {
        if (entry.letter == letter) {
            return entry.name;
        }
    }
    return nullptr;
}

/// H10: a literal Greek letter is its control word. The lower-case block runs
/// U+03B1..U+03C9 with final sigma at U+03C2, which is `varsigma` in this table's
/// naming; the capitals run U+0391..U+03A9 with U+03A2 unassigned.
const char* greek_from_code(char32_t code) {
    static const char* const kLower[] = {
        "alpha", "beta",     "gamma", "delta", "epsilon", "zeta",    "eta",  "theta",
        "iota",  "kappa",    "lambda", "mu",   "nu",      "xi",      "omicron", "pi",
        "rho",   "varsigma", "sigma", "tau",   "upsilon", "phi",     "chi",  "psi",
        "omega",
    };
    static const char* const kUpper[] = {
        "Alpha", "Beta",  "Gamma", "Delta",   "Epsilon", "Zeta", "Eta",   "Theta",
        "Iota",  "Kappa", "Lambda", "Mu",     "Nu",      "Xi",   "Omicron", "Pi",
        "Rho",   nullptr, "Sigma", "Tau",     "Upsilon", "Phi",  "Chi",   "Psi",
        "Omega",
    };
    if (code >= 0x3B1U && code <= 0x3C9U) {
        return kLower[code - 0x3B1U];
    }
    if (code >= 0x391U && code <= 0x3A9U) {
        return kUpper[code - 0x391U];
    }
    return nullptr;
}

/// True for the seven one-character escapes of §1.2. The other three are control
/// *words* with a trailing `{}` and are handled where names are read.
bool is_escape_char(char c) {
    return c == '#' || c == '$' || c == '%' || c == '&' || c == '_' || c == '{' ||
           c == '}';
}

/// Reverse of `escape` (notation_latex.cpp:49-74), over the raw bytes between a pair of
/// braces. Returns false and reports the offending byte offset when the text contains
/// something `escape` could not have produced -- an unescaped special, or a backslash
/// followed by anything but the ten spellings. Accepting those would mean inventing a
/// name for input the printer cannot emit.
bool unescape_name(std::string_view raw, std::string& out, std::size_t& bad_at) {
    out.clear();
    out.reserve(raw.size());
    std::size_t i = 0;
    while (i < raw.size()) {
        const char c = raw[i];
        if (c != '\\') {
            if (c == '{' || c == '}' || c == '$' || c == '&' || c == '#' || c == '%' ||
                c == '^' || c == '~' || c == '_') {
                bad_at = i;
                return false;
            }
            out.push_back(c);
            ++i;
            continue;
        }
        if (i + 1 < raw.size() && is_escape_char(raw[i + 1])) {
            out.push_back(raw[i + 1]);
            i += 2;
            continue;
        }
        const std::string_view rest = raw.substr(i);
        if (rest.rfind("\\textasciitilde{}", 0) == 0) {
            out.push_back('~');
            i += sizeof("\\textasciitilde{}") - 1;
            continue;
        }
        if (rest.rfind("\\textasciicircum{}", 0) == 0) {
            out.push_back('^');
            i += sizeof("\\textasciicircum{}") - 1;
            continue;
        }
        if (rest.rfind("\\textbackslash{}", 0) == 0) {
            out.push_back('\\');
            i += sizeof("\\textbackslash{}") - 1;
            continue;
        }
        bad_at = i;
        return false;
    }
    return true;
}

// --- The diagnostics ----------------------------------------------------------------
//
// Every string below is the wording `docs/LATEX_SUBSET.md` §3.2 gives for that
// rejection, carrying its code so that a caller can key on the code and a reader can
// find the row. The position goes in `ParseError::line`/`col` rather than into the text:
// `format_error` already prefixes "line L, column C: " to every ParseError, and a
// message that carried its own copy would print the position twice.

constexpr const char* kMsgEmptyInput =
    "[E-LATEX-0001] empty input; there is no expression to read";
constexpr const char* kMsgExpectedExpression = "[E-LATEX-0002] ";
constexpr const char* kMsgDelimiter = "[E-LATEX-0003] ";
constexpr const char* kMsgTrailing = "[E-LATEX-0004] ";
constexpr const char* kMsgTooDeep =
    "[E-LATEX-0005] the expression is nested too deeply to read; the limit is fixed so "
    "that a deep input is a diagnostic on every platform rather than a crash on the one "
    "with the smallest stack";
constexpr const char* kMsgNumeral = "[E-LATEX-0006] ";
constexpr const char* kMsgRagged = "[E-LATEX-0007] ";

constexpr const char* kMsgBraceless =
    "[E-LATEX-0011] a braceless argument takes only the next token: x^10 is x^{1} "
    "multiplied by 0 in TeX; write x^{10}";
constexpr const char* kMsgDoubleSuperscript =
    "[E-LATEX-0012] double superscript; TeX rejects this. Write x^{2 \\cdot 3} or "
    "(x^{2})^{3}";
constexpr const char* kMsgFunctionSuperscript =
    "[E-LATEX-0013] a superscript on a function name is the square of the value at 2 "
    "and the inverse function at -1; write (\\sin(x))^{2}, \\arcsin(x), or "
    "\\frac{1}{\\sin(x)}";
constexpr const char* kMsgUnfencedArgument =
    "[E-LATEX-0014] the extent of the argument is not written: \\sin x + 1 reads as "
    "(\\sin x) + 1 or \\sin(x + 1); write \\sin(x)";
constexpr const char* kMsgFunctionSubscript =
    "[E-LATEX-0015] a subscript on a function name is the base of the logarithm here "
    "and an index elsewhere; the subset has no based logarithm. Write "
    "\\frac{\\log(x)}{\\log(2)}";
constexpr const char* kMsgSlashExtent =
    "[E-LATEX-0016] the denominator ends at b or at c depending on the writer; use "
    "\\frac{a}{bc} or \\frac{a}{b} \\cdot c";
constexpr const char* kMsgFactorial =
    "[E-LATEX-0017] a postfix ! is a factorial, a double factorial when doubled, and a "
    "negation in some writing; write \\operatorname{factorial}(n)";
constexpr const char* kMsgPrime =
    "[E-LATEX-0018] a prime or a dot is a derivative with respect to an unwritten "
    "variable (and a prime is also a transpose); write "
    "\\frac{d}{dx} \\operatorname{f}(x)";
constexpr const char* kMsgPlusMinus =
    "[E-LATEX-0019] a \\pm b denotes two expressions at once; a MathScript expression "
    "is one value";
constexpr const char* kMsgRelation =
    "[E-LATEX-0020] the subset parses expressions, not equations; '=' has no expression "
    "head. Parse the two sides separately, or write the difference";
constexpr const char* kMsgBigOperator =
    "[E-LATEX-0021] \\sum_{i=1}^{n} has no expression head here; big operators are "
    "outside the subset";
constexpr const char* kMsgBinomial =
    "[E-LATEX-0022] \\binom{n}{k} has no expression head; write "
    "\\operatorname{binomial}(n, k)";
constexpr const char* kMsgIntegralBounds =
    "[E-LATEX-0023] an integral here records no bounds, so limits would be silently "
    "discarded; write \\int f \\, dx";
constexpr const char* kMsgLimitDirection =
    "[E-LATEX-0024] a limit here records no direction, so a one-sided "
    "limit would be read as a two-sided one";
constexpr const char* kMsgPartial =
    "[E-LATEX-0025] a derivative here records no distinction between a partial and a "
    "total one, so the two would be indistinguishable; write \\frac{d}{dx}. "
    "\\partial alone is the symbol named partial";
constexpr const char* kMsgBracket =
    "[E-LATEX-0026] \\{ \\} is a set or a case split, [a, b] is an interval, a list or "
    "a matrix row, and \\lfloor x \\rfloor is a floor; none has an expression head. Use "
    "( ) for grouping and \\operatorname{floor}(x) for a floor";
constexpr const char* kMsgEvaluationBar =
    "[E-LATEX-0027] an evaluation bar (a null delimiter with limits) has no expression "
    "head";
constexpr const char* kMsgNestedBar =
    "[E-LATEX-0028] a bare vertical bar cannot be paired: the open and close delimiter "
    "are the same character. Write \\left| ... \\right| or \\lvert ... \\rvert";
constexpr const char* kMsgDecoration =
    "[E-LATEX-0029] decoration is not part of a name in this subset: \\hat{x} and x are "
    "different symbols to a reader and the same name to the parser. Write x_{hat} or "
    "another name";
constexpr const char* kMsgFontVariant =
    "[E-LATEX-0030] \\varGamma is a font variant of \\Gamma, not a distinct symbol; "
    "write \\Gamma";
constexpr const char* kMsgScientific =
    "[E-LATEX-0032] 1e20 is 1 multiplied by Euler's number, plus 20, in math mode; "
    "write 1 \\times 10^{20}";
constexpr const char* kMsgLeibniz =
    "[E-LATEX-0045] \\frac{dy}{dx} is a derivative to a reader and a quotient to this "
    "grammar, whose derivative is \\frac{d}{dx} with the function after it. Read as a "
    "quotient the two d's cancel and it means y/x, which is not what you wrote; write "
    "\\frac{d}{dx} y";
constexpr const char* kMsgIntegralExtent =
    "[E-LATEX-0046] the integrand runs to the end of its group, so an integral with "
    "something after its differential does not say where it stops: \\int f \\, dx + 1 "
    "is (\\int f \\, dx) + 1 to a reader and \\int (f \\, dx + 1) here. Bracket the "
    "integral";
constexpr const char* kMsgInfimum =
    "[E-LATEX-0033] \\inf is the infimum operator, not infinity; write \\infty. (\\sup "
    "is a function name in this subset; \\inf is not.)";
constexpr const char* kMsgContinuedFraction =
    "[E-LATEX-0034] \\cfrac is continued-fraction layout with an optional alignment "
    "argument, not a distinct operation; write \\frac{a}{b}";
constexpr const char* kMsgMacro =
    "[E-LATEX-0035] macro expansion is not performed; TeX is Turing-complete. Expand "
    "the macro before parsing";
constexpr const char* kMsgEnvironment =
    "[E-LATEX-0036] \\begin{array}{cc} carries a column specification, which is "
    "presentation rather than structure; use pmatrix, and parse a matrix with the "
    "matrix entry point";
constexpr const char* kMsgMatrixInExpression =
    "[E-LATEX-0037] a matrix is not an expression here and has no head to be read into; "
    "parse it with parse_latex_matrix";
constexpr const char* kMsgCellSeparator =
    "[E-LATEX-0038] '&' is a matrix cell separator and '\\\\' a row separator; neither "
    "is an expression token";
constexpr const char* kMsgEmptyFunctionName =
    "[E-LATEX-0039] an empty function name carries no name to recover";
constexpr const char* kMsgStraySubscript =
    "[E-LATEX-0040] a subscript belongs to a symbol name; it has no meaning here";
constexpr const char* kMsgDecimalComma =
    "[E-LATEX-0041] 1,5 is one number with a decimal comma or two numbers in a list; "
    "write 1{,}5 and configure decimal_separator = ','";
constexpr const char* kMsgEmptyRadicand = "[E-LATEX-0042] empty radicand";
constexpr const char* kMsgSurd =
    "[E-LATEX-0043] \\surd is a glyph with no radicand; write \\sqrt{x}";

/// A control word that is spelled out in §3.2 by name, with the row it belongs to.
/// Looking the name up in one table rather than testing for each of them where it might
/// appear is what keeps the rejection independent of the position it is found in: a
/// `\sum` is outside the subset wherever it stands.
struct NamedRejection {
    const char* name;
    const char* message;
};

constexpr NamedRejection kNamedRejections[] = {
    {"sum", kMsgBigOperator},        {"prod", kMsgBigOperator},
    {"bigcup", kMsgBigOperator},     {"bigcap", kMsgBigOperator},
    {"coprod", kMsgBigOperator},     {"bigoplus", kMsgBigOperator},
    {"binom", kMsgBinomial},         {"dbinom", kMsgBinomial},
    {"tbinom", kMsgBinomial},        {"choose", kMsgBinomial},
    {"oint", kMsgIntegralBounds},    {"limsup", kMsgLimitDirection},
    {"liminf", kMsgLimitDirection},  {"le", kMsgRelation},
    {"leq", kMsgRelation},           {"ge", kMsgRelation},
    {"geq", kMsgRelation},           {"neq", kMsgRelation},
    {"ne", kMsgRelation},            {"approx", kMsgRelation},
    {"equiv", kMsgRelation},         {"sim", kMsgRelation},
    {"propto", kMsgRelation},        {"in", kMsgRelation},
    {"mid", kMsgRelation},           {"pm", kMsgPlusMinus},
    {"mp", kMsgPlusMinus},           {"dot", kMsgPrime},
    {"ddot", kMsgPrime},             {"text", kMsgDecoration},
    {"mathbf", kMsgDecoration},      {"mathbb", kMsgDecoration},
    {"mathcal", kMsgDecoration},     {"mathfrak", kMsgDecoration},
    {"mathsf", kMsgDecoration},      {"boldsymbol", kMsgDecoration},
    {"vec", kMsgDecoration},         {"hat", kMsgDecoration},
    {"bar", kMsgDecoration},         {"overline", kMsgDecoration},
    {"tilde", kMsgDecoration},       {"underline", kMsgDecoration},
    {"mathit", kMsgDecoration},      {"bm", kMsgDecoration},
    {"inf", kMsgInfimum},            {"cfrac", kMsgContinuedFraction},
    {"newcommand", kMsgMacro},       {"renewcommand", kMsgMacro},
    {"def", kMsgMacro},              {"let", kMsgMacro},
    {"providecommand", kMsgMacro},   {"surd", kMsgSurd},
    {"root", kMsgSurd},              {"langle", kMsgBracket},
    {"rangle", kMsgBracket},         {"lfloor", kMsgBracket},
    {"rfloor", kMsgBracket},         {"lceil", kMsgBracket},
    {"rceil", kMsgBracket},          {"lVert", kMsgBracket},
    {"rVert", kMsgBracket},          {"Vert", kMsgBracket},
    {"lbrace", kMsgBracket},         {"rbrace", kMsgBracket},
    {"lbrack", kMsgBracket},         {"rbrack", kMsgBracket},
};

/// `\varGamma` and `\upalpha` stand for whole families of font-variant control words,
/// so they are recognised by shape rather than listed: `\var` before a *capitalised*
/// Greek name (the lower-case `\varphi` family is real and is in `kGreekLower`), and
/// `\up` before any Greek name at all.
bool is_font_variant(std::string_view name) {
    if (name.size() > 3 && name.rfind("var", 0) == 0 &&
        name[3] >= 'A' && name[3] <= 'Z') {
        return true;
    }
    if (name.size() > 2 && name.rfind("up", 0) == 0) {
        const std::string_view rest = name.substr(2);
        if (lookup(kGreekLower, rest) != nullptr ||
            lookup(kGreekCapitals, rest) != nullptr) {
            return true;
        }
        for (const UprightCapital& entry : kUprightCapitals) {
            if (rest == entry.name) {
                return true;
            }
        }
    }
    return false;
}

/// A chain like `\frac{\frac{\frac{...}}}` recurses once per level, and how deep that
/// can go before the stack runs out is a property of the platform rather than of the
/// input: Linux gives a thread 8 MB and Windows gives it 1 MB. It segfaulted on the
/// small one and survived the large one, which is why `repl_engine_internal.cpp` carries
/// `kMaxScalarExprDepth = 1024` -- and why this file carries its own.
///
/// The count is raised in two places, `parse_expression` and `parse_primary`, so one
/// level of nesting costs two and this number permits 128 of them. A level is about
/// eight frames of this grammar -- Expression, Additive, Term, Power, Primary and
/// whichever construct recursed -- and measures at roughly 1.6 KB unoptimised, so the
/// deepest accepted input costs about 200 KB. That is a fivefold margin inside the 1 MB
/// a Windows thread gets, which is what it has to be: the parser is not at the bottom of
/// the stack when a REPL or a host program calls it. 128 levels is still far past
/// anything a person writes or the printer emits -- `to_latex` of an expression that
/// deep would be unreadable long before it was unparseable.
///
/// Guarding both functions rather than only `parse_expression` halves the depth and buys
/// the guarantee outright: it removes the need to prove that no path back into
/// `parse_primary` bypasses `parse_expression`. That proof holds today and would be one
/// refactor away from not holding.
constexpr int kMaxParseDepth = 256;

/// Puts a counter back to what it was, on the error path as well as the normal one.
class BarCountRestore {
public:
    BarCountRestore(int& counter, int value) : counter_(counter), value_(value) {}
    ~BarCountRestore() { counter_ = value_; }
    BarCountRestore(const BarCountRestore&) = delete;
    BarCountRestore& operator=(const BarCountRestore&) = delete;
    BarCountRestore(BarCountRestore&&) = delete;
    BarCountRestore& operator=(BarCountRestore&&) = delete;

private:
    int& counter_;
    int value_;
};

/// Raises the depth for as long as it is alive. The count has to come back down on the
/// error path too, and every parse function has several.
class DepthGuard {
public:
    explicit DepthGuard(int& counter) : counter_(counter) { ++counter_; }
    ~DepthGuard() { --counter_; }
    DepthGuard(const DepthGuard&) = delete;
    DepthGuard& operator=(const DepthGuard&) = delete;
    DepthGuard(DepthGuard&&) = delete;
    DepthGuard& operator=(DepthGuard&&) = delete;

private:
    int& counter_;
};

class LatexParser {
public:
    LatexParser(std::string_view text, const NotationOptions& options)
        : text_(text), toks_(lex::tokenize(text)),
          decimal_(options.decimal_separator == ',' ? ',' : '.') {
        limit_ = toks_.size() - 1;
        boundary_ = toks_[limit_];
    }

    Result<ExprRef> run_expression();
    Result<Matrix<double>> run_matrix();

private:
    // --- Cursor -------------------------------------------------------------------

    const Token& peek(std::size_t ahead = 0) const {
        const std::size_t at = pos_ + ahead;
        return at < limit_ ? toks_[at] : boundary_;
    }

    bool at_end() const { return pos_ >= limit_; }

    const Token& advance() {
        if (pos_ < limit_) {
            return toks_[pos_++];
        }
        return boundary_;
    }

    /// Narrow the parse to `[pos_, at)`. Every bounded construct -- a braced argument, a
    /// parenthesised group, one call argument, an integrand -- sets this, which is what
    /// gives "the end of the enclosing group" of §2.1 and §2.6 a single definition
    /// instead of one per caller. The token at the boundary is reported as end of input
    /// because to everything inside the group that is what it is.
    void set_limit(std::size_t at) {
        limit_ = at;
        boundary_ = toks_[at];
        boundary_.kind = TokKind::End;
    }

    // --- Diagnostics --------------------------------------------------------------

    ExprRef fail(const Token& at, const std::string& message) {
        if (!failed_) {
            failed_ = true;
            error_ = ParseError{at.line, at.col, message};
        }
        return nullptr;
    }

    ExprRef expected(const Token& at, const std::string& what) {
        return fail(at, std::string(kMsgExpectedExpression) + "expected " + what +
                            ", found " + lex::describe(at));
    }

    /// §5 rule 3: the position is the closer, because that is where the reader is, and
    /// the message carries the opener, because that is what they have to go back to.
    ExprRef unclosed(const Token& at, const Token& opener, const std::string& opener_text,
                     const std::string& what) {
        return fail(at, std::string(kMsgDelimiter) + "expected " + what + " to close " +
                            opener_text + " opened at line " +
                            std::to_string(opener.line) + ", column " +
                            std::to_string(opener.col) + ", found " + lex::describe(at));
    }

    ExprRef unclosed(const Token& at, const Token& opener, const std::string& what) {
        return unclosed(at, opener, lex::describe(opener), what);
    }

    /// The tokens that are in the language somewhere but not here. Naming them by their
    /// own row rather than as "unexpected" is the difference between a diagnostic that
    /// explains and one that only complains.
    bool stray_rejection(const Token& token) {
        if (token.kind == TokKind::Char && token.text.size() == 1) {
            switch (token.text.front()) {
            case '&': fail(token, kMsgCellSeparator); return true;
            case '|': fail(token, kMsgNestedBar); return true;
            case '=':
            case '<':
            case '>': fail(token, kMsgRelation); return true;
            case '!': fail(token, kMsgFactorial); return true;
            case '\'': fail(token, kMsgPrime); return true;
            case '[':
            case ']': fail(token, kMsgBracket); return true;
            case '_': fail(token, kMsgStraySubscript); return true;
            default: break;
            }
            return false;
        }
        if (lex::is_symbol(token, '\\')) {
            fail(token, kMsgCellSeparator);
            return true;
        }
        if (lex::is_symbol(token, '|')) {
            fail(token, kMsgNestedBar);
            return true;
        }
        if (token.kind != TokKind::ControlWord) {
            return false;
        }
        for (const NamedRejection& entry : kNamedRejections) {
            if (token.text == entry.name) {
                fail(token, entry.message);
                return true;
            }
        }
        if (is_font_variant(token.text)) {
            fail(token, kMsgFontVariant);
            return true;
        }
        if (token.text == "begin" || token.text == "end") {
            std::string environment;
            read_environment_name(pos_ + 1, environment);
            fail(token, lookup(kMatrixEnvironments, environment) != nullptr
                            ? kMsgMatrixInExpression
                            : kMsgEnvironment);
            return true;
        }
        return false;
    }

    /// The name inside the braces of a `\begin`/`\end`, without moving the cursor.
    /// Empty when the braces are missing, which `kMsgEnvironment` covers: an environment
    /// with no name is not one of the five either.
    void read_environment_name(std::size_t at, std::string& out) const {
        out.clear();
        if (at >= limit_ || !lex::is_char(toks_[at], '{')) {
            return;
        }
        const std::size_t close = find_brace_end(at + 1);
        if (close >= limit_) {
            return;
        }
        out = std::string(text_.substr(toks_[at].offset + 1,
                                       toks_[close].offset - toks_[at].offset - 1));
    }

    /// Step over the `{name}` of a `\begin`/`\end` the cursor is sitting on. The name
    /// is a run of letter tokens rather than one token, so its width is not fixed.
    void skip_environment_name() {
        if (at_end() || !lex::is_char(peek(), '{')) {
            return;
        }
        const std::size_t close = find_brace_end(pos_ + 1);
        pos_ = close >= limit_ ? limit_ : close + 1;
    }

    // --- Group scanning -------------------------------------------------------------

    /// The index of the `}` matching a `{` that has just been consumed, or `limit_`.
    /// Only braces count here, deliberately: a name may contain a comma or a bracket
    /// (`escape` passes both through), so `\mathrm{a,b}` is one group and a scan that
    /// stopped at punctuation would call it unclosed.
    std::size_t find_brace_end(std::size_t from) const {
        int depth = 0;
        for (std::size_t i = from; i < limit_; ++i) {
            if (lex::is_char(toks_[i], '{')) {
                ++depth;
            } else if (lex::is_char(toks_[i], '}')) {
                if (depth == 0) {
                    return i;
                }
                --depth;
            }
        }
        return limit_;
    }

    /// The index of the token that closes the group the cursor is inside, or `limit_`.
    /// Unlike `find_brace_end` this is the *expression* boundary, so it also stops at a
    /// comma that separates call arguments and at the bar that closes an unbraced
    /// absolute value -- §2.6 needs the end of the region an integrand may occupy, and
    /// those end it as surely as a parenthesis does.
    std::size_t find_group_end(std::size_t from) const {
        int depth = 0;
        for (std::size_t i = from; i < limit_; ++i) {
            const Token& token = toks_[i];
            if (token.kind == TokKind::Char && token.text.size() == 1) {
                const char c = token.text.front();
                if (c == '{' || c == '(' || c == '[') {
                    ++depth;
                } else if (c == '}' || c == ')' || c == ']') {
                    if (depth == 0) {
                        return i;
                    }
                    --depth;
                } else if (c == ',' && depth == 0) {
                    return i;
                } else if (c == '|' && depth == 0 && in_bare_bar_) {
                    return i;
                }
                continue;
            }
            if (lex::is_symbol(token, '|') && depth == 0 && in_bare_bar_) {
                return i;
                continue;
            }
            if (token.kind == TokKind::ControlWord &&
                lookup(kBigDelimiters, token.text) != nullptr) {
                // A sized delimiter is part of the closer it precedes, so the group ends
                // *before* it rather than at the parenthesis: `\big( x \big)` has to
                // leave both tokens of `\big)` to whoever opened the group.
                const Token& next = i + 1 < limit_ ? toks_[i + 1] : boundary_;
                if (depth == 0 && (lex::is_char(next, ')') ||
                                   (in_bare_bar_ && is_bar(next)))) {
                    return i;
                }
            } else if (lex::is_word(token, "left")) {
                ++depth;
                ++i;
            } else if (lex::is_word(token, "right")) {
                if (depth == 0) {
                    return i;
                }
                --depth;
                ++i;
            } else if (lex::is_word(token, "lvert")) {
                ++depth;
            } else if (lex::is_word(token, "rvert")) {
                if (depth == 0) {
                    return i;
                }
                --depth;
            } else if (lex::is_word(token, "end") && depth == 0) {
                return i;
            }
        }
        return limit_;
    }

    /// Whether the `\{` at the cursor is the opening brace of a set rather than the
    /// one-character Symbol named `{`.
    ///
    /// Both readings are real: `symbol("{")` prints `\{` and §4.1 promises that names
    /// with LaTeX specials round-trip, while A34 rejects `\{ ... \}` as a set. Nothing
    /// in the string tells them apart, so the tie is broken on whether a closing `\}`
    /// follows at the same depth -- a pair is a delimiter, a lone one is a name. The
    /// cost is that `mul(symbol("{"), symbol("}"))` is diagnosed as a set instead of
    /// round-tripping; the alternative cost was every set spelled `\{ x \}` being read
    /// as a product of three symbols, silently.
    bool closing_brace_escape_follows(std::size_t from) const {
        int depth = 0;
        for (std::size_t i = from; i < limit_; ++i) {
            const Token& token = toks_[i];
            if (lex::is_symbol(token, '}') && depth == 0) {
                return true;
            }
            if (token.kind != TokKind::Char || token.text.size() != 1) {
                continue;
            }
            const char c = token.text.front();
            if (c == '{' || c == '(') {
                ++depth;
            } else if (c == '}' || c == ')') {
                if (depth == 0) {
                    return false;
                }
                --depth;
            }
        }
        return false;
    }

    // --- Names ----------------------------------------------------------------------

    /// One `NameAtom` of §2.5, un-spelled back to the characters `spell_name` was given.
    ///
    /// `single_token` is TeX's one-token rule (H4): a braceless argument takes exactly
    /// the next token, so `x_12` is `x_1` followed by a stray 2 rather than `x_{12}`.
    /// The braced form takes a whole run of digits, because that is what `x_{12}` is.
    bool parse_name_atom(std::string& out, bool single_token) {
        const Token& token = peek();
        if (token.kind == TokKind::Char && token.text.size() == 1) {
            const char c = token.text.front();
            if (lex::is_ascii_letter(c)) {
                out.assign(1, c);
                advance();
                return true;
            }
            if (lex::is_ascii_digit(c)) {
                return parse_digit_name(out, single_token);
            }
            return false;
        }
        if (token.kind == TokKind::Char) {
            const char* greek = greek_from_code(token.code);
            if (greek == nullptr) {
                return false;
            }
            out = greek;
            advance();
            return true;
        }
        if (token.kind == TokKind::ControlSymbol) {
            if (token.text.size() != 1 || !is_escape_char(token.text.front())) {
                return false;
            }
            out.assign(1, token.text.front());
            advance();
            return true;
        }
        if (token.kind != TokKind::ControlWord) {
            return false;
        }
        char escaped = '\0';
        if (token.text == "textasciitilde") {
            escaped = '~';
        } else if (token.text == "textasciicircum") {
            escaped = '^';
        } else if (token.text == "textbackslash") {
            escaped = '\\';
        }
        if (escaped != '\0') {
            // The `{}` is part of the escape (§1.2), not an empty group after it.
            if (!lex::is_char(peek(1), '{') || !lex::is_char(peek(2), '}')) {
                return false;
            }
            out.assign(1, escaped);
            advance();
            advance();
            advance();
            return true;
        }
        const char* known = lookup(kGreekLower, token.text);
        if (known == nullptr) {
            known = lookup(kGreekCapitals, token.text);
        }
        if (known == nullptr) {
            known = lookup(kGlyphs, token.text);
        }
        if (known != nullptr) {
            out = known;
            advance();
            return true;
        }
        if (token.text == "mathrm" && !single_token) {
            return parse_upright_name(out);
        }
        return false;
    }

    bool parse_digit_name(std::string& out, bool single_token) {
        const Token& first = advance();
        out.assign(1, first.text.front());
        if (single_token) {
            if (lex::is_digit_token(peek()) && lex::adjacent(first, peek())) {
                fail(first, kMsgBraceless);
                return false;
            }
            return true;
        }
        const Token* previous = &first;
        while (lex::is_digit_token(peek()) && lex::adjacent(*previous, peek())) {
            previous = &advance();
            out.push_back(previous->text.front());
        }
        return true;
    }

    /// `\mathrm{...}`: either one of the thirteen Greek capitals that are Latin letters
    /// (A9) or a multi-letter name with the ten escapes reversed.
    bool parse_upright_name(std::string& out) {
        advance();
        if (!lex::is_char(peek(), '{')) {
            expected(peek(), "'{' after \\mathrm");
            return false;
        }
        const Token opener = advance();
        const std::size_t close = find_brace_end(pos_);
        if (close >= limit_) {
            unclosed(boundary_, opener, "'}'");
            return false;
        }
        const std::string_view raw =
            text_.substr(opener.offset + 1, toks_[close].offset - opener.offset - 1);
        pos_ = close + 1;
        if (raw.size() == 1) {
            const char* capital = upright_capital_name(raw.front());
            if (capital != nullptr) {
                out = capital;
                return true;
            }
        }
        std::size_t bad_at = 0;
        if (!unescape_name(raw, out, bad_at)) {
            fail(opener, std::string(kMsgExpectedExpression) +
                             "expected an escaped name inside \\mathrm{}, found '" +
                             std::string(raw.substr(bad_at, 1)) +
                             "' which the printer would have escaped");
            return false;
        }
        if (out.empty()) {
            fail(opener, std::string(kMsgExpectedExpression) +
                             "expected a name inside \\mathrm{}, found an empty group");
            return false;
        }
        return true;
    }

    /// `NameAtom [ "_" Argument ]`, rejoined the way `split_subscript` took it apart.
    /// The subscript is un-spelled as a *name*, so `x_{\pi}` is `x_pi` rather than a
    /// product with a constant in it (§2.5 step 2).
    bool parse_symbol_name(std::string& out) {
        std::string base;
        if (!parse_name_atom(base, false)) {
            return false;
        }
        if (!lex::is_char(peek(), '_')) {
            out = std::move(base);
            return true;
        }
        advance();
        std::string subscript;
        if (lex::is_char(peek(), '{')) {
            const Token opener = advance();
            const std::size_t close = find_brace_end(pos_);
            if (close >= limit_) {
                unclosed(boundary_, opener, "'}'");
                return false;
            }
            if (!parse_name_atom(subscript, false) || pos_ != close) {
                if (!failed_) {
                    expected(peek(), "a name in the subscript");
                }
                return false;
            }
            pos_ = close + 1;
        } else if (!parse_name_atom(subscript, true)) {
            if (!failed_) {
                expected(peek(), "a name in the subscript");
            }
            return false;
        }
        out = base + "_" + subscript;
        return true;
    }

    /// `parse_symbol_name` with the cursor and the error state put back on failure, for
    /// the two places that have to look before they commit: the derivative shape of A4
    /// and the differential scan of §2.6.
    bool speculative_symbol_name(std::string& out) {
        const std::size_t saved_pos = pos_;
        const bool saved_failed = failed_;
        ParseError saved_error = error_;
        if (parse_symbol_name(out) && !failed_) {
            return true;
        }
        pos_ = saved_pos;
        failed_ = saved_failed;
        error_ = std::move(saved_error);
        return false;
    }

    // --- Numerals -------------------------------------------------------------------

    /// The digits of one numeral. They must be adjacent in the source: `23` is a number
    /// and `2 3` and `2\,3` are a product (A14), which is what `2\,10^{n}` depends on.
    void read_digit_run(std::string& out) {
        const Token* previous = &advance();
        out.assign(1, previous->text.front());
        while (lex::is_digit_token(peek()) && lex::adjacent(*previous, peek())) {
            previous = &advance();
            out.push_back(previous->text.front());
        }
    }

    /// `strtod` over digits this parser assembled itself, so the only outcome worth
    /// checking for is the one the standard library reports by not moving the end
    /// pointer. An exponent past the range of a double gives HUGE_VAL, which is the
    /// infinity N22 says `2 \times 10^{5000}` re-reads as -- an overflow here is a
    /// value, not an error.
    static double to_double(const std::string& digits) {
        const char* begin = digits.c_str();
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        return end == begin ? 0.0 : value;
    }

    /// The decimal separator as §1.6 spells it: a `.` when configured `'.'`, and the
    /// three tokens `{` `,` `}` when configured `','` -- which is not a group, it is how
    /// `notation_latex.cpp:516-530` keeps a comma from being set as punctuation.
    bool match_decimal_point(const Token& before) {
        if (decimal_ == '.') {
            return lex::is_char(peek(), '.') && lex::adjacent(before, peek()) &&
                   lex::is_digit_token(peek(1)) && lex::adjacent(peek(), peek(1));
        }
        return lex::is_char(peek(), '{') && lex::adjacent(before, peek()) &&
               lex::is_char(peek(1), ',') && lex::adjacent(peek(), peek(1)) &&
               lex::is_char(peek(2), '}') && lex::adjacent(peek(1), peek(2)) &&
               lex::is_digit_token(peek(3)) && lex::adjacent(peek(2), peek(3));
    }

    /// `( INTLIT | DECLIT ) "\times" "10" "^" "{" [ "-" ] INTLIT "}"` -- §2.4's
    /// `RealNumeral`, matched leftmost-longest and therefore before the multiplicative
    /// rule gets a chance at the `\times` (A11, A12). Restores the cursor when the tail
    /// turns out to be an ordinary product such as `2 \times x`.
    bool match_scientific_tail(std::string& digits) {
        const std::size_t saved = pos_;
        if (!lex::is_word(peek(), "times")) {
            return false;
        }
        advance();
        if (!lex::is_char(peek(), '1') || !lex::is_char(peek(1), '0') ||
            !lex::adjacent(peek(), peek(1)) || !lex::is_char(peek(2), '^') ||
            !lex::is_char(peek(3), '{')) {
            pos_ = saved;
            return false;
        }
        advance();
        advance();
        advance();
        advance();
        std::string exponent;
        if (lex::is_char(peek(), '-')) {
            advance();
            exponent.push_back('-');
        }
        if (!lex::is_digit_token(peek())) {
            pos_ = saved;
            return false;
        }
        std::string run;
        read_digit_run(run);
        if (!lex::is_char(peek(), '}')) {
            pos_ = saved;
            return false;
        }
        advance();
        digits += "e" + exponent + run;
        return true;
    }

    /// Whether the tokens at `ahead` are the digits of a C-style exponent, with the
    /// optional sign `format_exact` writes in `1e-07`. A40 is about the whole shape:
    /// `1e-7` is `1 \cdot e - 7` in math mode, which is a different value rather than a
    /// worse spelling of the same one.
    bool adjacent_exponent_digits(std::size_t ahead) const {
        std::size_t at = ahead;
        if ((lex::is_char(peek(at), '-') || lex::is_char(peek(at), '+')) &&
            lex::adjacent(peek(at - 1), peek(at))) {
            ++at;
        }
        return lex::is_digit_token(peek(at)) && lex::adjacent(peek(at - 1), peek(at));
    }

    ExprRef parse_numeral() {
        std::string digits;
        read_digit_run(digits);
        const Token& previous = toks_[pos_ - 1];
        if (lex::is_letter_token(peek()) && lex::adjacent(previous, peek()) &&
            (peek().text.front() == 'e' || peek().text.front() == 'E') &&
            adjacent_exponent_digits(1)) {
            return fail(peek(), kMsgScientific);
        }
        bool inexact = false;
        if (match_decimal_point(previous)) {
            advance();
            if (decimal_ != '.') {
                advance();
                advance();
            }
            std::string fraction;
            read_digit_run(fraction);
            digits += "." + fraction;
            inexact = true;
        } else if (decimal_ == '.' && lex::is_char(peek(), '{') &&
                   lex::is_char(peek(1), ',') && lex::is_char(peek(2), '}')) {
            return fail(peek(), kMsgDecimalComma);
        } else if (lex::is_char(peek(), ',') && lex::adjacent(previous, peek()) &&
                   lex::is_digit_token(peek(1)) && lex::adjacent(peek(), peek(1))) {
            return fail(peek(), kMsgDecimalComma);
        }
        if (match_scientific_tail(digits)) {
            return real(to_double(digits));
        }
        if (inexact) {
            return real(to_double(digits));
        }
        return integer(bignum::BigInt(digits));
    }

    // --- Primary --------------------------------------------------------------------

    /// Whether a factor starts at `ahead` tokens from the cursor -- juxtaposition, so
    /// there is no operator to key on and the decision is the token itself.
    ///
    /// It needs the token after it for one case: a sized-delimiter control word begins a
    /// factor when it precedes `(` or `|` and *ends* one when it precedes `)`. Without
    /// the lookahead the `\\big` of `\\big( x \\big)` is read twice, once as the closer it
    /// is and once as the opener of a group that is not there.
    bool starts_factor(std::size_t ahead) const {
        const Token& token = peek(ahead);
        switch (token.kind) {
        case TokKind::End:
            return false;
        case TokKind::Char: {
            if (token.text.size() == 1) {
                const char c = token.text.front();
                return lex::is_ascii_digit(c) || lex::is_ascii_letter(c) || c == '(' ||
                       c == '{' || (c == '|' && !in_bare_bar_);
            }
            return token.code == 0x221EU || greek_from_code(token.code) != nullptr;
        }
        case TokKind::ControlSymbol:
            if (lex::is_symbol(token, '|')) {
                return !in_bare_bar_;
            }
            return token.text.size() == 1 && is_escape_char(token.text.front());
        case TokKind::ControlWord:
            break;
        }
        const std::string_view name = token.text;
        if (lookup(kBigDelimiters, name) != nullptr) {
            return lex::is_char(peek(ahead + 1), '(') ||
                   (is_bar(peek(ahead + 1)) && !in_bare_bar_);
        }
        if (lookup(kGreekLower, name) != nullptr ||
            lookup(kGreekCapitals, name) != nullptr || lookup(kGlyphs, name) != nullptr ||
            lookup(kOperators, name) != nullptr) {
            return true;
        }
        return name == "operatorname" || name == "frac" || name == "dfrac" ||
               name == "tfrac" || name == "sqrt" || name == "mathrm" || name == "left" ||
               name == "lvert" || name == "textasciitilde" || name == "textasciicircum" ||
               name == "textbackslash";
    }

    ExprRef parse_primary() {
        const DepthGuard guard(depth_);
        if (depth_ > kMaxParseDepth) {
            return fail(peek(), kMsgTooDeep);
        }
        const Token& token = peek();
        if (token.kind == TokKind::End) {
            return expected(token, "an expression");
        }
        if (token.kind == TokKind::Char) {
            return parse_char_primary(token);
        }
        if (token.kind == TokKind::ControlSymbol) {
            if (lex::is_symbol(token, '|')) {
                return parse_bar_absolute(advance());
            }
            if (lex::is_symbol(token, '{') && closing_brace_escape_follows(pos_ + 1)) {
                return fail(token, kMsgBracket);
            }
            if (token.text.size() == 1 && is_escape_char(token.text.front())) {
                return parse_symbol_or_constant();
            }
            if (stray_rejection(token)) {
                return nullptr;
            }
            return expected(token, "an expression");
        }
        return parse_control_word_primary(token);
    }

    ExprRef parse_char_primary(const Token& token) {
        if (token.text.size() == 1) {
            const char c = token.text.front();
            if (lex::is_ascii_digit(c)) {
                return parse_numeral();
            }
            if (lex::is_ascii_letter(c)) {
                return parse_symbol_or_constant();
            }
            if (c == '(') {
                return parse_paren_group(advance(), false);
            }
            if (c == '{') {
                return parse_braced_expression();
            }
            if (c == '|') {
                return parse_bar_absolute(advance());
            }
            if (stray_rejection(token)) {
                return nullptr;
            }
            return expected(token, "an expression");
        }
        if (token.code == 0x221EU) {
            advance();
            return constant("inf");
        }
        if (greek_from_code(token.code) != nullptr) {
            return parse_symbol_or_constant();
        }
        return expected(token, "an expression");
    }

    ExprRef parse_control_word_primary(const Token& token) {
        const std::string_view name = token.text;
        if (name == "frac" || name == "dfrac" || name == "tfrac") {
            return parse_fraction();
        }
        if (name == "sqrt") {
            return parse_root();
        }
        if (lookup(kOperators, name) != nullptr || name == "operatorname") {
            return parse_call();
        }
        if (name == "left") {
            return parse_left_group();
        }
        if (name == "lvert") {
            const Token& opener = advance();
            return parse_absolute(opener, BarKind::Vert);
        }
        if (lookup(kBigDelimiters, name) != nullptr) {
            const Token& opener = advance();
            if (lex::is_char(peek(), '(')) {
                advance();
                return parse_paren_group(opener, false);
            }
            if (at_bar()) {
                advance();
                return parse_bar_absolute(opener);
            }
            return expected(peek(), "'(' or '|' after a sized delimiter");
        }
        if (name == "int" || name == "iint" || name == "iiint" || name == "lim") {
            // §2.1 level 0. The body runs to the end of the enclosing group, so a
            // calculus operator standing in a factor slot would swallow the rest of the
            // product; the walker parenthesises one there for exactly that reason
            // (notation.cpp:330, :338, :344) and so must a reader.
            return fail(token, std::string(kMsgExpectedExpression) + "'\\" +
                                   std::string(name) +
                                   "' extends to the end of its group, so it is read "
                                   "only where a whole expression is expected; write "
                                   "(\\" + std::string(name) + " ...) to use it as a "
                                   "factor or a term");
        }
        if (lookup(kGreekLower, name) != nullptr ||
            lookup(kGreekCapitals, name) != nullptr || lookup(kGlyphs, name) != nullptr ||
            name == "mathrm" || name == "textasciitilde" ||
            name == "textasciicircum" || name == "textbackslash") {
            return parse_symbol_or_constant();
        }
        if (stray_rejection(token)) {
            return nullptr;
        }
        return fail(token, std::string("[E-LATEX-0044] unknown control sequence \\") +
                               std::string(name) +
                               "; the accepted subset is documented in "
                               "docs/LATEX_SUBSET.md");
    }

    /// A name, unless it is one of the seven Constant spellings standing on its own.
    /// A9 and A8: the Constant wins at expression level, and a subscript takes the
    /// spelling back out of that class -- `\pi_{1}` is `symbol("pi_1")`, which is what
    /// `split_subscript` and `spell_name` between them produce.
    ExprRef parse_symbol_or_constant() {
        const Token& token = peek();
        std::string name;
        if (!parse_symbol_name(name)) {
            if (failed_) {
                return nullptr;
            }
            return expected(token, "a name");
        }
        if (name == "pi" || name == "e" || name == "i") {
            return constant(name);
        }
        if (name == "infty") {
            return constant("inf");
        }
        if (name == "NaN") {
            return constant("nan");
        }
        if (name == "undefined") {
            return constant("undefined");
        }
        return symbol(name);
    }

    ExprRef parse_braced_expression() {
        const Token opener = advance();
        const std::size_t close = find_brace_end(pos_);
        if (close >= limit_) {
            return unclosed(boundary_, opener, "'}'");
        }
        if (close == pos_) {
            return expected(toks_[close], "an expression");
        }
        const std::size_t outer = limit_;
        set_limit(close);
        ExprRef inner = parse_expression();
        if (!inner) {
            return nullptr;
        }
        if (!at_end()) {
            return expected(peek(), "'}'");
        }
        set_limit(outer);
        pos_ = close + 1;
        return inner;
    }

    /// `(` has already been consumed. `sized` records that the opener was `\left(`, so
    /// that it can be required to close with `\right)` -- §2.4 asks the two to agree.
    ExprRef parse_paren_group(const Token& opener, bool sized) {
        const std::size_t close = find_group_end(pos_);
        const std::size_t outer = limit_;
        set_limit(close);
        ExprRef inner = parse_expression();
        if (!inner) {
            return nullptr;
        }
        if (!at_end()) {
            return expected(peek(), "')'");
        }
        set_limit(outer);
        pos_ = close;
        if (sized) {
            if (!lex::is_word(peek(), "right") || !lex::is_char(peek(1), ')')) {
                return unclosed(mismatch_site(true), opener, "'\\left('", "'\\right)'");
            }
            advance();
            advance();
            return inner;
        }
        if (lookup(kBigDelimiters, peek().text) != nullptr &&
            peek().kind == TokKind::ControlWord && lex::is_char(peek(1), ')')) {
            advance();
        }
        if (!lex::is_char(peek(), ')')) {
            return unclosed(mismatch_site(false), opener, lex::describe(opener), "')'");
        }
        advance();
        return inner;
    }

    ExprRef parse_left_group() {
        const Token opener = advance();
        if (lex::is_char(peek(), '.')) {
            return fail(peek(), kMsgEvaluationBar);
        }
        if (lex::is_char(peek(), '(')) {
            advance();
            return parse_paren_group(opener, true);
        }
        if (at_bar()) {
            advance();
            return parse_absolute(opener, BarKind::Right);
        }
        return fail(peek(), kMsgBracket);
    }

    enum class BarKind : std::uint8_t {
        Right, ///< opened by `\left|`, closed by `\right|`
        Vert,  ///< opened by `\lvert`, closed by `\rvert`
        Bare,  ///< opened by `|`, closed by `|`
    };

    /// A bare `|` opens an absolute value only where one cannot already be open: the
    /// open and close delimiter are the same character, so `||x||` and `|a|b|` have no
    /// reading a parser could defend (A36).
    /// The two spellings of a bar delimiter. `docs/LATEX_SUBSET.md` writes both of them
    /// `\|` -- §1.1 annotates that row "(U+007C)", which makes it a markdown-escaped
    /// pipe, while H3 and A36 read naturally only if it is the control symbol. Rather
    /// than pick, both are accepted: the bare `|` is what the printer emits
    /// (notation_latex.cpp:383) and so must round-trip, and `\|` is what H3 offers a
    /// human. Neither may nest, which is A36 and is the only claim the two readings
    /// agree on anyway.
    static bool is_bar(const Token& token) {
        return lex::is_char(token, '|') || lex::is_symbol(token, '|');
    }

    bool at_bar() const { return is_bar(peek()); }

    /// Where to point when a closing delimiter does not match its opener.
    ///
    /// A closer is a prefix plus a delimiter character, and WHICH prefix is allowed
    /// depends on the opener: `\left(` must close with `\right)`, while a bare `(`
    /// closes with `)` or with a sized `\big)`. When the prefix standing there is the
    /// one this opener licenses, the prefix is fine and the delimiter after it is the
    /// token out of place -- `\left( x \right]` is diagnosed on the `]`, the same
    /// choice A35 makes for `\left.`.
    ///
    /// When it is the other prefix, the prefix ITSELF is the token out of place, and
    /// skipping it produced a diagnostic that contradicted itself: `(x\right)` read
    /// "expected ')' ... found ')'" and pointed at a perfectly good `)`, while the
    /// `\right` six columns earlier -- the only thing wrong with the input -- was never
    /// named.
    const Token& mismatch_site(bool sized) const {
        const bool licensed = sized ? lex::is_word(peek(), "right")
                                    : (peek().kind == TokKind::ControlWord &&
                                       lookup(kBigDelimiters, peek().text) != nullptr);
        return licensed ? peek(1) : peek();
    }

    /// A36. An unbraced bar cannot be paired, so it is accepted only where there is
    /// nothing for it to be confused with: not inside another one (`\|\|x\|\|`) and not
    /// beside one that has already closed in the same term (`\|a\|b\|`, whose middle bar
    /// is both a closer and an opener). Two in separate terms -- `\|a\| + \|b\|` -- pair
    /// unambiguously and are left alone.
    ExprRef parse_bar_absolute(const Token& opener) {
        if (in_bare_bar_ || bars_in_term_ > 0) {
            return fail(opener, kMsgNestedBar);
        }
        return parse_absolute(opener, BarKind::Bare);
    }

    ExprRef parse_absolute(const Token& opener, BarKind kind) {
        if (lex::is_char(peek(), '_') || lex::is_char(peek(), '^')) {
            return fail(peek(), kMsgEvaluationBar);
        }
        const bool saved_bar = in_bare_bar_;
        if (kind == BarKind::Bare) {
            in_bare_bar_ = true;
        }
        ExprRef inner = parse_expression();
        if (!inner) {
            return nullptr;
        }
        in_bare_bar_ = saved_bar;
        switch (kind) {
        case BarKind::Right:
            if (!lex::is_word(peek(), "right") || !is_bar(peek(1))) {
                return unclosed(mismatch_site(true), opener, "'\\left|'", "'\\right|'");
            }
            advance();
            advance();
            break;
        case BarKind::Vert:
            if (!lex::is_word(peek(), "rvert")) {
                return unclosed(peek(), opener, "'\\rvert'");
            }
            advance();
            break;
        case BarKind::Bare:
            ++bars_in_term_;
            if (lookup(kBigDelimiters, peek().text) != nullptr &&
                peek().kind == TokKind::ControlWord && is_bar(peek(1))) {
                advance();
            }
            if (!at_bar()) {
                return unclosed(peek(), opener, lex::describe(opener), "'|'");
            }
            advance();
            break;
        }
        if (lex::is_char(peek(), '_')) {
            return fail(peek(), kMsgEvaluationBar);
        }
        return function("abs", {inner});
    }

    /// `{ Expression }` or TeX's one-token argument (H4).
    ///
    /// `last` says whether another argument follows. A digit touching a braceless digit
    /// argument is A19's `x^10` -- an argument that was silently cut to one token -- but
    /// only when nothing else is waiting for it: `\\frac12` is H4's own example and its
    /// `2` is the denominator, not a stray factor.
    ExprRef parse_argument(bool last = true) {
        if (lex::is_char(peek(), '{')) {
            return parse_braced_expression();
        }
        const Token& token = peek();
        if (lex::is_digit_token(token)) {
            const Token& digit = advance();
            if (last && lex::is_digit_token(peek()) && lex::adjacent(digit, peek())) {
                return fail(digit, kMsgBraceless);
            }
            return integer(static_cast<long long>(digit.text.front() - '0'));
        }
        std::string name;
        if (parse_name_atom(name, true)) {
            if (name == "pi" || name == "e" || name == "i") {
                return constant(name);
            }
            if (name == "infty") {
                return constant("inf");
            }
            return symbol(name);
        }
        if (failed_) {
            return nullptr;
        }
        if (stray_rejection(token)) {
            return nullptr;
        }
        return expected(token, "an argument");
    }

    /// A3: `\frac{A}{B}` is the quotient in every case. No case analysis on what A and B
    /// are, because the constructors canonicalise -- `div(3,4)` already *is*
    /// `rational(3,4)` and `div(1, x^{3})` already is `pow(x,-3)`, so a parser that tried
    /// to pick the right head could only pick a wrong one.
    ///
    /// The one thing it does decide is a product in the denominator, and that is to
    /// invert `walk_mul` rather than to interpret. The printer builds `\frac{x}{a \cdot
    /// b}` out of `mul(x, a^{-1}, b^{-1})` by collecting each negatively-exponentiated
    /// factor into a denominator list (notation.cpp:123-131); `div(x, mul(a, b))` is
    /// `mul(x, pow(mul(a, b), -1))`, and `pow` does not distribute over a Mul, so it is a
    /// different node from the one that was printed. Undistributing here is what closes
    /// the round trip. Doing it in `pow` instead would be the deeper fix and a change to
    /// the canonical form of every expression in the tree, which is not this file's to
    /// make.
    ExprRef parse_fraction() {
        advance();
        ExprRef numerator = parse_argument(false);
        if (!numerator) {
            return nullptr;
        }
        ExprRef denominator = parse_argument();
        if (!denominator) {
            return nullptr;
        }
        if (denominator->head != Head::Mul) {
            return div(std::move(numerator), std::move(denominator));
        }
        std::vector<ExprRef> factors;
        factors.reserve(denominator->args.size() + 1);
        factors.push_back(std::move(numerator));
        for (const ExprRef& factor : denominator->args) {
            factors.push_back(pow(factor, integer(-1)));
        }
        return mul(std::move(factors));
    }

    /// A2: `\sqrt{x}` is `pow(x, 1/2)`, which is what the printer emitted it from
    /// (notation_latex.cpp:369). A16 accepts a symbolic degree even though the printer
    /// never writes one, because `\sqrt[n]{x}` has only the one reading.
    ExprRef parse_root() {
        advance();
        ExprRef degree;
        if (lex::is_char(peek(), '[')) {
            advance();
            degree = parse_expression();
            if (!degree) {
                return nullptr;
            }
            if (!lex::is_char(peek(), ']')) {
                return expected(peek(), "']'");
            }
            advance();
        }
        if (lex::is_char(peek(), '{') && lex::is_char(peek(1), '}')) {
            return fail(peek(1), kMsgEmptyRadicand);
        }
        ExprRef radicand = parse_argument();
        if (!radicand) {
            return nullptr;
        }
        if (!degree) {
            return pow(radicand, rational(bignum::BigInt(1), bignum::BigInt(2)));
        }
        return pow(radicand, div(integer(1), degree));
    }

    /// A1: application is always *marked* in printer output, by `\operatorname{}` or by
    /// one of the 25 log-like control words. A bare letter or an upright word before a
    /// parenthesis is a product, and all three of `g(x + y)`, `\mathrm{foo}\,(x + y)`
    /// and `\operatorname{f}(y)x` are strings the printer emits.
    ExprRef parse_call() {
        const Token head = advance();
        std::string name;
        if (head.text == "operatorname") {
            if (lex::is_char(peek(), '*')) {
                advance(); // H8: `\operatorname*` differs only in where limits are set.
            }
            if (!lex::is_char(peek(), '{')) {
                return expected(peek(), "'{' after \\operatorname");
            }
            const Token opener = advance();
            const std::size_t close = find_brace_end(pos_);
            if (close >= limit_) {
                return unclosed(boundary_, opener, "'}'");
            }
            const std::string_view raw =
                text_.substr(opener.offset + 1, toks_[close].offset - opener.offset - 1);
            pos_ = close + 1;
            if (raw.empty()) {
                return fail(toks_[close], kMsgEmptyFunctionName);
            }
            std::size_t bad_at = 0;
            if (!unescape_name(raw, name, bad_at)) {
                return fail(opener, std::string(kMsgExpectedExpression) +
                                        "expected an escaped name inside "
                                        "\\operatorname{}, found '" +
                                        std::string(raw.substr(bad_at, 1)) + "'");
            }
        } else {
            name = std::string(head.text);
        }
        if (lex::is_char(peek(), '^')) {
            return fail(peek(), kMsgFunctionSuperscript);
        }
        if (lex::is_char(peek(), '_')) {
            return fail(peek(), kMsgFunctionSubscript);
        }
        bool sized = false;
        const Token open_token = peek();
        if (lex::is_word(peek(), "left") && lex::is_char(peek(1), '(')) {
            sized = true;
            advance();
            advance();
        } else if (peek().kind == TokKind::ControlWord &&
                   lookup(kBigDelimiters, peek().text) != nullptr &&
                   lex::is_char(peek(1), '(')) {
            advance();
            advance();
        } else if (lex::is_char(peek(), '(')) {
            advance();
        } else {
            return fail(peek(), kMsgUnfencedArgument);
        }
        std::vector<ExprRef> args;
        if (!close_call(open_token, sized, true)) {
            if (failed_) {
                return nullptr;
            }
            while (true) {
                const std::size_t outer = limit_;
                set_limit(find_group_end(pos_));
                ExprRef arg = parse_expression();
                if (!arg) {
                    return nullptr;
                }
                if (!at_end()) {
                    return expected(peek(), "',' or ')'");
                }
                set_limit(outer);
                args.push_back(std::move(arg));
                if (lex::is_char(peek(), ',')) {
                    advance();
                    continue;
                }
                break;
            }
            if (!close_call(open_token, sized, false)) {
                return nullptr;
            }
        }
        return function(name, std::move(args));
    }

    /// Consumes the closing parenthesis of a call when one is there. `probe` asks
    /// instead whether the argument list is empty -- `\max()` and `\operatorname{f}()`
    /// are both emitted (notation_latex.cpp:400 with no arguments) -- and reports
    /// nothing when it is not.
    bool close_call(const Token& opener, bool sized, bool probe) {
        if (sized) {
            if (lex::is_word(peek(), "right") && lex::is_char(peek(1), ')')) {
                advance();
                advance();
                return true;
            }
            if (!probe) {
                unclosed(mismatch_site(true), opener, "'\\left('", "'\\right)'");
            }
            return false;
        }
        std::size_t ahead = 0;
        if (peek().kind == TokKind::ControlWord &&
            lookup(kBigDelimiters, peek().text) != nullptr) {
            ahead = 1;
        }
        if (lex::is_char(peek(ahead), ')')) {
            pos_ += ahead + 1;
            return true;
        }
        if (!probe) {
            unclosed(peek(), opener, "')'");
        }
        return false;
    }

    // --- Operators ------------------------------------------------------------------

    /// The postfix checks live here rather than in `parse_primary` because they are
    /// about what follows a complete factor: `n!`, `f'(x)` and `\frac{a}{b}_{1}` are all
    /// well-formed up to the mark that has no expression head.
    ExprRef apply_power(ExprRef base) {
        if (!base) {
            return nullptr;
        }
        if (lex::is_char(peek(), '^')) {
            advance();
            ExprRef exponent = parse_argument();
            if (!exponent) {
                return nullptr;
            }
            base = pow(std::move(base), std::move(exponent));
            if (lex::is_char(peek(), '^')) {
                return fail(peek(), kMsgDoubleSuperscript);
            }
        }
        if (lex::is_char(peek(), '_')) {
            return fail(peek(), kMsgStraySubscript);
        }
        if (lex::is_char(peek(), '\'')) {
            return fail(peek(), kMsgPrime);
        }
        if (lex::is_char(peek(), '!')) {
            return fail(peek(), kMsgFactorial);
        }
        return base;
    }

    ExprRef parse_power() { return apply_power(parse_primary()); }

    bool at_multiplication() const {
        const Token& token = peek();
        return lex::is_word(token, "cdot") || lex::is_word(token, "times") ||
               lex::is_word(token, "ast") || lex::is_char(token, '*');
    }

    bool at_division() const {
        return lex::is_char(peek(), '/') || lex::is_word(peek(), "div");
    }

    bool at_minus() const {
        const Token& token = peek();
        return lex::is_char(token, '-') ||
               (token.kind == TokKind::Char && token.code == 0x2212U);
    }

    /// §2.1 level 3, with juxtaposition binding exactly as an explicit operator does.
    /// Division is folded into the accumulated product as it is met so that `a / b / c`
    /// is `(a/b)/c` rather than `a/(b/c)`.
    ExprRef continue_term(ExprRef first) {
        std::vector<ExprRef> factors;
        factors.push_back(std::move(first));
        while (!at_end()) {
            if (at_multiplication()) {
                advance();
                ExprRef factor = parse_power();
                if (!factor) {
                    return nullptr;
                }
                factors.push_back(std::move(factor));
                continue;
            }
            if (at_division()) {
                advance();
                ExprRef denominator = parse_power();
                if (!denominator) {
                    return nullptr;
                }
                if (starts_factor(0)) {
                    return fail(peek(), kMsgSlashExtent);
                }
                ExprRef numerator = factors.size() == 1 ? factors.front()
                                                        : mul(std::move(factors));
                factors.clear();
                factors.push_back(div(std::move(numerator), std::move(denominator)));
                continue;
            }
            // `\,` between two factors is the mark the printer puts where the page would
            // run them together (notation_latex.cpp:346). It separates; it never joins.
            if (lex::is_symbol(peek(), ',') && starts_factor(1)) {
                advance();
                ExprRef factor = parse_power();
                if (!factor) {
                    return nullptr;
                }
                factors.push_back(std::move(factor));
                continue;
            }
            if (!starts_factor(0)) {
                break;
            }
            ExprRef factor = parse_power();
            if (!factor) {
                return nullptr;
            }
            factors.push_back(std::move(factor));
        }
        if (factors.size() == 1) {
            return factors.front();
        }
        return mul(std::move(factors));
    }

    ExprRef parse_term() {
        ExprRef first = parse_power();
        if (!first) {
            return nullptr;
        }
        return continue_term(std::move(first));
    }

    /// `\infty` standing alone, which a leading unary minus joins rather than negates.
    bool at_infinity() const {
        const Token& token = peek();
        const bool infinity = lex::is_word(token, "infty") ||
                              (token.kind == TokKind::Char && token.code == 0x221EU);
        return infinity && !lex::is_char(peek(1), '_');
    }

    /// A Term with the optional unary minus of §2.1 level 2 in front of it.
    ///
    /// The minus is allowed after a binary operator as well as at the start, because the
    /// printer writes it there: `walk_add` asks `negated_for_display` whether a term is
    /// negative, and `constant("-inf")` is not a *number*, so it goes out as an ordinary
    /// term and `add(x, constant("-inf"))` reaches the page as `x + -\infty`. A grammar
    /// that only took a leading minus would reject the printer's own output.
    ExprRef parse_signed_term() {
        if (!at_minus()) {
            return parse_term();
        }
        advance();
        if (at_infinity()) {
            // A10 and N6. `constant("-inf")` and `neg(constant("inf"))` are printed the
            // same way and the document picks the Constant, which has to be built before
            // the minus is applied: the Constant is an atom, so `-\infty^{n}` is
            // `pow(constant("-inf"), n)` and not `neg(pow(constant("inf"), n))`.
            advance();
            ExprRef seed = apply_power(constant("-inf"));
            if (!seed) {
                return nullptr;
            }
            return continue_term(std::move(seed));
        }
        ExprRef term = parse_term();
        if (!term) {
            return nullptr;
        }
        return neg(std::move(term));
    }

    ExprRef parse_additive() {
        // Each term counts its own unbraced bars, and a nested expression gets a fresh
        // count that does not disturb the one it is nested inside.
        const int outer_bars = bars_in_term_;
        bars_in_term_ = 0;
        const BarCountRestore restore{bars_in_term_, outer_bars};
        std::vector<ExprRef> terms;
        ExprRef first = parse_signed_term();
        if (!first) {
            return nullptr;
        }
        terms.push_back(std::move(first));
        while (!at_end()) {
            const bool minus = at_minus();
            if (!minus && !lex::is_char(peek(), '+')) {
                break;
            }
            advance();
            bars_in_term_ = 0;
            ExprRef term = parse_signed_term();
            if (!term) {
                return nullptr;
            }
            terms.push_back(minus ? neg(std::move(term)) : std::move(term));
        }
        if (terms.size() == 1) {
            return terms.front();
        }
        return add(std::move(terms));
    }

    ExprRef parse_expression() {
        const DepthGuard guard(depth_);
        if (depth_ > kMaxParseDepth) {
            return fail(peek(), kMsgTooDeep);
        }
        if (failed_) {
            return nullptr;
        }
        if (at_fraction()) {
            if (lex::is_char(peek(1), '{') && lex::is_word(peek(2), "partial")) {
                return fail(peek(2), kMsgPartial);
            }
            std::vector<std::string> vars;
            std::string var;
            while (try_derivative_operator(var)) {
                vars.push_back(var);
            }
            if (failed_) {
                return nullptr;
            }
            if (!vars.empty()) {
                return finish_derivative(vars);
            }
            if (at_leibniz_fraction()) {
                return fail(peek(), kMsgLeibniz);
            }
        }
        if (lex::is_word(peek(), "int") || lex::is_word(peek(), "iint") ||
            lex::is_word(peek(), "iiint")) {
            return parse_integral();
        }
        if (lex::is_word(peek(), "lim")) {
            return parse_limit();
        }
        return parse_additive();
    }

    bool at_fraction() const {
        return lex::is_word(peek(), "frac") || lex::is_word(peek(), "dfrac") ||
               lex::is_word(peek(), "tfrac");
    }

    // --- Calculus -------------------------------------------------------------------

    /// `d` or `\mathrm{d}` in a differential position (H7).
    bool match_differential_d() {
        if (lex::is_char(peek(), 'd')) {
            advance();
            return true;
        }
        if (lex::is_word(peek(), "mathrm") && lex::is_char(peek(1), '{') &&
            lex::is_char(peek(2), 'd') && lex::is_char(peek(3), '}')) {
            pos_ += 4;
            return true;
        }
        return false;
    }

    /// A differential `d` at token index `at`, in either spelling, with `next` set past
    /// it. Written over the token vector rather than over the cursor because both
    /// callers are speculative and must not move it.
    bool differential_d_at(std::size_t at, std::size_t& next) const {
        if (at < limit_ && lex::is_char(toks_[at], 'd')) {
            next = at + 1;
            return true;
        }
        if (at + 3 < limit_ && lex::is_word(toks_[at], "mathrm") &&
            lex::is_char(toks_[at + 1], '{') && lex::is_char(toks_[at + 2], 'd') &&
            lex::is_char(toks_[at + 3], '}')) {
            next = at + 4;
            return true;
        }
        return false;
    }

    /// The index just past the `{...}` group opening at `at`, or `limit_` if it does not
    /// close inside the current bound.
    std::size_t brace_group_end(std::size_t at) const {
        int depth = 0;
        for (std::size_t i = at; i < limit_; ++i) {
            if (lex::is_char(toks_[i], '{')) {
                ++depth;
            } else if (lex::is_char(toks_[i], '}')) {
                --depth;
                if (depth == 0) {
                    return i + 1;
                }
            }
        }
        return limit_;
    }

    /// `\frac{dy}{dx}` and its relatives: both halves begin with a differential `d`, and
    /// the numerator is not the bare `{d}` that A4's derivative shape requires.
    ///
    /// A4 rules that `\frac{d}{dx} f` is the derivative and that anything else in that
    /// position is a quotient. The ruling is right and it left a hole, because the
    /// quotient reading of `\frac{dy}{dx}` is `(d \cdot y) / (d \cdot x)` -- and `mul`
    /// cancels the `d` at construction, so what came back was `y/x`.
    /// `\frac{d^{2}y}{dx^{2}}` came back `d*y/x^2`, carrying a factor of `d` the author
    /// never wrote, standing where the order of the derivative had been.
    ///
    /// Both readings are genuinely available, and almost nobody writing `\frac{dy}{dx}`
    /// means the quotient. That is the case §3.2 exists for.
    bool at_leibniz_fraction() const {
        if (!at_fraction()) {
            return false;
        }
        const std::size_t numerator = pos_ + 1;
        if (numerator >= limit_ || !lex::is_char(toks_[numerator], '{')) {
            return false;
        }
        std::size_t after_d = 0;
        if (!differential_d_at(numerator + 1, after_d)) {
            return false;
        }
        if (lex::is_char(toks_[after_d], '}')) {
            return false; // the bare `{d}`: A4's derivative, handled above.
        }
        const std::size_t denominator = brace_group_end(numerator);
        if (denominator >= limit_ || !lex::is_char(toks_[denominator], '{')) {
            return false;
        }
        std::size_t unused = 0;
        return differential_d_at(denominator + 1, unused);
    }

    /// `FracCS "{" DiffD "}" "{" DiffD Variable "}"`, matched speculatively: A4 turns on
    /// the denominator being `d` and a Symbol with nothing between them, so
    /// `\frac{d}{d \cdot x}` is the quotient and has to fall back to `parse_fraction`
    /// with the cursor where it started.
    bool try_derivative_operator(std::string& var) {
        const std::size_t saved_pos = pos_;
        const bool saved_failed = failed_;
        ParseError saved_error = error_;
        const bool matched = match_derivative_operator(var);
        if (matched && !failed_) {
            return true;
        }
        pos_ = saved_pos;
        failed_ = saved_failed;
        error_ = std::move(saved_error);
        return false;
    }

    bool match_derivative_operator(std::string& var) {
        if (!at_fraction()) {
            return false;
        }
        advance();
        if (!lex::is_char(peek(), '{')) {
            return false;
        }
        advance();
        if (!match_differential_d() || !lex::is_char(peek(), '}')) {
            return false;
        }
        advance();
        if (!lex::is_char(peek(), '{')) {
            return false;
        }
        advance();
        if (!match_differential_d()) {
            return false;
        }
        if (!speculative_symbol_name(var)) {
            return false;
        }
        if (!lex::is_char(peek(), '}')) {
            return false;
        }
        advance();
        return true;
    }

    /// A7 and N19: a run of adjacent operators is one flat `Head::Derivative` with the
    /// variables in written order. The nested encoding differs from the flat one by a
    /// single space, which math mode discards, so the two cannot be told apart and the
    /// flat reading is the one the printer produces.
    ExprRef finish_derivative(const std::vector<std::string>& vars) {
        ExprRef body = parse_expression();
        if (!body) {
            return nullptr;
        }
        std::vector<ExprRef> symbols;
        symbols.reserve(vars.size());
        for (const std::string& name : vars) {
            symbols.push_back(symbol(name));
        }
        return derivative(std::move(body), std::move(symbols));
    }

    /// §2.6's differential scan.
    ///
    /// The integrand has no closing delimiter, so its extent is decided from the right:
    /// everything up to the end of the enclosing group, less the maximal run of
    /// `\, d<var>` units that ends there. A left-to-right descent cannot do this. It
    /// would have to decide, on meeting `\,`, whether what follows is a differential or
    /// a pair of factors, and `\int \mathrm{aa}\,d` -- which N23 shows the printer
    /// emitting -- answers that question only at end of input.
    ExprRef parse_integral() {
        const Token first = peek();
        std::size_t signs = 0;
        while (true) {
            if (lex::is_word(peek(), "int")) {
                signs += 1;
            } else if (lex::is_word(peek(), "iint")) {
                signs += 2;
            } else if (lex::is_word(peek(), "iiint")) {
                signs += 3;
            } else {
                break;
            }
            advance();
            if (lex::is_word(peek(), "limits")) {
                advance(); // H7
            }
            if (lex::is_char(peek(), '_') || lex::is_char(peek(), '^')) {
                return fail(peek(), kMsgIntegralBounds);
            }
        }
        const std::size_t body_start = pos_;
        const std::size_t group_end = find_group_end(pos_);

        std::vector<std::size_t> separators;
        collect_thin_spaces(body_start, group_end, separators);

        std::vector<std::string> reversed;
        std::size_t boundary = group_end;
        while (!separators.empty()) {
            const std::size_t at = separators.back();
            std::string var;
            if (!match_differential_unit(at, boundary, var)) {
                break;
            }
            reversed.push_back(std::move(var));
            boundary = at;
            separators.pop_back();
        }

        const std::size_t found = reversed.size();
        // The `k == 0 && n == 1` exemption is the empty-variable-list form: `\int f`
        // with NO DIFFERENTIAL AT ALL, which `integral(f, {})` prints and which
        // therefore has to read back.
        //
        // The capitalised half is load-bearing and went unchecked. `\int x \, dx + 1`
        // has a differential, but the backwards scan requires each unit to end at the
        // group boundary and this one ends before `+ 1`, so `found` came back 0, the
        // exemption fired, and the whole line became the integrand -- `\, dx` included,
        // which juxtaposition then read as a product. The answer was
        // `integral(d*x^2 + 1)`: a factor of `d` the author never wrote, and the `+ 1`
        // swallowed into the integral it was added to.
        if (found == 0 && !separators.empty() &&
            differential_unit_starts_at(separators.back())) {
            return fail(first, kMsgIntegralExtent);
        }
        if (found != signs && !(found == 0 && signs == 1)) {
            return fail(first, std::string("[E-LATEX-0031] ") +
                                   std::to_string(signs) + " integral signs but " +
                                   std::to_string(found) +
                                   " differentials; write one \\, dx per \\int");
        }

        const std::size_t outer = limit_;
        pos_ = body_start;
        set_limit(boundary);
        ExprRef body = parse_expression();
        if (!body) {
            return nullptr;
        }
        if (!at_end()) {
            return expected(peek(), "the end of the integrand");
        }
        set_limit(outer);
        pos_ = group_end;

        std::vector<ExprRef> symbols;
        symbols.reserve(found);
        for (std::size_t i = found; i > 0; --i) {
            symbols.push_back(symbol(reversed[i - 1]));
        }
        return integral(std::move(body), std::move(symbols));
    }

    /// The indices of every `\,` at the top level of `[from, to)`. The depth matters:
    /// a thin space inside `\frac{...}` or inside a nested group belongs to that group's
    /// product, not to this integral's differential list.
    void collect_thin_spaces(std::size_t from, std::size_t to,
                             std::vector<std::size_t>& out) const {
        int depth = 0;
        for (std::size_t i = from; i < to; ++i) {
            const Token& token = toks_[i];
            if (token.kind == TokKind::Char && token.text.size() == 1) {
                const char c = token.text.front();
                if (c == '{' || c == '(' || c == '[') {
                    ++depth;
                } else if (c == '}' || c == ')' || c == ']') {
                    --depth;
                }
                continue;
            }
            if (lex::is_word(token, "left") || lex::is_word(token, "lvert")) {
                ++depth;
            } else if (lex::is_word(token, "right") || lex::is_word(token, "rvert")) {
                --depth;
            } else if (lex::is_symbol(token, ',') && depth == 0) {
                out.push_back(i);
            }
        }
    }

    /// Whether `[at, boundary)` is exactly one `\, DiffD Variable`. It has to end
    /// *exactly* at the boundary, which is what makes the walk maximal from the right
    /// and what stops `\, d` with nothing after it from being taken for a differential.
    /// Whether a `\,` at `at` begins a `d<name>` unit, without requiring it to reach any
    /// boundary. `match_differential_unit` answers the stricter question the backwards
    /// scan needs; this one is how the caller tells "there is no differential here" from
    /// "there is one and it is not at the end", which are the two cases the
    /// empty-variable-list exemption must not confuse.
    bool differential_unit_starts_at(std::size_t at) const {
        if (at + 1 >= limit_) {
            return false;
        }
        std::size_t after_d = 0;
        if (!differential_d_at(at + 1, after_d)) {
            return false;
        }
        return after_d < limit_ &&
               (toks_[after_d].kind == TokKind::Char ||
                toks_[after_d].kind == TokKind::ControlWord);
    }

    bool match_differential_unit(std::size_t at, std::size_t boundary, std::string& var) {
        if (at >= boundary) {
            return false;
        }
        const std::size_t saved_pos = pos_;
        pos_ = at;
        advance();
        bool matched = match_differential_d() && speculative_symbol_name(var) &&
                       pos_ == boundary;
        pos_ = saved_pos;
        return matched;
    }

    /// Whether the superscript starting at `at` is a bare `+` or `-`, braced or not.
    /// Neither is an expression, so this costs nothing that was otherwise readable.
    bool one_sided_marker_at(std::size_t at, std::size_t close) const {
        if (at >= close) {
            return false;
        }
        std::size_t sign = at;
        if (lex::is_char(toks_[at], '{')) {
            sign = at + 1;
            if (sign + 1 >= close || !lex::is_char(toks_[sign + 1], '}')) {
                return false;
            }
        }
        return sign < close &&
               (lex::is_char(toks_[sign], '+') || lex::is_char(toks_[sign], '-'));
    }

    /// `\lim_{x \to 0} f`. The body is unfenced and greedy: `\lim_{x \to 0} x + y` is
    /// `limit(x+y, x, 0)`, because the walker places a Limit at `kPrecAdd` and would
    /// have parenthesised it had the sum not been part of it.
    ExprRef parse_limit() {
        advance();
        if (lex::is_word(peek(), "limits")) {
            advance(); // H6
        }
        if (!lex::is_char(peek(), '_')) {
            return expected(peek(), "'_' after \\lim");
        }
        advance();
        if (!lex::is_char(peek(), '{')) {
            return expected(peek(), "'{' after \\lim_");
        }
        const Token opener = advance();
        const std::size_t close = find_brace_end(pos_);
        if (close >= limit_) {
            return unclosed(boundary_, opener, "'}'");
        }
        // A32: a one-sided limit is spelled with a sign in the superscript on the point,
        // and `Head::Limit` has nowhere to put the side, so it is refused rather than
        // widened into the two-sided limit it is not. It is the *sign* that marks the
        // side: `\lim_{y \to \infty^{2}}` carries an ordinary power in its point and is
        // a string the printer emits, so rejecting every superscript here would reject
        // the printer's own output.
        for (std::size_t i = pos_; i < close; ++i) {
            if (lex::is_char(toks_[i], '^') && one_sided_marker_at(i + 1, close)) {
                return fail(toks_[i], kMsgLimitDirection);
            }
        }
        std::string var;
        if (!parse_symbol_name(var)) {
            if (failed_) {
                return nullptr;
            }
            return expected(peek(), "the limit variable");
        }
        if (!lex::is_word(peek(), "to") && !lex::is_word(peek(), "rightarrow") &&
            !lex::is_word(peek(), "longrightarrow")) {
            return expected(peek(), "'\\to'");
        }
        advance();
        const std::size_t outer = limit_;
        set_limit(close);
        ExprRef point = parse_expression();
        if (!point) {
            return nullptr;
        }
        if (!at_end()) {
            return expected(peek(), "'}'");
        }
        set_limit(outer);
        pos_ = close + 1;
        ExprRef body = parse_expression();
        if (!body) {
            return nullptr;
        }
        return limit(std::move(body), symbol(var), std::move(point));
    }

    // --- Matrix ---------------------------------------------------------------------

    /// §2.7's `Cell`: a number, possibly signed, possibly in scientific notation,
    /// possibly an infinity or a NaN. Not a float regex -- `1e8` is printed
    /// `1 \times 10^{8}` and `0.5` under `decimal_separator = ','` is printed `0{,}5`,
    /// so the cell goes through the same numeral rules the expression parser uses.
    bool parse_matrix_cell(double& out) {
        bool negative = false;
        if (at_minus()) {
            advance();
            negative = true;
        }
        if (lex::is_word(peek(), "infty") ||
            (peek().kind == TokKind::Char && peek().code == 0x221EU)) {
            advance();
            out = negative ? -std::numeric_limits<double>::infinity()
                           : std::numeric_limits<double>::infinity();
            return true;
        }
        if (lex::is_word(peek(), "mathrm") && lex::is_char(peek(1), '{') &&
            lex::is_char(peek(2), 'N') && lex::is_char(peek(3), 'a') &&
            lex::is_char(peek(4), 'N') && lex::is_char(peek(5), '}')) {
            pos_ += 6;
            out = std::numeric_limits<double>::quiet_NaN();
            return true;
        }
        if (!lex::is_digit_token(peek())) {
            expected(peek(), "a number in the matrix cell");
            return false;
        }
        ExprRef value = parse_numeral();
        if (!value) {
            return false;
        }
        double magnitude = 0.0;
        if (!as_double(value, magnitude)) {
            fail(peek(), std::string(kMsgNumeral) + "the matrix cell is not a number");
            return false;
        }
        out = negative ? -magnitude : magnitude;
        return true;
    }

    // --- Entry points ---------------------------------------------------------------

    /// H9: one outer `$...$`, `$$...$$`, `\(...\)` or `\[...\]`, and only when the
    /// closer is the last token -- otherwise a `$` in the middle of the input would
    /// take the first and last tokens off an expression that never had a wrapper.
    void strip_math_delimiters() {
        if (limit_ == 0) {
            return;
        }
        std::size_t width = 0;
        if (lex::is_char(toks_[0], '$')) {
            width = (limit_ >= 2 && lex::is_char(toks_[1], '$')) ? 2 : 1;
            if (limit_ < 2 * width) {
                return;
            }
            for (std::size_t i = 0; i < width; ++i) {
                if (!lex::is_char(toks_[limit_ - 1 - i], '$')) {
                    return;
                }
            }
        } else if (lex::is_symbol(toks_[0], '(') || lex::is_symbol(toks_[0], '[')) {
            const char closer = lex::is_symbol(toks_[0], '(') ? ')' : ']';
            if (limit_ < 2 || !lex::is_symbol(toks_[limit_ - 1], closer)) {
                return;
            }
            width = 1;
        } else {
            return;
        }
        pos_ = width;
        set_limit(limit_ - width);
    }

    std::string_view text_;
    std::vector<Token> toks_;
    char decimal_ = '.';
    std::size_t pos_ = 0;
    std::size_t limit_ = 0;
    Token boundary_{};
    int depth_ = 0;
    bool failed_ = false;
    /// Whether an unbraced `|` is already open, which is what makes a second one a
    /// diagnostic rather than a nested absolute value (A36).
    bool in_bare_bar_ = false;
    /// How many unbraced bar groups the term being parsed has already closed.
    int bars_in_term_ = 0;
    ParseError error_{0, 0, {}};
};

Result<ExprRef> LatexParser::run_expression() {
    strip_math_delimiters();
    if (at_end()) {
        return std::unexpected(Error{ParseError{peek().line, peek().col, kMsgEmptyInput}});
    }
    ExprRef result = parse_expression();
    if (!result) {
        return std::unexpected(Error{std::move(error_)});
    }
    if (!at_end()) {
        if (!stray_rejection(peek())) {
            fail(peek(), std::string(kMsgTrailing) + "the expression ends here; " +
                             lex::describe(peek()) + " follows it");
        }
        return std::unexpected(Error{std::move(error_)});
    }
    return result;
}

Result<Matrix<double>> LatexParser::run_matrix() {
    strip_math_delimiters();
    if (at_end()) {
        return std::unexpected(Error{ParseError{peek().line, peek().col, kMsgEmptyInput}});
    }
    if (!lex::is_word(peek(), "begin")) {
        return std::unexpected(Error{ParseError{
            peek().line, peek().col,
            std::string(kMsgExpectedExpression) + "expected '\\begin' of a matrix "
            "environment, found " + lex::describe(peek())}});
    }
    const Token opener = advance();
    std::string environment;
    read_environment_name(pos_, environment);
    if (lookup(kMatrixEnvironments, environment) == nullptr) {
        // The name is what is wrong, not the `\begin`, and the name is where the reader
        // has to type. `\begin` is in the subset -- `\begin{pmatrix}` is the whole point
        // of this entry point -- so pointing at it would name the one token that is fine.
        const Token& name_at = peek(1).kind == TokKind::End ? opener : peek(1);
        return std::unexpected(Error{ParseError{name_at.line, name_at.col, kMsgEnvironment}});
    }
    skip_environment_name();

    std::vector<std::vector<double>> rows;
    if (!lex::is_word(peek(), "end")) {
        std::vector<double> row;
        while (true) {
            double cell = 0.0;
            if (!parse_matrix_cell(cell)) {
                return std::unexpected(Error{std::move(error_)});
            }
            row.push_back(cell);
            if (lex::is_char(peek(), '&')) {
                advance();
                continue;
            }
            if (lex::is_symbol(peek(), '\\')) {
                advance();
                rows.push_back(std::move(row));
                row.clear();
                continue;
            }
            break;
        }
        rows.push_back(std::move(row));
    }

    if (!lex::is_word(peek(), "end")) {
        return std::unexpected(Error{ParseError{
            peek().line, peek().col,
            std::string(kMsgDelimiter) + "expected '\\end{" + environment +
                "}' to close '\\begin{" + environment + "}' opened at line " +
                std::to_string(opener.line) + ", column " + std::to_string(opener.col) +
                ", found " + lex::describe(peek())}});
    }
    const Token closer = advance();
    std::string closing;
    read_environment_name(pos_, closing);
    if (closing != environment) {
        // §5 rule 3: the position is the closer and the message carries the opener.
        return std::unexpected(Error{ParseError{
            closer.line, closer.col,
            std::string(kMsgDelimiter) + "'\\end{" + closing + "}' closes '\\begin{" +
                environment + "}' opened at line " + std::to_string(opener.line) +
                ", column " + std::to_string(opener.col)}});
    }
    skip_environment_name();
    if (!at_end()) {
        return std::unexpected(Error{ParseError{
            peek().line, peek().col,
            std::string(kMsgTrailing) + "the matrix ends here; " +
                lex::describe(peek()) + " follows it"}});
    }

    const std::size_t columns = rows.empty() ? 0 : rows.front().size();
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].size() != columns) {
            return std::unexpected(Error{ParseError{
                opener.line, opener.col,
                std::string(kMsgRagged) + "the matrix is not rectangular: row 1 has " +
                    std::to_string(columns) + " cells and row " + std::to_string(i + 1) +
                    " has " + std::to_string(rows[i].size())}});
        }
    }
    Matrix<double> out(rows.size(), columns);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (std::size_t j = 0; j < columns; ++j) {
            out(i, j) = rows[i][j];
        }
    }
    return out;
}

} // namespace

Result<ExprRef> parse_latex(std::string_view text, const NotationOptions& options) {
    LatexParser parser(text, options);
    return parser.run_expression();
}

Result<Matrix<double>> parse_latex_matrix(std::string_view text,
                                          const NotationOptions& options) {
    LatexParser parser(text, options);
    return parser.run_matrix();
}

} // namespace ms::sym2
