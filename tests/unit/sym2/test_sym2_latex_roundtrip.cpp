// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §11.2's defining property, over generated expressions crossed with every option set:
//
//     parse_latex(to_latex(e, options), options) == e
//
// `docs/LATEX_SUBSET.md` does not describe the subset as a taste; it defines it as the
// image of the printer, which is what turns the sentence above from an aspiration into
// an assertion. Everything the printer can emit from a well-formed node must read back
// as that node, under every `NotationOptions` combination, because the options change
// the spelling and are documented not to change what is denoted.
//
// The generator is therefore constrained by §0 (well-formedness) and by §4.2 (the
// complete list of shapes where the printed form carries less than the node did). Both
// constraints are enforced by `round_trippable` below rather than by hoping the random
// draw misses them, and every §4.2 row has its own test further down that pins what
// *does* come back. An exception list nobody tests is an exception list that grows:
// once a row is only a sentence in a document, the next person to widen it has nothing
// to argue with.
//
// This file is written against the document and the header. It is deliberately not
// written against the parser, which is being implemented alongside it -- a test read
// off an implementation agrees with that implementation's bugs, and agreeing with a
// bug is indistinguishable, in a green test run, from being right.
//
// Where the document and `src/sym2/notation_latex.cpp` disagree, the code wins and the
// disagreement is named at the assertion that records it (N1 and N13 below).

#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "ms/core/format.hpp"
#include "ms/core/matrix.hpp"
#include "ms/sym2/expr.hpp"
#include "ms/sym2/latex_parse.hpp"
#include "ms/sym2/notation.hpp"

using ms::bignum::BigInt;
using namespace ms;
using namespace ms::sym2;

namespace {

constexpr int kTrials = 160;
constexpr std::uint64_t kSeedRoot = 0xD1B54A32D192ED03ull;
constexpr int kHeadCount = 12;

constexpr double kInfinity = std::numeric_limits<double>::infinity();

/// Written through `numeric_limits` rather than as `0.0 / 0.0`: the latter is a
/// constant expression and MSVC rejects it outright, where GCC folds it to a quiet NaN.
/// The existing printer tests record the same Windows break.
const double kNotANumber = std::numeric_limits<double>::quiet_NaN();

// --- Reporting -----------------------------------------------------------------------

std::string describe(const ms::Error& error) {
    if (const ms::ParseError* parse = std::get_if<ms::ParseError>(&error)) {
        return "latex:" + std::to_string(parse->line) + ":" + std::to_string(parse->col) +
               ": " + parse->msg;
    }
    return ms::format_error(error);
}

std::string describe(const Result<ExprRef>& result) {
    return result.has_value() ? to_string(*result) : describe(result.error());
}

/// The expression `text` denotes, asserted to parse at all. Used by the §4.2 tests,
/// which are about *what* comes back rather than about whether anything does.
ExprRef reparse(const std::string& text, const NotationOptions& options = {}) {
    const Result<ExprRef> back = parse_latex(text, options);
    if (!back.has_value()) {
        ADD_FAILURE() << "parse_latex(\"" << text << "\") failed: "
                      << describe(back.error());
        return undefined();
    }
    return *back;
}

// --- What §4.2 excludes ---------------------------------------------------------------

bool is_integer_value(const ExprRef& e, long long value) {
    return structurally_equal(e, integer(value));
}

bool is_infinite_constant(const ExprRef& e) {
    return structurally_equal(e, constant("inf")) || structurally_equal(e, constant("-inf"));
}

/// Whether `e` is outside every row of §4.2, so that the central property must hold for
/// it. Applied to the *constructed* node rather than to the generator's intent, because
/// the builders fold: `pow(real(0.25), integer(-1))` is `real(4.0)`, which is N1, and
/// nothing about the call that produced it says so.
bool round_trippable(const ExprRef& e) {
    switch (head_of(e)) {
    case Head::Real: {
        double value = 0.0;
        if (!as_double(e, value)) {
            return false;
        }
        // N1 through N5. A Real survives the trip only when its printed digits carry a
        // marker that it is one: LaTeX has no `Head::Real`, so bare digits come back as
        // an Integer and the non-finite values come back as the Constants they are
        // spelled with. §4.2 N1 gives the condition as "whole-valued below 1e6", which
        // is close but not the rule the printer follows -- `format_exact(1234567.0)` is
        // also bare digits, because the precision loop stops as soon as the value reads
        // back. The condition the code implements is the one used here.
        if (!std::isfinite(value)) {
            return false;
        }
        const std::string digits = ms::format_exact(value);
        return digits.find('.') != std::string::npos || digits.find('e') != std::string::npos;
    }
    case Head::Function: {
        const std::string* name = std::get_if<std::string>(&e->atom);
        if (name == nullptr || name->empty()) {
            return false; // N17, and §0 well-formedness.
        }
        if (*name == "sqrt" && e->args.size() == 1) {
            return false; // N16: `\sqrt{a}` is the spelling of `pow(a, 1/2)`.
        }
        break;
    }
    case Head::Mul: {
        // N6: a factor of -1 beside an infinity prints as `-\infty`, which ruling A10
        // reads as the Constant. The three-factor form (`-\infty \cdot x`) is the same
        // loss and is excluded here too, though §4.2 spells only the two-factor one.
        bool minus_one = false;
        bool infinite = false;
        for (const ExprRef& factor : e->args) {
            minus_one = minus_one || is_integer_value(factor, -1);
            infinite = infinite || is_infinite_constant(factor);
        }
        if (minus_one && infinite) {
            return false;
        }
        break;
    }
    case Head::Derivative:
        // N18: an empty variable list leaves no trace of the head in the output.
        // N19: a nested Derivative is spelled exactly like the flat one.
        if (e->args.size() < 2 || head_of(e->args[0]) == Head::Derivative) {
            return false;
        }
        break;
    case Head::Integral:
        // N20, as N19. The empty variable list is fine here -- `\int f` keeps its sign.
        if (head_of(e->args[0]) == Head::Integral) {
            return false;
        }
        break;
    default:
        break;
    }
    // §0 requires the variables of a Derivative, an Integral and a Limit to be Symbols,
    // because the walker places them in an unfenced slot (N25).
    if (head_of(e) == Head::Derivative || head_of(e) == Head::Integral) {
        for (std::size_t i = 1; i < e->args.size(); ++i) {
            if (head_of(e->args[i]) != Head::Symbol) {
                return false;
            }
        }
    }
    if (head_of(e) == Head::Limit && (e->args.size() < 2 || head_of(e->args[1]) != Head::Symbol)) {
        return false;
    }
    for (const ExprRef& arg : e->args) {
        if (!round_trippable(arg)) {
            return false;
        }
    }
    return true;
}

// --- The generator ---------------------------------------------------------------------

/// Symbol names chosen to sit inside §4.1's rule for `Head::Symbol` and to exercise the
/// parts of `spell_name` that are easy to invert wrongly: a Greek letter and its
/// capital, a named glyph, a subscript that is a name rather than an expression, a name
/// carrying LaTeX specials, a name carrying an embedded underscore, and a name carrying
/// raw UTF-8 (which `escape` passes through untouched, so the column of a later
/// diagnostic depends on counting scalars rather than bytes).
///
/// `d` is deliberately absent. Under `Multiplication::Juxtaposition` a factor named `d`
/// sitting behind an upright word inside an integrand is N23, the one case where the
/// printer emits a string that reads as a differential it did not mean.
const char* const kSymbolNames[] = {
    "x",   "y",     "z",    "t",    "n",    "foo",   "bar",
    "x_1", "x_max", "alpha", "Gamma", "Alpha", "hbar", "a%b",
    "a{b", "x_a_1", "\xc3\xa9t\xc3\xa9",
};

/// The seven of `notation_syntax.hpp`. Any other atom is spelled `\mathrm{name}`, which
/// is byte-for-byte a multi-letter Symbol (N24), so §0 excludes it.
const char* const kConstantNames[] = {"pi", "e", "i", "inf", "-inf", "nan", "undefined"};

/// Every one of these has a `.` or an `e` in `format_exact`, which is what keeps a Real
/// distinguishable from an Integer on the way back (N1). `1e6` is already
/// `1 \times 10^{6}` and `2.5e-13` exercises the exponent path with a mantissa.
const double kRealValues[] = {2.5, 0.25, -0.75, 0.1, 1.0 / 3.0, 1e20, 1e-7, 1e6, 123456.5, 2.5e-13};

/// Function names spanning the three spellings the table can produce: a `kOperators`
/// control word, an `\operatorname{}` fallback, a name that differs from a control word
/// only in case (so that `\operatorname{Sin}` and `\sin` must stay distinct), and `abs`,
/// which is written with bars instead of a call.
const char* const kFunctionNames[] = {"sin", "cos", "erf", "f", "Sin", "max", "g"};

ExprRef pick_symbol(std::mt19937_64& rng) {
    const int count = static_cast<int>(std::size(kSymbolNames));
    return symbol(kSymbolNames[std::uniform_int_distribution<int>(0, count - 1)(rng)]);
}

ExprRef generate(std::mt19937_64& rng, int depth);

ExprRef generate_atom(std::mt19937_64& rng) {
    switch (std::uniform_int_distribution<int>(0, 4)(rng)) {
    case 0:
        return integer(std::uniform_int_distribution<int>(-20, 20)(rng));
    case 1: {
        const int num = std::uniform_int_distribution<int>(-9, 9)(rng);
        const int den = std::uniform_int_distribution<int>(2, 9)(rng);
        return rational(BigInt(num), BigInt(den));
    }
    case 2: {
        const int count = static_cast<int>(std::size(kRealValues));
        return real(kRealValues[std::uniform_int_distribution<int>(0, count - 1)(rng)]);
    }
    case 3: {
        const int count = static_cast<int>(std::size(kConstantNames));
        return constant(kConstantNames[std::uniform_int_distribution<int>(0, count - 1)(rng)]);
    }
    default:
        return pick_symbol(rng);
    }
}

/// A variable for a calculus head. §0 requires a Symbol, and these four are short enough
/// that a failure message stays readable.
ExprRef generate_variable(std::mt19937_64& rng) {
    static const char* const kVariables[] = {"x", "y", "z", "t"};
    return symbol(kVariables[std::uniform_int_distribution<int>(0, 3)(rng)]);
}

std::vector<ExprRef> generate_arguments(std::mt19937_64& rng, int depth, int low, int high) {
    std::vector<ExprRef> args;
    const int count = std::uniform_int_distribution<int>(low, high)(rng);
    args.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        args.push_back(generate(rng, depth - 1));
    }
    return args;
}

ExprRef generate(std::mt19937_64& rng, int depth) {
    if (depth <= 0) {
        return generate_atom(rng);
    }
    switch (std::uniform_int_distribution<int>(0, 15)(rng)) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
        return generate_atom(rng);
    case 5:
        return add(generate_arguments(rng, depth, 2, 3));
    case 6:
        return mul(generate_arguments(rng, depth, 2, 3));
    case 7:
        // Small exponents on purpose. A base of ten under a large exact exponent is the
        // only way to reach N22, and the fold at `kMaxExactPowerExponent` means nothing
        // in between behaves differently.
        return pow(generate(rng, depth - 1),
                   integer(std::uniform_int_distribution<int>(-3, 4)(rng)));
    case 8:
        return pow(generate(rng, depth - 1), generate(rng, depth - 1));
    case 9:
        return neg(generate(rng, depth - 1));
    case 10:
        return sub(generate(rng, depth - 1), generate(rng, depth - 1));
    case 11:
        return div(generate(rng, depth - 1), generate(rng, depth - 1));
    case 12: {
        const int count = static_cast<int>(std::size(kFunctionNames));
        const std::string name =
            kFunctionNames[std::uniform_int_distribution<int>(0, count - 1)(rng)];
        // `abs` is spelled with bars and takes exactly one argument there; every other
        // arity of it goes through `\operatorname{abs}`. Zero arguments are legal and
        // are emitted, so the generator reaches them.
        const int low = name == "abs" ? 1 : 0;
        return function(name, generate_arguments(rng, depth, low, 2));
    }
    case 13: {
        // One or two variables: the second is what distinguishes the flat encoding from
        // the nested one, which is N19 and is excluded above.
        std::vector<ExprRef> args{generate(rng, depth - 1), generate_variable(rng)};
        if (std::uniform_int_distribution<int>(0, 1)(rng) == 1) {
            args.push_back(generate_variable(rng));
        }
        return derivative(args[0], {args.begin() + 1, args.end()});
    }
    case 14: {
        std::vector<ExprRef> vars;
        const int count = std::uniform_int_distribution<int>(0, 2)(rng);
        for (int i = 0; i < count; ++i) {
            vars.push_back(generate_variable(rng));
        }
        // The empty list is included deliberately: `\int f` with no differential is one
        // of the shapes §4.2's closing paragraph says does *not* belong on the list.
        return integral(generate(rng, depth - 1), vars);
    }
    default:
        return limit(generate(rng, depth - 1), generate_variable(rng), generate(rng, depth - 1));
    }
}

/// Draw until the draw is inside the property's domain. A rejection loop rather than a
/// generator that cannot produce an excluded shape, because the builders fold and the
/// shape that must be excluded is the one that was *built*, not the one that was asked
/// for.
ExprRef generate_round_trippable(std::mt19937_64& rng, int depth) {
    for (int attempt = 0; attempt < 200; ++attempt) {
        const ExprRef e = generate(rng, depth);
        if (round_trippable(e)) {
            return e;
        }
    }
    ADD_FAILURE() << "the generator could not produce an expression inside the property's "
                     "domain in 200 attempts -- the §4.2 exclusions have swallowed it";
    return symbol("x");
}

void mark_heads(const ExprRef& e, bool (&seen)[kHeadCount]) {
    seen[static_cast<int>(head_of(e))] = true;
    for (const ExprRef& arg : e->args) {
        mark_heads(arg, seen);
    }
}

const char* head_name(int head) {
    switch (static_cast<Head>(head)) {
    case Head::Integer: return "Integer";
    case Head::Rational: return "Rational";
    case Head::Real: return "Real";
    case Head::Symbol: return "Symbol";
    case Head::Constant: return "Constant";
    case Head::Add: return "Add";
    case Head::Mul: return "Mul";
    case Head::Pow: return "Pow";
    case Head::Function: return "Function";
    case Head::Derivative: return "Derivative";
    case Head::Integral: return "Integral";
    case Head::Limit: return "Limit";
    }
    return "?";
}

// --- The option cross product -----------------------------------------------------------

/// Every combination §11.2 must be independent of: `display` (`\frac` versus `\dfrac`),
/// `sized_delimiters` (`(` versus `\left(`), all three `Multiplication` spellings, both
/// legal decimal separators, and `roots_as_radicals` (`\sqrt{x}` versus
/// `x^{\frac{1}{2}}`). `matrix_environment` does not appear because no expression
/// contains a matrix; it is crossed in the matrix tests instead.
std::vector<NotationOptions> option_settings() {
    std::vector<NotationOptions> settings;
    for (const bool display : {false, true}) {
        for (const bool sized : {false, true}) {
            for (const NotationOptions::Multiplication multiplication :
                 {NotationOptions::Multiplication::Juxtaposition,
                  NotationOptions::Multiplication::Dot,
                  NotationOptions::Multiplication::Cross}) {
                for (const char separator : {'.', ','}) {
                    for (const bool radicals : {false, true}) {
                        NotationOptions options;
                        options.display = display;
                        options.sized_delimiters = sized;
                        options.multiplication = multiplication;
                        options.decimal_separator = separator;
                        options.roots_as_radicals = radicals;
                        settings.push_back(options);
                    }
                }
            }
        }
    }
    return settings;
}

std::string describe(const NotationOptions& options) {
    const char* multiplication = "juxtaposition";
    if (options.multiplication == NotationOptions::Multiplication::Dot) {
        multiplication = "dot";
    } else if (options.multiplication == NotationOptions::Multiplication::Cross) {
        multiplication = "cross";
    }
    return std::string("display=") + (options.display ? "1" : "0") + " sized=" +
           (options.sized_delimiters ? "1" : "0") + " mul=" + multiplication +
           " separator='" + options.decimal_separator + "' radicals=" +
           (options.roots_as_radicals ? "1" : "0");
}

// --- The property ------------------------------------------------------------------------

TEST(Sym2LatexRoundTrip, EveryPrintedFormReadsBackAsTheExpressionThatWasPrinted) {
    const std::vector<NotationOptions> settings = option_settings();
    ASSERT_EQ(settings.size(), 48U) << "the option cross product lost a dimension";

    bool head_seen[kHeadCount] = {};
    int checked = 0;
    for (int trial = 0; trial < kTrials; ++trial) {
        std::mt19937_64 rng{kSeedRoot + static_cast<std::uint64_t>(trial)};
        const ExprRef e = generate_round_trippable(rng, 3);
        mark_heads(e, head_seen);
        for (const NotationOptions& options : settings) {
            const std::string text = to_latex(e, options);
            ++checked;
            const Result<ExprRef> back = parse_latex(text, options);
            if (!back.has_value()) {
                ADD_FAILURE() << "printed as " << text << "\n  which does not parse: "
                              << describe(back.error()) << "\n  original: " << to_string(e)
                              << "\n  options: " << describe(options);
                continue;
            }
            EXPECT_TRUE(structurally_equal(*back, e))
                << "printed as " << text << "\n  which reads back as " << to_string(*back)
                << "\n  but denotes " << to_string(e) << "\n  options: " << describe(options);
        }
    }

    // Two guards against a vacuous pass. A property test whose generator quietly stops
    // producing anything goes green forever, and the green is indistinguishable from
    // the green it gave when it was working.
    EXPECT_EQ(checked, kTrials * static_cast<int>(settings.size()))
        << "the property did not run the intended number of times";
    for (int head = 0; head < kHeadCount; ++head) {
        EXPECT_TRUE(head_seen[head])
            << "the generator never produced a " << head_name(head)
            << " node, so nothing here asserts anything about that head";
    }
}

// A matrix is a separate entry point (§2.7), so it gets the property separately. The
// cells are `double`, so this is `to_latex(const Matrix<double>&)` rather than `walk()`.
TEST(Sym2LatexRoundTrip, EveryPrintedMatrixReadsBackAsTheMatrixThatWasPrinted) {
    const char* const kEnvironments[] = {"pmatrix", "bmatrix", "vmatrix", "Vmatrix", "matrix"};
    int checked = 0;
    for (int trial = 0; trial < 40; ++trial) {
        std::mt19937_64 rng{kSeedRoot + 0x1000ull + static_cast<std::uint64_t>(trial)};
        const std::size_t rows = std::uniform_int_distribution<std::size_t>(1, 4)(rng);
        const std::size_t cols = std::uniform_int_distribution<std::size_t>(1, 4)(rng);
        ms::Matrix<double> m(rows, cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                const int count = static_cast<int>(std::size(kRealValues));
                m(i, j) = kRealValues[std::uniform_int_distribution<int>(0, count - 1)(rng)];
            }
        }
        for (const char* environment : kEnvironments) {
            for (const char separator : {'.', ','}) {
                NotationOptions options;
                options.matrix_environment = environment;
                options.decimal_separator = separator;
                const std::string text = to_latex(m, options);
                ++checked;
                const Result<ms::Matrix<double>> back = parse_latex_matrix(text, options);
                if (!back.has_value()) {
                    ADD_FAILURE() << "printed as " << text
                                  << "\n  which does not parse: " << describe(back.error());
                    continue;
                }
                ASSERT_EQ(back->rows(), rows) << text;
                ASSERT_EQ(back->cols(), cols) << text;
                for (std::size_t i = 0; i < rows; ++i) {
                    for (std::size_t j = 0; j < cols; ++j) {
                        EXPECT_EQ((*back)(i, j), m(i, j))
                            << "cell " << i << "," << j << " of " << text;
                    }
                }
            }
        }
    }
    EXPECT_EQ(checked, 40 * 5 * 2) << "the matrix property did not run as intended";
}

// --- §4.2, one test per row ----------------------------------------------------------------
//
// Each of these is a case where the printed form genuinely carries less than the node
// did. The assertion is not "this fails" -- it is what the string actually denotes when
// it is read honestly, which is the only thing a reader of the output could have
// concluded. Writing them down is what keeps the list from growing by accident.

TEST(Sym2LatexException, N1_AWholeValuedRealComesBackAnInteger) {
    // LaTeX has no marker for `Head::Real`, and `format_exact` drops a fractional part
    // that is not there. The two heads are not interchangeable downstream: `is_exact`,
    // `is_zero` and Mul's zero-annihilation all treat them differently.
    //
    // §4.2 gives the condition as "whole-valued |v| < 1e6". That is the common case but
    // not the rule: `format_exact` raises its precision only until the value reads back,
    // so `1234567.0` is also spelled as bare digits. The code is what this asserts.
    for (const double value : {1.0, 2.0, 123456.0, 1234567.0}) {
        const ExprRef e = real(value);
        const std::string text = to_latex(e);
        EXPECT_EQ(text.find('.'), std::string::npos)
            << "expected bare digits for " << value << ", got " << text;
        const ExprRef back = reparse(text);
        EXPECT_EQ(head_of(back), Head::Integer) << text;
        EXPECT_FALSE(structurally_equal(back, e)) << text;
    }
}

TEST(Sym2LatexException, N2_NegativeZeroLosesItsSignAsWellAsItsHead) {
    const ExprRef e = real(-0.0);
    EXPECT_EQ(to_latex(e), "-0");
    const ExprRef back = reparse("-0");
    EXPECT_EQ(head_of(back), Head::Integer);
    EXPECT_TRUE(is_zero(back)) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
    // The second half of the loss, recorded but not asserted as a parse: `-0.0 < 0.0` is
    // false, so the walker reports the value atomic and it reaches a superscript bare.
    // What LaTeX then sets is `-(0^n)`, which is not the base that was printed.
    EXPECT_EQ(to_latex(pow(e, symbol("n"))), "-0^{n}");
}

TEST(Sym2LatexException, N3_APositiveInfiniteRealComesBackTheInfinityConstant) {
    const ExprRef e = real(kInfinity);
    EXPECT_EQ(to_latex(e), "\\infty");
    const ExprRef back = reparse("\\infty");
    EXPECT_TRUE(structurally_equal(back, constant("inf"))) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
}

TEST(Sym2LatexException, N4_ANegativeInfiniteRealComesBackTheNegativeInfinityConstant) {
    const ExprRef e = real(-kInfinity);
    EXPECT_EQ(to_latex(e), "-\\infty");
    const ExprRef back = reparse("-\\infty");
    EXPECT_TRUE(structurally_equal(back, constant("-inf"))) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
    // The Real and the Constant differ in precedence, so they are not even spelled the
    // same under a superscript -- and both spellings still land on the one Constant,
    // which is the loss this row is about.
    EXPECT_EQ(to_latex(pow(e, symbol("n"))), "(-\\infty)^{n}");
    EXPECT_EQ(to_latex(pow(constant("-inf"), symbol("n"))), "-\\infty^{n}");
}

TEST(Sym2LatexException, N5_ANaNRealComesBackTheNaNConstant) {
    const ExprRef e = real(kNotANumber);
    EXPECT_EQ(to_latex(e), "\\mathrm{NaN}");
    const ExprRef back = reparse("\\mathrm{NaN}");
    EXPECT_TRUE(structurally_equal(back, constant("nan"))) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
}

TEST(Sym2LatexException, N6_MinusOneTimesInfinityComesBackTheNegativeInfinityConstant) {
    const ExprRef e = mul({integer(-1), constant("inf")});
    ASSERT_EQ(head_of(e), Head::Mul) << "the builders would have made this row unreachable";
    EXPECT_EQ(to_latex(e), "-\\infty");
    const ExprRef back = reparse("-\\infty");
    EXPECT_TRUE(structurally_equal(back, constant("-inf"))) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
}

TEST(Sym2LatexException, N7_ASymbolNamedPiComesBackTheConstant) {
    // `greek_letter` maps the Symbol to `\pi`, which is also how the Constant is
    // spelled, so the string carries no way to tell a variable called pi from the number.
    EXPECT_EQ(to_latex(symbol("pi")), "\\pi");
    const ExprRef back = reparse("\\pi");
    EXPECT_TRUE(structurally_equal(back, constant("pi"))) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, symbol("pi")));
}

TEST(Sym2LatexException, N8_SymbolsNamedEAndIComeBackTheConstants) {
    // A one-letter name prints bare, and so does each of these two Constants.
    EXPECT_EQ(to_latex(symbol("e")), "e");
    EXPECT_EQ(to_latex(symbol("i")), "i");
    EXPECT_TRUE(structurally_equal(reparse("e"), constant("e")));
    EXPECT_TRUE(structurally_equal(reparse("i"), constant("i")));
    EXPECT_FALSE(structurally_equal(reparse("e"), symbol("e")));
}

TEST(Sym2LatexException, N9_ASymbolNamedInftyComesBackTheInfinityConstant) {
    // `kNamedGlyphs` gives the Symbol the same control word the Constant uses.
    EXPECT_EQ(to_latex(symbol("infty")), "\\infty");
    EXPECT_TRUE(structurally_equal(reparse("\\infty"), constant("inf")));
    EXPECT_FALSE(structurally_equal(reparse("\\infty"), symbol("infty")));
}

TEST(Sym2LatexException, N10_SymbolsNamedNaNAndUndefinedComeBackTheConstants) {
    // The multi-letter fallback `\mathrm{...}` collides with the Constant spellings.
    EXPECT_EQ(to_latex(symbol("NaN")), "\\mathrm{NaN}");
    EXPECT_EQ(to_latex(symbol("undefined")), "\\mathrm{undefined}");
    EXPECT_TRUE(structurally_equal(reparse("\\mathrm{NaN}"), constant("nan")));
    EXPECT_TRUE(structurally_equal(reparse("\\mathrm{undefined}"), constant("undefined")));
    EXPECT_FALSE(structurally_equal(reparse("\\mathrm{NaN}"), symbol("NaN")));
}

TEST(Sym2LatexException, N11_AnAllDigitSymbolComesBackAnInteger) {
    // `all_digits` short-circuits the `\mathrm{}` wrapper -- a convenience meant for a
    // subscript, applied to the base as well.
    EXPECT_EQ(to_latex(symbol("12")), "12");
    EXPECT_TRUE(structurally_equal(reparse("12"), integer(12)));
    EXPECT_FALSE(structurally_equal(reparse("12"), symbol("12")));
}

TEST(Sym2LatexException, N12_AnEmptySymbolNameVanishesFromTheOutput) {
    // §0 excludes this node; the row exists because the printer does not. `spell_name`
    // returns nothing for an empty name and the atom leaves no mark at all, so what the
    // parser is handed is an empty document.
    EXPECT_EQ(to_latex(symbol("")), "");
    const Result<ExprRef> back = parse_latex("");
    ASSERT_FALSE(back.has_value()) << "an empty input denotes no expression, and guessing "
                                      "one is the failure mode the subset exists to avoid";
    const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
    ASSERT_NE(error, nullptr) << describe(back.error());
    EXPECT_EQ(error->line, 1U);
    EXPECT_EQ(error->col, 1U);
    EXPECT_NE(error->msg.find("E-LATEX-0001"), std::string::npos) << error->msg;
}

TEST(Sym2LatexRoundTrip, AOneCharacterWhitespaceNameIsFencedRatherThanSetBare) {
    // §4.1 promises a round trip for every Symbol whose base is non-empty and not
    // all-digits. `symbol(" ")` is one of those, and it used to be a counterexample that
    // §4.2 did not list: a one-character name was set bare, so the whole printed form
    // was a single space -- which is not a name in math mode, it is EMPTY INPUT, and it
    // came back as E-LATEX-0001 rather than as the symbol.
    //
    // The fence is in the printer rather than in a new exception row, because the
    // guarantee as §4.1 words it is the one worth having.
    for (const char* name : {" ", "\t", "\n", "\r"}) {
        SCOPED_TRACE(name);
        const ExprRef atom = symbol(name);
        const std::string printed = to_latex(atom);
        EXPECT_NE(printed, std::string(name))
            << "set bare, so what reaches the parser is whitespace: [" << printed << "]";
        const Result<ExprRef> back = parse_latex(printed);
        ASSERT_TRUE(back.has_value())
            << "the printer's own output was rejected: [" << printed << "] -- "
            << describe(back.error());
        EXPECT_TRUE(structurally_equal(*back, atom))
            << "came back as " << to_string(*back) << " from [" << printed << "]";
    }
    // A name that is only partly whitespace was never at risk -- it is more than one
    // character, so it was already fenced -- but it is the neighbouring case.
    EXPECT_TRUE(structurally_equal(reparse(to_latex(symbol("a b"))), symbol("a b")));
}

TEST(Sym2LatexException, N13_ANonCanonicalGreekSpellingComesBackCanonical) {
    // `greek_letter` lowercases the whole name and keys the capital on the first
    // character alone, so four spellings collapse onto two.
    //
    // §4.2 N13 says `ALPHA` prints `\alpha` and returns `symbol("alpha")`. It does not:
    // the capital is decided by `std::isupper(base[0])` (notation.cpp:383), so `ALPHA`
    // is the capital and prints `\mathrm{A}`. The code is what this asserts, and the
    // document row is wrong on that one cell.
    EXPECT_EQ(to_latex(symbol("ALPHA")), "\\mathrm{A}");
    EXPECT_EQ(to_latex(symbol("aLPHA")), "\\alpha");
    EXPECT_EQ(to_latex(symbol("GaMmA")), "\\Gamma");
    EXPECT_EQ(to_latex(symbol("GAMMA")), "\\Gamma");
    EXPECT_TRUE(structurally_equal(reparse("\\mathrm{A}"), symbol("Alpha")));
    EXPECT_TRUE(structurally_equal(reparse("\\alpha"), symbol("alpha")));
    EXPECT_TRUE(structurally_equal(reparse("\\Gamma"), symbol("Gamma")));
    EXPECT_FALSE(structurally_equal(reparse("\\alpha"), symbol("aLPHA")));
    EXPECT_FALSE(structurally_equal(reparse("\\Gamma"), symbol("GAMMA")));
}

TEST(Sym2LatexException, N14_ACapitalisedVarFormComesBackItsPlainCapital) {
    // `\varepsilon` and `\epsilon` are two shapes of one letter and share the one
    // capital, so the six capitalised `var` names have no spelling of their own.
    const struct {
        const char* written;
        const char* printed;
        const char* recovered;
    } kCases[] = {
        {"Varepsilon", "\\mathrm{E}", "Epsilon"}, {"Vartheta", "\\Theta", "Theta"},
        {"Varpi", "\\Pi", "Pi"},                  {"Varrho", "\\mathrm{P}", "Rho"},
        {"Varsigma", "\\Sigma", "Sigma"},         {"Varphi", "\\Phi", "Phi"},
    };
    for (const auto& entry : kCases) {
        EXPECT_EQ(to_latex(symbol(entry.written)), entry.printed) << entry.written;
        const ExprRef back = reparse(entry.printed);
        EXPECT_TRUE(structurally_equal(back, symbol(entry.recovered)))
            << entry.written << " came back as " << to_string(back);
        EXPECT_FALSE(structurally_equal(back, symbol(entry.written))) << entry.written;
    }
}

TEST(Sym2LatexException, N15_TheSameCollapseHappensInsideASubscript) {
    // A subscript goes through `spell_name` too, so N13 and N14 reach it unchanged.
    EXPECT_EQ(to_latex(symbol("x_VARTHETA")), "x_{\\Theta}");
    const ExprRef back = reparse("x_{\\Theta}");
    EXPECT_TRUE(structurally_equal(back, symbol("x_Theta"))) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, symbol("x_VARTHETA")));
}

TEST(Sym2LatexException, N16_SqrtOfOneArgumentComesBackAHalfPower) {
    // `\sqrt{a}` is byte-identical whether it came from the Function or from the Pow,
    // and ruling A2 picks the Pow. Every other name and every other arity of `sqrt`
    // keeps its Function head.
    const ExprRef e = function("sqrt", {symbol("a")});
    EXPECT_EQ(to_latex(e), "\\sqrt{a}");
    const ExprRef back = reparse("\\sqrt{a}");
    EXPECT_TRUE(structurally_equal(back, pow(symbol("a"), rational(BigInt(1), BigInt(2)))))
        << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
    // Two arguments is a different node and is not spelled with a radical at all.
    EXPECT_EQ(to_latex(function("sqrt", {symbol("a"), symbol("b")})),
              "\\operatorname{sqrt}(a, b)");
}

TEST(Sym2LatexException, N16b_SqrtOfOneArgumentUnderAPowerLosesTheRootAsWell) {
    // N16 turns the base into `pow(a, 1/2)`, and then `(b^p)^n` folds. Two losses in
    // series leave nothing of either: what comes back is the radicand alone.
    const ExprRef e = pow(function("sqrt", {symbol("a")}), integer(2));
    ASSERT_EQ(head_of(e), Head::Pow) << "the builders would have made this row unreachable";
    EXPECT_EQ(to_latex(e), "\\sqrt{a}^{2}");
    const ExprRef back = reparse("\\sqrt{a}^{2}");
    EXPECT_TRUE(structurally_equal(back, symbol("a"))) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
}

TEST(Sym2LatexException, N17_AnEmptyFunctionNameIsRejectedRatherThanInvented) {
    // §0 excludes the node. The printer spells it anyway, so the string exists and the
    // parser has to have an answer for it: there is no name in `\operatorname{}` to
    // recover, and inventing one is the guess this subset refuses to make.
    const ExprRef e = function("", {symbol("x")});
    EXPECT_EQ(to_latex(e), "\\operatorname{}(x)");
    const Result<ExprRef> back = parse_latex("\\operatorname{}(x)");
    ASSERT_FALSE(back.has_value()) << "read back as " << describe(back);
    const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
    ASSERT_NE(error, nullptr) << describe(back.error());
    EXPECT_NE(error->msg.find("E-LATEX-0039"), std::string::npos) << error->msg;
}

TEST(Sym2LatexException, N18_ADerivativeWithNoVariablesLeavesNoTrace) {
    // The printer returns the body untouched, so there is nothing in the output for a
    // parser to recover the head from.
    const ExprRef body = pow(symbol("x"), integer(2));
    const ExprRef e = derivative(body, {});
    ASSERT_EQ(head_of(e), Head::Derivative);
    EXPECT_EQ(to_latex(e), "x^{2}");
    const ExprRef back = reparse("x^{2}");
    EXPECT_TRUE(structurally_equal(back, body)) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
}

TEST(Sym2LatexException, N19_ANestedDerivativeComesBackFlat) {
    // The flat and the nested encodings differ by one space, and math mode discards
    // source whitespace -- so the two strings are the same document. The variable order
    // inverts between them as well, which is why the flat node this returns names x
    // before y.
    const ExprRef f = symbol("f");
    const ExprRef nested = derivative(derivative(f, {symbol("y")}), {symbol("x")});
    EXPECT_EQ(to_latex(nested), "\\frac{d}{dx} \\frac{d}{dy} f");
    EXPECT_EQ(to_latex(derivative(f, {symbol("x"), symbol("y")})),
              "\\frac{d}{dx}\\frac{d}{dy} f");
    const ExprRef back = reparse("\\frac{d}{dx} \\frac{d}{dy} f");
    EXPECT_TRUE(structurally_equal(back, derivative(f, {symbol("x"), symbol("y")})))
        << to_string(back);
    EXPECT_FALSE(structurally_equal(back, nested));
}

TEST(Sym2LatexException, N20_ANestedIntegralComesBackFlat) {
    // Worse than N19: the two encodings are byte-identical, so there is not even a space
    // to tell them apart.
    const ExprRef f = symbol("f");
    const ExprRef nested = integral(integral(f, {symbol("x")}), {symbol("y")});
    const ExprRef flat = integral(f, {symbol("x"), symbol("y")});
    EXPECT_EQ(to_latex(nested), "\\int \\int f \\, dx \\, dy");
    EXPECT_EQ(to_latex(nested), to_latex(flat));
    const ExprRef back = reparse("\\int \\int f \\, dx \\, dy");
    EXPECT_TRUE(structurally_equal(back, flat)) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, nested));
}

TEST(Sym2LatexException, N21_AMatrixWithNoCellsComesBackZeroByZero) {
    // The row separator is appended only when the body is non-empty, so an m x 0 matrix
    // collapses to the same empty environment a 0 x 0 one produces. There is nothing in
    // `\begin{pmatrix}\end{pmatrix}` that records how many empty rows there were.
    const ms::Matrix<double> empty(0, 0);
    const ms::Matrix<double> no_columns(3, 0);
    const ms::Matrix<double> no_rows(0, 3);
    EXPECT_EQ(to_latex(empty), "\\begin{pmatrix}\\end{pmatrix}");
    EXPECT_EQ(to_latex(no_columns), to_latex(empty));
    EXPECT_EQ(to_latex(no_rows), to_latex(empty));
    const Result<ms::Matrix<double>> back = parse_latex_matrix("\\begin{pmatrix}\\end{pmatrix}");
    ASSERT_TRUE(back.has_value()) << describe(back.error());
    EXPECT_EQ(back->rows(), 0U);
    EXPECT_EQ(back->cols(), 0U);
}

TEST(Sym2LatexException, N22_ACrossProductPastTheExactPowerCapReadsAsOneNumeral) {
    // Below `kMaxExactPowerExponent` the power folds to digits and this is unreachable;
    // past it the printer emits a numeral shape it did not mean, and ruling A11 claims
    // `c \times 10^{k}` for the numeral. `strtod` of an exponent that large is infinity.
    NotationOptions cross;
    cross.multiplication = NotationOptions::Multiplication::Cross;
    const ExprRef e = mul({integer(2), pow(integer(10), integer(5000)), symbol("x")});
    EXPECT_EQ(to_latex(e, cross), "2 \\times 10^{5000} \\times x");
    const ExprRef back = reparse("2 \\times 10^{5000} \\times x", cross);
    EXPECT_TRUE(structurally_equal(back, mul({real(kInfinity), symbol("x")})))
        << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
}

TEST(Sym2LatexException, N23_AJuxtaposedFactorNamedDInsideAnIntegrandReadsAsADifferential) {
    // The factor sort puts `d` early, and the thin space the upright word needs is the
    // same mark the differential separator uses -- so a product ending in d times x,
    // under an integral sign, is spelled exactly like an integral in x.
    NotationOptions juxtaposed;
    juxtaposed.multiplication = NotationOptions::Multiplication::Juxtaposition;
    const ExprRef e = integral(mul({symbol("aa"), symbol("d"), symbol("x")}), {});
    EXPECT_EQ(to_latex(e, juxtaposed), "\\int \\mathrm{aa}\\,dx");
    const ExprRef back = reparse("\\int \\mathrm{aa}\\,dx", juxtaposed);
    EXPECT_TRUE(structurally_equal(back, integral(symbol("aa"), {symbol("x")})))
        << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
}

TEST(Sym2LatexException, N24_AConstantOutsideTheSevenComesBackASymbol) {
    // §0 excludes the node. An unnamed Constant is spelled with the same `\mathrm{}`
    // wrapper a multi-letter Symbol gets -- but only a Symbol that has no subscript and
    // is not Greek, and this is the collision the document originally overstated as
    // "byte-identical". A Constant never goes through `split_subscript`, so
    // `constant("gamma_E")` stays one name, while `symbol("gamma_E")` splits into base
    // `gamma` plus subscript `E` and, `gamma` being Greek, prints `\gamma_{E}` instead.
    // The exception is real either way: what the Constant prints reads back as a Symbol.
    const ExprRef e = constant("gamma_E");
    EXPECT_EQ(to_latex(e), "\\mathrm{gamma\\_E}");
    EXPECT_EQ(to_latex(symbol("gamma_E")), "\\gamma_{E}");
    EXPECT_NE(to_latex(e), to_latex(symbol("gamma_E")));
    // A name that does not split and is not Greek is where the two really do collide.
    EXPECT_EQ(to_latex(constant("kelvin")), to_latex(symbol("kelvin")));
    const ExprRef back = reparse("\\mathrm{gamma\\_E}");
    EXPECT_TRUE(structurally_equal(back, symbol("gamma_E"))) << to_string(back);
    EXPECT_FALSE(structurally_equal(back, e));
}

TEST(Sym2LatexException, N25_ACalculusVariableThatIsNotASymbolIsNeverFenced) {
    // §0 excludes these. The walker places a variable in an open slot, so a compound
    // one runs into whatever follows: the derivative's denominator swallows the `+ y`,
    // and the integral's differential leaves it outside the integrand entirely.
    const ExprRef x = symbol("x");
    const ExprRef bad_variable = add({x, symbol("y")});
    const ExprRef bad_derivative = derivative(x, {bad_variable});
    const ExprRef bad_integral = integral(x, {bad_variable});
    EXPECT_EQ(to_latex(bad_derivative), "\\frac{d}{dx + y} x");
    EXPECT_EQ(to_latex(bad_integral), "\\int x \\, dx + y");

    // The document says "parse error / wrong tree" and does not choose between them,
    // because the choice belongs to the parser rather than to the node. What must not
    // happen is the third possibility: the string reading back as the node it came from,
    // which would mean the unfenced variable had been recovered by guesswork.
    const Result<ExprRef> derivative_back = parse_latex("\\frac{d}{dx + y} x");
    if (derivative_back.has_value()) {
        EXPECT_FALSE(structurally_equal(*derivative_back, bad_derivative))
            << to_string(*derivative_back);
    }
    const Result<ExprRef> integral_back = parse_latex("\\int x \\, dx + y");
    if (integral_back.has_value()) {
        EXPECT_FALSE(structurally_equal(*integral_back, bad_integral))
            << to_string(*integral_back);
    }
}

TEST(Sym2LatexException, N26_AMatrixEnvironmentOutsideTheAllowListIsRejected) {
    // §0 excludes the option value. The environment name is interpolated with no
    // escaping and no validation, so a name carrying a brace produces a document whose
    // `\end` does not match its `\begin` -- and `array` is outside the subset anyway.
    NotationOptions forged;
    forged.matrix_environment = "array}{cc";
    const ms::Matrix<double> m{{1.0, 2.0}};
    const std::string text = to_latex(m, forged);
    EXPECT_EQ(text, "\\begin{array}{cc} 1 & 2 \\end{array}{cc}");
    const Result<ms::Matrix<double>> back = parse_latex_matrix(text, forged);
    ASSERT_FALSE(back.has_value())
        << "an environment outside the allow-list was read back as a matrix";
    const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
    ASSERT_NE(error, nullptr) << describe(back.error());
    EXPECT_NE(error->msg.find("E-LATEX-0036"), std::string::npos) << error->msg;
}

TEST(Sym2LatexException, N27_ANullReferencePrintsAsTheUndefinedConstant) {
    // `undefined()` round-trips, and for it the spelling is right. The loss is that a
    // null `ExprRef` -- and a head the walker does not recognise -- reach the page as
    // that same string, so what comes back claims to be a value where the printer had
    // nothing at all to print.
    EXPECT_EQ(to_latex(ExprRef{}), "\\mathrm{undefined}");
    EXPECT_EQ(to_latex(undefined()), "\\mathrm{undefined}");
    const ExprRef back = reparse("\\mathrm{undefined}");
    EXPECT_TRUE(structurally_equal(back, constant("undefined"))) << to_string(back);
    EXPECT_TRUE(is_undefined(back));
    EXPECT_NE(back.get(), nullptr) << "a null reference came back as a null reference";
}

} // namespace
