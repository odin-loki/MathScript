// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §11.2's cases that the round-trip property cannot reach.
//
// The property in `test_sym2_latex_roundtrip.cpp` asserts everything the printer emits.
// Three kinds of case sit outside its reach and are exactly the cases where a LaTeX
// reader normally goes wrong:
//
//   - **The rulings of §3.1.** Every row there is a string that two readings fit. The
//     property cannot test them, because it only ever sees strings the printer produced
//     from a node it already knows. A ruling is a decision about what a *reader* should
//     conclude, and it has to be asserted against the reading, not against the writer.
//   - **The rejections of §3.2.** A string outside the subset is never printed, so the
//     property never sees one. Each is asserted with its code *and its position*: a
//     diagnostic whose position is wrong is worse than one with no position at all,
//     because it sends the reader somewhere specific and wrong, and they believe it.
//   - **The human spellings of §2.9.** `\tfrac`, `\bigl`, `x^2`, `$...$` and the rest are
//     concessions to what people write; the printer emits none of them. Each is asserted
//     by parsing it *and* parsing the printer's own spelling of the same expression and
//     requiring the two to agree -- which is the only definition of "equivalent" that
//     does not depend on anyone restating the expected tree by hand.
//
// Plus the matrix entry point, which is a separate function because `Head` has no matrix
// member, and a deeply nested input, because the library is built with `-fno-exceptions`
// and a recursive descent that runs out of stack ends the process rather than the parse.
//
// **Where a position comes from.** §3.2 says only that `L:C` is "the line and column of
// the offending token", which leaves a choice on several rows. The choice made here,
// consistently, is:
//
//   1. a token that is outside the subset wherever it appears (`\sum`, `\pm`, `=`, `!`)
//      is reported at its own first character;
//   2. a modifier that makes an otherwise-legal construct unreadable (the `^` on a
//      function name, the `_` on `\int`, the second `^` of a double superscript) is
//      reported at the modifier, not at what it modifies;
//   3. a required piece that is empty (`\sqrt{}`, `\operatorname{}`) is reported where
//      the missing text should have started;
//   4. a construct whose whole shape is outside the subset (a matrix inside an
//      expression) is reported at the control sequence that introduces it.
//
// If the parser chose differently on a row, this file is where to have that argument,
// and it is worth having rather than relaxing: two readers disagreeing about where an
// error is means one of them is sending people to the wrong character.
//
// Columns count UTF-8 scalar values, not bytes (§5 rule 1), which
// `TheColumnCountsCharactersRatherThanBytes` below is the whole point of.

#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "ms/bignum/bignum.hpp"
#include "ms/core/matrix.hpp"
#include "ms/sym2/expr.hpp"
#include "ms/sym2/latex_parse.hpp"
#include "ms/sym2/notation.hpp"

using ms::bignum::BigInt;
using namespace ms;
using namespace ms::sym2;

namespace {

// The three characters of §2.9 H10 that are not ASCII, named rather than written inline:
// a hex escape followed by a hex digit would extend the escape, so a literal spelled in
// place would silently become a different character depending on what came after it.
const std::string kMinusSign = "\xe2\x88\x92";     // U+2212 MINUS SIGN
const std::string kGreekAlpha = "\xce\xb1";        // U+03B1 GREEK SMALL LETTER ALPHA
const std::string kInfinitySign = "\xe2\x88\x9e";  // U+221E INFINITY

ExprRef frac(long long numerator, long long denominator) {
    return rational(BigInt(numerator), BigInt(denominator));
}

NotationOptions with_comma() {
    NotationOptions options;
    options.decimal_separator = ',';
    return options;
}

std::string describe(const ms::Error& error) {
    if (const ms::ParseError* parse = std::get_if<ms::ParseError>(&error)) {
        return "latex:" + std::to_string(parse->line) + ":" + std::to_string(parse->col) +
               ": " + parse->msg;
    }
    return ms::format_error(error);
}

/// `text` denotes `expected`, and nothing else.
void expect_parses_as(const std::string& text, const ExprRef& expected,
                      const NotationOptions& options = {}) {
    SCOPED_TRACE(text);
    const Result<ExprRef> back = parse_latex(text, options);
    ASSERT_TRUE(back.has_value()) << "rejected: " << describe(back.error());
    EXPECT_TRUE(structurally_equal(*back, expected))
        << "read as " << to_string(*back) << ", expected " << to_string(expected);
}

/// `text` is outside the subset, and the diagnostic says so at `line`:`col` with `code`.
///
/// `fragment` is a verbatim slice of the §3.2 wording. It is checked by containment
/// rather than by equality because §5 rule 2 requires the message to also name what was
/// found, and what was found is not in the table.
void expect_rejected(const std::string& text, const std::string& code, std::size_t line,
                     std::size_t col, const std::string& fragment,
                     const NotationOptions& options = {}) {
    SCOPED_TRACE(text);
    const Result<ExprRef> back = parse_latex(text, options);
    ASSERT_FALSE(back.has_value()) << "accepted, as " << to_string(*back);
    const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
    ASSERT_NE(error, nullptr) << "rejected without a position: " << describe(back.error());
    EXPECT_NE(error->msg.find(code), std::string::npos)
        << "expected " << code << ", got: " << error->msg;
    EXPECT_NE(error->msg.find(fragment), std::string::npos)
        << "expected the documented wording \"" << fragment << "\", got: " << error->msg;
    EXPECT_EQ(error->line, line) << "reported at line " << error->line << ": " << error->msg;
    EXPECT_EQ(error->col, col) << "reported at column " << error->col << ": " << error->msg;
}

/// A §2.9 spelling means what the printer's own spelling of `node` means.
///
/// Both sides are parsed. Restating the expected tree by hand instead would make this a
/// test of the restatement: the claim in §2.9 is an equivalence between two strings, and
/// an equivalence is asserted by evaluating both of them.
void expect_means_the_same_as_printing(const std::string& human, const ExprRef& node,
                                       const NotationOptions& options = {}) {
    SCOPED_TRACE(human);
    const std::string printed = to_latex(node, options);
    const Result<ExprRef> from_printer = parse_latex(printed, options);
    ASSERT_TRUE(from_printer.has_value())
        << "the printer's own output was rejected: " << printed << " -- "
        << describe(from_printer.error());
    const Result<ExprRef> from_human = parse_latex(human, options);
    ASSERT_TRUE(from_human.has_value()) << "rejected: " << describe(from_human.error());
    EXPECT_TRUE(structurally_equal(*from_human, *from_printer))
        << human << " reads as " << to_string(*from_human) << ", but " << printed
        << " reads as " << to_string(*from_printer);
}

// --- §3.1: accepted, with a stated meaning ------------------------------------------------

TEST(Sym2LatexParse, A1_AnUnmarkedNameBeforeAParenthesisIsAProduct) {
    // Application is always marked in printer output -- a `kOperators` control word or
    // `\operatorname{}`. A bare italic letter or an upright word before a parenthesis is
    // therefore implicit multiplication, and all three of these strings are emitted for
    // products. Reading them as calls would hand back an expression nobody wrote, which
    // is the whole defect class this subset exists to keep out.
    const ExprRef x = symbol("x");
    const ExprRef y = symbol("y");
    expect_parses_as("g(x + y)", mul({symbol("g"), add({x, y})}));
    expect_parses_as("\\mathrm{foo}\\,(x + y)", mul({symbol("foo"), add({x, y})}));
    expect_parses_as("\\operatorname{f}(y)x", mul({function("f", {y}), x}));
    // The marked form is the call, and a user who means one writes it.
    expect_parses_as("\\operatorname{f}(x + y)", function("f", {add({x, y})}));
}

TEST(Sym2LatexParse, A2_ARadicalIsAHalfPowerRatherThanASqrtCall) {
    // The two producers are byte-identical, so the string cannot distinguish them and
    // the ruling picks the one that arithmetic can act on.
    expect_parses_as("\\sqrt{x}", pow(symbol("x"), frac(1, 2)));
    expect_parses_as("\\sqrt{x + 1}", pow(add({symbol("x"), integer(1)}), frac(1, 2)));
}

TEST(Sym2LatexParse, A3_AFractionIsADivisionInEveryCase) {
    // No case analysis on what the halves look like: `div` already canonicalises, so
    // `\frac{3}{4}` arrives as the Rational and `\frac{1}{x^{3}}` as the negative power
    // without the parser having to decide which it was.
    expect_parses_as("\\frac{3}{4}", frac(3, 4));
    expect_parses_as("\\frac{1}{x^{3}}", pow(symbol("x"), integer(-3)));
    expect_parses_as("\\frac{x + 1}{y}", div(add({symbol("x"), integer(1)}), symbol("y")));
    expect_parses_as("-\\frac{3}{4}", frac(-3, 4));
}

TEST(Sym2LatexParse, A4_TheDerivativeSpellingNeedsAnExactNumeratorAndDenominator) {
    const ExprRef f = symbol("f");
    const ExprRef x = symbol("x");
    expect_parses_as("\\frac{d}{dx} f", derivative(f, {x}));
    // The upright differential is the same operator (§2.9 H7).
    expect_parses_as("\\frac{\\mathrm{d}}{\\mathrm{d}x} f", derivative(f, {x}));
    // And the escape hatch the ruling promises: an explicit product in the denominator
    // is a quotient, because the denominator is then no longer `d` followed by a Symbol.
    //
    // What it is a quotient OF is the part worth stating. `parse_fraction` distributes
    // the reciprocal over a product denominator -- `\frac{a}{b \cdot c}` is
    // `a * b^-1 * c^-1`, not `a * (b c)^-1` -- and the two print identically, so one of
    // them cannot read back as itself. The distributed form is the one the printer emits
    // from a canonical node, which is why it wins; the other is N28 in
    // docs/LATEX_SUBSET.md.
    //
    // Here that means the `d` cancels, because `mul` collects `d^1 * d^-1` at
    // construction. `1/x` is what `d/(d x)` is, and a core whose constructors reduce
    // `x/3*3` to `x` was never going to hand this one back unreduced.
    expect_parses_as("\\frac{d}{d \\cdot x}", div(integer(1), symbol("x")));
}

// A53 and A54, both from an adversarial review of the parser after the grammar was
// written. Neither was a crash and neither was a rejection: each ACCEPTED the input and
// returned an expression nobody wrote, which is the failure this whole subset exists to
// prevent, arriving by the one route a grammar cannot be read for -- a ruling that is
// right about the shape it names and silent about the shape beside it.
TEST(Sym2LatexParse, A53_ALeibnizDerivativeIsNotAQuotient) {
    // Read as a quotient -- which is what A4's ruling makes it -- the two `d`s cancel in
    // `mul` at construction, so this used to come back `y/x`.
    expect_rejected("\\frac{dy}{dx}", "E-LATEX-0045", 1, 1, "is a derivative to a reader");
    expect_rejected("\\frac{\\mathrm{d}y}{\\mathrm{d}x}", "E-LATEX-0045", 1, 1,
                    "is a derivative to a reader");
    // And the one that invented a factor: this came back `d*y/x^2`, with the `d` left
    // over where the order of the derivative had been.
    expect_rejected("\\frac{d^{2}y}{dx^{2}}", "E-LATEX-0045", 1, 1,
                    "is a derivative to a reader");

    // The shapes either side of it are untouched: A4's derivative, and A4's own escape
    // hatch, whose N28 answer is `1/x`.
    expect_parses_as("\\frac{d}{dx} f", derivative(symbol("f"), {symbol("x")}));
    expect_parses_as("\\frac{d}{d \\cdot x}", div(integer(1), symbol("x")));
    // A fraction that merely starts with `d` on one side is a fraction.
    expect_parses_as("\\frac{dy}{2}", div(mul({symbol("d"), symbol("y")}), integer(2)));
    // Spelled as two divisions because that is what N28 says a product denominator
    // becomes: the reciprocal is distributed, so this is x * d^-1 * y^-1 and not
    // x * (d y)^-1. Writing the second and expecting it to match is the same slip A4's
    // test made before it.
    expect_parses_as("\\frac{x}{dy}", div(div(symbol("x"), symbol("d")), symbol("y")));
}

TEST(Sym2LatexParse, A54_AnIntegralSaysWhereItStops) {
    // The integrand runs to the end of its group, so this used to put the `+ 1` inside
    // the integral -- and with no differential then ending at the boundary, the
    // empty-variable-list exemption fired and `\, dx` became a product as well. The
    // answer was `integral(d*x^2 + 1)`.
    expect_rejected("\\int x \\, dx + 1", "E-LATEX-0046", 1, 1, "does not say where it stops");
    expect_rejected("\\int x \\, dx - 1", "E-LATEX-0046", 1, 1, "does not say where it stops");
    expect_rejected("\\int x \\, dx \\cdot 2", "E-LATEX-0046", 1, 1,
                    "does not say where it stops");

    // What the printer emits still reads, including the empty-variable-list form that
    // the exemption exists for.
    expect_parses_as("\\int x \\, dx", integral(symbol("x"), {symbol("x")}));
    expect_parses_as("\\int f", integral(symbol("f"), {}));
    expect_parses_as("(\\int x \\, dx) + 1",
                     add({integral(symbol("x"), {symbol("x")}), integer(1)}));
}

TEST(Sym2LatexParse, A5_AThinSpaceBeforeATrailingDIsADifferentialByPosition) {
    // The rule keys on position, never on the thin space itself: `\,` is multiplication
    // at one call site in the printer and a differential separator at another, so a rule
    // that keyed on the mark would get both wrong.
    expect_parses_as("\\int f \\, dx", integral(symbol("f"), {symbol("x")}));
    expect_parses_as("\\int x^{2} \\, dx", integral(pow(symbol("x"), integer(2)), {symbol("x")}));
    // No differential at all is the empty-variable form, which the printer emits.
    expect_parses_as("\\int f", integral(symbol("f"), {}));
}

TEST(Sym2LatexParse, A6_AdjacentIntegralSignsAreOneFlatIntegral) {
    // The flat and nested encodings are byte-identical, so there is nothing to choose
    // between them in the text and the flat one is what comes back.
    const ExprRef body = mul({symbol("x"), symbol("y")});
    expect_parses_as("\\int \\int x \\cdot y \\, dx \\, dy",
                     integral(body, {symbol("x"), symbol("y")}));
}

TEST(Sym2LatexParse, A7_AdjacentDerivativeOperatorsAreOneFlatDerivative) {
    // The two encodings differ by one space, and math mode discards source whitespace,
    // so both strings are the same document and both must give the same tree.
    const ExprRef f = symbol("f");
    const ExprRef flat = derivative(f, {symbol("x"), symbol("y")});
    expect_parses_as("\\frac{d}{dx}\\frac{d}{dy} f", flat);
    expect_parses_as("\\frac{d}{dx} \\frac{d}{dy} f", flat);
}

TEST(Sym2LatexParse, A8_TheNamedConstantsWinAtExpressionLevelAndAreNamesInASubscript) {
    expect_parses_as("\\pi", constant("pi"));
    expect_parses_as("e", constant("e"));
    expect_parses_as("i", constant("i"));
    expect_parses_as("\\infty", constant("inf"));
    expect_parses_as("\\mathrm{NaN}", constant("nan"));
    expect_parses_as("\\mathrm{undefined}", constant("undefined"));
    // A subscript is part of a name, so the same spellings are un-spelled as text there.
    // `x_{\pi}` is the symbol x_pi, not x subscript the number pi.
    expect_parses_as("x_{\\pi}", symbol("x_pi"));
    expect_parses_as("x_{e}", symbol("x_e"));
    expect_parses_as("x_{12}", symbol("x_12"));
    expect_parses_as("x_{\\mathrm{max}}", symbol("x_max"));
}

TEST(Sym2LatexParse, A9_AnUprightSingleCapitalIsAGreekCapitalRatherThanALatinSymbol) {
    // A one-letter symbol prints bare, so the upright spelling is free for the thirteen
    // Greek capitals that LaTeX gives no control sequence to. The two spellings are
    // distinct and both are emitted.
    expect_parses_as("\\mathrm{A}", symbol("Alpha"));
    expect_parses_as("A", symbol("A"));
    expect_parses_as("\\mathrm{X}", symbol("Chi"));
    expect_parses_as("\\mathrm{O}", symbol("Omicron"));
    expect_parses_as("\\Gamma", symbol("Gamma"));
}

TEST(Sym2LatexParse, A10_AMinusBeforeInfinityInAPrimarySlotIsTheNegativeInfinityConstant) {
    // Both `constant("-inf")` and `mul(integer(-1), constant("inf"))` print this string,
    // so one of them has to be chosen; the Constant is the one that is a single value.
    expect_parses_as("-\\infty", constant("-inf"));
    expect_parses_as("-\\infty \\cdot x", mul({constant("-inf"), symbol("x")}));
    // A *binary* minus is unaffected -- it is an operator there, not part of a numeral,
    // and this string is emitted for the subtraction.
    expect_parses_as("x - \\infty", sub(symbol("x"), constant("inf")));
}

TEST(Sym2LatexParse, A11_AMantissaTimesAPowerOfTenIsOneRealNumeral) {
    // This costs nothing, because the exact integer has a spelling of its own: `10^{20}`
    // with no mantissa in front of it is the Integer and stays exact.
    expect_parses_as("1 \\times 10^{20}", real(1e20));
    expect_parses_as("2.5 \\times 10^{-13}", real(2.5e-13));
    expect_parses_as("10^{20}", pow(integer(10), integer(20)));
}

TEST(Sym2LatexParse, A12_TheNumeralRuleIsMatchedBeforeTheMultiplicationRule) {
    // Leftmost-longest. `2 \times x` is a product because no numeral can be made of it;
    // `2 \times 10^{3}` is one numeral, and therefore a Real rather than the Integer
    // 2000 -- which is a distinction `is_exact` cares about downstream.
    expect_parses_as("2 \\times x", mul({integer(2), symbol("x")}));
    expect_parses_as("2 \\times 10^{3}", real(2000.0));
    const Result<ExprRef> numeral = parse_latex("2 \\times 10^{3}");
    ASSERT_TRUE(numeral.has_value()) << describe(numeral.error());
    EXPECT_EQ(head_of(*numeral), Head::Real);
    EXPECT_FALSE(structurally_equal(*numeral, integer(2000)));
}

TEST(Sym2LatexParse, A13_BareDigitsAreAnInteger) {
    // Not a Real and not a symbol whose name happens to be digits, even though the
    // printer emits this string for all three (§4.2 N1 and N11).
    expect_parses_as("12", integer(12));
    const Result<ExprRef> twelve = parse_latex("12");
    ASSERT_TRUE(twelve.has_value()) << describe(twelve.error());
    EXPECT_EQ(head_of(*twelve), Head::Integer);
    // Unbounded, because a BigInt never acquires an exponent on the way out.
    expect_parses_as("123456789012345678901234567890",
                     integer(BigInt("123456789012345678901234567890")));
}

TEST(Sym2LatexParse, A14_AnySeparatorBetweenDigitsIsAFactorBoundary) {
    // A numeral's digits are contiguous. This is required rather than tidy: the printer
    // emits `2\,10^{n}` for a product under juxtaposition, and reading the thin space as
    // nothing would turn it into the single numeral 210.
    expect_parses_as("2\\,3", integer(6));
    expect_parses_as("2 3", integer(6));
    const Result<ExprRef> spaced = parse_latex("2\\,3");
    ASSERT_TRUE(spaced.has_value()) << describe(spaced.error());
    EXPECT_FALSE(structurally_equal(*spaced, integer(23)));
    expect_parses_as("2\\,10^{n}", mul({integer(2), pow(integer(10), symbol("n"))}));
}

TEST(Sym2LatexParse, A15_BracesDecideBetweenADecimalCommaAndAnArgumentComma) {
    // The printer braces the separator because an unbraced comma in math mode is
    // punctuation and is set with a space after it. The braces are what make
    // `f(1{,}5, 2{,}5)` readable in both directions -- and this is only accepted at all
    // when the parser is configured for a decimal comma.
    expect_parses_as("\\operatorname{f}(1{,}5, 2{,}5)",
                     function("f", {real(1.5), real(2.5)}), with_comma());
    // The argument separator is never affected by the option.
    expect_parses_as("\\operatorname{f}(x, y)", function("f", {symbol("x"), symbol("y")}),
                     with_comma());
}

TEST(Sym2LatexParse, A16_ASymbolicRootDegreeIsAReciprocalExponent) {
    // Never emitted -- the printer only writes a degree when it is an exact integer --
    // but unambiguous, so accepting it costs nothing and rejecting it would surprise.
    expect_parses_as("\\sqrt[n]{x}", pow(symbol("x"), div(integer(1), symbol("n"))));
    expect_parses_as("\\sqrt[3]{x}", pow(symbol("x"), frac(1, 3)));
}

TEST(Sym2LatexParse, A17_UnaryMinusBindsLooserThanASuperscript) {
    // `-x^{2}` is the negative of a square, not the square of a negative. This is the
    // same rule the REPL parser was corrected to follow; getting it wrong changes the
    // sign of the value rather than the look of it.
    expect_parses_as("-x^{2}", neg(pow(symbol("x"), integer(2))));
    expect_parses_as("-2 \\cdot x", neg(mul({integer(2), symbol("x")})));
}

TEST(Sym2LatexParse, A18_AMinusInsideAnExponentIsOrdinaryNegation) {
    // Only a *numeric* negative exponent becomes a reciprocal; a symbolic one stays a
    // superscript, and all three of these strings are emitted.
    const ExprRef x = symbol("x");
    expect_parses_as("x^{-y}", pow(x, neg(symbol("y"))));
    expect_parses_as("x \\cdot y^{-n}", mul({x, pow(symbol("y"), neg(symbol("n")))}));
    // `x^{-\infty}` is where A18 and A10 overlap, and A10 is the more specific rule:
    // `-\infty` is a Primary, so the exponent slot holds the Constant rather than a
    // negation of the positive one. §4.2 N6 says the same thing from the other side --
    // `mul(integer(-1), constant("inf"))` comes back as the Constant wherever it stands.
    expect_parses_as("x^{-\\infty}", pow(x, constant("-inf")));
}

// --- §3.2: rejected, with the exact diagnostic ---------------------------------------------

TEST(Sym2LatexParse, A19_ABracelessArgumentOfMoreThanOneToken) {
    expect_rejected("x^10", "E-LATEX-0011", 1, 3,
                    "a braceless argument takes only the next token");
    expect_rejected("x_12", "E-LATEX-0011", 1, 3,
                    "a braceless argument takes only the next token");
}

TEST(Sym2LatexParse, A20_ADoubleSuperscript) {
    // TeX rejects this outright, so accepting it would mean reading a document that does
    // not compile and guessing which of the two groupings its author meant.
    expect_rejected("x^{2}^{3}", "E-LATEX-0012", 1, 6, "double superscript");
}

TEST(Sym2LatexParse, A21_ASuperscriptOnAFunctionName) {
    // The notation is genuinely overloaded: 2 means the square of the value and -1 means
    // the inverse function, and no rule recovers both.
    expect_rejected("\\sin^{2}(x)", "E-LATEX-0013", 1, 5,
                    "a superscript on a function name");
    expect_rejected("\\sin^{-1}(x)", "E-LATEX-0013", 1, 5,
                    "a superscript on a function name");
}

TEST(Sym2LatexParse, A22_AFunctionAppliedWithoutParentheses) {
    // `\sin x + 1` is read as `(\sin x) + 1` by some writers and as `\sin(x + 1)` by
    // others, and the extent of the argument is simply not in the text.
    expect_rejected("\\sin x", "E-LATEX-0014", 1, 6,
                    "the extent of the argument is not written");
}

TEST(Sym2LatexParse, A23_ASubscriptOnAFunctionName) {
    expect_rejected("\\log_{2}(x)", "E-LATEX-0015", 1, 5,
                    "a subscript on a function name");
}

TEST(Sym2LatexParse, A24_AJuxtaposedFactorAfterASlash) {
    // The denominator ends at `b` or at `c` depending on who is reading, so there is no
    // reading to pick. An explicit operator after the slash is fine (§2.9 H11).
    expect_rejected("a / b c", "E-LATEX-0016", 1, 7,
                    "the denominator ends at b or at c");
}

TEST(Sym2LatexParse, A25_APostfixExclamationMark) {
    expect_rejected("n!", "E-LATEX-0017", 1, 2, "a postfix ! is a factorial");
}

TEST(Sym2LatexParse, A26_APrimeOrADot) {
    // Both name a derivative with respect to a variable that is not written down, and a
    // prime is a transpose as well.
    expect_rejected("f'(x)", "E-LATEX-0018", 1, 2, "a prime or a dot is a derivative");
    expect_rejected("\\dot{x}", "E-LATEX-0018", 1, 1, "a prime or a dot is a derivative");
    expect_rejected("\\ddot{x}", "E-LATEX-0018", 1, 1, "a prime or a dot is a derivative");
}

TEST(Sym2LatexParse, A26TheAdviceParsesToWhatTheAuthorWasWriting) {
    // A rejection that hands out a remedy is only as good as the remedy. The advice is
    // taken out of the message rather than restated here, so the two cannot drift.
    //
    // It used to read `\frac{d}{dx} f(x)`, which in this subset is a juxtaposition and
    // therefore a PRODUCT (A1): following it gave the derivative of `f` times `x`, with
    // no call in it at all. Someone who wrote `f'(x)` and did what they were told got a
    // different expression and no second diagnostic to say so.
    const Result<ExprRef> back = parse_latex("f'(x)");
    ASSERT_FALSE(back.has_value()) << "accepted a prime";
    const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
    ASSERT_NE(error, nullptr) << describe(back.error());

    const std::size_t write_at = error->msg.find("write ");
    ASSERT_NE(write_at, std::string::npos) << "the message offers no remedy: " << error->msg;
    const std::string advice = error->msg.substr(write_at + std::strlen("write "));

    const Result<ExprRef> remedy = parse_latex(advice);
    ASSERT_TRUE(remedy.has_value())
        << "the advice does not parse: " << advice << " -- " << describe(remedy.error());
    EXPECT_EQ(*remedy, derivative(function("f", {symbol("x")}), {symbol("x")}))
        << "following the advice gives " << to_string(*remedy) << ", not the derivative "
        << "of f -- the advice was: " << advice;
}

TEST(Sym2LatexParse, A27_PlusOrMinus) {
    // `a \pm b` denotes two expressions at once and a MathScript expression is one value.
    expect_rejected("a \\pm b", "E-LATEX-0019", 1, 3, "denotes two expressions at once");
    expect_rejected("a \\mp b", "E-LATEX-0019", 1, 3, "denotes two expressions at once");
}

TEST(Sym2LatexParse, A28_ARelation) {
    // The subset parses expressions. `=` has no expression head, and inventing one --
    // a `Function("equals", ...)`, say -- would put a node in the tree that nothing else
    // in `ms::sym2` knows how to act on.
    const char* const kRelations[] = {"=",      "<",    ">",        "\\le",  "\\leq",
                                      "\\ge",   "\\neq", "\\ne",     "\\approx",
                                      "\\equiv", "\\sim", "\\propto", "\\in",  "\\mid"};
    for (const char* relation : kRelations) {
        expect_rejected(std::string("x ") + relation + " y", "E-LATEX-0020", 1, 3,
                        "the subset parses expressions, not equations");
    }
}

TEST(Sym2LatexParse, A29_ABigOperator) {
    const char* const kBigOperators[] = {"\\sum",      "\\prod",  "\\bigcup",
                                         "\\bigcap",  "\\coprod", "\\bigoplus"};
    for (const char* big : kBigOperators) {
        expect_rejected(std::string(big) + "_{i} i", "E-LATEX-0021", 1, 1,
                        "has no expression head here; big operators");
    }
}

TEST(Sym2LatexParse, A30_ABinomialCoefficient) {
    expect_rejected("\\binom{n}{k}", "E-LATEX-0022", 1, 1, "has no expression head");
}

TEST(Sym2LatexParse, A31_LimitsOnAnIntegral) {
    // `Head::Integral` records no bounds, so accepting the limits would discard them in
    // silence -- the expression would look like it had been read and would not be the
    // one that was written.
    expect_rejected("\\int_{a}^{b} f \\, dx", "E-LATEX-0023", 1, 5, "records no bounds");
    expect_rejected("\\oint f \\, dx", "E-LATEX-0023", 1, 1, "records no bounds");
}

TEST(Sym2LatexParse, A32_AOneSidedOrExtremalLimit) {
    // `Head::Limit` records no direction, so a one-sided limit would come back as a
    // two-sided one, which is a different claim about a function that may not have one.
    expect_rejected("\\lim_{x \\to 0^{+}} x", "E-LATEX-0024", 1, 14, "records no direction");
    expect_rejected("\\limsup_{x \\to 0} x", "E-LATEX-0024", 1, 1, "records no direction");
    expect_rejected("\\liminf_{x \\to 0} x", "E-LATEX-0024", 1, 1, "records no direction");
}

TEST(Sym2LatexParse, A33_APartialDerivative) {
    // `Head::Derivative` draws no distinction between a partial and a total derivative,
    // so reading `\partial` as one would assert something the tree cannot hold.
    expect_rejected("\\frac{\\partial}{\\partial x} f", "E-LATEX-0025", 1, 7,
                    "records no distinction between a partial and a total one");
    // Standing alone it is simply the symbol named partial, which the printer emits.
    expect_parses_as("\\partial", symbol("partial"));
}

TEST(Sym2LatexParse, A34_BracesBracketsAndTheOtherFences) {
    // Each of these means something structural that has no expression head: a set, a
    // case split, an interval, a list, a matrix row, a floor.
    expect_rejected("\\{ x \\}", "E-LATEX-0026", 1, 1, "is a set or a case split");
    expect_rejected("[a, b]", "E-LATEX-0026", 1, 1, "is a set or a case split");
    expect_rejected("\\lfloor x \\rfloor", "E-LATEX-0026", 1, 1, "is a set or a case split");
    expect_rejected("\\langle x \\rangle", "E-LATEX-0026", 1, 1, "is a set or a case split");
    expect_rejected("\\lceil x \\rceil", "E-LATEX-0026", 1, 1, "is a set or a case split");
    expect_rejected("\\lVert x \\rVert", "E-LATEX-0026", 1, 1, "is a set or a case split");
}

TEST(Sym2LatexParse, A35_AnEvaluationBarOrANullDelimiter) {
    // A null delimiter carrying limits is the "evaluated between" notation, which is a
    // substitution rather than a value and has no head here. The position is the `.`:
    // `\left` itself is in the subset, and the character after it is not.
    expect_rejected("\\left. x \\right.", "E-LATEX-0027", 1, 6,
                    "an evaluation bar (a null delimiter with limits)");
    // The same construct written with a sized bar and limits. Reported at the script,
    // which is what turns a legal absolute value into an evaluation bar.
    expect_rejected("f \\big\\|_{0}^{1}", "E-LATEX-0027", 1, 9,
                    "an evaluation bar (a null delimiter with limits)");
}

TEST(Sym2LatexParse, A36_ANestedBareDoubleBar) {
    // The open and close delimiter are the same character, so a second one cannot be
    // told from a close. The sized and the `\lvert` spellings both carry the direction
    // and are accepted (§2.9 H3).
    expect_rejected("\\|\\|x\\|\\|", "E-LATEX-0028", 1, 3, "a bare vertical bar cannot be paired");
    expect_rejected("\\|a\\|b\\|", "E-LATEX-0028", 1, 7, "a bare vertical bar cannot be paired");
}

TEST(Sym2LatexParse, A37_Decoration) {
    // `\hat{x}` and `x` are different symbols to a reader and the same name to a parser
    // that strips the decoration, which is how a decorated variable silently becomes its
    // undecorated self halfway through a document.
    const char* const kDecorations[] = {
        "\\text",  "\\mathbf",    "\\mathbb", "\\mathcal",  "\\mathfrak",
        "\\mathsf", "\\boldsymbol", "\\vec",   "\\hat",      "\\bar",
        "\\overline", "\\tilde",  "\\underline", "\\mathit", "\\bm"};
    for (const char* decoration : kDecorations) {
        expect_rejected(std::string(decoration) + "{x}", "E-LATEX-0029", 1, 1,
                        "decoration is not part of a name in this subset");
    }
}

TEST(Sym2LatexParse, A38_AFontVariantOfAGreekLetter) {
    // A font variant is not a distinct symbol, and treating it as one would give two
    // names for one letter that compare unequal everywhere.
    expect_rejected("\\varGamma", "E-LATEX-0030", 1, 1, "is a font variant of");
    expect_rejected("\\upalpha", "E-LATEX-0030", 1, 1, "is a font variant of");
}

TEST(Sym2LatexParse, A39_ADifferentialCountThatDoesNotMatchTheSignCount) {
    // Two signs and one differential is not a reading with a missing piece; it is two
    // readings, and the integrand is different in each.
    expect_rejected("\\int \\int x \\, dx", "E-LATEX-0031", 1, 1, "integral signs but");
    expect_rejected("\\int x \\, dx \\, dy", "E-LATEX-0031", 1, 1, "integral signs but");
}

TEST(Sym2LatexParse, A40_ScientificNotationInMathMode) {
    // `1e20` in math mode is one, times Euler's number, plus twenty -- and it typesets
    // perfectly, which is what makes it dangerous. The printer never emits it for
    // exactly this reason.
    expect_rejected("1e20", "E-LATEX-0032", 1, 2, "is 1 multiplied by Euler's number");
    expect_rejected("1e-7", "E-LATEX-0032", 1, 2, "is 1 multiplied by Euler's number");
}

TEST(Sym2LatexParse, A41_TheInfimumOperatorIsNotInfinity) {
    // `\sup` is a function name in this subset and `\inf` is not, which is exactly the
    // trap: the two look like a pair and only one of them is one.
    expect_rejected("\\inf", "E-LATEX-0033", 1, 1, "is the infimum operator, not infinity");
    expect_parses_as("\\sup(x)", function("sup", {symbol("x")}));
}

TEST(Sym2LatexParse, A42_AContinuedFractionLayout) {
    expect_rejected("\\cfrac{a}{b}", "E-LATEX-0034", 1, 1,
                    "is continued-fraction layout with an optional alignment argument");
}

TEST(Sym2LatexParse, A43_AMacroDefinition) {
    // TeX is Turing-complete, so expanding a macro is not a bounded piece of work and a
    // parser that tried would be a TeX implementation.
    const char* const kDefinitions[] = {"\\newcommand", "\\renewcommand", "\\def", "\\let",
                                        "\\providecommand"};
    for (const char* definition : kDefinitions) {
        expect_rejected(std::string(definition) + "{\\a}{1}", "E-LATEX-0035", 1, 1,
                        "macro expansion is not performed");
    }
}

TEST(Sym2LatexParse, A44_AnEnvironmentOutsideTheMatrixAllowList) {
    // `array` carries a column specification, which is presentation rather than
    // structure; `cases`, `aligned` and the rest are multi-expression layouts. Reported
    // at the environment name, because `\begin` itself is in the subset.
    const char* const kEnvironments[] = {"array",    "cases", "aligned",     "align",
                                         "equation", "gather", "split",      "smallmatrix"};
    for (const char* environment : kEnvironments) {
        const std::string text =
            std::string("\\begin{") + environment + "} 1 \\end{" + environment + "}";
        SCOPED_TRACE(text);
        const Result<ms::Matrix<double>> back = parse_latex_matrix(text);
        ASSERT_FALSE(back.has_value()) << "accepted an environment outside the allow-list";
        const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
        ASSERT_NE(error, nullptr) << describe(back.error());
        EXPECT_NE(error->msg.find("E-LATEX-0036"), std::string::npos) << error->msg;
        EXPECT_NE(error->msg.find("carries a column specification"), std::string::npos)
            << error->msg;
        EXPECT_EQ(error->line, 1U);
        EXPECT_EQ(error->col, 8U) << "expected the environment name: " << error->msg;
    }
}

TEST(Sym2LatexParse, A45_AMatrixInsideAnExpression) {
    // `expr.hpp` has no matrix head, so there is no node to build. The separate entry
    // point is the answer, and naming it is the whole content of the diagnostic.
    expect_rejected("x + \\begin{pmatrix} 1 \\end{pmatrix}", "E-LATEX-0037", 1, 5,
                    "a matrix is not an expression here");
}

TEST(Sym2LatexParse, A46_ACellOrRowSeparatorOutsideAMatrix) {
    expect_rejected("x & y", "E-LATEX-0038", 1, 3, "is a matrix cell separator");
    expect_rejected("x \\\\ y", "E-LATEX-0038", 1, 3, "is a matrix cell separator");
}

TEST(Sym2LatexParse, A47_AnEmptyFunctionName) {
    // The printer emits this from `function("", args)`, which §0 excludes. There is no
    // name in it to recover and no default worth inventing. Reported where the name
    // should have started.
    expect_rejected("\\operatorname{}(x)", "E-LATEX-0039", 1, 15,
                    "an empty function name carries no name to recover");
}

TEST(Sym2LatexParse, A48_ASubscriptOnSomethingThatIsNotAName) {
    // A subscript belongs to a symbol name. On a fraction or on a parenthesised group it
    // is a sequence index, a component, or a basis label, none of which has a head here.
    expect_rejected("\\frac{a}{b}_{1}", "E-LATEX-0040", 1, 12,
                    "a subscript belongs to a symbol name");
    expect_rejected("(x)_{1}", "E-LATEX-0040", 1, 4, "a subscript belongs to a symbol name");
}

TEST(Sym2LatexParse, A49_ADecimalCommaThatTheConfigurationDoesNotAllow) {
    // Configured for a decimal point, `1{,}5` is either one number or two, and which one
    // depends on a setting the text does not carry. The diagnostic names the setting.
    expect_rejected("1{,}5", "E-LATEX-0041", 1, 2,
                    "is one number with a decimal comma or two numbers in a list");
    expect_rejected("1,5", "E-LATEX-0041", 1, 2,
                    "is one number with a decimal comma or two numbers in a list");
    // A bare comma inside a numeral is rejected under the comma configuration too: the
    // braces are what distinguish it from an argument separator (§3.1 A15).
    expect_rejected("1,5", "E-LATEX-0041", 1, 2,
                    "is one number with a decimal comma or two numbers in a list",
                    with_comma());
}

TEST(Sym2LatexParse, A50_AnEmptyRadicand) {
    expect_rejected("\\sqrt{}", "E-LATEX-0042", 1, 7, "empty radicand");
}

TEST(Sym2LatexParse, A51_ARadicalGlyphWithNoRadicand) {
    expect_rejected("\\surd", "E-LATEX-0043", 1, 1, "is a glyph with no radicand");
    expect_rejected("\\root 3 \\of{x}", "E-LATEX-0043", 1, 1, "is a glyph with no radicand");
}

TEST(Sym2LatexParse, A52_AnyOtherControlSequence) {
    // The catch-all, and the reason the subset is closed rather than open: a control
    // sequence nobody wrote a rule for is refused by name instead of being skipped.
    expect_rejected("\\foo", "E-LATEX-0044", 1, 1, "unknown control sequence");
    expect_rejected("x + \\zzz", "E-LATEX-0044", 1, 5, "unknown control sequence");
    // §5 rule 2: the found text is quoted verbatim and never normalised.
    const Result<ExprRef> back = parse_latex("\\foo");
    ASSERT_FALSE(back.has_value());
    const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
    ASSERT_NE(error, nullptr) << describe(back.error());
    EXPECT_NE(error->msg.find("\\foo"), std::string::npos)
        << "the message does not name what was found: " << error->msg;
}

// --- The error model itself (§5) --------------------------------------------------------

TEST(Sym2LatexParse, ThePositionIsOnTheLineWhereTheProblemIs) {
    // A diagnostic that reports line 1 for a problem on line 2 is worse than one with no
    // line at all: the reader stops looking at the place it names.
    expect_rejected("x +\n  \\sum_{i} i", "E-LATEX-0021", 2, 3,
                    "has no expression head here; big operators");
}

TEST(Sym2LatexParse, TheColumnCountsCharactersRatherThanBytes) {
    // A symbol name may carry raw non-ASCII -- `escape` passes bytes through untouched --
    // so a byte column would point into the middle of a character and no editor would
    // put the cursor where the parser meant. `\mathrm{}` plus three characters plus a
    // space puts `\pm` at column 14; counted in bytes it would be 16.
    const std::string text = "\\mathrm{\xc3\xa9t\xc3\xa9} \\pm x";
    ASSERT_EQ(text.size(), 20U) << "the literal is not the byte length this test assumes";
    expect_rejected(text, "E-LATEX-0019", 1, 14, "denotes two expressions at once");
}

TEST(Sym2LatexParse, ADelimiterMismatchNamesTheOpenerAsWellAsTheCloser) {
    // §5 rule 3. The position is the closer, because that is where the reader is; the
    // message carries the opener, because that is what they have to go back to.
    const Result<ExprRef> back = parse_latex("\\left( x \\right]");
    ASSERT_FALSE(back.has_value()) << "accepted a mismatched delimiter pair";
    const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
    ASSERT_NE(error, nullptr) << describe(back.error());
    EXPECT_EQ(error->line, 1U);
    // Column 16 is the `]`, following the same choice A35 makes for `\left.`: `\right`
    // is in the subset and the character after it is not.
    EXPECT_EQ(error->col, 16U) << "expected the closing delimiter: " << error->msg;
    EXPECT_NE(error->msg.find("\\left("), std::string::npos)
        << "the message does not name the opener: " << error->msg;
}

TEST(Sym2LatexParse, AnUnlicensedCloserPrefixIsItselfTheTokenOutOfPlace) {
    // The skip in the test above is right only because `\left(` LICENSES `\right` as
    // its closer's prefix, so the prefix is fine and the delimiter after it is not.
    // Skipping unconditionally made the diagnostic contradict itself: a bare `(` does
    // not license `\right`, and `(x\right)` came back "expected ')' ... found ')'",
    // pointing at a perfectly good `)` while the `\right` went unnamed.
    {
        const Result<ExprRef> back = parse_latex("(x\\right)");
        ASSERT_FALSE(back.has_value()) << "accepted a bare '(' closed by '\\right)'";
        const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
        ASSERT_NE(error, nullptr) << describe(back.error());
        EXPECT_EQ(error->line, 1U);
        // Column 3 is the backslash of `\right`, which is the whole of what is wrong.
        EXPECT_EQ(error->col, 3U) << "expected the \\right: " << error->msg;
        EXPECT_NE(error->msg.find("\\right"), std::string::npos)
            << "the message does not name the token out of place: " << error->msg;
    }
    {
        // The mirror: `\left(` does not license a sized `\big` either.
        const Result<ExprRef> back = parse_latex("\\left(x\\big)");
        ASSERT_FALSE(back.has_value()) << "accepted '\\left(' closed by '\\big)'";
        const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
        ASSERT_NE(error, nullptr) << describe(back.error());
        EXPECT_EQ(error->line, 1U);
        EXPECT_EQ(error->col, 8U) << "expected the \\big: " << error->msg;
    }
    {
        // And the prefix each opener DOES license still closes it. Narrowing the skip
        // must not narrow what is accepted -- `mismatch_site` only chooses where to
        // point once something has already gone wrong.
        EXPECT_TRUE(parse_latex("(x\\big)").has_value()) << "'\\big)' no longer closes '('";
        EXPECT_TRUE(parse_latex("\\left(x\\right)").has_value())
            << "'\\right)' no longer closes '\\left('";
    }
}

TEST(Sym2LatexParse, AnUnclosedGroupNamesWhereItWasOpened) {
    const Result<ExprRef> back = parse_latex("\\frac{x}{y");
    ASSERT_FALSE(back.has_value()) << "accepted an unclosed group";
    const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
    ASSERT_NE(error, nullptr) << describe(back.error());
    // §5 rule 2: at end of input the message says so rather than quoting nothing, so
    // the position is one past the last character rather than the last character itself.
    EXPECT_FALSE(error->msg.empty());
    EXPECT_EQ(error->line, 1U);
    EXPECT_EQ(error->col, 11U) << "expected end of input: " << error->msg;
    // §5 rule 3: the opener is named, because that is where the reader has to go back to.
    EXPECT_NE(error->msg.find("{"), std::string::npos)
        << "the message does not name the opener: " << error->msg;
}

// --- §2.9: the twelve human spellings ------------------------------------------------------
//
// Each is compared against the printer's own spelling of the same expression, parsed.
// The claim in §2.9 is an equivalence between two strings and is asserted as one.

TEST(Sym2LatexParse, H1_TfracIsFrac) {
    expect_means_the_same_as_printing("\\tfrac{1}{2}", frac(1, 2));
    expect_means_the_same_as_printing("\\tfrac{x + 1}{y}",
                                      div(add({symbol("x"), integer(1)}), symbol("y")));
    // `\dfrac` is printer output rather than a concession: it is what `display` emits.
    NotationOptions display;
    display.display = true;
    expect_means_the_same_as_printing("\\dfrac{1}{2}", frac(1, 2), display);
}

TEST(Sym2LatexParse, H2_SizedDelimiterPrefixesCarryNoMeaning) {
    const ExprRef sum = add({symbol("x"), integer(1)});
    const char* const kPairs[][2] = {
        {"\\bigl", "\\bigr"},   {"\\Bigl", "\\Bigr"},   {"\\biggl", "\\biggr"},
        {"\\Biggl", "\\Biggr"}, {"\\big", "\\big"},     {"\\Big", "\\Big"},
        {"\\bigg", "\\bigg"},   {"\\Bigg", "\\Bigg"},
    };
    for (const auto& delimiters : kPairs) {
        expect_means_the_same_as_printing(
            std::string(delimiters[0]) + "( x + 1 " + delimiters[1] + ")", sum);
    }
    // And before a bar, where the pair is the absolute value rather than a grouping.
    expect_means_the_same_as_printing("\\bigl\\| x \\bigr\\|", function("abs", {symbol("x")}));
}

TEST(Sym2LatexParse, H3_TheOtherTwoSpellingsOfAnAbsoluteValue) {
    const ExprRef e = function("abs", {symbol("x")});
    expect_means_the_same_as_printing("\\lvert x \\rvert", e);
    expect_means_the_same_as_printing("\\|x\\|", e);
    // The printer's own spelling, for completeness: the bars always grow, whatever
    // `sized_delimiters` says, because a flat bar around a fraction is a broken render.
    expect_means_the_same_as_printing("\\left| x \\right|", e);
}

TEST(Sym2LatexParse, H4_TeXsOneTokenRuleForABracelessArgument) {
    expect_means_the_same_as_printing("x^2", pow(symbol("x"), integer(2)));
    expect_means_the_same_as_printing("\\frac12", frac(1, 2));
    expect_means_the_same_as_printing("\\sqrt2", pow(integer(2), frac(1, 2)));
    expect_means_the_same_as_printing("x_1", symbol("x_1"));
    expect_means_the_same_as_printing("\\frac\\alpha\\beta",
                                      div(symbol("alpha"), symbol("beta")));
}

TEST(Sym2LatexParse, H5_TheSpacingCommandsAreWhitespace) {
    // None of these is ever emitted, so accepting them cannot conflict with `\,`, which
    // is a token rather than whitespace precisely because it *is* emitted and is
    // load-bearing in two places.
    const ExprRef sum = add({symbol("x"), symbol("y")});
    const char* const kSpacings[] = {"\\;", "\\:", "\\!", "\\quad", "\\qquad",
                                     "\\thinspace", "\\ ", "~"};
    // The space before `y` is not decoration. A control WORD takes the longest run of
    // letters after the backslash, so `\\quady` is one control sequence named `quady` and
    // not `\\quad` followed by `y` -- which is what this loop wrote before, and what the
    // parser correctly rejected as unknown. The lexer's own header says the same thing
    // about `\\inftyx`. Control symbols like `\\;` and `~` do not have the problem, and
    // the space is harmless for them.
    for (const char* spacing : kSpacings) {
        expect_means_the_same_as_printing(
            std::string("x") + spacing + "+" + spacing + " y", sum);
    }
}

TEST(Sym2LatexParse, H6_TheOtherArrowsAndLimLimits) {
    const ExprRef e = limit(symbol("x"), symbol("x"), integer(0));
    expect_means_the_same_as_printing("\\lim_{x \\rightarrow 0} x", e);
    expect_means_the_same_as_printing("\\lim_{x \\longrightarrow 0} x", e);
    expect_means_the_same_as_printing("\\lim\\limits_{x \\to 0} x", e);
}

TEST(Sym2LatexParse, H7_TheMultipleIntegralSignsAndTheUprightDifferential) {
    const ExprRef f = symbol("f");
    const ExprRef x = symbol("x");
    const ExprRef y = symbol("y");
    const ExprRef z = symbol("z");
    expect_means_the_same_as_printing("\\iint f \\, dx \\, dy", integral(f, {x, y}));
    expect_means_the_same_as_printing("\\iiint f \\, dx \\, dy \\, dz",
                                      integral(f, {x, y, z}));
    expect_means_the_same_as_printing("\\int\\limits f \\, dx", integral(f, {x}));
    // `\mathrm{d}` in a differential position is the differential, not a symbol named d.
    expect_means_the_same_as_printing("\\int f \\, \\mathrm{d}x", integral(f, {x}));
    expect_means_the_same_as_printing("\\frac{\\mathrm{d}}{\\mathrm{d}x} f",
                                      derivative(f, {x}));
}

TEST(Sym2LatexParse, H8_TheOperatornameVariantsFoldOntoOneName) {
    // `\operatorname{X}` and `\X` are folded to one name before comparing, so a caller
    // writing the long form of a known operator gets the same node the short form gives.
    expect_means_the_same_as_printing("\\operatorname*{erf}(x)",
                                      function("erf", {symbol("x")}));
    expect_means_the_same_as_printing("\\operatorname{sin}(x)",
                                      function("sin", {symbol("x")}));
    // Case is still preserved, because `\operatorname{Sin}` is a different name and the
    // printer keeps it distinct.
    expect_means_the_same_as_printing("\\operatorname{Sin}(x)",
                                      function("Sin", {symbol("x")}));
    const Result<ExprRef> capitalised = parse_latex("\\operatorname{Sin}(x)");
    const Result<ExprRef> lowercase = parse_latex("\\sin(x)");
    ASSERT_TRUE(capitalised.has_value()) << describe(capitalised.error());
    ASSERT_TRUE(lowercase.has_value()) << describe(lowercase.error());
    EXPECT_FALSE(structurally_equal(*capitalised, *lowercase));
}

TEST(Sym2LatexParse, H9_OneOuterMathModeWrapperIsStripped) {
    const ExprRef sum = add({symbol("x"), integer(1)});
    expect_means_the_same_as_printing("$x + 1$", sum);
    expect_means_the_same_as_printing("$$x + 1$$", sum);
    expect_means_the_same_as_printing("\\(x + 1\\)", sum);
    expect_means_the_same_as_printing("\\[x + 1\\]", sum);
}

TEST(Sym2LatexParse, H10_CommentsAndTheThreeUnicodeSpellings) {
    expect_means_the_same_as_printing("x % this runs to the end of the line\n + 1",
                                      add({symbol("x"), integer(1)}));
    expect_means_the_same_as_printing("x " + kMinusSign + " 1",
                                      sub(symbol("x"), integer(1)));
    expect_means_the_same_as_printing(kGreekAlpha, symbol("alpha"));
    expect_means_the_same_as_printing(kInfinitySign, constant("inf"));
}

TEST(Sym2LatexParse, H11_SlashAndDivAreInfixDivision) {
    const ExprRef a = symbol("a");
    const ExprRef b = symbol("b");
    expect_means_the_same_as_printing("a / b", div(a, b));
    expect_means_the_same_as_printing("a \\div b", div(a, b));
    // Multiplication precedence, left-associative: the denominator is `b` alone, and the
    // `\cdot c` is a further factor. (With no operator at all this is A24 and rejected,
    // because nobody agrees where the denominator ends.)
    expect_parses_as("a / b \\cdot c", mul({div(a, b), symbol("c")}));
    expect_parses_as("a / b / c", div(div(a, b), symbol("c")));
}

TEST(Sym2LatexParse, H12_AsteriskAndAstAreCdot) {
    const ExprRef product = mul({integer(2), symbol("x")});
    expect_means_the_same_as_printing("2 * x", product);
    expect_means_the_same_as_printing("2 \\ast x", product);
}

// --- §2.7: the matrix entry point ------------------------------------------------------

TEST(Sym2LatexParseMatrix, EveryEnvironmentInTheAllowListIsRead) {
    // `matrix_environment` is ignored on input (§5 rule 6): all five are accepted
    // whatever the printer was configured with, which is what keeps the round trip
    // independent of the options.
    for (const char* environment : {"pmatrix", "bmatrix", "vmatrix", "Vmatrix", "matrix"}) {
        const std::string text = std::string("\\begin{") + environment +
                                 "} 1 & 2 \\\\ 3 & 0.5 \\end{" + environment + "}";
        SCOPED_TRACE(text);
        const Result<ms::Matrix<double>> m = parse_latex_matrix(text);
        ASSERT_TRUE(m.has_value()) << describe(m.error());
        ASSERT_EQ(m->rows(), 2U);
        ASSERT_EQ(m->cols(), 2U);
        EXPECT_EQ((*m)(0, 0), 1.0);
        EXPECT_EQ((*m)(0, 1), 2.0);
        EXPECT_EQ((*m)(1, 0), 3.0);
        EXPECT_EQ((*m)(1, 1), 0.5);
    }
}

TEST(Sym2LatexParseMatrix, TheEmptyBodyFormIsZeroByZero) {
    // The printer writes `\begin{ENV}\end{ENV}` with no spaces for an empty body, and
    // writes it identically for 0x0, m x 0 and 0 x n (§4.2 N21). The empty body is the
    // only form with no spaces around it, so it is a shape of its own to recognise.
    const Result<ms::Matrix<double>> m = parse_latex_matrix("\\begin{pmatrix}\\end{pmatrix}");
    ASSERT_TRUE(m.has_value()) << describe(m.error());
    EXPECT_EQ(m->rows(), 0U);
    EXPECT_EQ(m->cols(), 0U);
    // The same environment with the spaces the non-empty form uses is still empty.
    const Result<ms::Matrix<double>> spaced =
        parse_latex_matrix("\\begin{bmatrix} \\end{bmatrix}");
    ASSERT_TRUE(spaced.has_value()) << describe(spaced.error());
    EXPECT_EQ(spaced->rows(), 0U);
    EXPECT_EQ(spaced->cols(), 0U);
}

TEST(Sym2LatexParseMatrix, ARaggedBodyIsRejected) {
    // §2.7: rows must be rectangular. A short row could be padded with zeros or with
    // NaN, and both are a number the writer did not put there -- `Matrix<double>` has no
    // way to hold "this cell is absent", so the honest answer is to refuse.
    //
    // §3.2 assigns this rejection no code and no position, so only the rejection and the
    // presence of a 1-based position are pinned here. If the implementation gives it a
    // code, this is the place to record it.
    for (const char* text : {"\\begin{pmatrix} 1 & 2 \\\\ 3 \\end{pmatrix}",
                             "\\begin{pmatrix} 1 \\\\ 2 & 3 \\end{pmatrix}"}) {
        SCOPED_TRACE(text);
        const Result<ms::Matrix<double>> m = parse_latex_matrix(text);
        ASSERT_FALSE(m.has_value())
            << "accepted a ragged body as a " << m->rows() << "x" << m->cols() << " matrix";
        const ms::ParseError* error = std::get_if<ms::ParseError>(&m.error());
        ASSERT_NE(error, nullptr) << describe(m.error());
        EXPECT_GE(error->line, 1U) << error->msg;
        EXPECT_GE(error->col, 1U) << error->msg;
        EXPECT_FALSE(error->msg.empty());
    }
}

TEST(Sym2LatexParseMatrix, TheOpeningAndClosingEnvironmentMustBeTheSameString) {
    // §5 rule 3: the position is the closer and the message names the opener.
    const Result<ms::Matrix<double>> m =
        parse_latex_matrix("\\begin{pmatrix} 1 \\end{bmatrix}");
    ASSERT_FALSE(m.has_value()) << "accepted a mismatched environment pair";
    const ms::ParseError* error = std::get_if<ms::ParseError>(&m.error());
    ASSERT_NE(error, nullptr) << describe(m.error());
    EXPECT_NE(error->msg.find("pmatrix"), std::string::npos)
        << "the message does not name the opener: " << error->msg;
}

TEST(Sym2LatexParseMatrix, ACellIsANumeralRatherThanAFloatRegex) {
    // A cell goes through the same Real spelling an atom does, so at 1e8 it is already a
    // power of ten, and under a comma separator it is braced. A parser built on a plain
    // float pattern reads neither.
    const Result<ms::Matrix<double>> scientific =
        parse_latex_matrix("\\begin{pmatrix} 1 \\times 10^{8} & -2.5 \\end{pmatrix}");
    ASSERT_TRUE(scientific.has_value()) << describe(scientific.error());
    ASSERT_EQ(scientific->cols(), 2U);
    EXPECT_EQ((*scientific)(0, 0), 1e8);
    EXPECT_EQ((*scientific)(0, 1), -2.5);

    const Result<ms::Matrix<double>> european =
        parse_latex_matrix("\\begin{pmatrix} 0{,}5 & 1{,}25 \\end{pmatrix}", with_comma());
    ASSERT_TRUE(european.has_value()) << describe(european.error());
    ASSERT_EQ(european->cols(), 2U);
    EXPECT_EQ((*european)(0, 0), 0.5);
    EXPECT_EQ((*european)(0, 1), 1.25);

    // The non-finite cells, which the printer writes with the constant spellings.
    const Result<ms::Matrix<double>> special = parse_latex_matrix(
        "\\begin{pmatrix} \\infty & -\\infty & \\mathrm{NaN} \\end{pmatrix}");
    ASSERT_TRUE(special.has_value()) << describe(special.error());
    ASSERT_EQ(special->cols(), 3U);
    EXPECT_TRUE(std::isinf((*special)(0, 0)) && (*special)(0, 0) > 0.0);
    EXPECT_TRUE(std::isinf((*special)(0, 1)) && (*special)(0, 1) < 0.0);
    EXPECT_TRUE(std::isnan((*special)(0, 2)));
}

TEST(Sym2LatexParseMatrix, APrintedMatrixReadsBackThroughTheMatrixEntryPoint) {
    const ms::Matrix<double> m{{1.0, 2.0}, {3.0, 0.5}};
    const Result<ms::Matrix<double>> back = parse_latex_matrix(to_latex(m));
    ASSERT_TRUE(back.has_value()) << describe(back.error());
    ASSERT_EQ(back->rows(), m.rows());
    ASSERT_EQ(back->cols(), m.cols());
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            EXPECT_EQ((*back)(i, j), m(i, j)) << "cell " << i << "," << j;
        }
    }
}

TEST(Sym2LatexParseMatrix, TheExpressionEntryPointRefusesAMatrixAndNamesTheOtherOne) {
    // The two entry points exist because a matrix is not an expression here. A reader
    // who calls the wrong one is told which one to call, rather than getting a node that
    // pretends to be a matrix.
    expect_rejected("\\begin{pmatrix} 1 & 2 \\end{pmatrix}", "E-LATEX-0037", 1, 1,
                    "parse it with parse_latex_matrix");
}

// --- Depth ---------------------------------------------------------------------------
//
// The library is built with `-fno-exceptions`, so a recursive descent that runs off the
// end of the stack does not throw: it ends the process, taking the REPL, the server or
// the host program with it. Input arrives from files and from other programs, so depth
// is an input like any other and has to be reported like one.

TEST(Sym2LatexParse, AThousandNestedGroupsIsADiagnosticRatherThanACrash) {
    const int kDepth = 1000;
    std::string text;
    text.reserve(static_cast<std::size_t>(2 * kDepth) + 1);
    for (int i = 0; i < kDepth; ++i) {
        text += '{';
    }
    text += 'x';
    for (int i = 0; i < kDepth; ++i) {
        text += '}';
    }
    const Result<ExprRef> back = parse_latex(text);
    ASSERT_FALSE(back.has_value())
        << "a thousand nested groups were accepted; if the parser is iterative and this "
           "is genuinely safe, the limit this test exists to pin has moved and the test "
           "should say where to";
    const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
    ASSERT_NE(error, nullptr) << "rejected without a position: " << describe(back.error());
    EXPECT_GE(error->line, 1U);
    EXPECT_GE(error->col, 1U);
    EXPECT_FALSE(error->msg.empty()) << "a rejection with no message says nothing";
}

TEST(Sym2LatexParse, AThousandNestedFractionsIsADiagnosticRatherThanACrash) {
    const int kDepth = 1000;
    std::string text;
    for (int i = 0; i < kDepth; ++i) {
        text += "\\frac{";
    }
    text += "x";
    for (int i = 0; i < kDepth; ++i) {
        text += "}{1}";
    }
    const Result<ExprRef> back = parse_latex(text);
    ASSERT_FALSE(back.has_value()) << "a thousand nested fractions were accepted";
    const ms::ParseError* error = std::get_if<ms::ParseError>(&back.error());
    ASSERT_NE(error, nullptr) << "rejected without a position: " << describe(back.error());
    EXPECT_GE(error->line, 1U);
    EXPECT_GE(error->col, 1U);
    EXPECT_FALSE(error->msg.empty());
}

} // namespace
