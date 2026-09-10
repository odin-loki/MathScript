// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §11.1's LaTeX table, asserted on exact strings.
//
// Exact strings rather than properties, because a printer has no property to check
// against: the whole content of "this is LaTeX for that expression" is the string. What
// each expectation below is really asserting is that the string *renders* as the
// expression that went in -- so the cases that matter most are the ones where a
// plausible string renders as something else, and those are marked where they appear.

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include "ms/bignum/bignum.hpp"
#include "ms/core/matrix.hpp"
#include "ms/sym2/expr.hpp"
#include "ms/sym2/notation.hpp"

using ms::bignum::BigInt;
using namespace ms::sym2;

namespace {

ExprRef frac(long long numerator, long long denominator) {
    return rational(BigInt(numerator), BigInt(denominator));
}

// --- Atoms --------------------------------------------------------------------------

TEST(Sym2NotationLatex, IntegersPrintExactlyAndUnpadded) {
    EXPECT_EQ(to_latex(integer(42)), "42");
    EXPECT_EQ(to_latex(integer(-7)), "-7");
    // A BigInt is unbounded, and the digits are the value: an integer that does not fit
    // a double must not acquire an exponent on the way out.
    EXPECT_EQ(to_latex(integer(BigInt("123456789012345678901234567890"))),
              "123456789012345678901234567890");
}

TEST(Sym2NotationLatex, ExactRationalStaysAFraction) {
    EXPECT_EQ(to_latex(frac(3, 4)), "\\frac{3}{4}");
    // The sign goes outside the fraction. `\frac{-3}{4}` denotes the same number, but
    // the walker gives a negative Rational add-level precedence in a notation whose
    // quotient fences itself -- that is the precedence of a leading minus, so the table
    // is expected to produce one.
    EXPECT_EQ(to_latex(frac(-3, 4)), "-\\frac{3}{4}");
}

TEST(Sym2NotationLatex, RealsKeepEveryDigitTheyNeed) {
    EXPECT_EQ(to_latex(real(2.5)), "2.5");
    // A Real is an expression, not a display. Six decimals would print 0.333333, which
    // is a different double from 1/3 and therefore a different expression; the shortest
    // spelling that reads back as this same double is seventeen characters long.
    EXPECT_EQ(to_latex(real(1.0 / 3.0)), "0.3333333333333333");
    // A whole-valued double loses the ".000000" that std::to_string would have added.
    EXPECT_EQ(to_latex(real(1.0)), "1");
}

TEST(Sym2NotationLatex, ScientificNotationBecomesAPowerOfTen) {
    // The naive spelling is whatever the C library produced, `1e+20`. In math mode that
    // sets as 1 times *e* plus 20 -- and `e` is Euler's number in this very notation, so
    // the reader is handed a different expression that also happens to typeset cleanly.
    EXPECT_EQ(to_latex(real(1e20)), "1 \\times 10^{20}");
    // The exponent's own zero padding is not part of the number either.
    EXPECT_EQ(to_latex(real(1e-7)), "1 \\times 10^{-7}");
}

TEST(Sym2NotationLatex, SymbolsSplitAtTheirSubscript) {
    EXPECT_EQ(to_latex(symbol("x")), "x");
    // The braces are required rather than tidy: `x_12` is x subscript 1 followed by a
    // loose 2, so a subscript of more than one character changes meaning without them.
    EXPECT_EQ(to_latex(symbol("x_1")), "x_{1}");
    EXPECT_EQ(to_latex(symbol("x_12")), "x_{12}");
    // A subscript is a name too, so it is spelled by the same rules.
    EXPECT_EQ(to_latex(symbol("x_max")), "x_{\\mathrm{max}}");
    EXPECT_EQ(to_latex(symbol("x_alpha")), "x_{\\alpha}");
}

TEST(Sym2NotationLatex, GreekAndNamedGlyphsGetTheirControlSequence) {
    EXPECT_EQ(to_latex(symbol("alpha")), "\\alpha");
    EXPECT_EQ(to_latex(symbol("omega")), "\\omega");
    EXPECT_EQ(to_latex(symbol("Gamma")), "\\Gamma");
    EXPECT_EQ(to_latex(symbol("Omega")), "\\Omega");
    // There is no `\Alpha`: capital alpha *is* a Latin A, so LaTeX never gave it a
    // control sequence. It is written as the upright Latin letter it is -- upright
    // because a bare `A` in math mode is italic, and italic is how a variable named A is
    // spelled, which is a different thing from the Greek capital.
    EXPECT_EQ(to_latex(symbol("Alpha")), "\\mathrm{A}");
    EXPECT_EQ(to_latex(symbol("Rho")), "\\mathrm{P}");
    EXPECT_EQ(to_latex(symbol("hbar")), "\\hbar");
    EXPECT_EQ(to_latex(symbol("nabla")), "\\nabla");
    EXPECT_EQ(to_latex(symbol("alpha_1")), "\\alpha_{1}");
}

TEST(Sym2NotationLatex, MultiLetterNamesAreUpright) {
    // Math mode sets each letter as its own variable, so the naive `foo` typesets as the
    // product f*o*o, italic correction and all. That is a different expression.
    EXPECT_EQ(to_latex(symbol("foo")), "\\mathrm{foo}");
    EXPECT_EQ(to_latex(symbol("x1")), "\\mathrm{x1}");
}

TEST(Sym2NotationLatex, SpecialCharactersInANameAreEscaped) {
    // A symbol name is data: it can arrive from a saved session or a host program. An
    // unescaped `%` comments out the rest of the line it lands in, so the naive spelling
    // does not merely render wrongly, it deletes whatever followed it in the document.
    EXPECT_EQ(to_latex(symbol("a%b")), "\\mathrm{a\\%b}");
    EXPECT_EQ(to_latex(symbol("a&b")), "\\mathrm{a\\&b}");
    // `#`, `$` and a brace are the same class of hazard.
    EXPECT_EQ(to_latex(symbol("n#1")), "\\mathrm{n\\#1}");
    EXPECT_EQ(to_latex(symbol("a{b")), "\\mathrm{a\\{b}");
}

TEST(Sym2NotationLatex, EveryNamedConstantHasASpelling) {
    EXPECT_EQ(to_latex(constant("pi")), "\\pi");
    // `e` and `i` stay italic single letters: that is how every other one-letter name in
    // the notation is set, and the upright forms would read as the start of a word.
    EXPECT_EQ(to_latex(constant("e")), "e");
    EXPECT_EQ(to_latex(constant("i")), "i");
    EXPECT_EQ(to_latex(constant("inf")), "\\infty");
    EXPECT_EQ(to_latex(constant("-inf")), "-\\infty");
    // NaN and undefined are words, not variables, so they are upright for the same
    // reason `foo` is.
    EXPECT_EQ(to_latex(constant("nan")), "\\mathrm{NaN}");
    EXPECT_EQ(to_latex(constant("undefined")), "\\mathrm{undefined}");
}

TEST(Sym2NotationLatex, NonFiniteRealsUseTheConstantSpellings) {
    // `format_exact` hands a non-finite value to std::to_string, which gives the letters
    // `inf` -- a product of three italic variables on the page, one of them Euler's
    // number. A Real that overflowed has to print as the infinity it is.
    // Written through numeric_limits rather than as `1.0 / 0.0`: the latter is a
    // constant expression, and MSVC rejects it outright with C2124 ("divide or mod by
    // zero") where GCC folds it to the IEEE infinity. This built clean locally and
    // broke the Windows job.
    constexpr double kInfinity = std::numeric_limits<double>::infinity();
    EXPECT_EQ(to_latex(real(kInfinity)), "\\infty");
    EXPECT_EQ(to_latex(real(-kInfinity)), "-\\infty");
}

// --- Structure ----------------------------------------------------------------------

TEST(Sym2NotationLatex, SumsSpellTheirNegativeTermsAsSubtractions) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_latex(sub(x, symbol("y"))), "x - y");
    EXPECT_EQ(to_latex(sub(x, integer(1))), "x - 1");
    // Three terms, one of them subtracted. The order is the walker's: it prints the
    // node's canonical order with a leading numeric term moved to the end, because
    // `x^3 - 2x^2 + 5` is what a reader expects and `5 + ...` is not.
    EXPECT_EQ(to_latex(add({pow(x, integer(3)), neg(mul({integer(2), pow(x, integer(2))})),
                            integer(5)})),
              "-2 \\cdot x^{2} + x^{3} + 5");
}

TEST(Sym2NotationLatex, ProductsFollowTheMultiplicationOption) {
    const ExprRef e = mul({integer(2), symbol("x"), symbol("y")});
    EXPECT_EQ(to_latex(e), "2 \\cdot x \\cdot y");

    NotationOptions cross;
    cross.multiplication = NotationOptions::Multiplication::Cross;
    EXPECT_EQ(to_latex(e, cross), "2 \\times x \\times y");

    NotationOptions juxtaposed;
    juxtaposed.multiplication = NotationOptions::Multiplication::Juxtaposition;
    EXPECT_EQ(to_latex(e, juxtaposed), "2xy");
}

TEST(Sym2NotationLatex, JuxtapositionSeparatesWhatWouldRunTogether) {
    NotationOptions juxtaposed;
    juxtaposed.multiplication = NotationOptions::Multiplication::Juxtaposition;
    // Math mode discards the space between two factors, so the naive juxtaposition of
    // two upright names puts the single word "barfoo" on the page. (The order is the
    // walker's display sort, by the name of each factor's base.) A thin space is the
    // smallest mark that keeps them two factors.
    EXPECT_EQ(to_latex(mul({symbol("foo"), symbol("bar")}), juxtaposed),
              "\\mathrm{bar}\\,\\mathrm{foo}");
    EXPECT_EQ(to_latex(mul({integer(2), symbol("foo")}), juxtaposed), "2\\,\\mathrm{foo}");
    // Single italic letters are exactly the case juxtaposition exists for: a reader
    // takes each as its own variable, so nothing is inserted.
    EXPECT_EQ(to_latex(mul({symbol("x"), symbol("y")}), juxtaposed), "xy");
    EXPECT_EQ(to_latex(mul({integer(12), symbol("x")}), juxtaposed), "12x");
}

TEST(Sym2NotationLatex, QuotientsAreFractions) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_latex(div(x, symbol("y"))), "\\frac{x}{y}");
    // `\frac` fences both halves, so the sum in the numerator needs no parentheses --
    // this is what `quotient_is_fenced()` buys, and `(x + 1)/y` in ASCII does not get.
    EXPECT_EQ(to_latex(div(add({x, integer(1)}), symbol("y"))), "\\frac{x + 1}{y}");

    NotationOptions display;
    display.display = true;
    EXPECT_EQ(to_latex(div(x, symbol("y")), display), "\\dfrac{x}{y}");
}

TEST(Sym2NotationLatex, PowersBraceTheirExponent) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_latex(pow(x, integer(2))), "x^{2}");
    // The braces are the point of the whole method: `x^10` is x to the first power times
    // ten. It compiles, it renders, and it is a different number.
    EXPECT_EQ(to_latex(pow(x, integer(10))), "x^{10}");
    // The braces also fence a compound exponent, so the walker adds no parentheses.
    EXPECT_EQ(to_latex(pow(x, add({symbol("y"), integer(1)}))), "x^{y + 1}");
    // Nested, both ways. `^` is right-associative, so the tower needs nothing; the tower
    // written the other way round does, and the walker asks for it.
    EXPECT_EQ(to_latex(pow(x, pow(symbol("y"), symbol("z")))), "x^{y^{z}}");
    EXPECT_EQ(to_latex(pow(pow(x, symbol("y")), symbol("z"))), "(x^{y})^{z}");
}

TEST(Sym2NotationLatex, ABaseThatBindsLooselyIsParenthesised) {
    // The naive spelling of pow(x + 1, 2) is `x + 1^{2}`, which is a sum whose second
    // term is 1. The walker decides that the base needs a grouping; the table only says
    // what a grouping looks like.
    EXPECT_EQ(to_latex(pow(add({symbol("x"), integer(1)}), integer(2))), "(x + 1)^{2}");

    NotationOptions sized;
    sized.sized_delimiters = true;
    EXPECT_EQ(to_latex(pow(add({symbol("x"), integer(1)}), integer(2)), sized),
              "\\left(x + 1\\right)^{2}");
}

TEST(Sym2NotationLatex, FractionalExponentsBecomeRadicals) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_latex(pow(x, frac(1, 2))), "\\sqrt{x}");
    EXPECT_EQ(to_latex(pow(x, frac(1, 3))), "\\sqrt[3]{x}");
    // The vinculum runs the width of the radicand, so a sum under it needs no
    // parentheses either.
    EXPECT_EQ(to_latex(pow(add({x, integer(1)}), frac(1, 2))), "\\sqrt{x + 1}");
    // Only 1/n is a root. 2/3 stays a power, because a root of a square and a square of
    // a root differ for a negative base and the walker refuses to pick one.
    EXPECT_EQ(to_latex(pow(x, frac(2, 3))), "x^{\\frac{2}{3}}");

    // A caller feeding a parser rather than a reader turns the radicals off.
    NotationOptions plain;
    plain.roots_as_radicals = false;
    EXPECT_EQ(to_latex(pow(x, frac(1, 2)), plain), "x^{\\frac{1}{2}}");
}

TEST(Sym2NotationLatex, NegativeExponentsBecomeReciprocals) {
    const ExprRef x = symbol("x");
    // Nobody writes `x^{-3}` when they can write a fraction, and the walker has already
    // made that call before the table is asked anything.
    EXPECT_EQ(to_latex(pow(x, integer(-3))), "\\frac{1}{x^{3}}");
    EXPECT_EQ(to_latex(pow(x, integer(-1))), "\\frac{1}{x}");
    // Inside a product the same rule puts the factor under the rule bar.
    EXPECT_EQ(to_latex(mul({symbol("a"), pow(symbol("b"), integer(-1))})), "\\frac{a}{b}");
}

// --- Calls and unevaluated heads ------------------------------------------------------

TEST(Sym2NotationLatex, KnownFunctionsUseTheirOperatorName) {
    const ExprRef x = symbol("x");
    // `\sin` rather than `sin`: the naive spelling is the product s*i*n applied to
    // nothing, and it sets in italics with no operator spacing.
    EXPECT_EQ(to_latex(function("sin", {x})), "\\sin(x)");
    EXPECT_EQ(to_latex(function("arctan", {x})), "\\arctan(x)");
    EXPECT_EQ(to_latex(function("max", {x, symbol("y")})), "\\max(x, y)");
    EXPECT_EQ(to_latex(function("det", {symbol("A")})), "\\det(A)");
    // A name with no control sequence of its own gets the same upright shape and
    // spacing from `\operatorname`, which is the only reason the table above is a
    // shortlist rather than a correctness requirement.
    EXPECT_EQ(to_latex(function("erf", {x})), "\\operatorname{erf}(x)");
    // Including a one-character name: a function name is an operator name, and operator
    // names are upright however short they are -- `\det` and `\gcd` are no longer.
    EXPECT_EQ(to_latex(function("f", {x, symbol("y")})), "\\operatorname{f}(x, y)");
}

TEST(Sym2NotationLatex, AbsAndSqrtGetTheirOwnNotation) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_latex(function("sqrt", {x})), "\\sqrt{x}");
    // The bars grow whatever `sized_delimiters` says. A plain `|` around a fraction
    // stays one line high and the fraction hangs out of it, which is not a preference
    // about delimiters, it is a broken rendering.
    EXPECT_EQ(to_latex(function("abs", {div(x, symbol("y"))})),
              "\\left| \\frac{x}{y} \\right|");
}

TEST(Sym2NotationLatex, DerivativesUseTheTotalDerivativeSpelling) {
    const ExprRef x = symbol("x");
    // `\frac{d}{dx}`, not `\frac{\partial}{\partial x}`: `Head::Derivative` records no
    // distinction between a partial and a total derivative, so the partial spelling
    // would assert something the tree never said.
    EXPECT_EQ(to_latex(derivative(pow(x, integer(2)), {x})), "\\frac{d}{dx} x^{2}");
    EXPECT_EQ(to_latex(derivative(mul({x, symbol("y")}), {x, symbol("y")})),
              "\\frac{d}{dx}\\frac{d}{dy} x \\cdot y");
    // The operator carries a trailing body, so it binds loosely and the walker groups it
    // wherever something tighter follows -- otherwise the `+ y` would look like part of
    // what is being differentiated.
    EXPECT_EQ(to_latex(add({derivative(x, {x}), symbol("y")})), "y + (\\frac{d}{dx} x)");
}

TEST(Sym2NotationLatex, IntegralsCarryAThinSpaceBeforeTheDifferential) {
    const ExprRef x = symbol("x");
    // Without the `\,` the `d` sets hard against the integrand and `\int f dx` reads as
    // three factors under an integral sign rather than as an integral in x.
    EXPECT_EQ(to_latex(integral(pow(x, integer(2)), {x})), "\\int x^{2} \\, dx");
    EXPECT_EQ(to_latex(integral(mul({x, symbol("y")}), {x, symbol("y")})),
              "\\int \\int x \\cdot y \\, dx \\, dy");
}

TEST(Sym2NotationLatex, LimitsSubscriptTheirApproach) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_latex(limit(div(function("sin", {x}), x), x, integer(0))),
              "\\lim_{x \\to 0} \\frac{\\sin(x)}{x}");
    EXPECT_EQ(to_latex(limit(div(integer(1), x), x, constant("inf"))),
              "\\lim_{x \\to \\infty} \\frac{1}{x}");
}

TEST(Sym2NotationLatex, MatricesUseTheChosenEnvironment) {
    const ms::Matrix<double> m{{1.0, 2.0}, {3.0, 0.5}};
    EXPECT_EQ(to_latex(m), "\\begin{pmatrix} 1 & 2 \\\\ 3 & 0.5 \\end{pmatrix}");

    NotationOptions bracketed;
    bracketed.matrix_environment = "bmatrix";
    EXPECT_EQ(to_latex(m, bracketed), "\\begin{bmatrix} 1 & 2 \\\\ 3 & 0.5 \\end{bmatrix}");

    // Entries go through the Real spelling, so a matrix cannot quietly round: the same
    // round trip an atom gets applies to a cell.
    const ms::Matrix<double> exact{{1.0 / 3.0}};
    EXPECT_EQ(to_latex(exact), "\\begin{pmatrix} 0.3333333333333333 \\end{pmatrix}");
}

// --- Options that change the spelling and not the value -------------------------------

TEST(Sym2NotationLatex, TheDecimalSeparatorAppliesToRealsOnly) {
    NotationOptions european;
    european.decimal_separator = ',';
    // A comma in math mode is punctuation and is set with a space after it, so the naive
    // `2,5` reaches the page as `2, 5` -- a list of two numbers where one number was
    // meant. Bracing it makes it an ordinary symbol again.
    EXPECT_EQ(to_latex(real(2.5), european), "2{,}5");
    // The separator between arguments stays a comma whatever the option says, because
    // `f(1,5, 2,5)` cannot be read back by anything.
    EXPECT_EQ(to_latex(function("f", {real(1.5), real(2.5)}), european),
              "\\operatorname{f}(1{,}5, 2{,}5)");
    // An exact rational has no decimal point to move, and must not acquire one.
    EXPECT_EQ(to_latex(frac(1, 2), european), "\\frac{1}{2}");
}

TEST(Sym2NotationLatex, TheDocumentWrapperIsCompilable) {
    const std::string document = latex_document(to_latex(symbol("x")));
    EXPECT_NE(document.find("\\usepackage{amsmath}"), std::string::npos);
    EXPECT_NE(document.find("\\[\nx\n\\]"), std::string::npos);
}

TEST(Sym2NotationLatex, TheEnumAndNameDispatchReachTheSameTable) {
    const ExprRef e = div(symbol("x"), symbol("y"));
    EXPECT_EQ(to_notation(e, Notation::Latex), to_latex(e));
    const auto named = notation_from_name("LaTeX");
    ASSERT_TRUE(named.has_value());
    EXPECT_EQ(*named, Notation::Latex);
    EXPECT_EQ(notation_name(Notation::Latex), "latex");
}

// --- The plan's worked example --------------------------------------------------------

TEST(Sym2NotationLatex, TheWorkedExample) {
    // 2*x^2 - 3/y. Three of this file's themes at once: the subtraction is a term the
    // walker found under a -1 coefficient, the quotient is a factor it found under a
    // negative exponent, and the exponent is braced.
    const ExprRef e = add({mul({integer(2), pow(symbol("x"), integer(2))}),
                           neg(div(integer(3), symbol("y")))});
    EXPECT_EQ(to_latex(e), "-\\frac{3}{y} + 2 \\cdot x^{2}");

    NotationOptions juxtaposed;
    juxtaposed.multiplication = NotationOptions::Multiplication::Juxtaposition;
    EXPECT_EQ(to_latex(e, juxtaposed), "-\\frac{3}{y} + 2x^{2}");

    NotationOptions display;
    display.display = true;
    EXPECT_EQ(to_latex(e, display), "-\\dfrac{3}{y} + 2 \\cdot x^{2}");
}

// A value in scientific notation is a product, not a numeral, and a product in a
// superscript slot has to be grouped. Left ungrouped it is `1 \times 10^{20}^{n}`,
// which LaTeX rejects outright ("Double superscript") rather than setting wrongly.
//
// The exponent has to be symbolic to reach this at all: `pow(real(1e20), integer(2))`
// is folded to `real(1e40)` by the constructor, so a numeric exponent never leaves a
// Real as the base of a Pow. Writing the test with one asserted against an expression
// that cannot exist.
TEST(LatexNotation, ScientificNotationIsGroupedWhereAProductWouldBe) {
    EXPECT_EQ(to_latex(pow(real(1e20), symbol("n"))), "(1 \\times 10^{20})^{n}");
    // In a sum it needs nothing, because a product binds tighter than a sum.
    EXPECT_EQ(to_latex(add({symbol("x"), real(1e20)})), "x + 1 \\times 10^{20}");
    // A value that is not in scientific notation is still an atom.
    EXPECT_EQ(to_latex(pow(real(2.5), symbol("n"))), "2.5^{n}");
    // And the folding above is itself worth pinning: the printer never sees the power.
    EXPECT_EQ(to_latex(pow(real(1e20), integer(2))), "1 \\times 10^{40}");
}

} // namespace
