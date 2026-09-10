// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// §11.1's two MathML tables, asserted on exact strings.
//
// Exact strings rather than properties, because a notation *is* its exact strings: there
// is no weaker statement of "this is spelled correctly" available. What the properties
// would have covered -- precedence, display order, which factors are a denominator -- is
// the walker's, and is tested where the walker lives.
//
// The cases are grouped by what would go wrong if the spelling were the naive one, since
// that is what these are for. Markup is forgiving: nearly every wrong answer below is
// still well-formed MathML that a browser will render without complaint, and several of
// them render something that looks close enough to be believed.

#include <gtest/gtest.h>

#include <limits>
#include <string>

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

// --- Atoms ---------------------------------------------------------------------------

TEST(MathmlPresentation, Numbers) {
    EXPECT_EQ(to_presentation_mathml(integer(42)), "<mn>42</mn>");
    // A negative number's sign is an operator carrying U+2212 MINUS SIGN, not the
    // hyphen-minus that `BigInt::to_string` produces. `<mn>-5</mn>` is well-formed and
    // renders a hyphen, which is a narrower character sitting at a different height from
    // the minus in the `+` it has to line up with.
    EXPECT_EQ(to_presentation_mathml(integer(-5)), "<mrow><mo>&#x2212;</mo><mn>5</mn></mrow>");
    EXPECT_EQ(to_presentation_mathml(integer(BigInt("123456789012345678901234567890"))),
              "<mn>123456789012345678901234567890</mn>");
}

TEST(MathmlPresentation, ExactRationalKeepsItsSignOutsideTheBar) {
    EXPECT_EQ(to_presentation_mathml(frac(3, 4)), "<mfrac><mn>3</mn><mn>4</mn></mfrac>");
    // The minus belongs outside the fraction, not above the bar: a fraction whose
    // numerator is a negative number is a different thing to look at from the negation
    // of a fraction, and it is the second one that was meant.
    EXPECT_EQ(to_presentation_mathml(frac(-1, 3)),
              "<mrow><mo>&#x2212;</mo><mfrac><mn>1</mn><mn>3</mn></mfrac></mrow>");
}

TEST(MathmlPresentation, RealCarriesEveryDigitItNeeds) {
    // One third as a double needs sixteen significant digits to read back as itself.
    // This is an expression, not a display, so it may not be rounded for looks.
    EXPECT_EQ(to_presentation_mathml(real(1.0 / 3.0)), "<mn>0.3333333333333333</mn>");
    EXPECT_EQ(to_presentation_mathml(real(3.141592653589793)), "<mn>3.141592653589793</mn>");
    EXPECT_EQ(to_presentation_mathml(real(0.5)), "<mn>0.5</mn>");
}

TEST(MathmlPresentation, ExponentNotationIsTypesetRatherThanPrinted) {
    // `ms::format_exact` reaches for `%g`'s exponent form at a million already, so a
    // `<mn>` holding its output verbatim would print the literal characters `1e+06`.
    // That is source code; a presentation format exists to set it as a power of ten.
    EXPECT_EQ(to_presentation_mathml(real(1000000.0)),
              "<mrow><mn>1</mn><mo>&#xD7;</mo><msup><mn>10</mn><mn>6</mn></msup></mrow>");
    EXPECT_EQ(to_presentation_mathml(real(-1.5e-7)),
              "<mrow><mo>&#x2212;</mo><mn>1.5</mn><mo>&#xD7;</mo>"
              "<msup><mn>10</mn><mrow><mo>&#x2212;</mo><mn>7</mn></mrow></msup></mrow>");
}

TEST(MathmlPresentation, DecimalSeparatorReachesTheDigitsAndNothingElse) {
    NotationOptions options;
    options.decimal_separator = ',';
    // `<mn>` renders its content literally, so a comma in it is a comma on the page.
    EXPECT_EQ(to_presentation_mathml(real(0.5), options), "<mn>0,5</mn>");
}

TEST(MathmlPresentation, SymbolsSplitAndSpellTheirLetters) {
    EXPECT_EQ(to_presentation_mathml(symbol("x")), "<mi>x</mi>");
    // A numeric subscript is a number, so `<mn>`; a worded one is not.
    EXPECT_EQ(to_presentation_mathml(symbol("x_1")), "<msub><mi>x</mi><mn>1</mn></msub>");
    EXPECT_EQ(to_presentation_mathml(symbol("x_max")), "<msub><mi>x</mi><mi>max</mi></msub>");
    EXPECT_EQ(to_presentation_mathml(symbol("alpha")), "<mi>&#x3B1;</mi>");
    EXPECT_EQ(to_presentation_mathml(symbol("Omega")), "<mi>&#x3A9;</mi>");
    EXPECT_EQ(to_presentation_mathml(symbol("alpha_1")),
              "<msub><mi>&#x3B1;</mi><mn>1</mn></msub>");
    // U+210F, which `greek_letter` does not report because it is not Greek.
    EXPECT_EQ(to_presentation_mathml(symbol("hbar")), "<mi>&#x210F;</mi>");
}

TEST(MathmlPresentation, GreekVariantsFollowTheirTexNames) {
    // The two pairs that read backwards in English: TeX's \epsilon is the lunate U+03F5
    // and \varepsilon the ordinary U+03B5, and \phi is U+03D5 against \varphi's U+03C6.
    // Swapping either pair gives a valid document showing a different symbol.
    EXPECT_EQ(to_presentation_mathml(symbol("epsilon")), "<mi>&#x3F5;</mi>");
    EXPECT_EQ(to_presentation_mathml(symbol("varepsilon")), "<mi>&#x3B5;</mi>");
    EXPECT_EQ(to_presentation_mathml(symbol("phi")), "<mi>&#x3D5;</mi>");
    EXPECT_EQ(to_presentation_mathml(symbol("varphi")), "<mi>&#x3C6;</mi>");
}

TEST(MathmlPresentation, NamedConstants) {
    EXPECT_EQ(to_presentation_mathml(constant("pi")), "<mi>&#x3C0;</mi>");
    // Euler's number and the imaginary unit look like a sloped `e` and `i` in print, and
    // a one-character `<mi>` is drawn sloped. They are therefore indistinguishable from
    // the symbols of the same name -- which is right, because they are indistinguishable
    // on paper. Content MathML is where that difference is carried.
    EXPECT_EQ(to_presentation_mathml(constant("e")), "<mi>e</mi>");
    EXPECT_EQ(to_presentation_mathml(constant("e")), to_presentation_mathml(symbol("e")));
    EXPECT_EQ(to_presentation_mathml(constant("i")), "<mi>i</mi>");
    EXPECT_EQ(to_presentation_mathml(constant("inf")), "<mi>&#x221E;</mi>");
    EXPECT_EQ(to_presentation_mathml(constant("-inf")),
              "<mrow><mo>&#x2212;</mo><mi>&#x221E;</mi></mrow>");
    // Both are longer than one character, so MathML draws them upright without being
    // told to, which is what a word rather than a variable wants.
    EXPECT_EQ(to_presentation_mathml(constant("nan")), "<mi>NaN</mi>");
    EXPECT_EQ(to_presentation_mathml(constant("undefined")), "<mi>undefined</mi>");
}

TEST(MathmlPresentation, NonFiniteRealsAreSpelledAsTheConstantsTheyAre) {
    // `format_exact` falls back to `std::to_string` for these, giving the C library's
    // "inf" and "nan". A matrix entry is a Real, so this is reachable without anyone
    // constructing a Constant.
    EXPECT_EQ(to_presentation_mathml(real(std::numeric_limits<double>::infinity())),
              "<mi>&#x221E;</mi>");
    EXPECT_EQ(to_presentation_mathml(real(-std::numeric_limits<double>::infinity())),
              "<mrow><mo>&#x2212;</mo><mi>&#x221E;</mi></mrow>");
    EXPECT_EQ(to_presentation_mathml(real(std::numeric_limits<double>::quiet_NaN())),
              "<mi>NaN</mi>");
}

// --- Structure -----------------------------------------------------------------------

TEST(MathmlPresentation, SumsAndProducts) {
    const ExprRef x = symbol("x");
    const ExprRef y = symbol("y");
    // The signs and the terms are siblings in one `<mrow>`, which is what lets MathML
    // tell a prefix minus from an infix one and space them differently.
    EXPECT_EQ(to_presentation_mathml(sub(x, y)),
              "<mrow><mi>x</mi><mo>&#x2212;</mo><mi>y</mi></mrow>");
    EXPECT_EQ(to_presentation_mathml(add({x, neg(y), integer(1)})),
              "<mrow><mi>x</mi><mo>&#x2212;</mo><mi>y</mi><mo>+</mo><mn>1</mn></mrow>");
    EXPECT_EQ(to_presentation_mathml(mul({integer(2), x, y})),
              "<mrow><mn>2</mn><mo>&#x22C5;</mo><mi>x</mi><mo>&#x22C5;</mo><mi>y</mi></mrow>");
    EXPECT_EQ(to_presentation_mathml(neg(add({x, y}))),
              "<mrow><mo>&#x2212;</mo><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mi>y</mi>"
              "</mrow><mo>)</mo></mrow></mrow>");
}

TEST(MathmlPresentation, MultiplicationSignFollowsTheOption) {
    const ExprRef product = mul({integer(2), symbol("x")});
    NotationOptions options;
    options.multiplication = NotationOptions::Multiplication::Cross;
    EXPECT_EQ(to_presentation_mathml(product, options),
              "<mrow><mn>2</mn><mo>&#xD7;</mo><mi>x</mi></mrow>");
    // Juxtaposition is still an operator: U+2062 INVISIBLE TIMES prints nothing, and
    // printing nothing is the point -- but it is what makes `2x` a product rather than
    // two adjacent tokens, which is what a renderer spaces and a screen reader reads.
    options.multiplication = NotationOptions::Multiplication::Juxtaposition;
    EXPECT_EQ(to_presentation_mathml(product, options),
              "<mrow><mn>2</mn><mo>&#x2062;</mo><mi>x</mi></mrow>");
}

TEST(MathmlPresentation, QuotientsPowersAndRoots) {
    const ExprRef x = symbol("x");
    const ExprRef y = symbol("y");
    EXPECT_EQ(to_presentation_mathml(div(x, y)), "<mfrac><mi>x</mi><mi>y</mi></mfrac>");
    EXPECT_EQ(to_presentation_mathml(pow(x, integer(2))), "<msup><mi>x</mi><mn>2</mn></msup>");
    EXPECT_EQ(to_presentation_mathml(pow(x, pow(y, integer(2)))),
              "<msup><mi>x</mi><msup><mi>y</mi><mn>2</mn></msup></msup>");
    EXPECT_EQ(to_presentation_mathml(pow(x, frac(1, 2))), "<msqrt><mi>x</mi></msqrt>");
    // `<mroot>` takes the radicand first and the degree second, the reverse of the order
    // it is read aloud in. `<mroot><mn>3</mn><mi>x</mi></mroot>` is equally well-formed
    // and is the x-th root of three.
    EXPECT_EQ(to_presentation_mathml(pow(x, frac(1, 3))), "<mroot><mi>x</mi><mn>3</mn></mroot>");
    // A negative exponent is a reciprocal before it is a superscript -- the walker's
    // decision, and `<mfrac>` is how this table spells what it decided.
    EXPECT_EQ(to_presentation_mathml(pow(x, integer(-2))),
              "<mfrac><mn>1</mn><msup><mi>x</mi><mn>2</mn></msup></mfrac>");
    // Only 1/n is a root. `x^(2/3)` stays a power, and its fractional exponent needs no
    // grouping because `<msup>` takes exactly two children.
    EXPECT_EQ(to_presentation_mathml(pow(x, frac(2, 3))),
              "<msup><mi>x</mi><mfrac><mn>2</mn><mn>3</mn></mfrac></msup>");
}

TEST(MathmlPresentation, FencedElementsTakeNoParenthesesAndUnfencedSlotsDo) {
    const ExprRef sum = add({symbol("x"), integer(1)});
    const std::string rendered_sum = "<mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow>";
    // `<mfrac>` and `<msqrt>` hold their children directly, so a grouping inside one
    // would print a parenthesis a reader would take to be part of the expression.
    EXPECT_EQ(to_presentation_mathml(div(sum, add({symbol("y"), integer(2)}))),
              "<mfrac>" + rendered_sum + "<mrow><mi>y</mi><mo>+</mo><mn>2</mn></mrow></mfrac>");
    EXPECT_EQ(to_presentation_mathml(pow(sum, frac(1, 2))),
              "<msqrt>" + rendered_sum + "</msqrt>");
    // The base of a power and a factor of a product are not fenced by anything, so a sum
    // in either needs the parentheses drawn. Without them `x+1` squared is `x` plus one
    // squared, which is a different number.
    EXPECT_EQ(to_presentation_mathml(pow(sum, integer(2))),
              "<msup><mrow><mo>(</mo>" + rendered_sum + "<mo>)</mo></mrow><mn>2</mn></msup>");
    EXPECT_EQ(to_presentation_mathml(mul({sum, symbol("y")})),
              "<mrow><mrow><mo>(</mo>" + rendered_sum +
                  "<mo>)</mo></mrow><mo>&#x22C5;</mo><mi>y</mi></mrow>");
}

TEST(MathmlPresentation, ApplicationIsMarkedAsApplicationRatherThanLeftAdjacent) {
    // U+2061 FUNCTION APPLICATION between the name and the bracket is what says this is
    // a call. Leave it out and the markup says `sin` next to a bracketed `x`, which is a
    // product -- the exact ambiguity §11.2 gives as its first reason that presentation
    // markup cannot be parsed back. A renderer spaces the two differently and a screen
    // reader says two different sentences.
    EXPECT_EQ(to_presentation_mathml(function("sin", {symbol("x")})),
              "<mrow><mi>sin</mi><mo>&#x2061;</mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo>"
              "</mrow></mrow>");
    EXPECT_EQ(to_presentation_mathml(function("foo", {symbol("x"), symbol("y")})),
              "<mrow><mi>foo</mi><mo>&#x2061;</mo><mrow><mo>(</mo><mi>x</mi><mo>,</mo>"
              "<mi>y</mi><mo>)</mo></mrow></mrow>");
}

TEST(MathmlPresentation, UnevaluatedHeads) {
    const ExprRef x = symbol("x");
    const std::string sin_x =
        "<mrow><mi>sin</mi><mo>&#x2061;</mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>";
    EXPECT_EQ(to_presentation_mathml(derivative(function("sin", {x}), {x})),
              "<mrow><mfrac><mi>d</mi><mrow><mi>d</mi><mi>x</mi></mrow></mfrac>" + sin_x +
                  "</mrow>");
    EXPECT_EQ(to_presentation_mathml(integral(x, {x})),
              "<mrow><mo>&#x222B;</mo><mi>x</mi><mi>d</mi><mi>x</mi></mrow>");
    // `lim` is an `<mo>`, not an `<mi>`. The operator dictionary is where its upright
    // shape and the space after it come from; as an `<mi>` it would be an identifier
    // named "lim" sitting next to the body, spaced and read aloud as a product with it.
    EXPECT_EQ(to_presentation_mathml(limit(function("sin", {x}), x, integer(0))),
              "<mrow><munder><mo>lim</mo><mrow><mi>x</mi><mo>&#x2192;</mo><mn>0</mn></mrow>"
              "</munder>" + sin_x + "</mrow>");
}

TEST(MathmlPresentation, MatrixDelimitersFollowTheEnvironmentOption) {
    const ms::Matrix<double> m{{1.0, 2.0}, {3.0, 4.5}};
    const std::string table = "<mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr>"
                              "<mtr><mtd><mn>3</mn></mtd><mtd><mn>4.5</mn></mtd></mtr></mtable>";
    EXPECT_EQ(to_notation(m, Notation::PresentationMathml),
              "<mrow><mo>(</mo>" + table + "<mo>)</mo></mrow>");
    // The option is named for LaTeX, but what it actually names is the delimiters, and
    // MathML has no environments. Ignoring it here would leave one notation out of seven
    // quietly not honouring it.
    NotationOptions options;
    options.matrix_environment = "bmatrix";
    EXPECT_EQ(to_notation(m, Notation::PresentationMathml, options),
              "<mrow><mo>[</mo>" + table + "<mo>]</mo></mrow>");
}

TEST(MathmlPresentation, SymbolNamesAreEscaped) {
    // A name is arbitrary text. Unescaped, `<` opens an element that never closes and
    // the whole document stops being parseable -- and `&` is worse, because `a&b` is a
    // well-formed-looking entity reference that some parsers accept and others reject.
    EXPECT_EQ(to_presentation_mathml(symbol("a<b")), "<mi>a&lt;b</mi>");
    EXPECT_EQ(to_presentation_mathml(symbol("a&b")), "<mi>a&amp;b</mi>");
    EXPECT_EQ(to_presentation_mathml(function("a<b", {symbol("x")})),
              "<mrow><mi>a&lt;b</mi><mo>&#x2061;</mo><mrow><mo>(</mo><mi>x</mi><mo>)</mo>"
              "</mrow></mrow>");
}

// --- Content MathML ------------------------------------------------------------------

TEST(MathmlContent, Numbers) {
    EXPECT_EQ(to_content_mathml(integer(42)), "<cn>42</cn>");
    // A leading minus is inside the lexical form of a `<cn>`, so a negative integer
    // stays one number rather than becoming the negation of a positive one.
    EXPECT_EQ(to_content_mathml(integer(-5)), "<cn>-5</cn>");
    EXPECT_EQ(to_content_mathml(integer(BigInt("123456789012345678901234567890"))),
              "<cn>123456789012345678901234567890</cn>");
}

TEST(MathmlContent, RationalStaysASingleExactNumber) {
    // `<apply><divide/><cn>1</cn><cn>3</cn></apply>` denotes the same value and is the
    // wrong encoding: it says a division is to be performed, so a consumer that performs
    // it in floating point holds 0.333... and the exactness this tree exists to carry is
    // gone at the format boundary. `type="rational"` says the atom *is* the number.
    EXPECT_EQ(to_content_mathml(frac(1, 3)), "<cn type=\"rational\">1<sep/>3</cn>");
    EXPECT_EQ(to_content_mathml(frac(-1, 3)), "<cn type=\"rational\">-1<sep/>3</cn>");
    EXPECT_EQ(to_content_mathml(pow(symbol("x"), frac(2, 3))),
              "<apply><power/><ci>x</ci><cn type=\"rational\">2<sep/>3</cn></apply>");
}

TEST(MathmlContent, RealCarriesEveryDigitAndAnExponentGetsItsOwnType) {
    EXPECT_EQ(to_content_mathml(real(1.0 / 3.0)), "<cn>0.3333333333333333</cn>");
    // `<cn>1e+06</cn>` is outside the lexical space of a plain `<cn>`; MathML spells an
    // exponent with a type of its own and a `<sep/>` between the two halves.
    EXPECT_EQ(to_content_mathml(real(1000000.0)), "<cn type=\"e-notation\">1<sep/>6</cn>");
    EXPECT_EQ(to_content_mathml(real(-1.5e-7)), "<cn type=\"e-notation\">-1.5<sep/>-7</cn>");
}

TEST(MathmlContent, DecimalSeparatorIsNotAppliedBecauseItWouldNotBeANumber) {
    NotationOptions options;
    options.decimal_separator = ',';
    // The option is documented as a preference that may not alter the value, and here it
    // would: a `<cn>`'s content is a number in MathML's own lexical space, written with
    // a point. `<cn>0,5</cn>` does not denote a half. It does not denote a number.
    EXPECT_EQ(to_content_mathml(real(0.5), options), "<cn>0.5</cn>");
    EXPECT_EQ(to_content_mathml(real(0.5), options), to_content_mathml(real(0.5)));
}

TEST(MathmlContent, SymbolKeepsTheNameItWasGiven) {
    EXPECT_EQ(to_content_mathml(symbol("x")), "<ci>x</ci>");
    // Not U+03B1. Content MathML says what an expression means, and what this means is a
    // variable whose name is the six letters `alpha`; drawing the letter would hand a
    // consumer back a variable named with a Greek character, which is a different one.
    EXPECT_EQ(to_content_mathml(symbol("alpha")), "<ci>alpha</ci>");
    // The same argument for the subscript: `x_1` is one identifier. `<ci><msub>...`
    // is allowed by the spec and makes the name a tree that has to be flattened before
    // two of them can be compared.
    EXPECT_EQ(to_content_mathml(symbol("x_1")), "<ci>x_1</ci>");
    EXPECT_EQ(to_content_mathml(symbol("a<b")), "<ci>a&lt;b</ci>");
    EXPECT_EQ(to_content_mathml(symbol("a&b")), "<ci>a&amp;b</ci>");
}

TEST(MathmlContent, NamedConstants) {
    EXPECT_EQ(to_content_mathml(constant("pi")), "<pi/>");
    // Here the difference Presentation cannot carry is carried: Euler's number is an
    // element of its own, and the symbol `e` is not.
    EXPECT_EQ(to_content_mathml(constant("e")), "<exponentiale/>");
    EXPECT_EQ(to_content_mathml(symbol("e")), "<ci>e</ci>");
    EXPECT_EQ(to_content_mathml(constant("i")), "<imaginaryi/>");
    EXPECT_EQ(to_content_mathml(constant("inf")), "<infinity/>");
    // MathML has one infinity and no sign on it, so the sign has to be an operation.
    EXPECT_EQ(to_content_mathml(constant("-inf")), "<apply><minus/><infinity/></apply>");
    EXPECT_EQ(to_content_mathml(constant("nan")), "<notanumber/>");
    // Not `<notanumber/>`: NaN is a floating-point value that arithmetic produces and
    // propagates, where undefined is the absence of a value. MathML has no element for
    // the second, so a `csymbol` names it and says where its definition lives.
    EXPECT_EQ(to_content_mathml(constant("undefined")),
              "<csymbol cd=\"mathscript\">undefined</csymbol>");
    EXPECT_EQ(to_content_mathml(undefined()), "<csymbol cd=\"mathscript\">undefined</csymbol>");
}

TEST(MathmlContent, ApplyHasAFixedShapeSoNothingIsEverGrouped) {
    const ExprRef sum = add({symbol("x"), integer(1)});
    const std::string rendered_sum = "<apply><plus/><ci>x</ci><cn>1</cn></apply>";
    // `(x+1)^2` and `(x+1)*y` are where every other notation needs parentheses. Here the
    // nesting already says it, and a parenthesis would have to be written as a character
    // -- which in a semantic tree is a value, not a delimiter.
    EXPECT_EQ(to_content_mathml(pow(sum, integer(2))),
              "<apply><power/>" + rendered_sum + "<cn>2</cn></apply>");
    EXPECT_EQ(to_content_mathml(mul({sum, symbol("y")})),
              "<apply><times/>" + rendered_sum + "<ci>y</ci></apply>");
    EXPECT_EQ(to_content_mathml(neg(add({symbol("x"), symbol("y")}))),
              "<apply><minus/><apply><plus/><ci>x</ci><ci>y</ci></apply></apply>");
    EXPECT_EQ(to_content_mathml(pow(sum, integer(2))).find('('), std::string::npos);
}

TEST(MathmlContent, SumsProductsQuotientsAndPowers) {
    const ExprRef x = symbol("x");
    const ExprRef y = symbol("y");
    // A subtraction is `<plus/>` over a unary `<minus/>`. The binary
    // `<apply><minus/>a b</apply>` is the more idiomatic spelling of exactly two terms
    // and is not used, because `a - b - c` cannot take it: the general form has to exist
    // anyway, and one shape is one shape for a consumer to match.
    EXPECT_EQ(to_content_mathml(sub(x, y)),
              "<apply><plus/><ci>x</ci><apply><minus/><ci>y</ci></apply></apply>");
    EXPECT_EQ(to_content_mathml(add({x, y, symbol("z")})),
              "<apply><plus/><ci>x</ci><ci>y</ci><ci>z</ci></apply>");
    EXPECT_EQ(to_content_mathml(mul({integer(2), x, y})),
              "<apply><times/><cn>2</cn><ci>x</ci><ci>y</ci></apply>");
    EXPECT_EQ(to_content_mathml(div(x, y)), "<apply><divide/><ci>x</ci><ci>y</ci></apply>");
    EXPECT_EQ(to_content_mathml(pow(x, integer(2))),
              "<apply><power/><ci>x</ci><cn>2</cn></apply>");
    EXPECT_EQ(to_content_mathml(pow(x, pow(y, integer(2)))),
              "<apply><power/><ci>x</ci><apply><power/><ci>y</ci><cn>2</cn></apply></apply>");
    EXPECT_EQ(to_content_mathml(pow(x, integer(-2))),
              "<apply><divide/><cn>1</cn><apply><power/><ci>x</ci><cn>2</cn></apply></apply>");
}

TEST(MathmlContent, RootsPutTheDegreeQualifierFirst) {
    const ExprRef x = symbol("x");
    // A square root omits the degree rather than writing `<degree><cn>2</cn></degree>`,
    // because two is what `<root/>` already means.
    EXPECT_EQ(to_content_mathml(pow(x, frac(1, 2))), "<apply><root/><ci>x</ci></apply>");
    // The qualifier comes before the argument, which is MathML's order for qualifiers
    // and the reverse of what `<mroot>` wants above.
    EXPECT_EQ(to_content_mathml(pow(x, frac(1, 3))),
              "<apply><root/><degree><cn>3</cn></degree><ci>x</ci></apply>");
    // The same node whichever way it was written, which is the property that makes this
    // format the one worth reading back.
    EXPECT_EQ(to_content_mathml(function("sqrt", {x})), to_content_mathml(pow(x, frac(1, 2))));
}

TEST(MathmlContent, KnownFunctionsGetTheirElementAndLogIsNotTheBaseTenOne) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_content_mathml(function("sin", {x})), "<apply><sin/><ci>x</ci></apply>");
    EXPECT_EQ(to_content_mathml(function("abs", {x})), "<apply><abs/><ci>x</ci></apply>");
    EXPECT_EQ(to_content_mathml(function("asin", {x})), "<apply><arcsin/><ci>x</ci></apply>");
    EXPECT_EQ(to_content_mathml(function("ceil", {x})), "<apply><ceiling/><ci>x</ci></apply>");
    // The one worth reading twice. MathML's `<log/>` is base ten unless a `<logbase/>`
    // says otherwise, and `log` in this tree is the natural logarithm -- `evaluate`
    // sends it to `std::log`. The same-looking element would silently change the base of
    // every logarithm that went through here, and the result would still be valid MathML
    // that a consumer would happily evaluate to the wrong number.
    EXPECT_EQ(to_content_mathml(function("log", {x})), "<apply><ln/><ci>x</ci></apply>");
    // An unknown name is a `csymbol`, not a `ci`. `<ci>` means a variable, and applying
    // one says "whatever this variable holds, applied to these arguments" -- a different
    // claim from "the function this name denotes".
    EXPECT_EQ(to_content_mathml(function("foo", {x, symbol("y")})),
              "<apply><csymbol cd=\"mathscript\">foo</csymbol><ci>x</ci><ci>y</ci></apply>");
    EXPECT_EQ(to_content_mathml(function("a<b", {x})),
              "<apply><csymbol cd=\"mathscript\">a&lt;b</csymbol><ci>x</ci></apply>");
}

TEST(MathmlContent, UnevaluatedHeads) {
    const ExprRef x = symbol("x");
    const ExprRef y = symbol("y");
    EXPECT_EQ(to_content_mathml(derivative(function("sin", {x}), {x})),
              "<apply><diff/><bvar><ci>x</ci></bvar><apply><sin/><ci>x</ci></apply></apply>");
    // Several variables make it a partial derivative, which is the distinction MathML
    // draws between the two elements.
    EXPECT_EQ(to_content_mathml(derivative(function("f", {x, y}), {x, y})),
              "<apply><partialdiff/><bvar><ci>x</ci></bvar><bvar><ci>y</ci></bvar>"
              "<apply><csymbol cd=\"mathscript\">f</csymbol><ci>x</ci><ci>y</ci></apply></apply>");
    EXPECT_EQ(to_content_mathml(integral(x, {x})),
              "<apply><int/><bvar><ci>x</ci></bvar><ci>x</ci></apply>");
    // `<lowlimit>` rather than a `<condition>` holding a `<tendsto/>`: the condition form
    // exists to say how the variable approaches the point, and this tree records no
    // direction, so writing one would invent it.
    EXPECT_EQ(to_content_mathml(limit(function("sin", {x}), x, integer(0))),
              "<apply><limit/><bvar><ci>x</ci></bvar><lowlimit><cn>0</cn></lowlimit>"
              "<apply><sin/><ci>x</ci></apply></apply>");
}

TEST(MathmlContent, Matrix) {
    const ms::Matrix<double> m{{1.0, 2.0}, {3.0, 4.5}};
    EXPECT_EQ(to_notation(m, Notation::ContentMathml),
              "<matrix><matrixrow><cn>1</cn><cn>2</cn></matrixrow>"
              "<matrixrow><cn>3</cn><cn>4.5</cn></matrixrow></matrix>");
}

// --- The two halves side by side -------------------------------------------------------

TEST(MathmlNotation, TwoByXSquaredMinusThreeOverY) {
    // The expression §11.1 is checked against. Its display order is the walker's: the
    // node holds the two terms in canonical order, neither is a bare number, so neither
    // is moved, and `-3/y` sorts first. That is a structural decision and it belongs
    // there, not here.
    const ExprRef e = add({mul({integer(2), pow(symbol("x"), integer(2))}),
                           neg(div(integer(3), symbol("y")))});
    EXPECT_EQ(to_presentation_mathml(e),
              "<mrow><mo>&#x2212;</mo><mfrac><mn>3</mn><mi>y</mi></mfrac><mo>+</mo>"
              "<mrow><mn>2</mn><mo>&#x22C5;</mo><msup><mi>x</mi><mn>2</mn></msup></mrow></mrow>");
    EXPECT_EQ(to_content_mathml(e),
              "<apply><plus/><apply><minus/><apply><divide/><cn>3</cn><ci>y</ci></apply></apply>"
              "<apply><times/><cn>2</cn><apply><power/><ci>x</ci><cn>2</cn></apply></apply>"
              "</apply>");
    // `x*y^-1` became a denominator in both, which is the walker's decision made once
    // and spelled twice.
    EXPECT_NE(to_presentation_mathml(e).find("<mfrac>"), std::string::npos);
    EXPECT_NE(to_content_mathml(e).find("<divide/>"), std::string::npos);
}

TEST(MathmlNotation, DispatchByEnumAndTheDocumentWrapper) {
    const ExprRef e = pow(symbol("x"), integer(2));
    EXPECT_EQ(to_notation(e, Notation::PresentationMathml), to_presentation_mathml(e));
    EXPECT_EQ(to_notation(e, Notation::ContentMathml), to_content_mathml(e));
    // Neither table emits a `<math>` element or a namespace, so either output can be
    // nested inside another; the wrapper is a separate call for a caller writing a file.
    EXPECT_EQ(mathml_document(to_presentation_mathml(e)),
              "<math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"inline\">"
              "<msup><mi>x</mi><mn>2</mn></msup></math>");
}

} // namespace
