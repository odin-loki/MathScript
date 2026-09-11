// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The ASCII, Unicode, SymPy, Wolfram and source tables, at the level the round-trip
// property cannot reach.
//
// `test_sym2_notation_roundtrip` checks that every notation is total, deterministic,
// balanced and re-readable. None of that pins a *spelling*: mangled bytes still
// balance, and a table that emitted `1/3` where SymPy needs `Rational(1, 3)` would
// satisfy every structural property while handing a Python reader a float. So the
// spellings are asserted here, and the ones asserted hardest are those where the
// obvious string parses to a different expression than the one printed.

#include <string>

#include <gtest/gtest.h>

#include "ms/sym2/expr.hpp"
#include "ms/sym2/notation.hpp"

using ms::bignum::BigInt;
using namespace ms::sym2;

namespace {

ExprRef frac(long long num, long long den) { return rational(BigInt(num), BigInt(den)); }

// --- ASCII ----------------------------------------------------------------------------

// The ASCII table is what MathScript reads back, so these are also the shapes the
// REPL's own parser has to accept.
TEST(TextNotation, AsciiIsTheExpressionMathScriptWouldWrite) {
    const ExprRef x = symbol("x");
    const ExprRef y = symbol("y");
    EXPECT_EQ(to_notation(add({mul({integer(2), x}), integer(1)}), Notation::Ascii), "2*x + 1");
    EXPECT_EQ(to_notation(mul({add({x, integer(1)}), y}), Notation::Ascii), "(x + 1)*y");
    EXPECT_EQ(to_notation(sub(x, y), Notation::Ascii), "x - y");
    EXPECT_EQ(to_notation(neg(x), Notation::Ascii), "-x");
    EXPECT_EQ(to_notation(pow(x, integer(-1)), Notation::Ascii), "1/x");
    EXPECT_EQ(to_notation(frac(7, 3), Notation::Ascii), "7/3");
    EXPECT_EQ(to_notation(function("sin", {mul({integer(2), x})}), Notation::Ascii), "sin(2*x)");
    // A call fences its argument, so a sum under a square root needs no parentheses of
    // its own -- but a radical sign would, which is why the two tables answer the
    // walker differently about it.
    EXPECT_EQ(to_notation(pow(add({x, integer(1)}), frac(1, 2)), Notation::Ascii),
              "sqrt(x + 1)");
}

// --- Unicode --------------------------------------------------------------------------

TEST(TextNotation, UnicodeUsesRealGlyphs) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_notation(pow(x, integer(2)), Notation::Unicode), "x²");
    EXPECT_EQ(to_notation(pow(x, integer(3)), Notation::Unicode), "x³");
    EXPECT_EQ(to_notation(mul({integer(2), x}), Notation::Unicode), "2·x");
    EXPECT_EQ(to_notation(constant("pi"), Notation::Unicode), "π");
    EXPECT_EQ(to_notation(constant("inf"), Notation::Unicode), "∞");
    EXPECT_EQ(to_notation(symbol("alpha"), Notation::Unicode), "α");
    EXPECT_EQ(to_notation(symbol("Omega"), Notation::Unicode), "Ω");
    EXPECT_EQ(to_notation(symbol("x_1"), Notation::Unicode), "x₁");
    // A subscript that is not all digits has no Unicode form, so it keeps the
    // underscore rather than being partly transcribed.
    EXPECT_EQ(to_notation(symbol("x_max"), Notation::Unicode), "x_max");
}

// A radical sign in text has no vinculum, so it does not group what follows it. `√x + 1`
// is the square root of x, plus one. This is the assertion that would have caught the
// table answering the walker wrongly about fencing -- and the value, not the look, is
// what would have been wrong.
TEST(TextNotation, AUnicodeRadicalGroupsItsArgument) {
    EXPECT_EQ(to_notation(pow(add({symbol("x"), integer(1)}), frac(1, 2)), Notation::Unicode),
              "√(x + 1)");
    EXPECT_EQ(to_notation(pow(symbol("x"), frac(1, 2)), Notation::Unicode), "√x");
    EXPECT_EQ(to_notation(pow(symbol("x"), frac(1, 3)), Notation::Unicode), "∛x");
    // No glyph for a fifth root, so it falls back to a call rather than to a sign that
    // would read as the square root of five.
    EXPECT_EQ(to_notation(pow(symbol("x"), frac(1, 5)), Notation::Unicode), "root(x, 5)");
}

// A superscript is only used when every character of the exponent has one. A partial
// transcription would silently drop the characters that do not.
TEST(TextNotation, ASuperscriptIsAllOrNothing) {
    EXPECT_EQ(to_notation(pow(symbol("x"), symbol("n")), Notation::Unicode), "x^n");
    EXPECT_EQ(to_notation(pow(symbol("x"), integer(12)), Notation::Unicode), "x¹²");
    EXPECT_EQ(to_notation(pow(symbol("x"), add({symbol("n"), integer(1)})), Notation::Unicode),
              "x^(n + 1)");
}

// --- SymPy against Wolfram --------------------------------------------------------------

// The two traps, side by side, because they are opposites. In Python `1/3` is a float
// and the reader's expression stops being the printed one; in Wolfram Language the same
// three characters are exact.
TEST(TextNotation, AnExactRationalSurvivesBothReaders) {
    EXPECT_EQ(to_notation(frac(1, 3), Notation::SymPy), "Rational(1, 3)");
    EXPECT_EQ(to_notation(frac(1, 3), Notation::Mathematica), "1/3");
    // As a coefficient it is a numerator and a denominator by the time the table sees
    // it, so the quotient spelling is what both get -- and in Python `x/3` is true
    // division of a symbol, which stays symbolic.
    EXPECT_EQ(to_notation(mul({frac(1, 3), symbol("x")}), Notation::SymPy), "x/3");
    EXPECT_EQ(to_notation(mul({frac(1, 3), symbol("x")}), Notation::Mathematica), "x/3");
}

TEST(TextNotation, WolframAppliesWithBracketsAndGroupsWithParentheses) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_notation(function("sin", {x}), Notation::Mathematica), "Sin[x]");
    EXPECT_EQ(to_notation(pow(x, frac(1, 2)), Notation::Mathematica), "Sqrt[x]");
    // Parentheses are grouping only, and grouping is still needed.
    EXPECT_EQ(to_notation(pow(add({x, integer(1)}), integer(2)), Notation::Mathematica),
              "(x + 1)^2");
    EXPECT_EQ(to_notation(constant("pi"), Notation::Mathematica), "Pi");
    EXPECT_EQ(to_notation(constant("inf"), Notation::Mathematica), "Infinity");
    EXPECT_EQ(to_notation(constant("nan"), Notation::Mathematica), "Indeterminate");
}

// `e` is not an exponent marker in Wolfram Language input -- `1.5e-8` parses as a
// product involving a symbol named `e`, which evaluates to something else without
// complaint.
TEST(TextNotation, WolframScientificNotationUsesItsOwnMarker) {
    const std::string small = to_notation(real(1.5e-8), Notation::Mathematica);
    EXPECT_EQ(small, "1.5*^-8") << small;
    EXPECT_EQ(small.find('e'), std::string::npos) << small;
    EXPECT_EQ(to_notation(real(2.5), Notation::Mathematica), "2.5");
}

TEST(TextNotation, SympyUsesItsOwnNames) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_notation(pow(x, integer(2)), Notation::SymPy), "x**2");
    EXPECT_EQ(to_notation(constant("pi"), Notation::SymPy), "pi");
    EXPECT_EQ(to_notation(constant("inf"), Notation::SymPy), "oo");
    EXPECT_EQ(to_notation(constant("e"), Notation::SymPy), "E");
    // Python's builtin abs is not SymPy's and does not stay symbolic.
    EXPECT_EQ(to_notation(function("abs", {x}), Notation::SymPy), "Abs(x)");
    EXPECT_EQ(to_notation(derivative(pow(x, integer(2)), {x}), Notation::SymPy),
              "Derivative(x**2, x)");
}

// --- Source ------------------------------------------------------------------------------

// The trap that makes this table worth having: in C, `1/3` is `0`. An expression
// printed that way does not merely look different, it computes a different number.
TEST(SourceNotation, NoIntegerDivisionSurvivesIntoC) {
    EXPECT_EQ(to_source(frac(1, 3), SourceLanguage::C), "(1.0 / 3.0)");
    EXPECT_EQ(to_source(integer(2), SourceLanguage::C), "2.0");
    EXPECT_EQ(to_source(mul({integer(2), symbol("x")}), SourceLanguage::C), "2.0 * x");
    // format_exact gives "2" for 2.0, which is an int literal in C.
    EXPECT_EQ(to_source(real(2.0), SourceLanguage::C), "2.0");
    EXPECT_EQ(to_source(real(2.5), SourceLanguage::C), "2.5");
}

TEST(SourceNotation, PowerIsACallWhereThereIsNoOperator) {
    const ExprRef x = symbol("x");
    EXPECT_EQ(to_source(pow(x, integer(3)), SourceLanguage::C), "pow(x, 3.0)");
    EXPECT_EQ(to_source(pow(x, integer(3)), SourceLanguage::Cpp), "std::pow(x, 3.0)");
    EXPECT_EQ(to_source(pow(x, integer(3)), SourceLanguage::Python), "x**3.0");
    // A call fences both arguments, so nothing inside needs grouping.
    EXPECT_EQ(to_source(pow(add({x, integer(1)}), integer(2)), SourceLanguage::C),
              "pow(x + 1.0, 2.0)");
    // Python's ** does not fence, so it does.
    EXPECT_EQ(to_source(pow(add({x, integer(1)}), integer(2)), SourceLanguage::Python),
              "(x + 1.0)**2.0");
}

// C's abs() takes an int, so abs(-1.5) truncates to 1 before it is called: a wrong
// answer that compiles without a warning.
TEST(SourceNotation, AbsBecomesTheFloatingPointOne) {
    EXPECT_EQ(to_source(function("abs", {symbol("x")}), SourceLanguage::C), "fabs(x)");
    EXPECT_EQ(to_source(function("abs", {symbol("x")}), SourceLanguage::Cpp), "std::fabs(x)");
    EXPECT_EQ(to_source(function("abs", {symbol("x")}), SourceLanguage::Python), "abs(x)");
    EXPECT_EQ(to_source(function("sin", {symbol("x")}), SourceLanguage::Python), "math.sin(x)");
}

// A derivative has no source form. Emitting a plausible call would be a value a reader
// would take for an answer and is not one -- the defect class the audits were about --
// so what comes out does not compile and names the problem.
TEST(SourceNotation, SomethingUnevaluatedRefusesToCompile) {
    const std::string text = to_source(derivative(symbol("x"), {symbol("x")}), SourceLanguage::C);
    EXPECT_NE(text.find("ms_unevaluated_derivative"), std::string::npos) << text;
    const std::string imaginary = to_source(constant("i"), SourceLanguage::Cpp);
    EXPECT_NE(imaginary.find("ms_unevaluated_imaginary_unit"), std::string::npos) << imaginary;
    // The point is that neither is a number: a reader cannot mistake it for one.
    EXPECT_EQ(text.find_first_of("0123456789"), std::string::npos) << text;
}

// A MathScript name is not necessarily an identifier anywhere here.
TEST(SourceNotation, ANameThatIsNotAnIdentifierBecomesOne) {
    EXPECT_EQ(to_source(symbol("int"), SourceLanguage::C), "ms_int");
    EXPECT_EQ(to_source(symbol("lambda"), SourceLanguage::Python), "ms_lambda");
    EXPECT_EQ(to_source(symbol("x_1"), SourceLanguage::C), "x_1");
    // Anything illegal is escaped reversibly rather than dropped.
    const std::string odd = to_source(symbol("a.b"), SourceLanguage::C);
    EXPECT_EQ(odd, "a_x2eb") << odd;
}

TEST(SourceNotation, TheVariableArrayMapsOnlyIndexedNames) {
    NotationOptions options;
    options.source_variable_array = "v";
    EXPECT_EQ(to_source(symbol("x0"), SourceLanguage::C, options), "v[0]");
    EXPECT_EQ(to_source(symbol("x12"), SourceLanguage::C, options), "v[12]");
    // A name with no index has no slot, and inventing one would put two different
    // symbols in the same place on two different calls.
    EXPECT_EQ(to_source(symbol("y"), SourceLanguage::C, options), "y");
}

} // namespace
