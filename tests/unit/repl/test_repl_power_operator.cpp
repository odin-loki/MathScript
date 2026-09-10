// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// `^` in the REPL's scalar expressions.
//
// The golden corpus of §8.3 found that it was not there: `2^3` did not parse, and
// `pow(2, 3)` was the only spelling -- while `^` *was* an operator in every symbolic
// command and in the matrix literal syntax, so the same character meant different
// things on adjacent lines of one session.
//
// Two things about it are worth testing rather than assuming, because both have a
// second reading that is also plausible and gives a different number:
//
//   - **`^` is right-associative.** 2^3^2 is 2^(3^2) = 512, not (2^3)^2 = 64.
//   - **Unary minus binds looser than `^`.** -2^2 is -(2^2) = -4, not (-2)^2 = 4. This
//     is the convention in mathematics and in every language with a power operator,
//     and the first implementation here got it wrong in exactly the other direction.
//
// The same expressions are checked against `sym_parse`, which has had its own `^` all
// along. Two parsers in one program that disagree about what `-2^2` means is a defect
// whether or not either of them is wrong on its own.

#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "ms/interp/repl_engine.hpp"

#include "repl_test_helpers.hpp"

using namespace ms::interp;

namespace {

/// The value, read out of the interpreter's state rather than parsed back from what it
/// printed. The printed form is six decimals, so `2^0.5` comes back as 1.414214 and an
/// assertion made against it can only ever be as tight as the display -- which would
/// make this a test of `format_scalar` rather than of the operator.
double scalar(Interpreter& interp, const std::string& expression) {
    const auto result = interp.execute("ms_probe = " + expression);
    EXPECT_TRUE(result.has_value()) << expression;
    if (!result) {
        return std::nan("");
    }
    const auto found = interp.state().scalars.find("ms_probe");
    EXPECT_NE(found, interp.state().scalars.end()) << expression;
    return found == interp.state().scalars.end() ? std::nan("") : found->second;
}

TEST(ReplPowerOperator, ItExists) {
    Interpreter interp;
    EXPECT_NEAR(scalar(interp, "2^3"), 8.0, 1e-12);
    EXPECT_NEAR(scalar(interp, "2 ^ 3"), 8.0, 1e-12);
    expect_ok(interp, "x = 2");
    expect_ok(interp, "y = 3");
    EXPECT_NEAR(scalar(interp, "x^y"), 8.0, 1e-12);
    EXPECT_NEAR(scalar(interp, "2^0.5"), std::sqrt(2.0), 1e-12);
    // And it agrees with the call form that used to be the only one.
    EXPECT_NEAR(scalar(interp, "pow(2, 3)"), scalar(interp, "2^3"), 1e-12);
}

TEST(ReplPowerOperator, ItIsRightAssociative) {
    Interpreter interp;
    EXPECT_NEAR(scalar(interp, "2^3^2"), 512.0, 1e-9) << "2^(3^2) = 2^9";
    EXPECT_NEAR(scalar(interp, "(2^3)^2"), 64.0, 1e-9);
    EXPECT_NEAR(scalar(interp, "2^2^3"), 256.0, 1e-9);
}

// The case the first implementation got backwards. Both readings evaluate, and they
// differ by a sign, so nothing but an assertion distinguishes them.
TEST(ReplPowerOperator, UnaryMinusBindsLooserThanThePower) {
    Interpreter interp;
    EXPECT_NEAR(scalar(interp, "-2^2"), -4.0, 1e-12) << "-(2^2), not (-2)^2";
    EXPECT_NEAR(scalar(interp, "(-2)^2"), 4.0, 1e-12);
    expect_ok(interp, "x = 2");
    EXPECT_NEAR(scalar(interp, "-x^2"), -4.0, 1e-12);
    EXPECT_NEAR(scalar(interp, "-2^3"), -8.0, 1e-12);
    // The additive level is still resolved before the sign is taken as unary, which is
    // the defect this ordering had to preserve the fix for: -4 + 1 is -3, not -5.
    EXPECT_NEAR(scalar(interp, "-4 + 1"), -3.0, 1e-12);
    EXPECT_NEAR(scalar(interp, "-2^2 + 1"), -3.0, 1e-12);
}

// A minus straight after the caret is the exponent's sign, not an operator.
TEST(ReplPowerOperator, ANegativeExponentIsASign) {
    Interpreter interp;
    EXPECT_NEAR(scalar(interp, "2^-1"), 0.5, 1e-12);
    EXPECT_NEAR(scalar(interp, "2^-2"), 0.25, 1e-12);
    EXPECT_NEAR(scalar(interp, "10^-3"), 0.001, 1e-12);
}

TEST(ReplPowerOperator, ItBindsTighterThanMultiplicationAndDivision) {
    Interpreter interp;
    EXPECT_NEAR(scalar(interp, "2*3^2"), 18.0, 1e-12);
    EXPECT_NEAR(scalar(interp, "3^2*2"), 18.0, 1e-12);
    EXPECT_NEAR(scalar(interp, "1/2^2"), 0.25, 1e-12);
    EXPECT_NEAR(scalar(interp, "2^3*2"), 16.0, 1e-12);
    EXPECT_NEAR(scalar(interp, "1 + 2^3"), 9.0, 1e-12);
    EXPECT_NEAR(scalar(interp, "2^3 + 1"), 9.0, 1e-12);
}

// std::pow answers both of these with a number -- NaN and infinity -- and a NaN that
// reaches the display prints as a value. They are named instead.
TEST(ReplPowerOperator, TheTwoUndefinedCasesAreReported) {
    Interpreter interp;
    expect_error_contains(interp, "(-8)^0.5", "negative base");
    expect_error_contains(interp, "0^-1", "zero to a negative power");
    // A negative base under an integer exponent is perfectly well defined.
    EXPECT_NEAR(scalar(interp, "(-2)^3"), -8.0, 1e-12);
}

// Two parsers in one program that disagree about `-2^2` is a defect whichever of them
// is right, so the agreement is asserted rather than left to inspection.
TEST(ReplPowerOperator, TheSymbolicParserAgrees) {
    Interpreter interp;
    expect_contains(interp, "sym_simplify(\"-2^2\")", "-4");
    expect_contains(interp, "sym_simplify(\"2^3^2\")", "512");
    expect_contains(interp, "sym_eval(\"-x^2\", \"x=2\")", "-4");
}

} // namespace
