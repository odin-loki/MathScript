// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// The unsupported sentinel, and why it needs checking.
//
// Nine symbolic functions signal "no closed form" by returning sym_deriv(input,
// var). The convention is documented, but the sentinel is not inert: sym_eval
// evaluates a Deriv node by differentiating it. So a caller who integrates and then
// evaluates gets the derivative's value where the integral was asked for, and
// nothing anywhere reports a failure.
//
// These tests pin both halves: that the sentinel is detectable, and that the thing
// it protects against is real.

#include <gtest/gtest.h>

#include <cmath>
#include <map>

#include "ms/symbolic/symbolic.hpp"

using namespace ms;

TEST(SymbolicUnsupported, StructuralEqualityDistinguishesShapeValueAndName) {
    EXPECT_TRUE(sym_equal(sym_const(2.0), sym_const(2.0)));
    EXPECT_FALSE(sym_equal(sym_const(2.0), sym_const(3.0)));
    EXPECT_TRUE(sym_equal(sym_var("x"), sym_var("x")));
    EXPECT_FALSE(sym_equal(sym_var("x"), sym_var("y")));
    EXPECT_TRUE(sym_equal(sym_add(sym_var("x"), sym_const(1.0)),
                          sym_add(sym_var("x"), sym_const(1.0))));
    // Structural, not mathematical: these are equal in value and not in shape.
    EXPECT_FALSE(sym_equal(sym_add(sym_var("x"), sym_const(0.0)), sym_var("x")));
    // Operand order matters for the same reason.
    EXPECT_FALSE(sym_equal(sym_sub(sym_var("x"), sym_var("y")),
                           sym_sub(sym_var("y"), sym_var("x"))));
    // Same children, different operator.
    EXPECT_FALSE(sym_equal(sym_add(sym_var("x"), sym_var("y")),
                           sym_mul(sym_var("x"), sym_var("y"))));
}

TEST(SymbolicUnsupported, DetectsTheSentinelAndNothingElse) {
    // An input the table does not cover.
    auto hard = sym_log(sym_log(sym_var("x")));
    const auto declined = sym_integrate(hard, "x");
    EXPECT_TRUE(sym_is_unsupported(declined, "x"));

    // A supported input: the result is a real integral, not the sentinel.
    auto easy = sym_pow(sym_var("x"), sym_const(2.0));
    const auto solved = sym_integrate(easy, "x");
    EXPECT_FALSE(sym_is_unsupported(solved, "x"));

    // The right shape but the wrong variable is not this input's sentinel.
    EXPECT_FALSE(sym_is_unsupported(declined, "y"));
    // Nor is a derivative of something else.
    // A Deriv node whose operand is not the input is still a sentinel. It can only
    // have come from one -- SymOp::Deriv is built in one translation unit and only as
    // this marker -- and requiring it to match the input is exactly what let a
    // sentinel produced by one operand of a sum pass as an answer.
    EXPECT_TRUE(sym_is_unsupported(sym_deriv(sym_var("z"), "x"), "x"));
}

TEST(SymbolicUnsupported, TheSentinelEvaluatesToTheDerivativeIfNotChecked) {
    // This is the hazard, stated as a test rather than as a comment. An unchecked
    // sentinel does not produce a NaN or a zero that a caller might notice -- it
    // produces the derivative's value, which is a plausible number of the right
    // magnitude and the wrong quantity entirely.
    // 1/x integrates now, so the hazard needs an integrand that genuinely has no
    // elementary antiderivative. exp(x^2) is the canonical one.
    auto f = sym_exp(sym_pow(sym_var("x"), sym_const(2.0)));
    const auto declined = sym_integrate(f, "x");
    ASSERT_TRUE(sym_is_unsupported(declined, "x"));

    const std::map<std::string, double> at{{"x", 2.0}};
    const double evaluated = sym_eval(declined, at);
    // d/dx exp(x^2) = 2x*exp(x^2), which at x = 2 is 4*e^4.
    EXPECT_NEAR(evaluated, 4.0 * std::exp(4.0), 1e-9)
        << "the sentinel evaluates as d/dx(exp(x^2))";
    EXPECT_TRUE(std::isfinite(evaluated))
        << "and it is a plausible finite number, not a NaN a caller would notice";
}

TEST(SymbolicUnsupported, LaplaceHandlesNegativeRates) {
    // A negative literal parses as Neg(Const), never Const(-v), and every transform
    // table matched constants with an `op == SymOp::Const` test -- so no table entry
    // could match a negative coefficient. L{exp(-3t)} = 1/(s+3) is the decaying
    // exponential, which is most of the practical use of a Laplace transform.
    auto decay = sym_exp(sym_mul(sym_neg(sym_const(3.0)), sym_var("t")));
    const auto L = sym_laplace(decay, "t", "s");
    ASSERT_FALSE(sym_is_unsupported(L, "t"))
        << "L{exp(-3t)} declined: " << sym_to_string(L);

    // 1/(s - (-3)) evaluated at s = 1 is 1/4.
    const std::map<std::string, double> at{{"s", 1.0}};
    EXPECT_NEAR(sym_eval(L, at), 0.25, 1e-12);
}

TEST(SymbolicUnsupported, InverseLaplaceReadsBackTheForwardTable) {
    // The round trip is the property that caught these: the forward transform
    // emitted 1/s, 1/(s+a) and 1/s^n, and the inverse could not read any of them.
    struct Case { SymExpr f; const char* label; };
    std::vector<Case> cases;
    cases.push_back({sym_const(1.0), "1"});
    cases.push_back({sym_var("t"), "t"});
    cases.push_back({sym_pow(sym_var("t"), sym_const(2.0)), "t^2"});
    cases.push_back({sym_exp(sym_mul(sym_const(2.0), sym_var("t"))), "exp(2t)"});
    cases.push_back({sym_exp(sym_mul(sym_neg(sym_const(3.0)), sym_var("t"))), "exp(-3t)"});

    for (const auto& c : cases) {
        const auto L = sym_laplace(c.f, "t", "s");
        ASSERT_FALSE(sym_is_unsupported(L, "t")) << "forward declined " << c.label;
        const auto back = sym_ilaplace(L, "s", "t");
        ASSERT_FALSE(sym_is_unsupported(back, "s"))
            << "inverse could not read back its own forward output for " << c.label
            << ": L = " << sym_to_string(L);
        // Compare numerically rather than structurally: the round trip is allowed to
        // return a differently-shaped expression for the same function.
        const std::map<std::string, double> at{{"t", 0.7}};
        EXPECT_NEAR(sym_eval(back, at), sym_eval(c.f, at), 1e-9)
            << c.label << " did not survive the round trip";
    }
}

TEST(SymbolicUnsupported, InverseLaplaceScalesThePowerRule) {
    // 2/s^3 was accepted and 1/s^3 declined -- the same table entry, differently
    // scaled, because the numerator had to equal n! exactly.
    auto expr = sym_div(sym_const(1.0), sym_pow(sym_var("s"), sym_const(3.0)));
    const auto back = sym_ilaplace(expr, "s", "t");
    ASSERT_FALSE(sym_is_unsupported(back, "s"));
    const std::map<std::string, double> at{{"t", 2.0}};
    // L^-1{1/s^3} = t^2/2, which is 2 at t = 2.
    EXPECT_NEAR(sym_eval(back, at), 2.0, 1e-12);
}
