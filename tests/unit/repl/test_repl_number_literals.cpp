// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Exponent literals in scalar arithmetic.
//
// The REPL's scalar expression evaluator splits a line at its lowest-precedence
// top-level operator. It found that operator by scanning for '+' and '-' without
// knowing that either can be the sign of an exponent, so `1e-09 * 2` was split at
// the minus into `1e` and `09 * 2` and reported as "could not parse". The same
// happened to `1e+16` on the plus.
//
// That mattered on its own, before any question of how numbers are printed: `vars`
// already prints small magnitudes in exponent form, so a session could show a value
// that it could not then accept back in an expression.

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

#include "repl_test_helpers.hpp"

using namespace ms::interp;

namespace {

double scalar_of(Interpreter& interp, const std::string& command) {
    const auto result = interp.execute(command);
    EXPECT_TRUE(result.has_value()) << command;
    if (!result.has_value()) {
        return std::nan("");
    }
    // "name = 1.5" or a bare "1.5"; take whatever follows the last '='.
    const std::string text = *result;
    const std::size_t eq = text.rfind('=');
    return std::stod(eq == std::string::npos ? text : text.substr(eq + 1));
}

}  // namespace

TEST(ReplNumberLiterals, ExponentLiteralsSurviveArithmetic) {
    // Every expression here is scaled so its result lands in a magnitude the REPL
    // can print faithfully. The printer is a separate defect -- it renders anything
    // below 5e-7 as 0.000000 -- and reading a tiny result back through it would
    // measure that, not the parse.
    Interpreter interp;
    struct Case {
        const char* expression;
        double expected;
    };
    const Case cases[] = {
        {"1e-09 * 2 * 1e9", 2.0},   {"2 * 1e-09 * 1e9", 2.0},  {"1e+16 / 1e16", 1.0},
        {"1.5e-3 * 4 * 1e3", 6.0},  {"2e3 - 1", 1999.0},       {"1e2 + 1e2", 200.0},
        {"1E-2 * 4", 0.04},         {"-1e-3 * 2 * 1e3", -2.0}, {"1e3 / 1e1", 100.0},
        {"1e+2 - 1", 99.0},         {"2e0 * 3", 6.0},
    };
    for (const Case& c : cases) {
        EXPECT_NEAR(scalar_of(interp, c.expression), c.expected,
                    1e-9 * std::max(1.0, std::abs(c.expected)))
            << c.expression;
    }
}

TEST(ReplNumberLiterals, LeadingUnarySignAppliesToOneOperandOnly) {
    // Both scalar evaluators tested `expr.front() == '-'` before looking for a
    // binary operator, so the sign was applied to everything that followed:
    // -4 + 1 evaluated to -(4 + 1) = -5, and with x = 4, -x + y came out as -6.
    Interpreter interp;
    expect_ok(interp, "x = 4");
    expect_ok(interp, "y = 2");
    struct Case {
        const char* expression;
        double expected;
    };
    const Case cases[] = {
        {"-4 + 1", -3.0},      {"-4 - 1", -5.0},      {"-4 + 1 + 2", -1.0},
        {"-x + 1", -3.0},      {"-x - 1", -5.0},      {"-x + y", -2.0},
        {"-x - y", -6.0},      {"-2 * 3", -6.0},      {"-2 * 3 + 1", -5.0},
        {"+4 - 1", 3.0},       {"-4 / 2 + 1", -1.0},  {"-x / 2 - 1", -3.0},
        // The spellings that were already right have to stay right.
        {"(-4) + 1", -3.0},    {"1 + -4", -3.0},      {"-(4+1)", -5.0},
        {"2 - -3", 5.0},       {"-4", -4.0},          {"-x", -4.0},
    };
    for (const Case& c : cases) {
        EXPECT_NEAR(scalar_of(interp, c.expression), c.expected, 1e-12) << c.expression;
    }
}

TEST(ReplNumberLiterals, OrdinarySubtractionIsUnaffected) {
    // The exponent guard must not swallow a real operator. These are the cases that
    // would break if it matched too eagerly.
    Interpreter interp;
    expect_ok(interp, "x = 5");
    struct Case {
        const char* expression;
        double expected;
    };
    const Case cases[] = {
        {"3 - 2", 1.0},        {"2*3-1", 5.0},        {"x - 1", 4.0},
        {"10 - 2 - 3", 5.0},   {"(2 - 5) * 2", -6.0}, {"1.5 - 0.5", 1.0},
        {"8 / 2 / 2", 2.0},    {"1 - 2 + 3", 2.0},
    };
    for (const Case& c : cases) {
        EXPECT_NEAR(scalar_of(interp, c.expression), c.expected, 1e-12) << c.expression;
    }
}

TEST(ReplNumberLiterals, ExponentLiteralsRoundTripThroughAVariable) {
    // The point of the fix: what the session prints, the session can read back.
    Interpreter interp;
    expect_ok(interp, "tiny = 1e-09");
    EXPECT_NEAR(scalar_of(interp, "tiny * 1e9"), 1.0, 1e-9);
    expect_ok(interp, "huge = 1e+16");
    EXPECT_NEAR(scalar_of(interp, "huge / 1e16"), 1.0, 1e-12);
}
