// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
//
// Counts that do not fit in 64 bits must be reported, not printed.
//
// ms::combo signals an unrepresentable count by returning UINT64_MAX. Its headers say
// so. The REPL did not read them: it cast the sentinel to double and printed it, so
// `combo_factorial(25)` answered 18446744073709551615 -- a twenty-digit number, where
// 25! is about 1.55e25. Nothing in the output said it was a marker rather than a
// count.
//
// The counting functions had a second problem underneath that one. binomial() advanced
// with r = r * (n - i) / (i + 1), whose intermediate product needs one bit more than
// the result, so it wrapped while the answer still fitted: C(67,33) is
// 14226520737620288370 and the loop returned 8829174638479413. permutations() and
// multinomial() wrapped the same way. Those are fixed at the source, so the values
// below are checked as values, and only the genuinely unrepresentable ones error.
//
// Every expected number here was computed in exact arithmetic outside this program.

#include <gtest/gtest.h>

#include <string>

#include "ms/interp/repl_engine.hpp"

#include "repl_test_helpers.hpp"

using namespace ms::interp;

namespace {

constexpr const char* kOverflowMessage = "does not fit in 64 bits";

// The sentinel spelled out, so a regression that prints it is caught by the text as
// well as by the missing error.
constexpr const char* kSentinelDigits = "18446744073709551615";

void expect_overflow(Interpreter& interp, const std::string& call) {
    // Both surfaces: the bare call that echoes a value, and the assignment that
    // stores one. They are separate dispatch paths in the engine.
    expect_error_contains(interp, call, kOverflowMessage);
    expect_error_contains(interp, "q = " + call, kOverflowMessage);
    EXPECT_EQ(interp.state().scalars.count("q"), 0U) << call;
}

void expect_count(Interpreter& interp, const std::string& call, const std::string& digits) {
    expect_contains(interp, call, digits);
    const auto echoed = interp.execute(call);
    ASSERT_TRUE(echoed.has_value()) << call;
    EXPECT_EQ(echoed->find(kSentinelDigits), std::string::npos) << call << " -> " << *echoed;
}

TEST(ReplComboOverflow, FactorialsStopAtTheEdgeOfTheType) {
    Interpreter interp;
    expect_count(interp, "combo_factorial(20)", "2432902008176640000");
    expect_overflow(interp, "combo_factorial(21)");
    expect_overflow(interp, "combo_factorial(25)");

    expect_count(interp, "combo_double_factorial(33)", "6332659870762850625");
    expect_overflow(interp, "combo_double_factorial(34)");

    expect_count(interp, "combo_subfactorial(20)", "895014631192902121");
    expect_overflow(interp, "combo_subfactorial(21)");
}

TEST(ReplComboOverflow, SequenceCountsStopAtTheirDocumentedLimits) {
    Interpreter interp;
    expect_count(interp, "combo_catalan(36)", "11959798385860453492");
    expect_overflow(interp, "combo_catalan(37)");

    expect_count(interp, "combo_bell(25)", "4638590332229999353");
    expect_overflow(interp, "combo_bell(26)");
    expect_overflow(interp, "combo_bell_num(26)");

    expect_count(interp, "combo_motzkin(45)", "13603677110519480289");
    expect_overflow(interp, "combo_motzkin(46)");

    expect_count(interp, "combo_involutions(31)", "3666624057550245376");
    expect_overflow(interp, "combo_involutions(32)");
}

TEST(ReplComboOverflow, TwoArgumentCountsStopToo) {
    Interpreter interp;
    expect_count(interp, "combo_stirling2(26, 13)", "1850568574253550060");
    expect_overflow(interp, "combo_stirling2(27, 13)");

    expect_count(interp, "combo_stirling1(21, 10)", "10142299865511450");
    expect_overflow(interp, "combo_stirling1(30, 15)");

    expect_count(interp, "combo_eulerian(21, 10)", "14950368791471452636");
    expect_overflow(interp, "combo_eulerian(22, 11)");
}

// The binomial cases are the ones the old loop got wrong while still fitting.
TEST(ReplComboOverflow, BinomialCoefficientsAreExactUpToTheLimit) {
    Interpreter interp;
    expect_count(interp, "combo_binomial(67, 33)", "14226520737620288370");
    expect_count(interp, "combo_nchoosek(67, 33)", "14226520737620288370");
    expect_count(interp, "combo_binomial(62, 31)", "465428353255261088");
    expect_count(interp, "combo_binomial(52, 5)", "2598960");
    expect_overflow(interp, "combo_binomial(68, 34)");
    expect_overflow(interp, "combo_nchoosek(100, 50)");
}

TEST(ReplComboOverflow, ArrangementsAreExactUpToTheLimit) {
    Interpreter interp;
    expect_count(interp, "combo_permutations(20, 20)", "2432902008176640000");
    expect_count(interp, "combo_permutations(25, 10)", "11861676288000");
    expect_overflow(interp, "combo_permutations(21, 21)");
    expect_overflow(interp, "combo_permutations(30, 30)");

    expect_count(interp, "combo_combinations_with_rep(50, 10)", "62828356305");
    expect_overflow(interp, "combo_combinations_with_rep(200, 100)");
}

TEST(ReplComboOverflow, MultinomialsPastTwentyFactorialAreStillCounted) {
    Interpreter interp;
    // 30!/(10!)^3 is 5550996791340 -- it fits, even though 30! does not, and the old
    // factorial-then-divide route could not reach it.
    expect_contains(interp, "combo_multinomial(30, [10; 10; 10])", "5550996791340");
    expect_contains(interp, "combo_multinomial(25, [5; 5; 5; 5; 5])", "623360743125120");
    // 60!/(20!)^3 is about 5.6e26 and genuinely does not fit.
    expect_error_contains(interp, "combo_multinomial(60, [20; 20; 20])", kOverflowMessage);
}

TEST(ReplComboOverflow, RanksStopWhereRanksStopFitting) {
    Interpreter interp;
    expect_contains(interp, "combo_rank_permutation([2; 1; 0])", "5");
    expect_error_contains(interp, "combo_unrank_permutation(3, 6)", "no permutation with that rank");
    expect_error_contains(interp, "combo_unrank_combination(4, 2, 6)", "no combination with that rank");
    expect_error_contains(interp, "combo_unrank_combination(100, 50, 0)", "no combination with that rank");
}

} // namespace
