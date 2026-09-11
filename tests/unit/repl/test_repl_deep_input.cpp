// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Odin Loch
// Deeply nested and very long input.
//
// Three stack overflows, all reached from a single REPL line:
//   * the symbolic parser is recursive descent, so sin(sin(sin(...))) recursed once per
//     nesting level and crashed at about 10000 levels;
//   * parse_add and parse_mul LOOP over their operands rather than recursing, so
//     x+x+x+... did not hit a depth limit -- but it built a left spine one node deep per
//     term, and ~SymExpr walks that spine recursively through its unique_ptr children, so
//     100000 terms overflowed the stack on DESTRUCTION;
//   * command dispatch runs the line through a chain of std::regex matches, and
//     libstdc++'s executor recurses once per input character through repetition
//     operators, so a six-figure line crashed inside the regex before any of the
//     interpreter's own code ran.
//
// All three now report instead. The point of each expectation is that execute() returns --
// a value or an error, either is fine -- rather than taking the process down.

#include <gtest/gtest.h>

#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/symbolic/symbolic.hpp"

using namespace ms::interp;

namespace {

std::string nested_calls(const char* fn, int depth) {
    std::string s = "1";
    for (int i = 0; i < depth; ++i) s = std::string(fn) + "(" + s + ")";
    return s;
}

std::string chained_sum(int terms) {
    std::string s = "x";
    for (int i = 0; i < terms; ++i) s += "+x";
    return s;
}

}  // namespace

TEST(ReplDeepInput, DeeplyNestedSymbolicExpressionsAreRejectedNotCrashed) {
    for (const int depth : {50, 200, 1000, 5000}) {
        const auto parsed = ms::sym_parse(nested_calls("sin", depth));
        // Shallow nesting still parses; deep nesting reports "nested too deeply".
        if (depth <= 50) {
            EXPECT_TRUE(parsed.has_value()) << "depth " << depth;
        }
        // Either way it must have returned.
        SUCCEED();
    }
    // A depth that used to crash.
    const auto deep = ms::sym_parse(nested_calls("sin", 5000));
    EXPECT_FALSE(deep.has_value());
}

TEST(ReplDeepInput, LongOperandChainsAreBounded) {
    // The left spine that ~SymExpr has to unwind.
    const auto few = ms::sym_parse(chained_sum(100));
    EXPECT_TRUE(few.has_value());
    const auto many = ms::sym_parse(chained_sum(50000));
    EXPECT_FALSE(many.has_value()) << "a 50000-term chain must be refused";

    // Nested parentheses are the same shape through parse_primary.
    const std::string deep_parens = std::string(5000, '(') + "1" + std::string(5000, ')');
    EXPECT_FALSE(ms::sym_parse(deep_parens).has_value());
    EXPECT_TRUE(ms::sym_parse("((((1))))").has_value());
}

TEST(ReplDeepInput, OverlongCommandLinesAreRefused) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = 2").has_value());
    // Under the cap the line still runs.
    const std::string ok = "s = " + chained_sum(100);
    EXPECT_TRUE(interp.execute(ok).has_value());

    // Over it, the interpreter reports rather than handing a six-figure string to
    // std::regex.
    for (const std::string& cmd : {std::string("y = ") + chained_sum(100000),
                                   std::string("sym_simplify(\"") + std::string(100000, 'x'),
                                   std::string(100000, 'a') + " = 1",
                                   std::string("x = ") + std::string(100000, '(') + "1"}) {
        const auto r = interp.execute(cmd);
        EXPECT_FALSE(r.has_value()) << "length " << cmd.size();
    }

    // And the session is still usable afterwards.
    EXPECT_TRUE(interp.execute("z = [1, 2; 3, 4]").has_value());
    EXPECT_TRUE(interp.execute("det(z)").has_value());
}

TEST(ReplDeepInput, ModeratelyLargeButLegalInputStillWorks) {
    // The caps must not get in the way of an expression a person might actually write.
    Interpreter interp;
    ASSERT_TRUE(interp.execute("x = 3").has_value());
    EXPECT_TRUE(interp.execute("p = " + chained_sum(500)).has_value());
    EXPECT_TRUE(ms::sym_parse(nested_calls("sin", 40)).has_value());
    EXPECT_TRUE(ms::sym_parse(chained_sum(2000)).has_value());
    const auto d = ms::sym_diff(*ms::sym_parse("sin(cos(tan(x)))"), "x");
    EXPECT_NE(ms::sym_to_string(d), "");
}
