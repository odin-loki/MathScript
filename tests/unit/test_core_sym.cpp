#include <gtest/gtest.h>
#include "ms/symbolic/symbolic.hpp"
#include <map>
#include <string>

using namespace ms;

TEST(SymTest, create_variable) {
    SymExpr v = sym_var("x");
    EXPECT_EQ(v.op, SymOp::Var);
    EXPECT_EQ(v.name, "x");
}

TEST(SymTest, add_constants) {
    SymExpr result = sym_add(sym_const(2.0), sym_const(3.0));
    const std::map<std::string, double> env;
    EXPECT_NEAR(sym_eval(result, env), 5.0, 1e-10);
}

TEST(SymTest, simplify_constant_fold) {
    SymExpr expr = sym_add(sym_const(4.0), sym_const(6.0));
    SymExpr simplified = sym_simplify(std::move(expr));
    const std::map<std::string, double> env;
    EXPECT_NEAR(sym_eval(simplified, env), 10.0, 1e-10);
}

TEST(SymTest, to_string_variable) {
    SymExpr v = sym_var("x");
    EXPECT_EQ(sym_to_string(v), "x");
}

TEST(SymTest, to_string_expression) {
    SymExpr expr = sym_add(sym_var("x"), sym_const(1.0));
    const std::string s = sym_to_string(expr);
    EXPECT_FALSE(s.empty());
}
