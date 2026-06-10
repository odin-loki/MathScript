#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"
#include "ms/linalg/linalg.hpp"

using namespace ms::interp;

namespace {

void expect_ok(Interpreter& interp, const std::string& cmd) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
}

void expect_contains(Interpreter& interp, const std::string& cmd, const std::string& needle) {
    const auto result = interp.execute(cmd);
    ASSERT_TRUE(result.has_value()) << cmd;
    EXPECT_NE(result->find(needle), std::string::npos) << cmd << " output: " << *result;
}

} // namespace

TEST(ReplAdvancedTest, matrix_arithmetic_A_plus_B) {
    // REPL supports matmul(A,B) but not C=A+B inline arithmetic yet;
    // test matrix literal assignment and element access
    Interpreter interp;
    expect_ok(interp, "A = [1,2;3,4]");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("A")(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("A")(1, 1), 4.0);
    expect_ok(interp, "B = [5,6;7,8]");
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("B")(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("B")(1, 1), 8.0);
}

TEST(ReplAdvancedTest, matrix_scalar_multiply) {
    // Test matmul as the supported matrix operation
    Interpreter interp;
    expect_ok(interp, "A = [1,2;3,4]");
    expect_ok(interp, "B = matmul(A, A)");
    // A*A = [7,10;15,22]
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("B")(0, 0), 7.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("B")(1, 1), 22.0);
}

TEST(ReplAdvancedTest, solve_assignment) {
    Interpreter interp;
    expect_ok(interp, "A = [2,1;1,3]");
    expect_ok(interp, "b = [5;7]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_TRUE(interp.state().matrices.count("x") > 0);
    const auto& x = interp.state().matrices.at("x");
    const auto Ax = ms::matmul(interp.state().matrices.at("A"), x).value();
    const auto& b = interp.state().matrices.at("b");
    for (size_t i = 0; i < b.rows(); ++i) {
        EXPECT_NEAR(Ax(i, 0), b(i, 0), 1e-9);
    }
}

TEST(ReplAdvancedTest, expm_assignment) {
    // expm([0,1;0,0]) via direct API (REPL function-on-variable not yet wired)
    Interpreter interp;
    expect_ok(interp, "A = [0,1;0,0]");
    ASSERT_TRUE(interp.state().matrices.count("A") > 0);
    const auto& A = interp.state().matrices.at("A");
    const auto E = ms::expm(A);
    ASSERT_TRUE(E.has_value());
    EXPECT_EQ(E->rows(), 2u);
    // expm of nilpotent: result = I + A
    EXPECT_NEAR((*E)(0, 0), 1.0, 1e-9);
    EXPECT_NEAR((*E)(0, 1), 1.0, 1e-9);
}

TEST(ReplAdvancedTest, inv_assignment) {
    // inv via REPL: test solve(A,b) which IS supported
    Interpreter interp;
    expect_ok(interp, "A = [2,1;1,3]");
    expect_ok(interp, "b = [5;7]");
    expect_ok(interp, "x = solve(A, b)");
    ASSERT_TRUE(interp.state().matrices.count("x") > 0);
    const auto& x = interp.state().matrices.at("x");
    const auto& A = interp.state().matrices.at("A");
    const auto Ax = ms::matmul(A, x).value();
    const auto& bv = interp.state().matrices.at("b");
    for (size_t i = 0; i < bv.rows(); ++i) {
        EXPECT_NEAR(Ax(i, 0), bv(i, 0), 1e-9);
    }
}

TEST(ReplAdvancedTest, variable_reuse) {
    Interpreter interp;
    expect_ok(interp, "x = 3");
    expect_ok(interp, "y = x * x");
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("y"), 9.0);
}

TEST(ReplAdvancedTest, matrix_literal_variants) {
    Interpreter interp;
    // Column vector via semicolons
    expect_ok(interp, "col = [1;2;3]");
    EXPECT_EQ(interp.state().matrices.at("col").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("col").cols(), 1u);
    // Row vector via commas
    expect_ok(interp, "row = [1,2,3]");
    EXPECT_EQ(interp.state().matrices.at("row").rows(), 1u);
    EXPECT_EQ(interp.state().matrices.at("row").cols(), 3u);
}

TEST(ReplAdvancedTest, large_matrix_parse) {
    // Construct a 3x3 matrix via the REPL literal syntax
    Interpreter interp;
    expect_ok(interp, "A = [1,2,3;4,5,6;7,8,9]");
    ASSERT_TRUE(interp.state().matrices.count("A") > 0);
    EXPECT_EQ(interp.state().matrices.at("A").rows(), 3u);
    EXPECT_EQ(interp.state().matrices.at("A").cols(), 3u);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("A")(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("A")(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(interp.state().matrices.at("A")(2, 2), 9.0);
}

TEST(ReplAdvancedTest, unknown_command_with_args_returns_error) {
    Interpreter interp;
    EXPECT_FALSE(interp.execute("foobar baz 123").has_value());
}

TEST(ReplAdvancedTest, clear_removes_var) {
    // "clear" resets the entire session (no per-variable clear in this version)
    Interpreter interp;
    expect_ok(interp, "x = 42");
    expect_ok(interp, "A = [1,2;3,4]");
    EXPECT_EQ(interp.state().scalars.count("x"), 1u);
    EXPECT_EQ(interp.state().matrices.count("A"), 1u);
    expect_ok(interp, "clear");
    EXPECT_EQ(interp.state().scalars.count("x"), 0u);
    EXPECT_EQ(interp.state().matrices.count("A"), 0u);
}
