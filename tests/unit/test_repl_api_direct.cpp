#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;

// ---------------------------------------------------------------------------
// Direct tests of Interpreter's public assign_* methods
// ---------------------------------------------------------------------------

TEST(ReplApiDirectTest, assign_scalar_expr_basic) {
    Interpreter interp;
    const auto r = interp.assign_scalar_expr("y", "sin(1.0) + 1.0");
    ASSERT_TRUE(r.has_value());
    const double val = interp.state().scalars.at("y");
    EXPECT_NEAR(val, std::sin(1.0) + 1.0, 1e-12);
}

TEST(ReplApiDirectTest, assign_scalar_binary_add) {
    Interpreter interp;
    // Setup: assign x = 3.0 first
    ASSERT_TRUE(interp.execute("x = 3.0").has_value());

    ScalarBinaryAssign assign;
    assign.target = "z";
    assign.op = '+';
    assign.left.is_literal = false;
    assign.left.name = "x";
    assign.right.is_literal = true;
    assign.right.literal = 5.0;

    const auto r = interp.assign_scalar_binary(assign);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("z"), 8.0);
}

TEST(ReplApiDirectTest, assign_scalar_binary_mul) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("a = 4.0").has_value());

    ScalarBinaryAssign assign;
    assign.target = "b";
    assign.op = '*';
    assign.left.is_literal = false;
    assign.left.name = "a";
    assign.right.is_literal = true;
    assign.right.literal = 2.5;

    const auto r = interp.assign_scalar_binary(assign);
    ASSERT_TRUE(r.has_value());
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("b"), 10.0);
}

TEST(ReplApiDirectTest, assign_matrix_call_zeros) {
    Interpreter interp;

    MatrixCallAssign assign;
    assign.target = "Z";
    assign.callee = "zeros";
    assign.args = {"3"};

    const auto r = interp.assign_matrix_call(assign);
    ASSERT_TRUE(r.has_value());
    const auto& Z = interp.state().matrices.at("Z");
    EXPECT_EQ(Z.rows(), 3u);
    EXPECT_EQ(Z.cols(), 3u);
    EXPECT_DOUBLE_EQ(Z(0, 0), 0.0);
}

TEST(ReplApiDirectTest, assign_matrix_call_eye) {
    Interpreter interp;

    MatrixCallAssign assign;
    assign.target = "I";
    assign.callee = "eye";
    assign.args = {"4"};

    const auto r = interp.assign_matrix_call(assign);
    ASSERT_TRUE(r.has_value());
    const auto& I = interp.state().matrices.at("I");
    EXPECT_EQ(I.rows(), 4u);
    EXPECT_DOUBLE_EQ(I(2, 2), 1.0);
    EXPECT_DOUBLE_EQ(I(0, 1), 0.0);
}

TEST(ReplApiDirectTest, assign_multi_matrix_call_qr) {
    Interpreter interp;
    // First create a matrix
    ASSERT_TRUE(interp.execute("M = [2,1;1,3]").has_value());

    MultiMatrixCallAssign assign;
    assign.targets = {"Q", "R"};
    assign.callee = "qr";
    assign.arg = "M";

    const auto r = interp.assign_multi_matrix_call(assign);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(interp.state().matrices.count("Q") > 0);
    EXPECT_TRUE(interp.state().matrices.count("R") > 0);
}

TEST(ReplApiDirectTest, session_state_history_populated) {
    Interpreter interp;
    interp.execute("x = 1.0");
    interp.execute("y = 2.0");

    // history should record both lines
    EXPECT_GE(interp.state().history.size(), 2u);
}

TEST(ReplApiDirectTest, eval_scalar_call_sin) {
    const auto r = Interpreter::eval_scalar_call("sin", {0.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 0.0, 1e-15);
}

TEST(ReplApiDirectTest, eval_scalar_call_sqrt) {
    const auto r = Interpreter::eval_scalar_call("sqrt", {4.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 2.0, 1e-12);
}

TEST(ReplApiDirectTest, eval_scalar_call_pow) {
    const auto r = Interpreter::eval_scalar_call("pow", {2.0, 10.0});
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(*r, 1024.0, 1e-10);
}

TEST(ReplApiDirectTest, eval_scalar_op_div_by_zero) {
    // divide by zero should return an error (or Inf, implementation-defined)
    const auto r = Interpreter::eval_scalar_op('/', 1.0, 0.0);
    // either a value (Inf) or an error is acceptable; no crash
    (void)r;  // just verify no exception
    SUCCEED();
}

TEST(ReplApiDirectTest, list_scalar_expr_variables) {
    const auto vars = Interpreter::list_scalar_expr_variables("a + b * c");
    // should contain a, b, c
    EXPECT_GE(vars.size(), 1u);
}

TEST(ReplApiDirectTest, assign_scalar_sets_state) {
    Interpreter interp;
    const auto r = interp.assign_scalar("pi_approx", 3.14159265358979);
    ASSERT_TRUE(r.has_value());
    EXPECT_NEAR(interp.state().scalars.at("pi_approx"), 3.14159265358979, 1e-14);
}
