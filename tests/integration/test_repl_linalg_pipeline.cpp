// MathScript Integration Tests: REPL Interpreter – Matrix & Linear Algebra Pipeline

#include <gtest/gtest.h>
#include <cmath>
#include <string>

#include "ms/interp/repl_engine.hpp"

using namespace ms::interp;
using ms::Matrix;

// ---------------------------------------------------------------------------
// Matrix assignment
// ---------------------------------------------------------------------------

// REPL matrix syntax: rows separated by ';', elements by ','
// e.g. "[1,2;3,4]" is a 2x2 matrix

TEST(ReplLinalgPipeline, MatrixAssignment_PopulatesState) {
    Interpreter interp;
    auto result = interp.execute("A = [1,2;3,4]");
    ASSERT_TRUE(result.has_value()) << "Expected successful parse of matrix literal";
    EXPECT_GT(interp.state().matrices.count("A"), 0u)
        << "Matrix 'A' should be present in session state after assignment";
}

// ---------------------------------------------------------------------------
// trace
// ---------------------------------------------------------------------------

TEST(ReplLinalgPipeline, Trace_OfTwoByTwo) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("A = [1,2;3,4]").has_value());

    auto result = interp.execute("x = trace(A)");
    ASSERT_TRUE(result.has_value()) << "trace(A) should succeed";

    const auto& scalars = interp.state().scalars;
    ASSERT_GT(scalars.count("x"), 0u) << "Scalar 'x' should be set after trace(A)";
    EXPECT_DOUBLE_EQ(scalars.at("x"), 5.0);
}

// ---------------------------------------------------------------------------
// det
// ---------------------------------------------------------------------------

TEST(ReplLinalgPipeline, Det_OfTwoByTwo) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("A = [1,2;3,4]").has_value());

    auto result = interp.execute("b = det(A)");
    ASSERT_TRUE(result.has_value()) << "det(A) should succeed";

    const auto& scalars = interp.state().scalars;
    ASSERT_GT(scalars.count("b"), 0u) << "Scalar 'b' should be set after det(A)";
    EXPECT_DOUBLE_EQ(scalars.at("b"), -2.0);
}

// ---------------------------------------------------------------------------
// norm
// ---------------------------------------------------------------------------

TEST(ReplLinalgPipeline, Norm_IsFiniteAndPositive) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("A = [1,2;3,4]").has_value());

    auto result = interp.execute("y = norm(A)");
    ASSERT_TRUE(result.has_value()) << "norm(A) should succeed";

    const auto& scalars = interp.state().scalars;
    ASSERT_GT(scalars.count("y"), 0u) << "Scalar 'y' should be set after norm(A)";
    double n = scalars.at("y");
    EXPECT_TRUE(std::isfinite(n)) << "norm result should be finite";
    EXPECT_GT(n, 0.0) << "norm result should be positive";
}

// ---------------------------------------------------------------------------
// Mixed assignment sequence – scalars and matrices
// ---------------------------------------------------------------------------

TEST(ReplLinalgPipeline, MixedAssignmentSequence) {
    Interpreter interp;

    // Scalar literal
    ASSERT_TRUE(interp.execute("s = 3.14").has_value());
    EXPECT_GT(interp.state().scalars.count("s"), 0u);
    EXPECT_NEAR(interp.state().scalars.at("s"), 3.14, 1e-10);

    // Matrix literal (comma-separated elements, semicolon row separator)
    ASSERT_TRUE(interp.execute("B = [2,0;0,2]").has_value());
    EXPECT_GT(interp.state().matrices.count("B"), 0u);

    // Scalar derived from matrix
    ASSERT_TRUE(interp.execute("t = trace(B)").has_value());
    EXPECT_GT(interp.state().scalars.count("t"), 0u);
    EXPECT_DOUBLE_EQ(interp.state().scalars.at("t"), 4.0);

    // Second matrix
    ASSERT_TRUE(interp.execute("C = [5,6;7,8]").has_value());
    EXPECT_GT(interp.state().matrices.count("C"), 0u);

    // Both matrices remain in state
    EXPECT_GT(interp.state().matrices.count("B"), 0u);
    EXPECT_GT(interp.state().matrices.count("C"), 0u);
}

// ---------------------------------------------------------------------------
// Invalid input returns an error (has_value == false)
// ---------------------------------------------------------------------------

TEST(ReplLinalgPipeline, InvalidLine_ReturnsError) {
    Interpreter interp;
    auto result = interp.execute("@@@invalid$$$");
    EXPECT_FALSE(result.has_value())
        << "Executing invalid input should return an error Result";
}

TEST(ReplLinalgPipeline, UnknownFunction_ReturnsError) {
    Interpreter interp;
    ASSERT_TRUE(interp.execute("A = [1,2;3,4]").has_value());
    auto result = interp.execute("z = notafunc(A)");
    EXPECT_FALSE(result.has_value())
        << "Calling an unknown function should return an error Result";
}
