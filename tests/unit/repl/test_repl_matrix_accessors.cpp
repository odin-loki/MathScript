// The REPL could compute matrices but never look inside one: there was no way
// to read its shape, pull out an element, take a row or a column, reshape it,
// or cut out a block. A computed matrix could only ever be printed whole.
//
// These pin the accessor set in both the assignment and the printing form, and
// pin the errors, because an out-of-range index must be a reported error and
// never a read past the end of the buffer.

#include <gtest/gtest.h>

#include <string>

#include "ms/interp/repl_engine.hpp"

#include "repl_test_helpers.hpp"

using namespace ms::interp;

namespace {

void seed(Interpreter& interp) {
    // A = [1 2 3; 4 5 6]
    expect_ok(interp, "A = [1, 2, 3; 4, 5, 6]");
}

}  // namespace

TEST(ReplMatrixAccessors, ShapeAsScalars) {
    Interpreter interp;
    seed(interp);

    expect_ok(interp, "r = mat_rows(A)");
    expect_ok(interp, "c = mat_cols(A)");
    expect_ok(interp, "n = mat_numel(A)");
    EXPECT_EQ(interp.state().scalars.at("r"), 2.0);
    EXPECT_EQ(interp.state().scalars.at("c"), 3.0);
    EXPECT_EQ(interp.state().scalars.at("n"), 6.0);

    expect_contains(interp, "mat_rows(A)", "2");
    expect_contains(interp, "mat_cols(A)", "3");
    expect_contains(interp, "mat_numel(A)", "6");
}

TEST(ReplMatrixAccessors, ElementAccess) {
    Interpreter interp;
    seed(interp);

    expect_ok(interp, "a00 = mat_at(A, 0, 0)");
    expect_ok(interp, "a12 = mat_at(A, 1, 2)");
    EXPECT_EQ(interp.state().scalars.at("a00"), 1.0);
    EXPECT_EQ(interp.state().scalars.at("a12"), 6.0);

    expect_contains(interp, "mat_at(A, 0, 2)", "3");
}

TEST(ReplMatrixAccessors, ElementAccessRejectsBadIndices) {
    Interpreter interp;
    seed(interp);

    expect_error_contains(interp, "b = mat_at(A, 2, 0)", "out of range");
    expect_error_contains(interp, "b = mat_at(A, 0, 3)", "out of range");
    expect_error_contains(interp, "b = mat_at(A, -1, 0)", "non-negative integer");
    expect_error_contains(interp, "b = mat_at(A, 0.5, 0)", "non-negative integer");
}

TEST(ReplMatrixAccessors, RowAndColumn) {
    Interpreter interp;
    seed(interp);

    expect_ok(interp, "r1 = mat_row(A, 1)");
    const auto& r1 = interp.state().matrices.at("r1");
    ASSERT_EQ(r1.rows(), 1u);
    ASSERT_EQ(r1.cols(), 3u);
    EXPECT_EQ(r1(0, 0), 4.0);
    EXPECT_EQ(r1(0, 2), 6.0);

    expect_ok(interp, "c2 = mat_col(A, 2)");
    const auto& c2 = interp.state().matrices.at("c2");
    ASSERT_EQ(c2.rows(), 2u);
    ASSERT_EQ(c2.cols(), 1u);
    EXPECT_EQ(c2(0, 0), 3.0);
    EXPECT_EQ(c2(1, 0), 6.0);

    expect_error_contains(interp, "b = mat_row(A, 2)", "out of range");
    expect_error_contains(interp, "b = mat_col(A, 3)", "out of range");
}

TEST(ReplMatrixAccessors, ReshapeKeepsRowOrder) {
    Interpreter interp;
    seed(interp);

    expect_ok(interp, "B = mat_reshape(A, 3, 2)");
    const auto& B = interp.state().matrices.at("B");
    ASSERT_EQ(B.rows(), 3u);
    ASSERT_EQ(B.cols(), 2u);
    // A read in row order is 1 2 3 4 5 6, so a 3x2 refill gives [1 2; 3 4; 5 6].
    EXPECT_EQ(B(0, 0), 1.0);
    EXPECT_EQ(B(0, 1), 2.0);
    EXPECT_EQ(B(1, 0), 3.0);
    EXPECT_EQ(B(2, 1), 6.0);

    expect_error_contains(interp, "C = mat_reshape(A, 4, 2)", "does not fit");
}

TEST(ReplMatrixAccessors, Submatrix) {
    Interpreter interp;
    seed(interp);

    expect_ok(interp, "S = mat_submatrix(A, 0, 1, 2, 2)");
    const auto& S = interp.state().matrices.at("S");
    ASSERT_EQ(S.rows(), 2u);
    ASSERT_EQ(S.cols(), 2u);
    EXPECT_EQ(S(0, 0), 2.0);
    EXPECT_EQ(S(0, 1), 3.0);
    EXPECT_EQ(S(1, 0), 5.0);
    EXPECT_EQ(S(1, 1), 6.0);

    expect_error_contains(interp, "T = mat_submatrix(A, 1, 1, 2, 2)", "past the end");
}

TEST(ReplMatrixAccessors, PrintingFormsWorkWithoutATarget) {
    Interpreter interp;
    seed(interp);

    expect_contains(interp, "mat_row(A, 0)", "_ =");
    expect_contains(interp, "mat_col(A, 0)", "_ =");
    expect_contains(interp, "mat_reshape(A, 6, 1)", "_ =");
    expect_contains(interp, "mat_submatrix(A, 0, 0, 1, 1)", "_ =");
    // The same generic path also prints the matrix constructors, which had no
    // no-assignment form at all before.
    expect_contains(interp, "eye(2)", "_ =");
    expect_contains(interp, "zeros(2, 3)", "_ =");
}

TEST(ReplMatrixAccessors, AccessorsComposeWithExpressions) {
    Interpreter interp;
    seed(interp);

    // A scalar pulled out of a matrix is an ordinary session scalar.
    expect_ok(interp, "v = mat_at(A, 1, 1)");
    expect_ok(interp, "w = v * 2 + 1");
    EXPECT_EQ(interp.state().scalars.at("w"), 11.0);

    // And an index may itself be an expression.
    expect_ok(interp, "i = 1");
    expect_ok(interp, "u = mat_at(A, i, i)");
    EXPECT_EQ(interp.state().scalars.at("u"), 5.0);
}
