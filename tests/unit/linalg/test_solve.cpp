// MathScript Linear Solver Unit Test

#include <gtest/gtest.h>
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(SolveTest, basic_2x2) {
    DMatrix A{{4, 3}, {6, 3}};
    DMatrix b{{1}, {1}};

    auto x = solve(A, b).value();

    DMatrix Ax = A * x;
    for (size_t i = 0; i < b.rows(); ++i) {
        EXPECT_NEAR(Ax(i, 0), b(i, 0), 1e-10);
    }
}

TEST(SolveTest, 3x3_identity) {
    DMatrix A = eye<double>(3);
    DMatrix b{{1}, {2}, {3}};

    auto x = solve(A, b).value();

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(x(i, 0), b(i, 0), 1e-10);
    }
}

TEST(SolveTest, overdetermined_leastsquares) {
    DMatrix A2{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}};
    DMatrix b2{{14}, {32}, {50}};

    auto x = solve(A2, b2).value();

    DMatrix Ax = A2 * x;
    for (size_t i = 0; i < b2.rows(); ++i) {
        EXPECT_NEAR(Ax(i, 0), b2(i, 0), 1e-10);
    }
}

TEST(SolveTest, multiple_right_hand_sides) {
    DMatrix A{{2, 1}, {1, 3}};
    DMatrix B{{4, 1}, {7, 5}};

    auto x = solve(A, B).value();
    const DMatrix Ax = A * x;

    for (std::size_t i = 0; i < A.rows(); ++i) {
        for (std::size_t j = 0; j < B.cols(); ++j) {
            EXPECT_NEAR(Ax(i, j), B(i, j), 1e-10);
        }
    }
}

TEST(SolveTest, singular_matrix) {
    DMatrix A{{1, 2}, {2, 4}};
    DMatrix b{{5}, {10}};

    auto result = solve(A, b);
    EXPECT_FALSE(result.has_value());
}

TEST(SolveTest, dimension_mismatch_matrix_and_rhs) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}};
    DMatrix b{{1}, {2}};
    EXPECT_FALSE(solve(A, b).has_value());
    DMatrix B{{1, 2}, {3, 4}, {5, 6}};
    EXPECT_FALSE(solve(A, B).has_value());
}
