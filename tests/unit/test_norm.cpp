// MathScript Matrix Norm Unit Test

#include <gtest/gtest.h>
#include <cmath>
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(NormTest, frobenius) {
    DMatrix A{{1, 2}, {3, 4}};
    auto n = norm(A, 2).value();
    EXPECT_NEAR(n, std::sqrt(30.0), 1e-10);
}

TEST(NormTest, l1) {
    DMatrix A{{1, 2}, {3, 4}};
    auto n = norm(A, 1).value();
    EXPECT_NEAR(n, 10.0, 1e-10);
}

TEST(NormTest, inf) {
    DMatrix A{{1, 2}, {3, 4}};
    auto n = norm(A, -1).value();
    EXPECT_NEAR(n, 7.0, 1e-10);
}

TEST(NormTest, zero_matrix) {
    DMatrix A = zeros<double>(3, 3);
    auto n = norm(A).value();
    EXPECT_NEAR(n, 0.0, 1e-10);
}
