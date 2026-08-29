// MathScript LU Decomposition Unit Test

#include <gtest/gtest.h>
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(LUTest, basic_2x2) {
    DMatrix A{{4, 3}, {6, 3}};
    auto [L, U, P] = lu(A).value();

    EXPECT_DOUBLE_EQ(L(0, 0), 1.0);
    EXPECT_NEAR(L(1, 0), 4.0 / 6.0, 1e-10);
    EXPECT_DOUBLE_EQ(L(1, 1), 1.0);

    EXPECT_DOUBLE_EQ(U(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(U(0, 1), 3.0);
    EXPECT_NEAR(U(1, 1), 1.0, 1e-10);
}

TEST(LUTest, 3x3_identity) {
    DMatrix I = eye<double>(3);
    auto [L, U, P] = lu(I).value();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(L(i, j), (i == j) ? 1.0 : 0.0, 1e-10);
            EXPECT_NEAR(U(i, j), (i == j) ? 1.0 : 0.0, 1e-10);
        }
    }
}

TEST(LUTest, singular_matrix) {
    DMatrix A{{1, 2}, {2, 4}};
    auto result = lu(A);
    EXPECT_FALSE(result.has_value());
}

TEST(LUTest, 3x3_general) {
    DMatrix A{{4, 3, 2}, {6, 7, 5}, {2, 8, 3}};
    auto [L, U, P] = lu(A).value();

    DMatrix PA = P * A;
    DMatrix LU = L * U;
    DMatrix diff = PA - LU;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(diff(i, j), 0.0, 1e-10);
        }
    }
}
