#include <gtest/gtest.h>
#include "ms/core/operations.hpp"
#include "ms/core/matrix.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(TransposeTest, square_2x2) {
    DMatrix A{{1, 2}, {3, 4}};
    auto T = transpose(A);

    EXPECT_DOUBLE_EQ(T(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(T(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(T(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(T(1, 1), 4.0);
}

TEST(TransposeTest, rectangular_2x3) {
    DMatrix A{{1, 2, 3}, {4, 5, 6}};
    auto T = transpose(A);

    EXPECT_EQ(T.rows(), 3u);
    EXPECT_EQ(T.cols(), 2u);

    EXPECT_DOUBLE_EQ(T(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(T(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(T(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(T(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(T(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(T(2, 1), 6.0);
}

TEST(TransposeTest, row_vector) {
    DMatrix v{{1, 2, 3, 4}};
    EXPECT_EQ(v.rows(), 1u);
    EXPECT_EQ(v.cols(), 4u);

    auto T = transpose(v);
    EXPECT_EQ(T.rows(), 4u);
    EXPECT_EQ(T.cols(), 1u);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(T(i, 0), static_cast<double>(i + 1));
    }
}

TEST(TransposeTest, double_transpose) {
    // transpose(A)(j,i) == A(i,j) -- verify both directions
    DMatrix A{{1, 2, 3}, {4, 5, 6}};
    auto T = transpose(A);
    EXPECT_EQ(T.rows(), 3u);
    EXPECT_EQ(T.cols(), 2u);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_DOUBLE_EQ(T(j, i), A(i, j));
        }
    }
}
