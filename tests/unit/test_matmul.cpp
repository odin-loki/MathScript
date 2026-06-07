// MathScript Matrix Multiply Unit Test

#include <gtest/gtest.h>
#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(MatmulTest, basic_2x2) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix B{{5, 6}, {7, 8}};
    auto C = matmul(A, B).value();

    EXPECT_DOUBLE_EQ(C(0, 0), 19);
    EXPECT_DOUBLE_EQ(C(0, 1), 22);
    EXPECT_DOUBLE_EQ(C(1, 0), 43);
    EXPECT_DOUBLE_EQ(C(1, 1), 50);
}

TEST(MatmulTest, 3x3_identity) {
    DMatrix I = eye<double>(3);
    DMatrix A{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto C = matmul(I, A).value();

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(C(i, j), A(i, j), 1e-12);
        }
    }
}

TEST(MatmulTest, dimension_mismatch) {
    DMatrix A{{1, 2}, {3, 4}};
    DMatrix B{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    auto result = matmul(A, B);
    EXPECT_FALSE(result.has_value());
}
