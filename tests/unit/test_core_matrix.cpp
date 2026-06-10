#include <gtest/gtest.h>
#include <sstream>

#include "ms/core/matrix.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

TEST(MatrixTest, MatrixCtorFill_CorrectShape) {
    DMatrix m(2, 3, 7.0);
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 3u);
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            EXPECT_DOUBLE_EQ(m(i, j), 7.0);
        }
    }
}

TEST(MatrixTest, MatrixCtorDefault_Empty) {
    DMatrix m;
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.size(), 0u);
}

TEST(MatrixTest, MatrixScalarMultiply_Commutative) {
    DMatrix A{{1, 2}, {3, 4}};
    const DMatrix left = 2.0 * A;
    const DMatrix right = A * 2.0;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_DOUBLE_EQ(left(i, j), right(i, j));
        }
    }
}

TEST(MatrixTest, MatrixPrint_DoesNotThrow) {
    DMatrix m{{1, 2}, {3, 4}};
    std::ostringstream os;
    std::streambuf* old = std::cout.rdbuf(os.rdbuf());
    m.print();
    std::cout.rdbuf(old);
    EXPECT_FALSE(os.str().empty());
}

TEST(MatrixTest, MatrixOperatorSubscript_CorrectValues) {
    DMatrix A{{1, 2}, {3, 4}};
    EXPECT_DOUBLE_EQ(A(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(A(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(A(1, 1), 4.0);
}

TEST(MatrixTest, MatrixEye_IsIdentity) {
    const DMatrix I = eye<double>(3);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            const double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_DOUBLE_EQ(I(i, j), expected);
        }
    }
}

TEST(MatrixTest, MatrixZeros_AllZero) {
    const DMatrix Z = zeros<double>(2, 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(Z(i, j), 0.0);
        }
    }
}

TEST(MatrixTest, MatrixOnes_AllOne) {
    const DMatrix O = ones<double>(2, 3);
    EXPECT_EQ(O.rows(), 2u);
    EXPECT_EQ(O.cols(), 3u);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(O(i, j), 1.0);
        }
    }
}
