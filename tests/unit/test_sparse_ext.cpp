#include <gtest/gtest.h>
#include <cmath>
#include "ms/core/sparse.hpp"

using namespace ms;

TEST(SparseExtTest, empty_construction) {
    Sparse<double> S(4, 5);
    EXPECT_EQ(S.rows(), 4u);
    EXPECT_EQ(S.cols(), 5u);
    EXPECT_EQ(S.nnz(), 0u);
}

TEST(SparseExtTest, csc_construction_nnz) {
    // 3x3 sparse with 3 nonzeros on diagonal
    Sparse<double> S(3, 3, {0, 1, 2}, {0, 1, 2}, {1.0, 2.0, 3.0});
    EXPECT_EQ(S.nnz(), 3u);
    EXPECT_EQ(S.rows(), 3u);
    EXPECT_EQ(S.cols(), 3u);
}

TEST(SparseExtTest, to_dense_diagonal) {
    Sparse<double> S(3, 3, {0, 1, 2}, {0, 1, 2}, {5.0, 6.0, 7.0});
    const auto D = S.to_dense();
    EXPECT_EQ(D.rows(), 3u);
    EXPECT_EQ(D.cols(), 3u);
    EXPECT_DOUBLE_EQ(D(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(D(1, 1), 6.0);
    EXPECT_DOUBLE_EQ(D(2, 2), 7.0);
    EXPECT_DOUBLE_EQ(D(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(D(1, 0), 0.0);
}

TEST(SparseExtTest, to_dense_off_diagonal) {
    // Row 0, col 1 = 3.0; Row 1, col 0 = 4.0
    Sparse<double> S(2, 2, {0, 1}, {1, 0}, {3.0, 4.0});
    const auto D = S.to_dense();
    EXPECT_DOUBLE_EQ(D(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(D(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(D(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(D(1, 1), 0.0);
}

TEST(SparseExtTest, spmv_identity) {
    // Identity 3x3
    Sparse<double> I(3, 3, {0, 1, 2}, {0, 1, 2}, {1.0, 1.0, 1.0});
    ColMatrix<double> x(3, 1);
    x(0, 0) = 2.0; x(1, 0) = 3.0; x(2, 0) = 4.0;
    const auto y = I.spmv(x);
    EXPECT_EQ(y.rows(), 3u);
    EXPECT_DOUBLE_EQ(y(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(y(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(y(2, 0), 4.0);
}

TEST(SparseExtTest, empty_to_dense_all_zero) {
    Sparse<double> S(2, 2);
    const auto D = S.to_dense();
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_DOUBLE_EQ(D(i, j), 0.0);
}
