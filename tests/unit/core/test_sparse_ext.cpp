#include <gtest/gtest.h>
#include <cmath>
#include "ms/core/sparse.hpp"
#include "ms/core/matrix.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

namespace {

DMatrix dense_add(const DMatrix& a, const DMatrix& b) {
    DMatrix c(a.rows(), a.cols(), 0.0);
    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            c(i, j) = a(i, j) + b(i, j);
        }
    }
    return c;
}

void expect_sparse_matches_dense(const Sparse<double>& s, const DMatrix& expected) {
    const auto d = s.to_dense();
    ASSERT_EQ(d.rows(), expected.rows());
    ASSERT_EQ(d.cols(), expected.cols());
    for (size_t i = 0; i < expected.rows(); ++i) {
        for (size_t j = 0; j < expected.cols(); ++j) {
            EXPECT_NEAR(d(i, j), expected(i, j), 1e-12);
        }
    }
}

} // namespace

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

TEST(SparseAddTest, empty_plus_empty) {
    Sparse<double> a(3, 3);
    Sparse<double> b(3, 3);
    const auto c = sparse_add(a, b);
    EXPECT_EQ(c.rows(), 3u);
    EXPECT_EQ(c.cols(), 3u);
    EXPECT_EQ(c.nnz(), 0u);
}

TEST(SparseAddTest, empty_plus_nonzero) {
    Sparse<double> a(2, 2);
    Sparse<double> b(2, 2, {0, 1}, {1, 0}, {3.0, 4.0});
    const auto c = sparse_add(a, b);
    expect_sparse_matches_dense(c, b.to_dense());
}

TEST(SparseAddTest, nonoverlapping_indices) {
    Sparse<double> a(2, 2, {0}, {0}, {1.0});
    Sparse<double> b(2, 2, {1}, {1}, {2.0});
    const auto c = sparse_add(a, b);
    EXPECT_EQ(c.nnz(), 2u);
    expect_sparse_matches_dense(c, dense_add(a.to_dense(), b.to_dense()));
}

TEST(SparseAddTest, overlapping_indices_sum) {
    Sparse<double> a(2, 2, {0, 1}, {0, 1}, {1.0, 2.0});
    Sparse<double> b(2, 2, {0, 1}, {0, 1}, {5.0, 7.0});
    const auto c = sparse_add(a, b);
    EXPECT_EQ(c.nnz(), 2u);
    const auto d = c.to_dense();
    EXPECT_NEAR(d(0, 0), 6.0, 1e-12);
    EXPECT_NEAR(d(1, 1), 9.0, 1e-12);
}

TEST(SparseAddTest, matches_dense_2x2) {
    Sparse<double> a(2, 2, {0, 1}, {0, 1}, {1.0, 2.0});
    Sparse<double> b(2, 2, {0, 1}, {1, 0}, {3.0, 4.0});
    const auto c = sparse_add(a, b);
    expect_sparse_matches_dense(c, dense_add(a.to_dense(), b.to_dense()));
}

TEST(SparseAddTest, matches_dense_3x3_diagonal) {
    Sparse<double> a(3, 3, {0, 1, 2}, {0, 1, 2}, {1.0, 2.0, 3.0});
    Sparse<double> b(3, 3, {0, 1, 2}, {0, 1, 2}, {4.0, 5.0, 6.0});
    const auto c = sparse_add(a, b);
    expect_sparse_matches_dense(c, dense_add(a.to_dense(), b.to_dense()));
}

TEST(SparseAddTest, dimension_mismatch_returns_empty) {
    Sparse<double> a(2, 2, {0}, {0}, {1.0});
    Sparse<double> b(3, 2, {0}, {0}, {2.0});
    const auto c = sparse_add(a, b);
    EXPECT_EQ(c.rows(), 0u);
    EXPECT_EQ(c.cols(), 0u);
    EXPECT_EQ(c.nnz(), 0u);
}

TEST(SparseAddTest, cancellation_drops_zero_entry) {
    Sparse<double> a(2, 2, {0}, {0}, {5.0});
    Sparse<double> b(2, 2, {0}, {0}, {-5.0});
    const auto c = sparse_add(a, b);
    EXPECT_EQ(c.nnz(), 0u);
    expect_sparse_matches_dense(c, DMatrix(2, 2, 0.0));
}

TEST(SparseAddTest, partial_overlap_mixed_pattern) {
    Sparse<double> a(3, 3, {0, 1, 2}, {1, 0, 2}, {2.0, 3.0, 4.0});
    Sparse<double> b(3, 3, {0, 1, 2}, {1, 2, 2}, {1.0, 5.0, 6.0});
    const auto c = sparse_add(a, b);
    expect_sparse_matches_dense(c, dense_add(a.to_dense(), b.to_dense()));
}
