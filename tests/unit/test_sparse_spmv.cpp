#include <gtest/gtest.h>
#include <cmath>
#include "ms/core/sparse.hpp"
#include "ms/cuda/sparse.hpp"

using namespace ms;

TEST(SparseSpmvTest, diagonal_action) {
    Sparse<double> S(3, 3, {0, 1, 2}, {0, 1, 2}, {2.0, 3.0, 4.0});
    ColMatrix<double> x{{1}, {1}, {1}};
    auto y = S.spmv(x);
    EXPECT_DOUBLE_EQ(y(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(y(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(y(2, 0), 4.0);
}

TEST(SparseSpmvTest, cuda_matches_cpu) {
    Sparse<double> S(2, 2, {0, 1}, {0, 1}, {5.0, 7.0});
    ColMatrix<double> x{{2}, {3}};
    auto cpu = S.spmv(x);
    auto gpu = cuda::spmv_coo(2, 2, {0, 1}, {0, 1}, {5.0, 7.0}, x).value();
    EXPECT_NEAR(cpu(0, 0), gpu(0, 0), 1e-12);
    EXPECT_NEAR(cpu(1, 0), gpu(1, 0), 1e-12);
}

TEST(SparseSpmvTest, to_dense) {
    Sparse<double> S(2, 2, {0, 1}, {1, 0}, {3.0, 4.0});
    auto dense = S.to_dense();
    EXPECT_DOUBLE_EQ(dense(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(dense(1, 0), 4.0);
}

namespace {

ColMatrix<double> dense_matvec(const ColMatrix<double>& A, const ColMatrix<double>& x) {
    ColMatrix<double> y(A.rows(), 1, 0.0);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            y(i, 0) += A(i, j) * x(j, 0);
        }
    }
    return y;
}

Sparse<double> sparse_from_dense(const ColMatrix<double>& A, double zero_tol = 0.0) {
    std::vector<size_t> rows;
    std::vector<size_t> cols;
    std::vector<double> vals;
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            if (std::abs(A(i, j)) > zero_tol) {
                rows.push_back(i);
                cols.push_back(j);
                vals.push_back(A(i, j));
            }
        }
    }
    return Sparse<double>(A.rows(), A.cols(), std::move(rows), std::move(cols), std::move(vals));
}

Sparse<double> transpose_sparse(const Sparse<double>& S) {
    const auto dense = S.to_dense();
    ColMatrix<double> AT(S.cols(), S.rows(), 0.0);
    for (size_t i = 0; i < S.rows(); ++i) {
        for (size_t j = 0; j < S.cols(); ++j) {
            AT(j, i) = dense(i, j);
        }
    }
    return sparse_from_dense(AT);
}

} // namespace

TEST(SparseSpmvTest, csr_matmul) {
    Sparse<double> S(2, 3, {0, 0, 1}, {0, 2, 1}, {1.0, 2.0, 3.0});
    ColMatrix<double> B{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    const auto dense = S.to_dense();
    ColMatrix<double> ref(2, 2, 0.0);
    for (size_t k = 0; k < 2; ++k) {
        ColMatrix<double> x{{B(0, k)}, {B(1, k)}, {B(2, k)}};
        const auto y = S.spmv(x);
        ref(0, k) = y(0, 0);
        ref(1, k) = y(1, 0);
    }
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double expected = 0.0;
            for (size_t t = 0; t < 3; ++t) {
                expected += dense(i, t) * B(t, j);
            }
            EXPECT_NEAR(ref(i, j), expected, 1e-12);
        }
    }
}

TEST(SparseSpmvTest, nnz_count) {
    Sparse<double> S(4, 4, {0, 1, 2, 3}, {0, 1, 2, 3}, {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(S.nnz(), 4u);
}

TEST(SparseSpmvTest, sparsity_ratio) {
    Sparse<double> S(10, 10, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    const double sparsity = 1.0 - static_cast<double>(S.nnz()) / static_cast<double>(S.rows() * S.cols());
    EXPECT_NEAR(sparsity, 0.9, 1e-12);
}

TEST(SparseSpmvTest, spmv_transpose) {
    Sparse<double> S(2, 3, {0, 0, 1}, {0, 2, 1}, {1.0, 2.0, 3.0});
    const Sparse<double> ST = transpose_sparse(S);
    ColMatrix<double> x{{4.0}, {5.0}};
    const auto y = ST.spmv(x);
    const auto ref = dense_matvec(ST.to_dense(), x);
    for (size_t i = 0; i < ST.cols(); ++i) {
        EXPECT_NEAR(y(i, 0), ref(i, 0), 1e-12);
    }
}

TEST(SparseSpmvTest, from_dense_roundtrip) {
    ColMatrix<double> A{{1.0, 0.0, 2.0}, {0.0, 3.0, 0.0}, {4.0, 0.0, 5.0}};
    const Sparse<double> S = sparse_from_dense(A);
    const auto rebuilt = S.to_dense();
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            EXPECT_NEAR(rebuilt(i, j), A(i, j), 1e-12);
        }
    }
}

TEST(SparseSpmvTest, empty_sparse) {
    Sparse<double> S(0, 0);
    EXPECT_EQ(S.rows(), 0u);
    EXPECT_EQ(S.cols(), 0u);
    EXPECT_EQ(S.nnz(), 0u);
    const ColMatrix<double> x;
    const auto y = S.spmv(x);
    EXPECT_EQ(y.rows(), 0u);
}
