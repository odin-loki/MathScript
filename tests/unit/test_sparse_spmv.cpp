#include <gtest/gtest.h>
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
