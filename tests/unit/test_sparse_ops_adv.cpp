// MathScript: Advanced Sparse Matrix Operation Tests (Wave 47)
// Tests for spmv with non-square and larger matrices, various sparsity patterns

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/core/sparse.hpp"
#include "ms/core/matrix.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// Sparse matrix spmv: rectangular 3x4
// ---------------------------------------------------------------------------

TEST(SparseOpsAdv, SPMV_Rect_3x4_AllOnes) {
    // 3x4: (0,0)=1, (0,1)=2, (0,2)=3, (0,3)=4 | (1,1)=1, (1,3)=1 | (2,0)=2, (2,2)=2
    std::vector<size_t> r = {0,0,0,0, 1,1, 2,2};
    std::vector<size_t> c = {0,1,2,3, 1,3, 0,2};
    std::vector<double> v = {1,2,3,4, 1,1, 2,2};
    Sparse<double> A(3, 4, r, c, v);
    DMatrix x({{1.0},{1.0},{1.0},{1.0}});
    auto y = A.spmv(x);
    EXPECT_EQ(y.rows(), 3u);
    EXPECT_NEAR(y(0,0), 10.0, 1e-10);  // 1+2+3+4
    EXPECT_NEAR(y(1,0), 2.0,  1e-10);  // 1+1
    EXPECT_NEAR(y(2,0), 4.0,  1e-10);  // 2+2
}

TEST(SparseOpsAdv, SPMV_Rect_4x3_AllFinite) {
    std::vector<size_t> r = {0,1,2,3};
    std::vector<size_t> c = {0,1,2,0};
    std::vector<double> v = {1.0,1.0,1.0,1.0};
    Sparse<double> A(4, 3, r, c, v);
    DMatrix x({{2.0},{3.0},{4.0}});
    auto y = A.spmv(x);
    ASSERT_EQ(y.rows(), 4u);
    for (size_t i = 0; i < y.rows(); ++i) EXPECT_TRUE(std::isfinite(y(i,0)));
}

// ---------------------------------------------------------------------------
// Larger 5x5 diagonal
// ---------------------------------------------------------------------------

TEST(SparseOpsAdv, SPMV_5x5_Diagonal) {
    size_t N = 5;
    std::vector<size_t> ri, ci;
    std::vector<double> vi;
    for (size_t i = 0; i < N; ++i) {
        ri.push_back(i); ci.push_back(i); vi.push_back(static_cast<double>(i + 1));
    }
    Sparse<double> A(N, N, ri, ci, vi);
    DMatrix x(N, 1, 2.0);
    auto y = A.spmv(x);
    ASSERT_EQ(y.rows(), N);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(y(i,0), static_cast<double>((i + 1) * 2), 1e-10);
}

// ---------------------------------------------------------------------------
// 6x6 tridiagonal
// ---------------------------------------------------------------------------

TEST(SparseOpsAdv, SPMV_6x6_Tridiagonal) {
    size_t N = 6;
    std::vector<size_t> ri, ci;
    std::vector<double> vi;
    for (size_t i = 0; i < N; ++i) {
        ri.push_back(i); ci.push_back(i); vi.push_back(2.0);
        if (i > 0) { ri.push_back(i); ci.push_back(i-1); vi.push_back(-1.0); }
        if (i + 1 < N) { ri.push_back(i); ci.push_back(i+1); vi.push_back(-1.0); }
    }
    Sparse<double> A(N, N, ri, ci, vi);
    DMatrix x(N, 1, 1.0);
    auto y = A.spmv(x);
    ASSERT_EQ(y.rows(), N);
    EXPECT_NEAR(y(0,0), 1.0, 1e-10);
    for (size_t i = 1; i + 1 < N; ++i)
        EXPECT_NEAR(y(i,0), 0.0, 1e-10);
    EXPECT_NEAR(y(N-1,0), 1.0, 1e-10);
}

// ---------------------------------------------------------------------------
// Zero vector input
// ---------------------------------------------------------------------------

TEST(SparseOpsAdv, SPMV_ZeroVector_IsZero) {
    size_t N = 4;
    std::vector<size_t> r = {0, 1, 2, 3};
    std::vector<size_t> c = {0, 1, 2, 3};
    std::vector<double> v = {5, 5, 5, 5};
    Sparse<double> A(N, N, r, c, v);
    DMatrix x(N, 1, 0.0);
    auto y = A.spmv(x);
    for (size_t i = 0; i < N; ++i) EXPECT_NEAR(y(i,0), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// NNZ properties
// ---------------------------------------------------------------------------

TEST(SparseOpsAdv, NNZ_MatchesInputCount) {
    std::vector<size_t> r = {0, 1, 2};
    std::vector<size_t> c = {0, 1, 2};
    std::vector<double> v = {1.0, 2.0, 3.0};
    Sparse<double> A(5, 5, r, c, v);
    EXPECT_EQ(A.nnz(), 3u);
}

TEST(SparseOpsAdv, NNZ_Empty_IsZero) {
    Sparse<double> A(5, 5, {}, {}, {});
    EXPECT_EQ(A.nnz(), 0u);
}

TEST(SparseOpsAdv, NNZ_Full_3x3) {
    size_t N = 3;
    std::vector<size_t> r, c;
    std::vector<double> v;
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j) {
            r.push_back(i); c.push_back(j); v.push_back(1.0);
        }
    Sparse<double> A(N, N, r, c, v);
    EXPECT_EQ(A.nnz(), N * N);
}

// ---------------------------------------------------------------------------
// to_dense: shape and values
// ---------------------------------------------------------------------------

TEST(SparseOpsAdv, ToDense_Shape_3x4) {
    Sparse<double> A(3, 4, {0,1}, {0,1}, {1.0,1.0});
    auto D = A.to_dense();
    EXPECT_EQ(D.rows(), 3u);
    EXPECT_EQ(D.cols(), 4u);
}

TEST(SparseOpsAdv, ToDense_CorrectValues_OffDiag) {
    // (0,2)=5, (1,0)=3, (2,1)=7
    std::vector<size_t> r = {0, 1, 2};
    std::vector<size_t> c = {2, 0, 1};
    std::vector<double> v = {5.0, 3.0, 7.0};
    Sparse<double> A(3, 3, r, c, v);
    auto D = A.to_dense();
    EXPECT_NEAR(D(0, 2), 5.0, 1e-10);
    EXPECT_NEAR(D(1, 0), 3.0, 1e-10);
    EXPECT_NEAR(D(2, 1), 7.0, 1e-10);
    EXPECT_NEAR(D(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(D(1, 1), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// spmv linearity: A*(alpha*x) = alpha * A*x
// ---------------------------------------------------------------------------

TEST(SparseOpsAdv, SPMV_Linearity) {
    size_t N = 4;
    std::vector<size_t> ri = {0,1,2,3,0,1};
    std::vector<size_t> ci = {0,1,2,3,1,2};
    std::vector<double> vi = {2,3,4,5,1,1};
    Sparse<double> A(N, N, ri, ci, vi);

    DMatrix x({{1.0},{2.0},{3.0},{4.0}});
    double alpha = 3.0;
    DMatrix ax({{alpha*1.0},{alpha*2.0},{alpha*3.0},{alpha*4.0}});

    auto y1 = A.spmv(x);
    auto y2 = A.spmv(ax);

    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(y2(i,0), alpha * y1(i,0), 1e-9);
}
