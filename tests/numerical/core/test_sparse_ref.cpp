// MathScript Sparse Matrix Numerical Reference Tests
// Covers Sparse construction, nnz, to_dense, spmv operations

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "ms/core/matrix.hpp"
#include "ms/core/sparse.hpp"
#include "ms/core/operations.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;
using SpMatrix = Sparse<double>;

// ---------------------------------------------------------------------------
// Sparse matrix construction
// ---------------------------------------------------------------------------

TEST(SparseRef, DefaultConstruct_ZeroNNZ) {
    const SpMatrix S(3, 4);
    EXPECT_EQ(S.rows(), 3u);
    EXPECT_EQ(S.cols(), 4u);
    EXPECT_EQ(S.nnz(), 0u);
}

TEST(SparseRef, CoordConstruct_KnownNNZ) {
    // Diagonal 3x3: (0,0)=1, (1,1)=2, (2,2)=3
    const SpMatrix S(3, 3,
        {0, 1, 2},     // row
        {0, 1, 2},     // col
        {1.0, 2.0, 3.0});  // values
    EXPECT_EQ(S.nnz(), 3u);
    EXPECT_EQ(S.rows(), 3u);
    EXPECT_EQ(S.cols(), 3u);
}

TEST(SparseRef, CoordConstruct_SingleEntry) {
    const SpMatrix S(2, 2, {0}, {1}, {5.0});
    EXPECT_EQ(S.nnz(), 1u);
}

// ---------------------------------------------------------------------------
// to_dense
// ---------------------------------------------------------------------------

TEST(SparseRef, ToDense_ZeroSparse_IsZeroMatrix) {
    const SpMatrix S(3, 3);
    const DMatrix D = S.to_dense();
    ASSERT_EQ(D.rows(), 3u);
    ASSERT_EQ(D.cols(), 3u);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(D(i, j), 0.0, 1e-15);
}

TEST(SparseRef, ToDense_Diagonal) {
    const SpMatrix S(3, 3, {0,1,2}, {0,1,2}, {1.0,2.0,3.0});
    const DMatrix D = S.to_dense();
    EXPECT_NEAR(D(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(D(1, 1), 2.0, 1e-12);
    EXPECT_NEAR(D(2, 2), 3.0, 1e-12);
    EXPECT_NEAR(D(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(D(1, 0), 0.0, 1e-12);
}

TEST(SparseRef, ToDense_OffDiagonal) {
    // 2x2: (0,1)=7, (1,0)=3
    const SpMatrix S(2, 2, {0, 1}, {1, 0}, {7.0, 3.0});
    const DMatrix D = S.to_dense();
    EXPECT_NEAR(D(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(D(0, 1), 7.0, 1e-12);
    EXPECT_NEAR(D(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(D(1, 1), 0.0, 1e-12);
}

TEST(SparseRef, ToDense_AllFinite) {
    const SpMatrix S(4, 4, {0,1,2,3}, {0,1,2,3}, {1.0,2.0,3.0,4.0});
    const DMatrix D = S.to_dense();
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_TRUE(std::isfinite(D(i, j)));
}

// ---------------------------------------------------------------------------
// spmv: sparse matrix-vector multiplication
// ---------------------------------------------------------------------------

TEST(SparseRef, SpMV_Identity_Times_Vector) {
    // I * x = x
    const SpMatrix I(3, 3, {0,1,2}, {0,1,2}, {1.0,1.0,1.0});
    DMatrix x(3, 1);
    x(0,0)=1.0; x(1,0)=2.0; x(2,0)=3.0;
    const DMatrix result = I.spmv(x);
    ASSERT_EQ(result.rows(), 3u);
    ASSERT_EQ(result.cols(), 1u);
    EXPECT_NEAR(result(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(result(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(result(2, 0), 3.0, 1e-12);
}

TEST(SparseRef, SpMV_DiagonalScaling) {
    // D = diag(2,3,4), x = [1,1,1] => D*x = [2,3,4]
    const SpMatrix D(3, 3, {0,1,2}, {0,1,2}, {2.0,3.0,4.0});
    DMatrix x(3, 1);
    x(0,0)=1.0; x(1,0)=1.0; x(2,0)=1.0;
    const DMatrix result = D.spmv(x);
    EXPECT_NEAR(result(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(result(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(result(2, 0), 4.0, 1e-12);
}

TEST(SparseRef, SpMV_ZeroSparse_IsZero) {
    const SpMatrix Z(3, 3);
    DMatrix x(3, 1);
    x(0,0)=1.0; x(1,0)=2.0; x(2,0)=3.0;
    const DMatrix result = Z.spmv(x);
    ASSERT_EQ(result.rows(), 3u);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(result(i, 0), 0.0, 1e-15);
}

TEST(SparseRef, SpMV_Known_2x2) {
    // A = [[1,2],[3,4]], x = [1,-1] => [1*1+2*(-1), 3*1+4*(-1)] = [-1, -1]
    const SpMatrix A(2, 2, {0,0,1,1}, {0,1,0,1}, {1.0,2.0,3.0,4.0});
    DMatrix x(2, 1);
    x(0,0)=1.0; x(1,0)=-1.0;
    const DMatrix result = A.spmv(x);
    EXPECT_NEAR(result(0, 0), -1.0, 1e-12);
    EXPECT_NEAR(result(1, 0), -1.0, 1e-12);
}

TEST(SparseRef, SpMV_Agrees_With_Dense_MatMul) {
    // Build sparse and dense versions of same matrix, compare spmv vs matmul
    const SpMatrix S(3, 3,
        {0,0,1,2,2},
        {0,2,1,0,2},
        {1.0,3.0,5.0,2.0,4.0});
    const DMatrix D = S.to_dense();
    DMatrix x(3, 1);
    x(0,0)=2.0; x(1,0)=-1.0; x(2,0)=3.0;
    const DMatrix sp_result = S.spmv(x);
    const auto dense_result = matmul(D, x);
    ASSERT_TRUE(dense_result.has_value());
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(sp_result(i, 0), (*dense_result)(i, 0), 1e-10);
}

// ---------------------------------------------------------------------------
// Sparsity properties
// ---------------------------------------------------------------------------

TEST(SparseRef, Sparsity_IsNNZ_Divided_By_Total) {
    // nnz=3 out of 3x3=9 => sparsity 1/3
    const SpMatrix S(3, 3, {0,1,2}, {0,1,2}, {1.0,2.0,3.0});
    const double expected_density = 3.0 / 9.0;
    EXPECT_NEAR(static_cast<double>(S.nnz()) / (S.rows() * S.cols()), expected_density, 1e-12);
}

TEST(SparseRef, DenseMatrixEquality_AfterRoundtrip) {
    // Build dense, create sparse from known entries, convert back, compare
    const SpMatrix S(2, 2, {0,1}, {0,1}, {7.0, 11.0});
    const DMatrix D = S.to_dense();
    EXPECT_NEAR(D(0,0), 7.0, 1e-12);
    EXPECT_NEAR(D(1,1), 11.0, 1e-12);
    EXPECT_NEAR(D(0,1), 0.0, 1e-12);
    EXPECT_NEAR(D(1,0), 0.0, 1e-12);
}
