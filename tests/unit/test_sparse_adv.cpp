// MathScript: Advanced Sparse Matrix Tests
// Tests for Sparse class: construction, nnz, to_dense, spmv with larger matrices

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "ms/core/sparse.hpp"
#include "ms/core/matrix.hpp"

using namespace ms;
using DMatrix = Matrix<double>;

// ---------------------------------------------------------------------------
// Construction and basic properties
// ---------------------------------------------------------------------------

TEST(SparseAdv, DefaultConstruct_EmptyMatrix) {
    Sparse<double> S(4, 4);
    EXPECT_EQ(S.rows(), 4u);
    EXPECT_EQ(S.cols(), 4u);
    EXPECT_EQ(S.nnz(), 0u);
}

TEST(SparseAdv, Construct_WithData_NNZ) {
    // 3x3 identity in COO format
    std::vector<size_t> row = {0, 1, 2};
    std::vector<size_t> col = {0, 1, 2};
    std::vector<double> val = {1.0, 1.0, 1.0};
    Sparse<double> S(3, 3, row, col, val);
    EXPECT_EQ(S.nnz(), 3u);
    EXPECT_EQ(S.rows(), 3u);
    EXPECT_EQ(S.cols(), 3u);
}

TEST(SparseAdv, Construct_Single_Entry) {
    std::vector<size_t> row = {1};
    std::vector<size_t> col = {2};
    std::vector<double> val = {5.0};
    Sparse<double> S(4, 5, row, col, val);
    EXPECT_EQ(S.nnz(), 1u);
}

TEST(SparseAdv, Construct_Multiple_Entries_In_Row) {
    std::vector<size_t> row = {0, 0, 1, 2};
    std::vector<size_t> col = {0, 2, 1, 2};
    std::vector<double> val = {1.0, 2.0, 3.0, 4.0};
    Sparse<double> S(3, 3, row, col, val);
    EXPECT_EQ(S.nnz(), 4u);
}

// ---------------------------------------------------------------------------
// to_dense() conversion
// ---------------------------------------------------------------------------

TEST(SparseAdv, ToDense_Identity_3x3) {
    std::vector<size_t> row = {0, 1, 2};
    std::vector<size_t> col = {0, 1, 2};
    std::vector<double> val = {1.0, 1.0, 1.0};
    Sparse<double> S(3, 3, row, col, val);
    auto D = S.to_dense();
    EXPECT_EQ(D.rows(), 3u);
    EXPECT_EQ(D.cols(), 3u);
    // Diagonal should be 1
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(D(i, i), 1.0, 1e-10);
    }
    // Off-diagonal should be 0
    EXPECT_NEAR(D(0, 1), 0.0, 1e-10);
    EXPECT_NEAR(D(1, 2), 0.0, 1e-10);
}

TEST(SparseAdv, ToDense_Empty_AllZeros) {
    Sparse<double> S(3, 4);
    auto D = S.to_dense();
    EXPECT_EQ(D.rows(), 3u);
    EXPECT_EQ(D.cols(), 4u);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(D(i, j), 0.0, 1e-10);
}

TEST(SparseAdv, ToDense_SingleEntry) {
    std::vector<size_t> row = {1};
    std::vector<size_t> col = {2};
    std::vector<double> val = {7.5};
    Sparse<double> S(3, 4, row, col, val);
    auto D = S.to_dense();
    EXPECT_NEAR(D(1, 2), 7.5, 1e-10);
    EXPECT_NEAR(D(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(D(2, 3), 0.0, 1e-10);
}

TEST(SparseAdv, ToDense_OffDiagonal_Matrix) {
    std::vector<size_t> row = {0, 1, 1};
    std::vector<size_t> col = {1, 0, 2};
    std::vector<double> val = {2.0, 3.0, 4.0};
    Sparse<double> S(3, 3, row, col, val);
    auto D = S.to_dense();
    EXPECT_NEAR(D(0, 1), 2.0, 1e-10);
    EXPECT_NEAR(D(1, 0), 3.0, 1e-10);
    EXPECT_NEAR(D(1, 2), 4.0, 1e-10);
    EXPECT_NEAR(D(0, 0), 0.0, 1e-10);
}

// ---------------------------------------------------------------------------
// spmv() - sparse matrix-vector multiplication
// ---------------------------------------------------------------------------

TEST(SparseAdv, SPMV_Identity_TimesVector) {
    std::vector<size_t> row = {0, 1, 2};
    std::vector<size_t> col = {0, 1, 2};
    std::vector<double> val = {1.0, 1.0, 1.0};
    Sparse<double> S(3, 3, row, col, val);

    DMatrix x({{2.0}, {3.0}, {5.0}});
    auto result = S.spmv(x);
    EXPECT_EQ(result.rows(), 3u);
    EXPECT_NEAR(result(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(result(1, 0), 3.0, 1e-10);
    EXPECT_NEAR(result(2, 0), 5.0, 1e-10);
}

TEST(SparseAdv, SPMV_DiagonalScaling) {
    // Diagonal matrix scales vector entries
    std::vector<size_t> row = {0, 1, 2};
    std::vector<size_t> col = {0, 1, 2};
    std::vector<double> val = {2.0, 3.0, 4.0};
    Sparse<double> S(3, 3, row, col, val);

    DMatrix x({{1.0}, {1.0}, {1.0}});
    auto result = S.spmv(x);
    EXPECT_NEAR(result(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(result(1, 0), 3.0, 1e-10);
    EXPECT_NEAR(result(2, 0), 4.0, 1e-10);
}

TEST(SparseAdv, SPMV_Tridiagonal_System) {
    // Tridiagonal [[2,-1,0],[-1,2,-1],[0,-1,2]] * [1,1,1] = [1,0,1]
    std::vector<size_t> row = {0, 0, 1, 1, 1, 2, 2};
    std::vector<size_t> col = {0, 1, 0, 1, 2, 1, 2};
    std::vector<double> val = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
    Sparse<double> S(3, 3, row, col, val);

    DMatrix x({{1.0}, {1.0}, {1.0}});
    auto result = S.spmv(x);
    EXPECT_NEAR(result(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(result(1, 0), 0.0, 1e-10);
    EXPECT_NEAR(result(2, 0), 1.0, 1e-10);
}

TEST(SparseAdv, SPMV_ZeroVector) {
    std::vector<size_t> row = {0, 1};
    std::vector<size_t> col = {0, 1};
    std::vector<double> val = {5.0, 7.0};
    Sparse<double> S(2, 2, row, col, val);

    DMatrix x({{0.0}, {0.0}});
    auto result = S.spmv(x);
    EXPECT_NEAR(result(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(result(1, 0), 0.0, 1e-10);
}

TEST(SparseAdv, SPMV_LargerMatrix_5x5) {
    // 5x5 diagonal matrix
    std::vector<size_t> rows_v = {0, 1, 2, 3, 4};
    std::vector<size_t> cols_v = {0, 1, 2, 3, 4};
    std::vector<double> vals_v = {1.0, 2.0, 3.0, 4.0, 5.0};
    Sparse<double> S(5, 5, rows_v, cols_v, vals_v);

    DMatrix x({{1.0}, {1.0}, {1.0}, {1.0}, {1.0}});
    auto result = S.spmv(x);
    EXPECT_EQ(result.rows(), 5u);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(result(i, 0), static_cast<double>(i + 1), 1e-10);
    }
}

TEST(SparseAdv, SPMV_EmptyMatrix_ZeroOutput) {
    Sparse<double> S(3, 3);
    DMatrix x({{1.0}, {2.0}, {3.0}});
    auto result = S.spmv(x);
    EXPECT_EQ(result.rows(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(result(i, 0), 0.0, 1e-10);
    }
}

TEST(SparseAdv, SPMV_Rectangular_2x3) {
    // 2x3 sparse matrix * 3-vector
    std::vector<size_t> row = {0, 1};
    std::vector<size_t> col = {0, 2};
    std::vector<double> val = {2.0, 3.0};
    Sparse<double> S(2, 3, row, col, val);

    DMatrix x({{1.0}, {0.0}, {4.0}});
    auto result = S.spmv(x);
    EXPECT_EQ(result.rows(), 2u);
    EXPECT_NEAR(result(0, 0), 2.0, 1e-10);   // 2*1
    EXPECT_NEAR(result(1, 0), 12.0, 1e-10);  // 3*4
}
