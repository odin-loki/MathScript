#include <cmath>
#include <gtest/gtest.h>

#include "ms/linalg/linalg.hpp"

using namespace ms;
using DMatrix = ColMatrix<double>;

// ---------------------------------------------------------------------------
// LDL decomposition
// ---------------------------------------------------------------------------

TEST(LdlTest, spd_2x2_positive_diagonal) {
    // [4, 2; 2, 3] is SPD; D is stored as column vector (n x 1)
    DMatrix A(2, 2);
    A(0, 0) = 4; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 3;
    const auto ldl_res = ldl(A);
    ASSERT_TRUE(ldl_res.has_value());
    // D is a column vector: D(i, 0) is the i-th diagonal element
    EXPECT_GT(ldl_res->D(0, 0), 0.0);
    EXPECT_GT(ldl_res->D(1, 0), 0.0);
}

TEST(LdlTest, ldl_3x3_l_unit_triangular) {
    // Verify L has ones on the diagonal
    DMatrix A{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}};
    const auto ldl_res = ldl(A);
    ASSERT_TRUE(ldl_res.has_value());
    EXPECT_EQ(ldl_res->L.rows(), 3u);
    EXPECT_NEAR(ldl_res->L(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(ldl_res->L(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(ldl_res->L(2, 2), 1.0, 1e-12);
}

TEST(LdlTest, nonsymmetric_does_not_crash) {
    DMatrix A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;
    const auto ldl_res = ldl(A);
    (void)ldl_res; // result validity is implementation-defined; must not crash
}

// ---------------------------------------------------------------------------
// Hessenberg reduction
// ---------------------------------------------------------------------------

TEST(HessTest, output_shape_3x3) {
    DMatrix A(3, 3);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            A(i, j) = static_cast<double>(i * 3 + j + 1);
    const auto H = hess(A);
    ASSERT_TRUE(H.has_value());
    EXPECT_EQ(H->rows(), 3u);
    EXPECT_EQ(H->cols(), 3u);
}

TEST(HessTest, diagonal_matrix_preserved) {
    // A diagonal matrix is already Hessenberg; diagonal entries must be unchanged.
    DMatrix A = eye<double>(3);
    A(0, 0) = 3.0; A(1, 1) = 2.0; A(2, 2) = 1.0;
    const auto H = hess(A);
    ASSERT_TRUE(H.has_value());
    EXPECT_NEAR((*H)(0, 0), 3.0, 1e-9);
    EXPECT_NEAR((*H)(1, 1), 2.0, 1e-9);
    EXPECT_NEAR((*H)(2, 2), 1.0, 1e-9);
}

TEST(HessTest, subdiagonal_zero_below_first) {
    // H is upper-Hessenberg: H(i, j) == 0 for i > j + 1
    DMatrix A{{2, 1, 3}, {1, 4, 2}, {3, 2, 6}};
    const auto H = hess(A);
    ASSERT_TRUE(H.has_value());
    EXPECT_NEAR((*H)(2, 0), 0.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Bidiagonalization
// ---------------------------------------------------------------------------

TEST(BidiagTest, square_3x3_shape) {
    DMatrix A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 9;
    const auto bd = bidiag(A);
    ASSERT_TRUE(bd.has_value());
    EXPECT_EQ(bd->B.rows(), 3u);
    EXPECT_EQ(bd->B.cols(), 3u);
}

TEST(BidiagTest, tall_3x2_shape) {
    DMatrix A{{1, 2}, {3, 4}, {5, 6}};
    const auto bd = bidiag(A);
    ASSERT_TRUE(bd.has_value());
    EXPECT_EQ(bd->B.rows(), 3u);
    EXPECT_EQ(bd->B.cols(), 2u);
}

// ---------------------------------------------------------------------------
// General eigenvalue
// ---------------------------------------------------------------------------

TEST(EigExtTest, diagonal_2x2_eigenvalues) {
    DMatrix A(2, 2);
    A(0, 0) = 3.0; A(0, 1) = 0.0;
    A(1, 0) = 0.0; A(1, 1) = 1.0;
    const auto e = eig(A);
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->values.rows(), 2u);
    // Eigenvalues of a diagonal matrix are the diagonal entries
    EXPECT_NEAR(e->values(0, 0), 3.0, 1e-9);
    EXPECT_NEAR(e->values(1, 0), 1.0, 1e-9);
}

TEST(EigExtTest, identity_3x3_all_ones) {
    const DMatrix A = eye<double>(3);
    const auto e = eig(A);
    ASSERT_TRUE(e.has_value());
    EXPECT_EQ(e->values.rows(), 3u);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(e->values(i, 0), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Least-squares
// ---------------------------------------------------------------------------

TEST(LsqExtTest, overdetermined_3x2) {
    // 3-equation, 2-unknown system
    DMatrix A(3, 2);
    A(0, 0) = 1; A(0, 1) = 0;
    A(1, 0) = 0; A(1, 1) = 1;
    A(2, 0) = 1; A(2, 1) = 1;
    DMatrix b(3, 1);
    b(0, 0) = 1; b(1, 0) = 2; b(2, 0) = 3;
    const auto x = lsq(A, b);
    ASSERT_TRUE(x.has_value());
    EXPECT_EQ(x->rows(), 2u);
    EXPECT_EQ(x->cols(), 1u);
}

TEST(LsqExtTest, consistent_2x2_matches_solve) {
    // For a square, full-rank system lsq should give the exact solution
    DMatrix A{{2, 1}, {1, 3}};
    DMatrix b(2, 1);
    b(0, 0) = 5; b(1, 0) = 10;
    const auto x_lsq = lsq(A, b);
    const auto x_sol = solve(A, b);
    ASSERT_TRUE(x_lsq.has_value());
    ASSERT_TRUE(x_sol.has_value());
    EXPECT_NEAR((*x_lsq)(0, 0), (*x_sol)(0, 0), 1e-9);
    EXPECT_NEAR((*x_lsq)(1, 0), (*x_sol)(1, 0), 1e-9);
}
